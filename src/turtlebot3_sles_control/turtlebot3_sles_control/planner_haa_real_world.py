#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped

from tf_transformations import euler_from_quaternion
import tf2_ros
import os
import atexit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import time
from datetime import datetime
from scipy import ndimage

try:
    from shapely.geometry import LineString
    _HAS_SHAPELY = True
except Exception:
    LineString = None
    _HAS_SHAPELY = False

# MPPI Implementation from test_mppi_map.py
class MPPI:
    def __init__(self, sigma: float = 0.1, temperature: float = 0.05, 
                 num_nodes: int = 20, num_rollouts: int = 100, 
                 use_noise_ramp: bool = False, noise_ramp: float = 1.0, nu: int = 2,
                 a_limit: float = 0.25, alpha_limit: float = 0.5,
                 v_limit_haa: float = 0.14, v_limit_hpa: float = 0.2,
                 omega_limit_haa: float = 0.9, omega_limit_hpa: float = 1.0,
                 dt: float = 0.1):
        self.sigma = sigma # Control noise standard deviation
        self.temperature = temperature # Temperature for softmax weighting
        self.num_nodes = num_nodes # Planning horizon
        self.num_rollouts = num_rollouts # More samples to ensure valid trajectories
        self.use_noise_ramp = use_noise_ramp # Whether to use noise ramp
        self.noise_ramp = noise_ramp # Noise ramp factor
        self.nu = nu # Number of control inputs
        self.a_limit = a_limit
        self.alpha_limit = alpha_limit
        self.v_limit_haa = v_limit_haa
        self.v_limit_hpa = v_limit_hpa
        self.omega_limit_haa = omega_limit_haa
        self.omega_limit_hpa = omega_limit_hpa
        #self.w_limit = omega_limit
        self.dt = dt
    
    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample control knots with adaptive noise based on control scale."""
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        _sigma = self.sigma

        # Adaptive noise: scale by control magnitude
        control_magnitude = np.abs(nominal_knots)
        
        # Define control_magnitude_2 with same shape as control_magnitude
        # First column all equals 0.1, second column all equals 0.5
        control_magnitude_2 = np.ones_like(control_magnitude)
        control_magnitude_2[:, 0] = 0.2  # Set first column to 0.4
        control_magnitude_2[:, 1] = 0.4  # Set second column to 0.4
        
        # Avoid division by zero and ensure minimum noise
        min_noise_scale = 0.1 * _sigma
        #adaptive_sigma = np.maximum(control_magnitude * _sigma, min_noise_scale)
        adaptive_sigma = np.maximum(control_magnitude_2 * _sigma, min_noise_scale)
        
        if self.use_noise_ramp:
            # Create a ramp that starts at 1/num_nodes and rises to 1.0 (increasing noise toward the end)
            # ramp shape: (num_nodes, 1), adaptive_sigma shape: (num_nodes, nu)
            # Broadcasting ramp * adaptive_sigma gives (num_nodes, nu), then add rollout dimension
            ramp = self.noise_ramp * np.linspace(1 / num_nodes, 1, num_nodes, endpoint=True)[:, None]  # shape: (num_nodes, 1)
            sigma = (ramp * adaptive_sigma)[None, :, :]  # shape: (1, num_nodes, nu)
        else:
            sigma = adaptive_sigma[None, :, :]  # Add rollout dimension
        
        # Ensure sigma has the correct shape for broadcasting
        sigma = np.broadcast_to(sigma, (num_rollouts - 1, num_nodes, self.nu))
        # Generate noise with adaptive scaling
        noise = np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        noised_knots = nominal_knots[None, :, :] + sigma * noise
        
        
        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update nominal knots with rewards."""
        costs = -rewards
        beta = np.min(costs) # Minimum cost
        _weights = np.exp(-(costs - beta) / self.temperature) # Softmax weighting
        weights = _weights / np.sum(_weights) # Normalize weights
        nominal_knots = np.sum(weights[:, None, None] * sampled_knots, axis=0) # Update nominal knots
        return nominal_knots

@dataclass
class RobotState:
    """Turtlebot state: [x, y, theta, v, w]"""
    x: float      # x position (m)
    y: float      # y position (m)
    theta: float  # orientation (rad)
    v: float      # linear velocity (m/s)
    w: float      # angular velocity (rad/s)

class TurtlebotDynamics:
    """Turtlebot dynamics with acceleration control."""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.nu = 2 # Number of control inputs
        self.nx = 5 # Number of state variables
    
    def step(self, state: RobotState, control: np.ndarray) -> RobotState:
        """Step dynamics: control = [a, alpha] (accelerations)"""
        a, alpha = control  # linear and angular acceleration
        
        # Update velocities
        new_v = state.v + a * self.dt
        new_w = state.w + alpha * self.dt
        
        # Update position and orientation
        new_x = state.x + state.v * np.cos(state.theta) * self.dt
        new_y = state.y + state.v * np.sin(state.theta) * self.dt
        new_theta = state.theta + state.w * self.dt
        
        return RobotState(new_x, new_y, new_theta, new_v, new_w)

class OccupancyGridMap:
    """Occupancy grid map for collision checking."""
    
    def __init__(self, width: float = 4.0, height: float = 4.0, resolution: float = 0.02):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # Initialize empty map (all free space)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        
        # World coordinates
        self.x_min = -width / 2
        self.y_min = -height / 2
        self.x_max = width / 2
        self.y_max = height / 2
        
        # Goal and start positions
        self.goal = None
        self.start = None
    
    def world_to_grid(self, x: float, y: float) -> tuple:
        """Convert world coordinates to grid coordinates."""
        grid_x = int((x - self.x_min) / self.resolution)
        grid_y = int((y - self.y_min) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> tuple:
        """Convert grid coordinates to world coordinates."""
        x = self.x_min + (grid_x + 0.5) * self.resolution
        y = self.y_min + (grid_y + 0.5) * self.resolution
        return x, y
    
    def dilate_grid_new(self, robot_radius_cells):
        """Optimized dilation using distance transform - much faster for large radii."""
        obstacle_mask = self.occupancy_grid > 50
        
        # Early exit: no obstacles
        if not np.any(obstacle_mask):
            return np.zeros_like(obstacle_mask, dtype=bool)
        
        # Use distance transform method (faster than binary_dilation for large radii)
        distance_map = ndimage.distance_transform_edt(~obstacle_mask)
        dilated_grid = distance_map <= robot_radius_cells
        
        return dilated_grid
    
    def check_collision_new(self, x: np.ndarray, y: np.ndarray, robot_radius: float = 0.22) -> np.ndarray:
        """Fully vectorized collision checking using dilated occupancy grid."""
        # Convert to grid coordinates
        grid_x = ((x - self.x_min) / self.resolution).astype(int)
        grid_y = ((y - self.y_min) / self.resolution).astype(int)
        
        # Check bounds
        out_of_bounds = ((grid_x < 0) | (grid_x >= self.grid_width) | 
                        (grid_y < 0) | (grid_y >= self.grid_height))
        
        # Initialize result array
        collisions = np.zeros(len(x), dtype=bool)
        collisions[out_of_bounds] = True
        
        # Process valid positions
        valid_mask = ~out_of_bounds
        if not np.any(valid_mask):
            return collisions
        
        # Get dilated grid once
        robot_radius_cells = int(robot_radius / self.resolution)
        dilate_grid = self.dilate_grid_new(robot_radius_cells)
        
        # For valid positions, check if they collide with dilated obstacles
        valid_x = grid_x[valid_mask]
        valid_y = grid_y[valid_mask]
        
        # Check collisions: if dilate_grid[valid_y, valid_x] is True, then collision
        collisions[valid_mask] = dilate_grid[valid_y, valid_x]
        
        return collisions

    
    def get_distance_to_goal(self, x: float, y: float) -> float:
        """Get distance to goal."""
        if self.goal is None:
            return 0.0
        return np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)
    
    def visualize(self, ax=None):
        """Visualize the occupancy grid."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Convert to binary: > 50 = occupied (1), otherwise = free (0)
        binary_grid = (self.occupancy_grid > 50).astype(int)
        
        # Plot occupancy grid
        im = ax.imshow(binary_grid, 
                      origin='lower', 
                      # Use "gray_r": 0 => white (free), 1 => black (occupied)
                      cmap='gray_r',
                      extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                      vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Occupancy Probability')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Occupancy Grid Map')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def set_goal(self, x: float, y: float):
        """Set goal position."""
        self.goal = np.array([x, y])
    
    def set_start(self, x: float, y: float):
        """Set start position."""
        self.start = np.array([x, y])

class MPPIPlanner:
    """MPPI Planner for Turtlebot navigation."""

    def __init__(self, mppi: MPPI, dynamics: TurtlebotDynamics, occupancy_map: OccupancyGridMap,
                 v_min: float = -0.26, v_max: float = 0.26, w_min: float = -1.82, w_max: float = 1.82,
                 a_min: float = -1.0, a_max: float = 1.0, alpha_min: float = -1.0, alpha_max: float = 1.0,
                 smoothing_weight: float = 0.1, logger=None):
        self.mppi = mppi
        self.dynamics = dynamics
        self.occupancy_map = occupancy_map
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.a_min = a_min
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.smoothing_weight = smoothing_weight
        self._logger = logger  # rclpy logger from the parent Node, or None

        # Initialize nominal control sequence
        self.nominal_controls = np.zeros((mppi.num_nodes, mppi.nu))

        # Warm start: store previous optimal control sequence
        self.previous_controls = None

    def _log_info(self, msg: str):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(f'[MPPIPlanner/INFO] {msg}')

    def _log_warn(self, msg: str):
        if self._logger is not None:
            self._logger.warn(msg)
        else:
            print(f'[MPPIPlanner/WARN] {msg}')

    def _log_error(self, msg: str):
        if self._logger is not None:
            self._logger.error(msg)
        else:
            print(f'[MPPIPlanner/ERROR] {msg}')
    
    def update_occupancy_map(self, new_occupancy_map: OccupancyGridMap):
        """Update the occupancy map while preserving control history."""
        self.occupancy_map = new_occupancy_map
        # Keep all other state (nominal_controls, previous_controls, etc.) intact
    
    def debug_plot_no_valid_trajectories(self, start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask, title_override=None):
        """Debug function to plot trajectories when no valid ones are found."""
        try:
            num_valid = np.sum(valid_mask)
            num_total = len(sampled_controls)
            
            # Detailed analysis of why trajectories are invalid
            collision_only = 0
            linear_velocity_only = 0
            angular_velocity_only = 0
            collision_linearv = 0
            collision_angularv = 0
            control_only = 0
            other_violations = 0
            self._log_info(f"Nominal controls: {nominal_controls}")
            self._log_info(f"start state: {start_state}")
            
            for i in range(num_total):
                if valid_mask[i]:
                    continue  # Skip valid trajectories
                
                trajectory = all_trajectories[i]
                control_seq = sampled_controls[i]
                
                # Check collision
                has_collision = False
                collision_states = []
                xs = trajectory[:, 0]
                ys = trajectory[:, 1]
                collisions = self.occupancy_map.check_collision_new(xs, ys)
                if np.any(collisions):
                    has_collision = True
                    # Don't break - collect all collision states for debugging
                
                # Check velocity violations
                has_linear_velocity_violation = False
                v_violations = np.any((trajectory[:, 3] < self.v_min) | (trajectory[:, 3] > self.v_max))
                if v_violations:
                    has_linear_velocity_violation = True
                
                has_angular_velocity_violation = False
                w_violations = np.any((trajectory[:, 4] < self.w_min) | (trajectory[:, 4] > self.w_max))
                if w_violations:
                    has_angular_velocity_violation = True
                
                # Check control violations
                has_control_violation = False
                a_violations = np.any((control_seq[:, 0] < self.a_min) | (control_seq[:, 0] > self.a_max))
                alpha_violations = np.any((control_seq[:, 1] < self.alpha_min) | (control_seq[:, 1] > self.alpha_max))
                if a_violations or alpha_violations:
                    has_control_violation = True
                
                # Categorize violations with more detailed breakdown
                if has_collision and not has_linear_velocity_violation and not has_angular_velocity_violation and not has_control_violation:
                    collision_only += 1
                elif has_linear_velocity_violation and not has_collision and not has_angular_velocity_violation and not has_control_violation:
                    linear_velocity_only += 1
                elif has_angular_velocity_violation and not has_collision and not has_linear_velocity_violation and not has_control_violation:
                    angular_velocity_only += 1
                elif has_collision and has_linear_velocity_violation:
                    # collision + linear v
                    collision_linearv += 1
                elif has_collision and has_angular_velocity_violation:
                    # collision + angular v
                    collision_angularv += 1
                elif has_control_violation and not has_collision and not has_linear_velocity_violation and not has_angular_velocity_violation:
                    control_only += 1
                else:
                    other_violations += 1

            self._log_warn(f"MPPI trajectory breakdown — total={num_total}, valid={num_valid} ({num_valid/num_total*100:.1f}%)")
            self._log_warn(f"  Collision only:          {collision_only:4d} ({collision_only/num_total*100:.1f}%)")
            self._log_warn(f"  Linear-velocity only:    {linear_velocity_only:4d} ({linear_velocity_only/num_total*100:.1f}%)")
            self._log_warn(f"  Angular-velocity only:   {angular_velocity_only:4d} ({angular_velocity_only/num_total*100:.1f}%)")
            self._log_warn(f"  Collision + linear-vel:  {collision_linearv:4d} ({collision_linearv/num_total*100:.1f}%)")
            self._log_warn(f"  Collision + angular-vel: {collision_angularv:4d} ({collision_angularv/num_total*100:.1f}%)")
            self._log_warn(f"  Control only:            {control_only:4d} ({control_only/num_total*100:.1f}%)")
            self._log_warn(f"  Other:                   {other_violations:4d} ({other_violations/num_total*100:.1f}%)")
            # ── Create debug figure: left=map+trajectories, right=violation breakdown ──
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

            # ── ax1: occupancy map with inflation zones as background ──────────────
            omap = self.occupancy_map
            raw  = omap.occupancy_grid
            res  = omap.resolution
            _robot_r   = 0.15    # keep in sync with robot_radius param
            _safe_dist = 0.10    # keep in sync with plan() safe_distance
            hard_r = int(_robot_r / res)
            soft_r = int((_robot_r + _safe_dist) / res)
            obstacle_mask = raw > 10
            if np.any(obstacle_mask):
                dist_c = ndimage.distance_transform_edt(~obstacle_mask)
            else:
                dist_c = np.full(raw.shape, 9999.0)
            rgba = np.ones((*raw.shape, 4), dtype=float)
            rgba[dist_c <= soft_r] = [1.0, 1.0, 0.45, 1.0]   # soft zone: light yellow
            rgba[dist_c <= hard_r] = [1.0, 0.55, 0.10, 1.0]   # hard zone: orange
            rgba[obstacle_mask]    = [0.15, 0.15, 0.15, 1.0]  # obstacle: near-black
            ax1.imshow(rgba, extent=[omap.x_min, omap.x_max, omap.y_min, omap.y_max],
                       origin='lower', aspect='equal', zorder=0)

            # ── ax1: plot up to 120 invalid trajectories (thin red, transparent) ──
            invalid_indices = np.where(~valid_mask)[0]
            for i in invalid_indices[:120]:
                traj = all_trajectories[i]
                ax1.plot(traj[:, 0], traj[:, 1], color='red', alpha=0.18, linewidth=0.6, zorder=1)

            # ── ax1: plot up to 30 valid trajectories (thin green) ────────────────
            valid_indices = np.where(valid_mask)[0]
            for i in valid_indices[:30]:
                traj = all_trajectories[i]
                ax1.plot(traj[:, 0], traj[:, 1], color='limegreen', alpha=0.55, linewidth=0.9, zorder=2)

            # ── ax1: highlight nominal trajectory (index 0) in bold blue ──────────
            nom_traj = all_trajectories[0]
            ax1.plot(nom_traj[:, 0], nom_traj[:, 1], color='royalblue', linewidth=2.5,
                     zorder=3, label='_nolegend_')

            # ── ax1: robot position + orientation arrow ───────────────────────────
            ax1.plot(start_state.x, start_state.y, 'o', color='cyan',
                     markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            arr_len = max(0.15, _robot_r * 1.2)
            ax1.annotate('', xy=(start_state.x + arr_len * np.cos(start_state.theta),
                                  start_state.y + arr_len * np.sin(start_state.theta)),
                          xytext=(start_state.x, start_state.y),
                          arrowprops=dict(arrowstyle='->', color='cyan', lw=2.0), zorder=5)

            # ── ax1: hard-zone circle around robot (shows what MPPI discards) ─────
            ax1.add_patch(patches.Circle((start_state.x, start_state.y), _robot_r,
                                          fill=False, edgecolor='orange',
                                          linestyle='--', linewidth=1.8, zorder=4))
            ax1.add_patch(patches.Circle((start_state.x, start_state.y), _robot_r + _safe_dist,
                                          fill=False, edgecolor='gold',
                                          linestyle=':', linewidth=1.5, zorder=4))

            # ── ax1: goal marker ──────────────────────────────────────────────────
            ax1.plot(goal[0], goal[1], 'r*', markersize=16,
                     markeredgecolor='darkred', markeredgewidth=1, zorder=5)

            # ── ax1: legend with proxy artists ───────────────────────────────────
            proxy_invalid  = plt.Line2D([0], [0], color='red',       lw=1.5, alpha=0.6,  label=f'Invalid ({len(invalid_indices)}, showing ≤120)')
            proxy_valid    = plt.Line2D([0], [0], color='limegreen',  lw=1.5, alpha=0.8,  label=f'Valid ({num_valid}, showing ≤30)')
            proxy_nominal  = plt.Line2D([0], [0], color='royalblue',  lw=2.5,             label='Nominal trajectory')
            proxy_robot    = plt.Line2D([0], [0], marker='o', color='cyan', lw=0, markersize=9, label='Robot position')
            proxy_goal     = plt.Line2D([0], [0], marker='*', color='red',  lw=0, markersize=12, label='Goal')
            proxy_hard     = patches.Patch(facecolor=[1.0,0.55,0.10], label=f'Hard zone (r={_robot_r}m) — trajectories touching this are DISCARDED')
            proxy_soft     = patches.Patch(facecolor=[1.0,1.0,0.45], label=f'Soft zone (+{_safe_dist}m) — trajectories here are PENALISED')
            proxy_obs      = patches.Patch(facecolor=[0.15,0.15,0.15], label='Obstacle (raw map)')
            ax1.legend(handles=[proxy_invalid, proxy_valid, proxy_nominal,
                                  proxy_robot, proxy_goal,
                                  proxy_hard, proxy_soft, proxy_obs],
                        fontsize=8, loc='upper right')
            _plot_title = title_override if title_override else f'MPPI — No Valid Trajectories  ({num_valid}/{num_total} valid)'
            ax1.set_title(f'{_plot_title}\nnominal traj in blue; up to 120 invalid (red) + 30 valid (green)',
                           fontsize=10)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.25)

            # ── ax2: violation breakdown bar chart ────────────────────────────────
            categories  = ['Collision\nonly', 'Lin-vel\nonly', 'Ang-vel\nonly',
                           'Collision\n+lin-vel', 'Collision\n+ang-vel',
                           'Control\nonly', 'Other']
            counts      = [collision_only, linear_velocity_only, angular_velocity_only,
                           collision_linearv, collision_angularv, control_only, other_violations]
            bar_colors  = ['#d62728', '#ff7f0e', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
            bars = ax2.bar(categories, counts, color=bar_colors, edgecolor='black', linewidth=0.7)

            # Annotate bar tops with count + percentage
            for bar, cnt in zip(bars, counts):
                if cnt > 0:
                    pct = cnt / num_total * 100
                    ax2.text(bar.get_x() + bar.get_width() / 2.0,
                              bar.get_height() + num_total * 0.005,
                              f'{cnt}\n({pct:.1f}%)',
                              ha='center', va='bottom', fontsize=9)

            ax2.set_ylabel('Number of trajectories')
            ax2.set_title(f'Why trajectories were invalidated\n'
                           f'Total={num_total}  |  Valid={num_valid}  |  Invalid={num_total - num_valid}',
                           fontsize=10)
            ax2.grid(axis='y', alpha=0.4)

            # State info text box below the bar chart
            state_text = (f'Robot state at planning instant:\n'
                          f'  x = {start_state.x:.3f} m\n'
                          f'  y = {start_state.y:.3f} m\n'
                          f'  θ = {start_state.theta:.3f} rad  ({np.degrees(start_state.theta):.1f}°)\n'
                          f'  v = {start_state.v:.3f} m/s\n'
                          f'  ω = {start_state.w:.3f} rad/s\n\n'
                          f'Goal: ({goal[0]:.3f}, {goal[1]:.3f}) m\n'
                          f'Dist to goal: {np.hypot(start_state.x - goal[0], start_state.y - goal[1]):.3f} m')
            ax2.text(0.02, 0.98, state_text,
                      transform=ax2.transAxes,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                      fontsize=9, verticalalignment='top', family='monospace')
            
            plt.tight_layout()
            
            # Save the plot in the specified directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_path = os.path.join(os.getcwd(), f'mppi_no_valid_trajectories_debug_{timestamp}.png')
            plt.savefig(debug_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            self._log_info(f"MPPI debug plot saved to: {debug_path}")

        except Exception as e:
            self._log_error(f"Error in MPPI debug plot: {e}")
    
    def simulate_trajectories_vectorized(self, start_state: RobotState, control_sequences: np.ndarray) -> np.ndarray:
        """Vectorized trajectory simulation for all control sequences at once."""
        num_rollouts, horizon, nu = control_sequences.shape
        trajectories = np.zeros((num_rollouts, horizon + 1, 5))
        
        # Initialize all trajectories with start state
        trajectories[:, 0, 0] = start_state.x
        trajectories[:, 0, 1] = start_state.y
        trajectories[:, 0, 2] = start_state.theta
        trajectories[:, 0, 3] = start_state.v
        trajectories[:, 0, 4] = start_state.w
        
        # Vectorized simulation
        for i in range(horizon):
            # Current states
            x = trajectories[:, i, 0]
            y = trajectories[:, i, 1]
            theta = trajectories[:, i, 2]
            v = trajectories[:, i, 3]
            w = trajectories[:, i, 4]
            
            # Control inputs
            a = control_sequences[:, i, 0]
            alpha = control_sequences[:, i, 1]
            
            # Update velocities and clamp to hardware limits (mirrors real motor behaviour;
            # the hardware caps velocity — it does not "fail" when the limit is reached)
            new_v = np.clip(v + a * self.dynamics.dt, self.v_min, self.v_max)
            new_w = np.clip(w + alpha * self.dynamics.dt, self.w_min, self.w_max)

            # Update positions and orientation
            new_x = x + v * np.cos(theta) * self.dynamics.dt
            new_y = y + v * np.sin(theta) * self.dynamics.dt
            new_theta = theta + w * self.dynamics.dt
            
            # Store next states
            trajectories[:, i + 1, 0] = new_x
            trajectories[:, i + 1, 1] = new_y
            trajectories[:, i + 1, 2] = new_theta
            trajectories[:, i + 1, 3] = new_v
            trajectories[:, i + 1, 4] = new_w
        
        return trajectories
    
    def validate_trajectories_vectorized(self, trajectories: np.ndarray, robot_radius: float = 0.22) -> np.ndarray:
        """Vectorized trajectory validation for all trajectories at once."""
        num_rollouts, num_steps, _ = trajectories.shape
        valid_mask = np.ones(num_rollouts, dtype=bool)
        
        # Velocity constraints are now enforced by clamping inside simulate_trajectories_vectorized,
        # so no trajectory can violate them. Collision checking is the sole discard criterion.

        # Vectorized collision checking using occupancy grid
        if np.any(valid_mask):
            # Extract all positions for valid trajectories
            valid_trajectories = trajectories[valid_mask]
            all_x = valid_trajectories[:, :, 0].flatten()  # All x positions
            all_y = valid_trajectories[:, :, 1].flatten()  # All y positions
            
            # Simple collision checking for each position
            collision_results = np.zeros(len(all_x), dtype=bool)
            collision_results = self.occupancy_map.check_collision_new(all_x, all_y, robot_radius = robot_radius)
            # Reshape back to trajectory format
            collision_matrix = collision_results.reshape(valid_trajectories.shape[0], num_steps)
            
            # Check if any step in each trajectory has collision
            trajectory_collisions = np.any(collision_matrix, axis=1)
            
            # Update valid mask
            valid_indices = np.where(valid_mask)[0]
            valid_mask[valid_indices[trajectory_collisions]] = False
        
        return valid_mask
    
    def compute_rewards_vectorized(self, trajectories: np.ndarray, goal: np.ndarray, 
                                 safety_dilated_grid: np.ndarray = None, 
                                 control_sequences: np.ndarray = None) -> np.ndarray:
        """Vectorized reward computation for all trajectories at once."""
        num_rollouts, num_steps, _ = trajectories.shape
        rewards = np.zeros(num_rollouts)
        
        # Compute distance to goal for all states at once
        x = trajectories[:, :, 0]
        y = trajectories[:, :, 1]
        distances = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        
        # Sum negative distances as reward (like test_mppi_map.py)
        rewards = -np.sum(distances * 10.0, axis=1)
        
        final_x = trajectories[:, -1, 0]
        final_y = trajectories[:, -1, 1]
        terminal_distances = np.sqrt((final_x - goal[0]) ** 2 + (final_y - goal[1]) ** 2)
        # Subtract terminal error (weighted, e.g., 100.0)
        rewards -= terminal_distances * 100.0
        
        # === NEAR-OBSTACLE PENALTY ===
        # Add penalty for trajectories that get too close to obstacles
        if safety_dilated_grid is not None:
            # Check all trajectory points for near-obstacle violations
            all_x = x.flatten()  # All x positions across all trajectories and time steps
            all_y = y.flatten()  # All y positions across all trajectories and time steps
            
            # Convert world coordinates to grid coordinates
            grid_x = ((all_x - self.occupancy_map.x_min) / self.occupancy_map.resolution).astype(int)
            grid_y = ((all_y - self.occupancy_map.y_min) / self.occupancy_map.resolution).astype(int)
            
            # Check bounds
            out_of_bounds = ((grid_x < 0) | (grid_x >= self.occupancy_map.grid_width) | 
                            (grid_y < 0) | (grid_y >= self.occupancy_map.grid_height))
            
            # Initialize near-obstacle violations
            near_obstacle_violations = np.zeros(len(all_x), dtype=bool)
            
            # Check valid positions for near-obstacle violations
            valid_mask = ~out_of_bounds
            if np.any(valid_mask):
                valid_x = grid_x[valid_mask]
                valid_y = grid_y[valid_mask]
                
                # Check if positions are too close to obstacles (using pre-computed safety-dilated grid)
                near_obstacle_violations[valid_mask] = safety_dilated_grid[valid_y, valid_x]
            
            # Reshape violations back to trajectory format (num_rollouts, num_steps)
            violation_matrix = near_obstacle_violations.reshape(num_rollouts, num_steps)
            
            # Count violations per trajectory and apply penalty
            violations_per_trajectory = np.sum(violation_matrix, axis=1)  # Sum violations per trajectory
            near_obstacle_penalty = violations_per_trajectory * 5  # Per-step soft-zone penalty; higher = wider detours, lower = tighter paths
            rewards -= near_obstacle_penalty
        
        # Add control smoothing penalty if control sequences are provided
        if control_sequences is not None:
            # Compute control differences (rate of change)
            control_diff = np.diff(control_sequences, axis=1)  # Shape: (num_rollouts, num_steps-1, 2)
            
            # Compute squared control differences for smoothing penalty
            control_smoothness_penalty = np.sum(control_diff**2, axis=(1, 2))  # Sum over time and control dimensions
            
            # Add smoothing penalty to rewards (negative because we want to minimize it)
            rewards -= self.smoothing_weight * control_smoothness_penalty
        
        return rewards
    
    def plan(self, start_state: RobotState, goal: np.ndarray, debug_plot: bool = False, robot_radius: float = 0.22) -> tuple:
        """Plan using MPPI and return the control sequence and the resulting trajectory.
        
        Args:
            start_state: Initial robot state
            goal: Target position [x, y]
            debug_plot: If True, generate debug plots when no valid trajectories are found
        """
        # Start timing
        start_time = time.time()
        
        # Warm start: reuse previous solution if available
        warm_start_time = time.time()
        if self.previous_controls is not None and len(self.previous_controls) >= self.mppi.num_nodes:
            # Shift previous solution: remove first control, add zero at end
            nominal_controls = np.vstack([
                self.previous_controls[1:],  # Remove first control
                np.zeros((1, 2))            # Add zero control at end
            ])
            # if abs(self.v_max - start_state.v) < 0.05:
            #     nominal_controls[:, 0] = 0
            # if abs(start_state.v - self.v_min) < 0.05:
            #     nominal_controls[:, 0] = 0.02
            # if abs(start_state.w - self.w_min) < 0.2:
            #     nominal_controls[:, 1] = 0.0
            # if abs(start_state.w - self.w_max) < 0.2:
            #     nominal_controls[:, 1] = 0.0
        else:
            # Cold start: initialize based on goal direction
            cold_start_time = time.time()
            # Calculate angle to goal in robot frame
            goal_x = goal[0]
            goal_y = goal[1]
            
            # Vector from robot to goal in world frame
            dx_world = goal_x - start_state.x
            dy_world = goal_y - start_state.y
            
            # Transform to robot frame
            cos_theta = np.cos(start_state.theta)
            sin_theta = np.sin(start_state.theta)
            dx_robot = dx_world * cos_theta + dy_world * sin_theta
            dy_robot = -dx_world * sin_theta + dy_world * cos_theta
            
            # Calculate angle to goal in robot frame
            angle_to_goal = np.arctan2(dy_robot, dx_robot)
            
            # Initialize control sequence
            nominal_controls = np.zeros((self.mppi.num_nodes, 2))
            
            # Set linear acceleration: 0.1 * a_max (always forward)
            nominal_controls[:, 0] = 0.1 * self.a_max
            
            # Set angular acceleration based on goal direction
            if angle_to_goal < 0:  # Target in negative horizontal axis (left side)
                nominal_controls[:, 1] = -0.3 * self.alpha_max
            else:  # Target in positive horizontal axis (right side)
                nominal_controls[:, 1] = 0.3 * self.alpha_max
            
            #nominal_controls = np.zeros((self.mppi.num_nodes, 2))
        
        # MPPI iterations
        mppi_iterations_time = time.time()
        for iteration in range(1):  # 3 iterations per planning cycle for better convergence around obstacles
            iteration_start = time.time()
            
            # Sample control sequences
            sample_time = time.time()
            sampled_controls = self.mppi.sample_control_knots(nominal_controls)
            
            # Apply control limits to all sampled controls BEFORE simulation
            clip_time = time.time()
            sampled_controls[:, :, 0] = np.clip(sampled_controls[:, :, 0], self.a_min, self.a_max)
            sampled_controls[:, :, 1] = np.clip(sampled_controls[:, :, 1], self.alpha_min, self.alpha_max)
            
            # Vectorized trajectory simulation and validation
            simulation_time = time.time()
            all_trajectories = self.simulate_trajectories_vectorized(start_state, sampled_controls)
            
            validation_time = time.time()
            valid_mask = self.validate_trajectories_vectorized(all_trajectories, robot_radius = robot_radius)
            
            reward_time = time.time()
            if np.any(valid_mask):
                valid_controls = sampled_controls[valid_mask]
                valid_trajectories = all_trajectories[valid_mask]
                
                # Pre-compute safety-dilated grid for near-obstacle penalty (compute once per planning iteration)
                safe_distance = 0.050  # meters beyond robot_radius — soft penalty zone for near-obstacle cost
                total_safety_radius = robot_radius + safe_distance
                safety_radius_cells = int(total_safety_radius / self.occupancy_map.resolution)
                safety_dilated_grid = self.occupancy_map.dilate_grid_new(safety_radius_cells)
                
                valid_rewards = self.compute_rewards_vectorized(valid_trajectories, goal, safety_dilated_grid, valid_controls)
            else:
                valid_controls = np.array([])
                valid_rewards = np.array([])
            
            # If we have valid trajectories, update nominal controls
            update_time = time.time()
            if len(valid_controls) > 0:
                # Update nominal controls using only valid samples
                nominal_controls = self.mppi.update_nominal_knots(valid_controls, valid_rewards)
                
                # Apply control limits to updated nominal controls
                nominal_controls[:, 0] = np.clip(nominal_controls[:, 0], self.a_min, self.a_max)
                nominal_controls[:, 1] = np.clip(nominal_controls[:, 1], self.alpha_min, self.alpha_max)
                
                # Store optimal controls for warm start in next iteration
                self.previous_controls = nominal_controls.copy()
                
            else:
                # If no valid trajectories found, clear warm-start so next cycle cold-starts
                # toward the goal instead of repeating the same stuck nominal.
                self.previous_controls = None
                self._log_warn("No valid trajectories found!")
                if debug_plot:
                    self.debug_plot_no_valid_trajectories(start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask)
                end_time = time.time()
                return None, None
        
        # Propagate optimal controls through system dynamics to get consistent trajectory
        final_simulation_time = time.time()
        # Use vectorized method with single control sequence (add batch dimension)
        control_sequence_batch = nominal_controls.reshape(1, -1, 2)
        trajectory_batch = self.simulate_trajectories_vectorized(start_state, control_sequence_batch)
        best_trajectory = trajectory_batch[0]  # Extract single trajectory
        
        end_time = time.time()
        total_time = end_time - start_time
        return best_trajectory, nominal_controls

    def sample_debug_trajectories(self, start_state: RobotState, goal: np.ndarray, robot_radius: float):
        """Sample a fresh batch of trajectories for debug visualisation.
        Does NOT update self.previous_controls — safe to call at any time."""
        nominal = self.previous_controls.copy() if self.previous_controls is not None \
                  else np.zeros((self.mppi.num_nodes, 2))
        sampled = self.mppi.sample_control_knots(nominal)
        sampled[:, :, 0] = np.clip(sampled[:, :, 0], self.a_min, self.a_max)
        sampled[:, :, 1] = np.clip(sampled[:, :, 1], self.alpha_min, self.alpha_max)
        all_trajs  = self.simulate_trajectories_vectorized(start_state, sampled)
        valid_mask = self.validate_trajectories_vectorized(all_trajs, robot_radius)
        return sampled, nominal, all_trajs, valid_mask


class KanayamaController:
    """Kanayama/Samson-style unicycle tracking controller with integral action."""
    
    def __init__(self, kx: float = 1.0, ky: float = 1.0, kth: float = 2.0, kv: float = 1.0, kw: float = 1.0,
                 kix: float = 0.1, kiy: float = 0.1, kith: float = 0.1, max_integral: float = 1.0):
        """
        Initialize Kanayama controller with integral terms.
        
        Args:
            kx: Longitudinal position error gain
            ky: Lateral position error gain  
            kth: Orientation error gain
            kv: Linear velocity error gain
            kw: Angular velocity error gain
            kix: Longitudinal position integral gain
            kiy: Lateral position integral gain
            kith: Orientation integral gain
            max_integral: Maximum integral term to prevent windup
        """
        self.kx = kx  # Longitudinal position error gain
        self.ky = ky  # Lateral position error gain
        self.kth = kth  # Orientation error gain
        self.kv = kv  # Linear velocity error gain
        self.kw = kw  # Angular velocity error gain
        
        # Integral gains
        self.kix = kix  # Longitudinal position integral gain
        self.kiy = kiy  # Lateral position integral gain
        self.kith = kith  # Orientation integral gain
        
        # Integral windup protection
        self.max_integral = max_integral
        
        # Integral error accumulators
        self.integral_ex = 0.0  # Longitudinal position error integral
        self.integral_ey = 0.0  # Lateral position error integral
        self.integral_eth = 0.0  # Orientation error integral
        
        # Previous errors for derivative terms (if needed later)
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_eth = 0.0
        
        # Time step for integral calculation
        self.dt = 0.02  # 50Hz control loop
        
        self.v_limit_haa = 0.20
        self.omega_limit_haa = 0.9
        
        # Previous reference velocities for feedforward
        self.prev_v_ref = 0.0
        self.prev_w_ref = 0.0
        
    def compute_control(self, current_state, ref_state):
        """
        Compute control inputs using Kanayama tracking law with integral action.
        
        Args:
            current_state: [x, y, theta, v, w] - current robot state
            ref_state: [x_ref, y_ref, theta_ref, v_ref, w_ref] - reference state
            
        Returns:
            v_cmd, w_cmd: commanded linear and angular velocities
        """
        x, y, theta, v, w = current_state
        xr, yr, thr, vr, wr = ref_state
        
        # Position errors in world frame
        dx, dy = xr - x, yr - y
        
        # Transform position errors to robot frame
        ex = np.cos(theta) * dx + np.sin(theta) * dy
        ey = -np.sin(theta) * dx + np.cos(theta) * dy
        
        # Orientation error with wrapping
        eth = thr - theta
        eth = np.arctan2(np.sin(eth), np.cos(eth))  # wrap to [-pi, pi]
        
        # === INTEGRAL CONTROLLER ===
        # Update integral terms using trapezoidal rule
        self.integral_ex += (ex + self.prev_ex) * self.dt / 2.0
        self.integral_ey += (ey + self.prev_ey) * self.dt / 2.0
        self.integral_eth += (eth + self.prev_eth) * self.dt / 2.0
        
        # Integral windup protection - clamp integral terms
        self.integral_ex = np.clip(self.integral_ex, -self.max_integral, self.max_integral)
        self.integral_ey = np.clip(self.integral_ey, -self.max_integral, self.max_integral)
        self.integral_eth = np.clip(self.integral_eth, -self.max_integral, self.max_integral)
        
        # Store current errors for next iteration
        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_eth = eth
        
        # === ENHANCED KANAYAMA CONTROL LAW ===
        # Original Kanayama terms
        v_cmd = vr * np.cos(eth) + self.kx * ex + self.kv * (vr - v)
        w_cmd = wr + vr * self.ky * ey + vr * self.kth * np.sin(eth) + self.kw * (wr - w)
        
        # Add integral terms
        v_cmd += self.kix * self.integral_ex  # Longitudinal position integral
        w_cmd += self.kiy * self.integral_ey + self.kith * self.integral_eth  # Lateral and orientation integrals
        
        # Clip the control inputs using HAA limits
        v_cmd = np.clip(v_cmd, 0, self.v_limit_haa)
        w_cmd = np.clip(w_cmd, -self.omega_limit_haa, self.omega_limit_haa)
        
        return v_cmd, w_cmd
    
    def reset_integral_errors(self):
        """Reset all integral error accumulators to zero."""
        self.integral_ex = 0.0
        self.integral_ey = 0.0
        self.integral_eth = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_eth = 0.0
        #self.get_logger().info("Integral errors reset to zero")

class HAANavigationNode(Node):
    def __init__(self):
        super().__init__('HAA_only')
        # Declare all ROS2 parameters with defaults.
        self.declare_parameter('horizon_haa', 40)
        self.declare_parameter('horizon_hpa', 20)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_limit_haa', 0.26)
        self.declare_parameter('v_limit_hpa', 0.2)
        self.declare_parameter('omega_limit_haa', 1.82)
        self.declare_parameter('omega_limit_hpa', 1.0)
        self.declare_parameter('a_limit', 1.0)
        self.declare_parameter('alpha_limit', 1.0)
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('goal', '[-1.5, -1.5, 0, 0, 0]')
        self.declare_parameter('scenario_has_random_obstacles', False)
        self.declare_parameter('kx', 1.0)
        self.declare_parameter('ky', 1.0)
        self.declare_parameter('kth', 1.0)
        self.declare_parameter('kv', 1.0)
        self.declare_parameter('kw', 1.0)
        self.declare_parameter('kix', 0.1)
        self.declare_parameter('kiy', 0.1)
        self.declare_parameter('kith', 0.1)
        self.declare_parameter('max_integral', 1.0)
        self.declare_parameter('Q_diag', '[10,10,0,0,0]')
        self.declare_parameter('R_diag', '[0,0]')
        self.declare_parameter('failure_path', os.path.join(os.path.expanduser('~'), 'mppi_failure_data.npz'))
        self.declare_parameter('K_feedback', '[]')
        for i in range(1, 17):
            self.declare_parameter(f'random_x_{i}', 0.0)
            self.declare_parameter(f'random_y_{i}', 0.0)
            self.declare_parameter(f'random_size_{i}', 0.2)
            self.declare_parameter(f'random_shape_{i}', 'rectangle')


        # Load params
        self.N_haa = self.get_parameter('horizon_haa').value  # default: 40
        self.dt = self.get_parameter('dt').value  # default: 0.1
        self.v_limit_haa = self.get_parameter('v_limit_haa').value  # default: 0.26
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value  # default: 1.82
        self.a_limit = self.get_parameter('a_limit').value  # default: 1.0
        self.alpha_limit = self.get_parameter('alpha_limit').value  # default: 1.0
        self.robot_radius = self.get_parameter('robot_radius').value  # default: 0.22

        # Goal
        raw_goal = self.get_parameter('goal').value  # default: [0, 0, 0, 0, 0]
        import ast
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal
        self.scenario_has_random_obstacles = bool(self.get_parameter('scenario_has_random_obstacles').value)
        self.goal_temp = self.goal.copy()
        self.goal_orig_history = []
        self.goal_temp_history = []
        self.state_traj = []

        # State
        self.HAA_map_ready = False
        self.state_ready = False
        self.step = 0
        self.HAA_mppi_planner = None
        self.HAA_occupancy_map = None
        
        # Storage for raw occupancy
        self.HAA_raw_occ = None
        
        # Trajectory tracking variables
        self.current_trajectory = None
        self.current_controls = None
        self.trajectory_start_time = None
        self.trajectory_ready = False
        
        self.shutdown_requested = False # Flag to stop loops immediately

        # Failure save path
        default_path = os.path.join(os.path.expanduser('~'), 'mppi_failure_data.npz')
        self.failure_path = self.get_parameter('failure_path').value  # default: default_path
        
        # List to store data for plotting (lower level control)
        self.plot_data = [] 
        self.plot_path = os.path.join(os.path.expanduser('~'), 'plot_data_lowlvlctrl.npz')

        self.trajectory_path = os.path.join(os.path.expanduser('~'), 'robot_trajectory.png')
        self.target_reached = False # Flag to track if target is reached
        # Xopt history storage
        self.Xopt_history = []
        
        # Flag to track first plan
        self.first_plan = True
        
        # Control command history storage
        self.control_command_history = []
        
        # Low-pass filter state and rate-limiter memory for v_cmd / w_cmd
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0   # previous published v (for rate limiting)
        self.w_cmd_prev = 0.0

        # Sliding-window pose history for TF2-based velocity estimation
        # Stores (t_sec, x, y, theta) tuples; window = 0.2 s
        self._pose_history = []
        
        # Kanayama controller parameters
        self.kx = self.get_parameter('kx').value  # default: 1.0
        self.ky = self.get_parameter('ky').value  # default: 1.0
        self.kth = self.get_parameter('kth').value  # default: 1.0
        self.kv = self.get_parameter('kv').value  # default: 1.0
        self.kw = self.get_parameter('kw').value  # default: 1.0
        
        # Integral controller parameters
        self.kix = self.get_parameter('kix').value  # default: 0.1
        self.kiy = self.get_parameter('kiy').value  # default: 0.1
        self.kith = self.get_parameter('kith').value  # default: 0.1
        self.max_integral = self.get_parameter('max_integral').value  # default: 1.0
        
        # Feedback gain matrix K for reference state computation (2x5: 2 controls, 5 states)
        # Default: diagonal gains for position/velocity errors
        K_default = [[0.0, 0.0, 0.0, 0.5, 0.0],  # Linear acceleration gains [x, y, theta, v, w]
                     [0.0, 0.0, 0.5, 0.0, 0.5]]  # Angular acceleration gains [x, y, theta, v, w]
        K_param = self.get_parameter('K_feedback').value  # default: K_default
        self.K_feedback = np.array(K_param) if isinstance(K_param, list) else K_param
        
        self.kanayama_controller = KanayamaController(
            self.kx, self.ky, self.kth, self.kv, self.kw,
            self.kix, self.kiy, self.kith, self.max_integral
        )

        # Initial pose/velocity state (set by TF2 timer and odom callback)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        # World-frame velocity: computed by differencing map-frame TF2 pose at 100 Hz
        self.vx_world = 0.0   # m/s in map +X direction
        self.vy_world = 0.0   # m/s in map +Y direction
        self.w_world  = 0.0   # rad/s yaw rate (same value in body and world frame)
        self._x_prev   = None  # previous TF2 x for differentiation
        self._y_prev   = None
        self._th_prev  = None

        # Goal management: wait for explicit goal from RViz2 before planning
        self.goal_received = False
        self._goal_wait_logged = False  # log "waiting for goal" only once per wait cycle

        # Stuck detection: abort goal if robot makes no position progress for too long
        self._stuck_timeout_sec   = 10.0   # seconds without progress → abort
        self._stuck_threshold_m   = 0.1   # metres — "progress" means moving at least this far
        self._stuck_last_prog_time = None  # wall-clock seconds at last progress event
        self._stuck_last_prog_pos  = None  # (x, y) at last progress event

        # TF2 for map-frame robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ROS pubs/subs
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.inflation_map_pub = self.create_publisher(OccupancyGrid, '/inflation_map', 1)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)

        # Timers: state update (100Hz), planning (10Hz), control (50Hz)
        self.state_timer = self.create_timer(0.01, self.state_update_cb)   # 100Hz TF2 pose
        self.planning_timer = self.create_timer(0.1, self.planning_loop)   # 10Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)    # 50Hz       

        self.get_logger().info("Real-world HAA planner node started:")
        self.get_logger().info("  - Velocity source: TF2 sliding window (map→base_footprint)")
        self.get_logger().info("  - State update: 100Hz  (TF2 map→base_footprint pose)")
        self.get_logger().info("  - Planning loop: 10Hz (MPPI, 4s horizon)")
        self.get_logger().info("  - Control loop:  50Hz (Kanayama tracker)")
        self.get_logger().info("  - Map source: /map  (Cartographer OccupancyGrid)")
        self.get_logger().info("  - Goal source: /move_base_simple/goal  (RViz2 '2D Goal Pose')")
        self.get_logger().info("Waiting for /map and TF2 (map→base_footprint) to become available...")
        # rclpy.spin() called in main()

    def _time_to_sec(self, ros_time):
        return float(ros_time.nanoseconds) * 1e-9

    def _stop_timers(self):
        if hasattr(self, 'planning_timer') and self.planning_timer is not None:
            try:
                self.planning_timer.cancel()
            except Exception:
                pass
        if hasattr(self, 'control_timer') and self.control_timer is not None:
            try:
                self.control_timer.cancel()
            except Exception:
                pass
    
    def save_data_on_exit(self):
        """Save the occupancy grid, goal, and state information when the program exits."""
        if hasattr(self, 'x') and hasattr(self, 'y') and hasattr(self, 'theta'):
            print(self.failure_path)
            x0 = [self.x, self.y, self.theta, self.v, self.omega]
            # Save occupancy grid data
            haa_occ_data = self.HAA_occupancy_map.occupancy_grid if self.HAA_occupancy_map else None
            np.savez(self.failure_path, x0=x0, goal=self.goal_temp, 
                     haa_occ=haa_occ_data,
                     goal_orig_history=self.goal_orig_history, goal_temp_history=self.goal_temp_history, 
                     state_traj=self.state_traj, control_command_history=self.control_command_history)
            
            # Save plot data
            np.savez(self.plot_path, data=self.plot_data)
            
            # Save Xopt history
            if hasattr(self, 'Xopt_history') and len(self.Xopt_history) > 0:
                xopt_history_path = os.path.join('/home/rant3/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch', 'Xopt_history.npz')
                # Convert Xopt history to numpy arrays for saving
                xopt_arrays = []
                uopt_arrays = []
                start_states = []
                goals = []
                steps = []
                timestamps = []
                
                for entry in self.Xopt_history:
                    xopt_arrays.append(entry['Xopt'])
                    uopt_arrays.append(entry['Uopt'])
                    start_states.append(entry['start_state'])
                    goals.append(entry['goal'])
                    steps.append(entry['step'])
                    timestamps.append(entry['timestamp'])
                
                np.savez(xopt_history_path, 
                        Xopt_history=xopt_arrays,
                        Uopt_history=uopt_arrays,
                        start_states=start_states,
                        goals=goals,
                        steps=steps,
                        timestamps=timestamps)
                self.get_logger().info(f"Saved Xopt history with {len(self.Xopt_history)} entries to: {xopt_history_path}")
        else:
            self.get_logger().warn("Data not saved on exit: Missing state information.")
    
    def save_trajectory_plot(self):
        """Save a plot of the robot trajectory with position, velocity, and angular velocity."""
        if len(self.state_traj) < 2:
            self.get_logger().warn("Not enough trajectory data to plot")
            return
            
        try:
            # Extract data from trajectory
            x_coords = [state[0] for state in self.state_traj]
            y_coords = [state[1] for state in self.state_traj]
            theta_coords = [state[2] for state in self.state_traj]
            v_coords = [state[3] for state in self.state_traj]
            w_coords = [state[4] for state in self.state_traj]

            # Unwrap orientation to keep theta continuous over time
            theta_coords_unwrapped = np.unwrap(np.array(theta_coords))
            
            # Create time array
            time_steps = np.arange(len(self.state_traj)) * self.dt
            
            # Create subplots (2x3 layout to include controller commands)
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

            # --- Occupancy map background with hard/soft inflation zones ---
            if self.HAA_occupancy_map is not None:
                omap = self.HAA_occupancy_map
                raw  = omap.occupancy_grid
                res  = omap.resolution
                _safe_dist = 0.10   # must match plan() safe_distance
                hard_r = int(self.robot_radius / res)
                soft_r = int((self.robot_radius + _safe_dist) / res)
                obstacle_mask = raw > 10
                if np.any(obstacle_mask):
                    dist_c = ndimage.distance_transform_edt(~obstacle_mask)
                else:
                    dist_c = np.full(raw.shape, 9999.0)
                # RGBA: start as white opaque, paint zones from outermost in
                rgba = np.ones((*raw.shape, 4), dtype=float)
                rgba[dist_c <= soft_r] = [1.0, 1.0, 0.45, 1.0]   # soft: light yellow
                rgba[dist_c <= hard_r] = [1.0, 0.55, 0.10, 1.0]   # hard: orange
                rgba[obstacle_mask]    = [0.15, 0.15, 0.15, 1.0]  # obstacle: near-black
                ax1.imshow(rgba,
                           extent=[omap.x_min, omap.x_max, omap.y_min, omap.y_max],
                           origin='lower', aspect='equal', zorder=0)
                # Store proxy patches for legend — do NOT add_patch (Patch base class is abstract)
                _p_soft = patches.Rectangle((0,0),1,1, facecolor=[1.0,1.0,0.45], label=f'Soft zone (+{_safe_dist}m)')
                _p_hard = patches.Rectangle((0,0),1,1, facecolor=[1.0,0.55,0.10], label=f'Hard zone (robot_r={self.robot_radius}m)')
                _p_obs  = patches.Rectangle((0,0),1,1, facecolor=[0.15,0.15,0.15], label='Obstacle')
            else:
                _p_soft = _p_hard = _p_obs = None

            # Plot 1: Position trajectory (x, y)
            ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Robot Trajectory')
            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start Position')
            ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=14, label='Target Position')

            # Draw tube around trajectory with robot radius (Shapely buffer if available)
            positions = np.column_stack([x_coords, y_coords])
            if len(positions) > 1:
                if _HAS_SHAPELY:
                    buffered = LineString(positions).buffer(self.robot_radius)
                    if buffered.geom_type == 'Polygon':
                        bx, by = buffered.exterior.xy
                        tube_polygon = np.column_stack([bx, by])
                        tube_patch = patches.Polygon(
                            tube_polygon,
                            facecolor='lightblue',
                            edgecolor='blue',
                            alpha=0.35,
                            linewidth=1,
                            label='Robot Radius'
                        )
                        ax1.add_patch(tube_patch)
                    elif buffered.geom_type == 'MultiPolygon':
                        for j, poly in enumerate(buffered.geoms):
                            bx, by = poly.exterior.xy
                            tube_polygon = np.column_stack([bx, by])
                            tube_patch = patches.Polygon(
                                tube_polygon,
                                facecolor='lightblue',
                                edgecolor='blue',
                                alpha=0.35,
                                linewidth=1,
                                label='Robot Radius' if j == 0 else ''
                            )
                            ax1.add_patch(tube_patch)
                else:
                    # Fallback approximation: draw a few radius circles along the path
                    step = max(1, len(positions) // 50)
                    for idx in range(0, len(positions), step):
                        circ = patches.Circle(
                            (positions[idx, 0], positions[idx, 1]),
                            self.robot_radius,
                            facecolor='lightblue',
                            edgecolor='none',
                            alpha=0.08,
                            label='Robot Radius' if idx == 0 else ''
                        )
                        ax1.add_patch(circ)
            
            # Plot intermediate goals if available
            if self.goal_temp_history:
                temp_x = [goal[0] for goal in self.goal_temp_history]
                temp_y = [goal[1] for goal in self.goal_temp_history]
                ax1.scatter(temp_x, temp_y, c='green', marker='o', alpha=0.7, s=50, label='Intermediate Goals')
            # Plot original goals if available
            if self.goal_orig_history:
                orig_x = [goal[0] for goal in self.goal_orig_history]
                orig_y = [goal[1] for goal in self.goal_orig_history]
                ax1.scatter(orig_x, orig_y, c='red', marker='s', alpha=0.7, s=50, label='Original Goals')
            
            # Plot obstacles from ROS2 parameters (loaded from runtime YAML)
            try:
                obstacle_count = 0
                if self.scenario_has_random_obstacles:
                    for i in range(1, 17):
                        obs_x = float(self.get_parameter(f'random_x_{i}').value)
                        obs_y = float(self.get_parameter(f'random_y_{i}').value)
                        obs_size = float(self.get_parameter(f'random_size_{i}').value)
                        obs_shape = str(self.get_parameter(f'random_shape_{i}').value)

                        obstacle_bottom_left_x = obs_x - obs_size / 2
                        obstacle_bottom_left_y = obs_y - obs_size / 2

                        if obs_shape == 'rectangle':
                            obstacle_rect = patches.Rectangle(
                                (obstacle_bottom_left_x, obstacle_bottom_left_y),
                                obs_size,
                                obs_size,
                                facecolor='gray',
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=0.7,
                                label='Obstacles' if obstacle_count == 0 else ""
                            )
                            ax1.add_patch(obstacle_rect)
                        elif obs_shape == 'hexagon':
                            hex_radius = obs_size / 2
                            hex_vertices = []
                            for j in range(6):
                                angle = j * np.pi / 3
                                vx = obs_x + hex_radius * np.cos(angle)
                                vy = obs_y + hex_radius * np.sin(angle)
                                hex_vertices.append((vx, vy))
                            hexagon = patches.Polygon(
                                hex_vertices,
                                facecolor='gray',
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=0.7,
                                label='Obstacles' if obstacle_count == 0 else ""
                            )
                            ax1.add_patch(hexagon)
                        elif obs_shape == 'triangle':
                            base = obs_size
                            half_base = base / 2.0
                            height_tri = base * np.sqrt(3) / 2.0
                            tri_vertices = [
                                (obs_x - half_base, obs_y),
                                (obs_x + half_base, obs_y),
                                (obs_x, obs_y + height_tri)
                            ]
                            triangle = patches.Polygon(
                                tri_vertices,
                                facecolor='gray',
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=0.7,
                                label='Obstacles' if obstacle_count == 0 else ""
                            )
                            ax1.add_patch(triangle)
                        else:
                            obstacle_rect = patches.Rectangle(
                                (obstacle_bottom_left_x, obstacle_bottom_left_y),
                                obs_size,
                                obs_size,
                                facecolor='gray',
                                edgecolor='black',
                                linewidth=1.5,
                                alpha=0.7,
                                label='Obstacles' if obstacle_count == 0 else ""
                            )
                            ax1.add_patch(obstacle_rect)

                        obstacle_count += 1

                if obstacle_count > 0:
                    self.get_logger().info(f"Plotted {obstacle_count} obstacles on trajectory plot")
            except Exception as e:
                self.get_logger().warn(f"Could not plot obstacles: {e}")
            
            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('Robot Position Trajectory')
            # Include zone proxy patches in legend if the occupancy map was available
            _extra_handles = [h for h in [_p_soft, _p_hard, _p_obs] if h is not None]
            _leg_handles, _ = ax1.get_legend_handles_labels()
            ax1.legend(handles=_leg_handles + _extra_handles)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Plot 2: Orientation over time (unwrapped)
            ax2.plot(time_steps, theta_coords_unwrapped, 'g-', linewidth=2, label='Orientation (θ)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Orientation (rad)')
            ax2.set_title('Robot Orientation vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Linear velocity over time
            ax3.plot(time_steps, v_coords, 'r-', linewidth=2, label='Linear Velocity (v)')
            ax3.axhline(y=self.v_limit_haa+0.01, color='r', linestyle='--', alpha=0.7, label=f'HAA Max Velocity)')
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='HAA Min Velocity')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Linear Velocity (m/s)')
            ax3.set_title('Linear Velocity vs Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Angular velocity over time
            ax4.plot(time_steps, w_coords, 'm-', linewidth=2, label='Angular Velocity (ω)')
            ax4.axhline(y=self.omega_limit_haa+0.1, color='m', linestyle='--', alpha=0.7, label=f'Max Angular Velocity')
            ax4.axhline(y=-self.omega_limit_haa-0.1, color='m', linestyle='--', alpha=0.7, label=f'Min Angular Velocity')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Angular Velocity (rad/s)')
            ax4.set_title('Angular Velocity vs Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Plot 5: Controller Commands - Linear Velocity
            if self.control_command_history:
                cmd_timestamps = [cmd['timestamp'] for cmd in self.control_command_history]
                cmd_v = [cmd['v_cmd'] for cmd in self.control_command_history]
                cmd_w = [cmd['w_cmd'] for cmd in self.control_command_history]
                
                # Convert timestamps to relative time (seconds from start)
                start_time = cmd_timestamps[0] if cmd_timestamps else 0
                cmd_time_steps = [(t - start_time) for t in cmd_timestamps]
                
                ax5.plot(cmd_time_steps, cmd_v, 'b-', linewidth=2, label='Linear Velocity Command')
                ax5.axhline(y=self.v_limit_haa, color='r', linestyle='--', alpha=0.7, label=f'Max Command Velocity')
                ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7, label=f'Min Command Velocity')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Linear Velocity Command (m/s)')
                ax5.set_title('Controller Linear Velocity Commands')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Controller Commands - Angular Velocity
                ax6.plot(cmd_time_steps, cmd_w, 'g-', linewidth=2, label='Angular Velocity Command')
                ax6.axhline(y=self.omega_limit_haa, color='m', linestyle='--', alpha=0.7, label=f'Max Command Angular Velocity')
                ax6.axhline(y=-self.omega_limit_haa, color='m', linestyle='--', alpha=0.7, label=f'Min Command Angular Velocity')
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Angular Velocity Command (rad/s)')
                ax6.set_title('Controller Angular Velocity Commands')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                # No control command data available
                ax5.text(0.5, 0.5, 'No Control Command Data', transform=ax5.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
                ax5.set_title('Controller Linear Velocity Commands')
                ax5.grid(True, alpha=0.3)
                
                ax6.text(0.5, 0.5, 'No Control Command Data', transform=ax6.transAxes, 
                        ha='center', va='center', fontsize=12, color='red')
                ax6.set_title('Controller Angular Velocity Commands')
                ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(self.trajectory_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f"Trajectory plot saved to {self.trajectory_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory plot: {e}")

    def save_first_plan_plot(self, Xopt, Uopt, start_state, goal):
        """Save a plot of the first HAA plan for debugging."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Trajectory in world coordinates
            ax1.plot(Xopt[:, 0], Xopt[:, 1], 'b-', linewidth=2, label='HAA Trajectory')
            ax1.plot(start_state.x, start_state.y, 'go', markersize=10, label='Start')
            ax1.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('First HAA Plan - Trajectory')
            ax1.legend()
            ax1.grid(True)
            ax1.axis('equal')
            
            # Plot 2: Control inputs over time
            time_steps_u = np.arange(len(Uopt)) * self.dt
            ax2.plot(time_steps_u, Uopt[:, 0], 'b-', linewidth=2, label='Linear Acceleration (m/s^2)')
            ax2.plot(time_steps_u, Uopt[:, 1], 'r-', linewidth=2, label='Angular Acceleration (rad/s^2)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Control Input')
            ax2.set_title('First HAA Plan - Control Sequence')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Robot orientation over time
            time_steps_x = np.arange(len(Xopt)) * self.dt
            ax3.plot(time_steps_x, Xopt[:, 2], 'g-', linewidth=2, label='Theta (rad)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Orientation (rad)')
            ax3.set_title('First HAA Plan - Robot Orientation')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Robot velocities over time
            ax4.plot(time_steps_x, Xopt[:, 3], 'b-', linewidth=2, label='Linear Vel (m/s)')
            ax4.plot(time_steps_x, Xopt[:, 4], 'r-', linewidth=2, label='Angular Vel (rad/s)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Velocity')
            ax4.set_title('First HAA Plan - Robot Velocities')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join('/home/rant3/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch', 'first_haa_plan.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            self.get_logger().info(f"First HAA plan plot saved to: {plot_path}")
            self.get_logger().info(f"Trajectory shape: {Xopt.shape}, Control shape: {Uopt.shape}")
            self.get_logger().info(f"Start state: x={start_state.x:.3f}, y={start_state.y:.3f}, theta={start_state.theta:.3f}")
            self.get_logger().info(f"Goal: x={goal[0]:.3f}, y={goal[1]:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error in first HAA plan plot: {e}")

    def map_cb(self, msg: OccupancyGrid):
        # Save original occupancy grid (integer values)
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.HAA_raw_occ = data.copy()

        # Build HAA occupancy grid map (dynamic map from lidar)
        self.HAA_occupancy_map = OccupancyGridMap(
            width=msg.info.width * msg.info.resolution,
            height=msg.info.height * msg.info.resolution,
            resolution=msg.info.resolution
        )
        
        # Set occupancy grid
        self.HAA_occupancy_map.occupancy_grid = data
        self.HAA_occupancy_map.x_min = msg.info.origin.position.x
        self.HAA_occupancy_map.y_min = msg.info.origin.position.y
        self.HAA_occupancy_map.x_max = msg.info.origin.position.x + msg.info.width * msg.info.resolution
        self.HAA_occupancy_map.y_max = msg.info.origin.position.y + msg.info.height * msg.info.resolution
        
        self.HAA_map_ready = True
        self.get_logger().debug("HAA map received: dynamic occupancy grid map built.")
        self._publish_inflation_map(msg)

    def _publish_inflation_map(self, source_msg: OccupancyGrid):
        """Publish /inflation_map for RViz2 visualisation of obstacle inflation zones.

        Cell values (OccupancyGrid 0-100 scale):
          0   – free space          (white in RViz2)
          20  – soft safety zone    (robot_radius + safe_distance, ~0.37 m) — light colour
          60  – hard inflation zone (within robot_radius, ~0.22 m)           — darker colour
          100 – raw obstacle cell   (reported by Cartographer SLAM)          — black
        """
        omap = self.HAA_occupancy_map
        raw = omap.occupancy_grid                          # shape (H, W), int values
        res = omap.resolution

        # Fixed soft-zone distance kept in sync with plan() parameter
        _soft_extra = 0.05          # metres beyond robot_radius
        hard_radius_cells = int(self.robot_radius / res)
        soft_radius_cells = int((self.robot_radius + _soft_extra) / res)

        # Distance (in cells) from every free cell to the nearest obstacle cell
        obstacle_mask = raw > 50   # True = obstacle (>0 means occupied; -1 = unknown → treated as free)
        if np.any(obstacle_mask):
            dist_cells = ndimage.distance_transform_edt(~obstacle_mask)
        else:
            dist_cells = np.full(raw.shape, 9999.0)

        # Build visualisation layer
        viz = np.zeros(raw.shape, dtype=np.int8)
        viz[dist_cells <= soft_radius_cells] = 20    # soft safety zone (light)
        viz[dist_cells <= hard_radius_cells] = 60    # hard inflation zone (darker)
        viz[obstacle_mask] = 100                     # raw obstacle

        out = OccupancyGrid()
        out.header = source_msg.header               # same frame_id ("map") and stamp
        out.info   = source_msg.info                 # same resolution, width, height, origin
        out.data   = viz.flatten().tolist()
        self.inflation_map_pub.publish(out)

    def state_update_cb(self):
        """Look up map→base_footprint TF at 100 Hz; estimate v, omega from 200ms sliding window."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time()
            )
            x_new  = t.transform.translation.x
            y_new  = t.transform.translation.y
            q      = t.transform.rotation
            _, _, th_new = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # --- sliding-window velocity estimate ---
            # Only add a new entry to pose_history when TF2 actually returned a NEW transform.
            # Cartographer publishes map→odom at ~10-50 Hz.  Our timer runs at 100 Hz.
            # When TF2 hasn't updated yet, lookup_transform returns the SAME pose as the
            # previous call.  Adding that duplicate to pose_history and then applying an EMA
            # against a "measured velocity" of 0 would decay self.v toward 0 between TF2
            # updates — which is exactly the wrong behaviour.
            TF2_CHANGE_THRESHOLD = 0.0003   # 0.3 mm; below this treat position as unchanged
            pos_changed = (self._x_prev is None or
                           abs(x_new - self._x_prev) > TF2_CHANGE_THRESHOLD or
                           abs(y_new - self._y_prev) > TF2_CHANGE_THRESHOLD or
                           abs(((th_new - self._th_prev) + np.pi) % (2*np.pi) - np.pi) > 0.0003)

            if pos_changed:
                now_sec = self.get_clock().now().nanoseconds * 1e-9
                self._pose_history.append((now_sec, x_new, y_new, th_new))

                # Drop entries older than 300 ms
                cutoff = now_sec - 0.30
                while self._pose_history and self._pose_history[0][0] < cutoff:
                    self._pose_history.pop(0)

                # Need at least 80 ms of actual movement data to compute a stable estimate
                if len(self._pose_history) >= 2:
                    t0, x0h, y0h, th0h = self._pose_history[0]
                    dt_win = now_sec - t0
                    if dt_win >= 0.08:
                        vx_w = (x_new - x0h) / dt_win
                        vy_w = (y_new - y0h) / dt_win
                        dth  = ((th_new - th0h) + np.pi) % (2 * np.pi) - np.pi
                        w_w  = dth / dt_win

                        # Project world-frame velocity onto robot heading → body-frame forward speed
                        v_body = vx_w * np.cos(th_new) + vy_w * np.sin(th_new)

                        # Store world-frame components for logging
                        self.vx_world = vx_w
                        self.vy_world = vy_w
                        self.w_world  = w_w

                        # EMA filter — only applied when TF2 gave fresh data (no artificial decay)
                        _a = 0.5
                        self.v     = _a * v_body + (1.0 - _a) * self.v
                        self.omega = _a * w_w    + (1.0 - _a) * self.omega
            # If pos_changed is False: self.v and self.omega are HELD (not decayed)

            self.x = x_new
            self.y = y_new
            self.theta = th_new
            self._x_prev  = x_new
            self._y_prev  = y_new
            self._th_prev = th_new
            self.state_ready = True
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass  # TF not available yet — keep waiting silently

    def goal_cb(self, msg: PoseStamped):
        """Receive a new goal from RViz2 '2D Goal Pose' button."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.goal = [x, y, theta, 0.0, 0.0]
        self.goal_temp = self.goal.copy()
        self.goal_received = True
        self.target_reached = False
        self.trajectory_ready = False
        self.HAA_mppi_planner = None   # reset warm-start so MPPI plans fresh toward new goal
        self.step = 0
        self.state_traj = []
        self._goal_wait_logged = False
        self.kanayama_controller.reset_integral_errors()
        # Reset stuck tracker for the new goal
        self._stuck_last_prog_time = self.get_clock().now().nanoseconds * 1e-9
        self._stuck_last_prog_pos  = (self.x, self.y)
        self.get_logger().info(
            f"New goal from RViz2: x={x:.3f} m, y={y:.3f} m, theta={np.degrees(theta):.1f} deg"
        )

    def planning_loop(self, event=None):
        """HAA-only planning loop: Use HAA planner all the time, stop if no feasible solution."""
        # Check if shutdown has been requested
        if self.shutdown_requested:
            return

        # Check if all required data is ready
        if not (self.HAA_map_ready and self.state_ready):
            if not self.HAA_map_ready:
                self.get_logger().warn("HAA map not ready — waiting for /map from Cartographer")
            if not self.state_ready:
                self.get_logger().warn("State not ready — waiting for TF2 map→base_footprint")
            return

        # Wait for an explicit goal from RViz2 before planning
        if not self.goal_received:
            if not self._goal_wait_logged:
                self.get_logger().info(
                    "Ready. Set a goal using '2D Goal Pose' button in RViz2 "
                    "(publishes to /move_base_simple/goal)."
                )
                self._goal_wait_logged = True
            return

        # current state
        # Limit velocities to avoid planning failures due to noises
        # It is similar to Tube MPC where the initial condition is also a decision varaible
        # x_true is the true state of the robot, may violate constraints
        x_true = [self.x, self.y, self.theta, self.v, self.omega]
        #print(x_true)
        if self.v > self.v_limit_haa - 0.005:
            self.v = self.v_limit_haa - 0.005
            #self.get_logger().warn(f"Velocity limited to {self.v_limit_haa} m/s")
        if self.v < 0:
            self.v = 0.01;
        if self.omega > self.omega_limit_haa:
            self.omega = self.omega_limit_haa - 0.005
            #self.get_logger().warn(f"Angular velocity limited to {self.omega_limit_haa} rad/s")
        if self.omega < -self.omega_limit_haa:
            self.omega = -self.omega_limit_haa + 0.005
            #self.get_logger().warn(f"Angular velocity limited to {-self.omega_limit_haa} rad/s")
        #x0 is the initial state of the nominal system for solving MPC, satisfy all constraints
        x0 = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(x_true.copy())

        # Check if target is reached
        dist = np.linalg.norm(np.array(x_true[:2]) - np.array(self.goal_temp[:2]))
        if dist < 0.1 and not self.target_reached:
            self.target_reached = True
            self.publish_stop_command()
            self.save_trajectory_plot()
            # Reset for next goal — keep node alive
            self.goal_received = False
            self.trajectory_ready = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            self.get_logger().info(
                f"Goal reached! (distance={dist:.3f} m). "
                "Robot stopped. Set next goal via '2D Goal Pose' in RViz2."
            )
            return

        # ── Stuck-timeout check ───────────────────────────────────────────────────
        if self._stuck_last_prog_time is not None:
            _now = self.get_clock().now().nanoseconds * 1e-9
            _disp = np.hypot(self.x - self._stuck_last_prog_pos[0],
                             self.y - self._stuck_last_prog_pos[1])
            if _disp >= self._stuck_threshold_m:
                # Robot made progress — reset reference point and timer
                self._stuck_last_prog_time = _now
                self._stuck_last_prog_pos  = (self.x, self.y)
            elif (_now - self._stuck_last_prog_time) > self._stuck_timeout_sec:
                _elapsed = _now - self._stuck_last_prog_time
                self.get_logger().warn(
                    f"STUCK TIMEOUT: robot has not moved {self._stuck_threshold_m}m "
                    f"in {_elapsed:.1f}s. Aborting goal and saving debug plot."
                )
                # Generate debug plot using a fresh sample from the current nominal
                if self.HAA_mppi_planner is not None and self.HAA_map_ready:
                    try:
                        _stuck_state = RobotState(
                            x=self.x, y=self.y, theta=self.theta, v=self.v, w=self.omega)
                        _s_ctrl, _s_nom, _s_trajs, _s_mask = \
                            self.HAA_mppi_planner.sample_debug_trajectories(
                                _stuck_state, np.array(self.goal_temp[:2]), self.robot_radius)
                        self.HAA_mppi_planner.debug_plot_no_valid_trajectories(
                            _stuck_state, np.array(self.goal_temp[:2]),
                            _s_ctrl, _s_nom, _s_trajs, _s_mask,
                            title_override=f'STUCK — no progress for {_elapsed:.0f}s '
                                           f'(threshold {self._stuck_threshold_m}m)')
                    except Exception as _pe:
                        self.get_logger().error(f"Stuck debug plot failed: {_pe}")
                self.publish_stop_command()
                self.goal_received      = False
                self.trajectory_ready   = False
                self.HAA_mppi_planner   = None
                self._goal_wait_logged  = False
                self._stuck_last_prog_time = None
                return
        # ─────────────────────────────────────────────────────────────────────────

        try:
            self.goal_temp = self.goal.copy()

            # === HAA PLANNER (4-second horizon, dynamic map) ===
            # Initialize HAA planner if it doesn't exist
            if self.HAA_mppi_planner is None:
                dynamics = TurtlebotDynamics(dt=self.dt)
                # HAA: 4-second horizon (40 nodes at 0.1s dt)
                haa_horizon = self.N_haa  # 4 seconds
                haa_mppi = MPPI(
                    sigma=2,           # higher noise = more diverse path exploration
                    temperature=0.3,
                    num_nodes=haa_horizon,
                    num_rollouts=2000,
                    use_noise_ramp=False,
                    nu=2
                )
                
                self.HAA_mppi_planner = MPPIPlanner(
                    mppi=haa_mppi,
                    dynamics=dynamics,
                    occupancy_map=self.HAA_occupancy_map,
                    v_min=-0.05, v_max=self.v_limit_haa,
                    w_min=-self.omega_limit_haa, w_max=self.omega_limit_haa,
                    a_min=-self.a_limit, a_max=self.a_limit,
                    alpha_min=-self.alpha_limit, alpha_max=self.alpha_limit,
                    smoothing_weight=0.1,
                    logger=self.get_logger()
                )
                self.get_logger().info("HAA planner initialized (4s horizon, dynamic map).")
            
            # Update HAA planner with latest dynamic map
            if self.HAA_mppi_planner.occupancy_map != self.HAA_occupancy_map:
                self.HAA_mppi_planner.update_occupancy_map(self.HAA_occupancy_map)
            
            # === HAA PLANNING ===
            #all planning uses x0, not x_true
            haa_current_state = RobotState(x=x0[0], y=x0[1], theta=x0[2], v=x0[3], w=x0[4])

            # # --- diagnostic: clearance from robot centre to nearest obstacle ---
            # omap = self.HAA_occupancy_map
            # gx, gy = omap.world_to_grid(x0[0], x0[1])
            # in_bounds = (0 <= gx < omap.grid_width and 0 <= gy < omap.grid_height)
            # if in_bounds:
            #     occ_val = int(omap.occupancy_grid[gy, gx])
            #     obstacle_mask = omap.occupancy_grid > 0.01
            #     dist_cells = ndimage.distance_transform_edt(~obstacle_mask)[gy, gx]
            #     clearance_m = dist_cells * omap.resolution
            #     inflation_cells = int(self.robot_radius / omap.resolution)
            #     # Use one-cell tolerance to avoid false positives from grid quantization.
            #     # E.g. at 0.05 m/cell the nearest discrete distance below 0.22 m is 0.212 m
            #     # (sqrt(18)×0.05), which may occur even when the robot is physically safe.
            #     start_in_collision = clearance_m < (self.robot_radius - omap.resolution)
            #     # World-frame speed magnitude and yaw rate (from TF2 pose differentiation)
            #     speed_world = np.hypot(self.vx_world, self.vy_world)
            #     # Log nav status every step (10 Hz) — concise single line
            #     self.get_logger().info(
            #         f"step={self.step:4d} | pos=({x0[0]:.3f},{x0[1]:.3f}) "
            #         f"θ={np.degrees(x0[2]):.1f}° "
            #         f"vx={self.vx_world:.3f} vy={self.vy_world:.3f} spd={speed_world:.3f} w={self.w_world:.3f} | "
            #         f"dist_goal={dist:.3f}m | clearance={clearance_m:.3f}m "
            #         f"(inflation={inflation_cells} cells={self.robot_radius:.2f}m) | "
            #         f"grid=({gx},{gy}) occ={occ_val}"
            #         + (" *** IN COLLISION ZONE ***" if start_in_collision else "")
            #     )
            #     if start_in_collision:
            #         self.get_logger().error(
            #             f"Robot is inside the inflated obstacle zone "
            #             f"(clearance {clearance_m:.3f}m < robot_radius {self.robot_radius:.3f}m). "
            #             "All MPPI trajectories will fail. "
            #             "Manually back the robot away from the obstacle, then set a new goal."
            #         )
            # else:
            #     self.get_logger().error(
            #         f"Robot position ({x0[0]:.3f},{x0[1]:.3f}) is OUTSIDE map bounds "
            #         f"[x={omap.x_min:.2f}..{omap.x_max:.2f}, y={omap.y_min:.2f}..{omap.y_max:.2f}]. "
            #         "TF2 pose or map origin may be incorrect."
            #     )

            haa_start_time = time.time()
            haa_Xopt, haa_Uopt = self.HAA_mppi_planner.plan(haa_current_state, np.array(self.goal_temp[:2]), debug_plot=True, robot_radius=self.robot_radius)
            haa_end_time = time.time()

            # Check HAA feasibility
            haa_feasible = (haa_Xopt is not None and haa_Uopt is not None and
                          len(haa_Xopt) > 0 and len(haa_Uopt) > 0)

            if haa_feasible:
                self.current_trajectory = haa_Xopt
                self.current_controls = haa_Uopt
                self.trajectory_start_time = self.get_clock().now()
                self.trajectory_ready = True
                self.get_logger().debug(f"HAA planning time: {haa_end_time - haa_start_time:.6f}s")
                
                # Plot first plan
                if self.first_plan:
                    #self.save_first_plan_plot(haa_Xopt, haa_Uopt, haa_current_state, self.goal_temp[:2])
                    self.first_plan = False  # Only plot the first plan
            else:
                # HAA planning failed — stop robot and wait for new goal
                self.get_logger().warn("HAA planning failed — no feasible trajectory found")
                self.get_logger().info(
                    f'State at failure: x={haa_current_state.x:.3f}, y={haa_current_state.y:.3f}, '
                    f'theta={haa_current_state.theta:.3f}, v={haa_current_state.v:.3f}, w={haa_current_state.w:.3f}'
                )
                self.trajectory_ready = False
                self.publish_stop_command()
                # Reset so user can set a new goal
                self.goal_received = False
                self.HAA_mppi_planner = None
                self._goal_wait_logged = False
                self.get_logger().warn(
                    "Robot stopped. Check map/obstacles and set a new goal via RViz2."
                )
                return
            
            # Store Xopt history
            if self.trajectory_ready and self.current_trajectory is not None:
                xopt_data = {
                    'Xopt': self.current_trajectory.copy(),
                    'Uopt': self.current_controls.copy(),
                    'start_state': [x0[0], x0[1], x0[2], x0[3], x0[4]],
                    'goal': self.goal_temp[:2].copy(),
                    'step': self.step,
                    'timestamp': self._time_to_sec(self.get_clock().now()),
                    'planner': 'HAA'
                }
                self.Xopt_history.append(xopt_data)

            # Reset integral errors for new trajectory
            self.kanayama_controller.reset_integral_errors()
            
        except Exception as e:
            self.get_logger().error(f"Planning exception at step {self.step}: {e}")
            np.savez(self.failure_path, x0=x0, goal=self.goal_temp, occ=self.HAA_raw_occ)
            self.get_logger().info(f"Failure data saved to {self.failure_path}")
            self.publish_stop_command()
            # Reset so user can set a new goal without restarting the node
            self.goal_received = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            return

        self.step += 1
    
    def get_reference_state(self, current_time):
        """Get reference state by applying feedback control to current state.
        
        Computes: u = u_nominal + K * (x_true - x_nominal)
        Then applies u to x_true using system dynamics.
        """
        if not self.trajectory_ready or self.current_trajectory is None or self.current_controls is None:
            return None
        
        if len(self.current_trajectory) == 0 or len(self.current_controls) == 0:
            return None
        
        # Current true state
        x_true = np.array([self.x, self.y, self.theta, self.v, self.omega])
        
        # Nominal state and control from planned trajectory
        x_nominal = np.array(self.current_trajectory[0])  # First state in trajectory
        u_nominal = np.array(self.current_controls[0])    # First control in sequence
        
        # State error
        error = x_true - x_nominal
        
        K_feedback = [[0.0, 0.0, 0.0, -2, 0.0], [0.0, 0.0, 0, 0.0, -0.6]]
        # Feedback control: u = u_nominal + K * error
        u = u_nominal + K_feedback @ error
        
        # Apply control to current state using dynamics
        if self.HAA_mppi_planner is not None and self.HAA_mppi_planner.dynamics is not None:
            dynamics = self.HAA_mppi_planner.dynamics
        else:
            # Fallback: create dynamics instance if planner not ready
            dynamics = TurtlebotDynamics(dt=self.dt)
        #dynamics = TurtlebotDynamics(0.2)
        current_robot_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2], 
                                        v=x_true[3], w=x_true[4])
        ref_robot_state = dynamics.step(current_robot_state, u)
        
        ref_robot_state = dynamics.step(ref_robot_state, self.current_controls[1])
        # Convert RobotState to array format [x, y, theta, v, w]
        ref_state = np.array([ref_robot_state.x, ref_robot_state.y, ref_robot_state.theta,
                             ref_robot_state.v, ref_robot_state.w])
        
        ref_state = self.current_trajectory[2] #tested that works well

        return ref_state
    
    def control_loop(self, event=None):
        """Control loop running at 50Hz using Kanayama tracking controller."""
        # Check if shutdown has been requested
        if self.shutdown_requested:
            return
        
        if not self.state_ready:
            self.get_logger().warn("Control loop: state not ready (TF2 unavailable)")
            return
        if not self.trajectory_ready:
            return  # silently wait — either no goal yet or planning in progress
        
        current_time = self.get_clock().now()
        ref_state = self.get_reference_state(current_time)
        
        if ref_state is None:
            # No valid reference, stop robot
            self.publish_stop_command()
            return
        
        # Current robot state
        current_state = [self.x, self.y, self.theta, self.v, self.omega]
        
        # Compute control using Kanayama controller
        v_cmd, w_cmd = self.kanayama_controller.compute_control(
            current_state, ref_state
        )
        
        # --- EMA low-pass filter (alpha=0.6: 60% previous, 40% new — stronger smoothing for angular) ---
        _alpha = 0.6
        v_smooth = _alpha * self.v_cmd_filtered + (1.0 - _alpha) * v_cmd
        w_smooth = _alpha * self.w_cmd_filtered + (1.0 - _alpha) * w_cmd

        # --- Rate limiter: cap change per 20 ms tick to avoid step jumps ---
        _dt_ctrl = 0.02
        _max_dv = self.a_limit     * _dt_ctrl   # e.g. 0.5 m/s² → 0.010 m/s per tick
        _max_dw = self.alpha_limit * _dt_ctrl   # e.g. 1.0 rad/s² → 0.020 rad/s per tick
        self.v_cmd_filtered = float(np.clip(v_smooth,
                                            self.v_cmd_prev - _max_dv,
                                            self.v_cmd_prev + _max_dv))
        self.w_cmd_filtered = float(np.clip(w_smooth,
                                            self.w_cmd_prev - _max_dw,
                                            self.w_cmd_prev + _max_dw))
        self.v_cmd_prev = self.v_cmd_filtered
        self.w_cmd_prev = self.w_cmd_filtered
        
        # Publish filtered command
        twist = Twist()
        twist.linear.x = self.v_cmd_filtered
        twist.angular.z = self.w_cmd_filtered
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"cmd_vel: v={self.v_cmd_filtered:.3f} w={self.w_cmd_filtered:.3f} | "
            f"raw: v={v_cmd:.3f} w={w_cmd:.3f} | "
            f"state v={self.v:.3f} w={self.omega:.3f}",
            throttle_duration_sec=0.2,
        )
        
        # Store control command history (store filtered values that were actually published)
        control_data = {
            'timestamp': self._time_to_sec(current_time),
            'v_cmd': self.v_cmd_filtered,
            'w_cmd': self.w_cmd_filtered,
            'current_state': current_state.copy(),
            'ref_state': ref_state.copy()
        }
        self.control_command_history.append(control_data)

    
    def publish_stop_command(self):
        """Publish stop command."""
        # Reset filter and rate-limiter state when stopping
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
    

def main(args=None):
    rclpy.init(args=args)
    node = HAANavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

