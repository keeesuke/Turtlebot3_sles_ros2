#!/usr/bin/env python3
"""
ROS2 MPC+NN Switching Planner for TurtleBot3 — Real World

Dual-planner simplex architecture:
  HPA (High-Performance Autonomous) = Neural Network (fast, reactive)
  HAA (High-Assurance Autonomous)   = MPPI MPC (conservative, safe)

While in HPA: NN drives; each cycle checks HAA feasibility from predicted state.
              If infeasible → 2-loop braking → switch to HAA.
While in HAA: MPPI plans; checks safety margin → switch back to HPA when safe.

State estimation : TF2 map → base_footprint (100 Hz) + /odom velocity (EMA)
Map source       : /map (Cartographer SLAM occupancy grid)
LiDAR input      : /scan (real TurtleBot3 LDS-01, clipped to training range)
Goal input       : /move_base_simple/goal (RViz2 '2D Goal Pose')
Control output   : /cmd_vel (50 Hz)

Based on planner_haa_real_world.py (MPC infrastructure) and
planner_nn_real_world.py (NN model loading and LiDAR processing),
with switching logic from planner_switch_mpc_nn.py.

Prerequisites (must already be running):
  1. turtlebot3_node       (hw driver — /scan, /tf odom→base_footprint)
  2. turtlebot3_cartographer  (SLAM — /tf map→odom, /map)
  3. rviz2                 (optional — for interactive goal input)
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import os
import atexit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import time
from datetime import datetime
from scipy import ndimage

import torch
import torch.nn as nn

try:
    from shapely.geometry import LineString
    _HAS_SHAPELY = True
except Exception:
    LineString = None
    _HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# NN architecture — must match the architecture used during training
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Multi-Layer Perceptron matching training architecture."""
    def __init__(self, input_dim=364, hidden_dims=None, output_dim=2, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# MPPI Implementation (from planner_haa_real_world.py)
# ---------------------------------------------------------------------------
class MPPI:
    def __init__(self, sigma: float = 0.1, temperature: float = 0.05,
                 num_nodes: int = 20, num_rollouts: int = 100,
                 use_noise_ramp: bool = False, noise_ramp: float = 1.0, nu: int = 2,
                 a_limit: float = 0.25, alpha_limit: float = 0.5,
                 v_limit_haa: float = 0.14, v_limit_hpa: float = 0.2,
                 omega_limit_haa: float = 0.9, omega_limit_hpa: float = 1.0,
                 dt: float = 0.1):
        self.sigma = sigma
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.num_rollouts = num_rollouts
        self.use_noise_ramp = use_noise_ramp
        self.noise_ramp = noise_ramp
        self.nu = nu
        self.a_limit = a_limit
        self.alpha_limit = alpha_limit
        self.v_limit_haa = v_limit_haa
        self.v_limit_hpa = v_limit_hpa
        self.omega_limit_haa = omega_limit_haa
        self.omega_limit_hpa = omega_limit_hpa
        self.dt = dt

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample control knots with adaptive noise based on control scale."""
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        _sigma = self.sigma

        control_magnitude = np.abs(nominal_knots)

        control_magnitude_2 = np.ones_like(control_magnitude)
        control_magnitude_2[:, 0] = 0.2
        control_magnitude_2[:, 1] = 0.4

        min_noise_scale = 0.1 * _sigma
        adaptive_sigma = np.maximum(control_magnitude_2 * _sigma, min_noise_scale)

        if self.use_noise_ramp:
            # Increasing noise toward the end of horizon (from planner_haa_real_world.py)
            ramp = self.noise_ramp * np.linspace(1 / num_nodes, 1, num_nodes, endpoint=True)[:, None]
            sigma = (ramp * adaptive_sigma)[None, :, :]
        else:
            sigma = adaptive_sigma[None, :, :]

        sigma = np.broadcast_to(sigma, (num_rollouts - 1, num_nodes, self.nu))
        noise = np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        noised_knots = nominal_knots[None, :, :] + sigma * noise

        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update nominal knots with rewards."""
        costs = -rewards
        beta = np.min(costs)
        _weights = np.exp(-(costs - beta) / self.temperature)
        weights = _weights / np.sum(_weights)
        nominal_knots = np.sum(weights[:, None, None] * sampled_knots, axis=0)
        return nominal_knots


@dataclass
class RobotState:
    """Turtlebot state: [x, y, theta, v, w]"""
    x: float
    y: float
    theta: float
    v: float
    w: float


class TurtlebotDynamics:
    """Turtlebot dynamics with acceleration control."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.nu = 2
        self.nx = 5

    def step(self, state: RobotState, control: np.ndarray) -> RobotState:
        """Step dynamics: control = [a, alpha] (accelerations)"""
        a, alpha = control
        new_v = state.v + a * self.dt
        new_w = state.w + alpha * self.dt
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
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        self.x_min = -width / 2
        self.y_min = -height / 2
        self.x_max = width / 2
        self.y_max = height / 2
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
        """Optimized dilation using distance transform."""
        obstacle_mask = self.occupancy_grid > 50
        if not np.any(obstacle_mask):
            return np.zeros_like(obstacle_mask, dtype=bool)
        distance_map = ndimage.distance_transform_edt(~obstacle_mask)
        dilated_grid = distance_map <= robot_radius_cells
        return dilated_grid

    def check_collision_new(self, x: np.ndarray, y: np.ndarray, robot_radius: float = 0.22) -> np.ndarray:
        """Fully vectorized collision checking using dilated occupancy grid."""
        grid_x = ((x - self.x_min) / self.resolution).astype(int)
        grid_y = ((y - self.y_min) / self.resolution).astype(int)
        out_of_bounds = ((grid_x < 0) | (grid_x >= self.grid_width) |
                        (grid_y < 0) | (grid_y >= self.grid_height))
        collisions = np.zeros(len(x), dtype=bool)
        collisions[out_of_bounds] = True
        valid_mask = ~out_of_bounds
        if not np.any(valid_mask):
            return collisions
        robot_radius_cells = int(robot_radius / self.resolution)
        dilate_grid = self.dilate_grid_new(robot_radius_cells)
        valid_x = grid_x[valid_mask]
        valid_y = grid_y[valid_mask]
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
        binary_grid = (self.occupancy_grid > 50).astype(int)
        im = ax.imshow(binary_grid, origin='lower', cmap='gray_r',
                      extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                      vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Occupancy Probability')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Occupancy Grid Map')
        ax.grid(True, alpha=0.3)
        return ax

    def set_goal(self, x: float, y: float):
        self.goal = np.array([x, y])

    def set_start(self, x: float, y: float):
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
        self._logger = logger
        self.nominal_controls = np.zeros((mppi.num_nodes, mppi.nu))
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

    def debug_plot_no_valid_trajectories(self, start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask, title_override=None):
        """Debug function to plot trajectories when no valid ones are found."""
        try:
            num_valid = np.sum(valid_mask)
            num_total = len(sampled_controls)

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
                    continue
                trajectory = all_trajectories[i]
                control_seq = sampled_controls[i]
                has_collision = False
                xs = trajectory[:, 0]
                ys = trajectory[:, 1]
                collisions = self.occupancy_map.check_collision_new(xs, ys)
                if np.any(collisions):
                    has_collision = True
                has_linear_velocity_violation = False
                v_violations = np.any((trajectory[:, 3] < self.v_min) | (trajectory[:, 3] > self.v_max))
                if v_violations:
                    has_linear_velocity_violation = True
                has_angular_velocity_violation = False
                w_violations = np.any((trajectory[:, 4] < self.w_min) | (trajectory[:, 4] > self.w_max))
                if w_violations:
                    has_angular_velocity_violation = True
                has_control_violation = False
                a_violations = np.any((control_seq[:, 0] < self.a_min) | (control_seq[:, 0] > self.a_max))
                alpha_violations = np.any((control_seq[:, 1] < self.alpha_min) | (control_seq[:, 1] > self.alpha_max))
                if a_violations or alpha_violations:
                    has_control_violation = True
                if has_collision and not has_linear_velocity_violation and not has_angular_velocity_violation and not has_control_violation:
                    collision_only += 1
                elif has_linear_velocity_violation and not has_collision and not has_angular_velocity_violation and not has_control_violation:
                    linear_velocity_only += 1
                elif has_angular_velocity_violation and not has_collision and not has_linear_velocity_violation and not has_control_violation:
                    angular_velocity_only += 1
                elif has_collision and has_linear_velocity_violation:
                    collision_linearv += 1
                elif has_collision and has_angular_velocity_violation:
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

            omap = self.occupancy_map
            raw  = omap.occupancy_grid
            res  = omap.resolution
            _robot_r   = 0.15
            _safe_dist = 0.10
            hard_r = int(_robot_r / res)
            soft_r = int((_robot_r + _safe_dist) / res)
            obstacle_mask = raw > 50
            if np.any(obstacle_mask):
                dist_c = ndimage.distance_transform_edt(~obstacle_mask)
            else:
                dist_c = np.full(raw.shape, 9999.0)
            rgba = np.ones((*raw.shape, 4), dtype=float)
            rgba[dist_c <= soft_r] = [1.0, 1.0, 0.45, 1.0]
            rgba[dist_c <= hard_r] = [1.0, 0.55, 0.10, 1.0]
            rgba[obstacle_mask]    = [0.15, 0.15, 0.15, 1.0]
            ax1.imshow(rgba, extent=[omap.x_min, omap.x_max, omap.y_min, omap.y_max],
                       origin='lower', aspect='equal', zorder=0)

            invalid_indices = np.where(~valid_mask)[0]
            for i in invalid_indices[:120]:
                traj = all_trajectories[i]
                ax1.plot(traj[:, 0], traj[:, 1], color='red', alpha=0.18, linewidth=0.6, zorder=1)

            valid_indices = np.where(valid_mask)[0]
            for i in valid_indices[:30]:
                traj = all_trajectories[i]
                ax1.plot(traj[:, 0], traj[:, 1], color='limegreen', alpha=0.55, linewidth=0.9, zorder=2)

            nom_traj = all_trajectories[0]
            ax1.plot(nom_traj[:, 0], nom_traj[:, 1], color='royalblue', linewidth=2.5,
                     zorder=3, label='_nolegend_')

            ax1.plot(start_state.x, start_state.y, 'o', color='cyan',
                     markersize=10, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
            arr_len = max(0.15, _robot_r * 1.2)
            ax1.annotate('', xy=(start_state.x + arr_len * np.cos(start_state.theta),
                                  start_state.y + arr_len * np.sin(start_state.theta)),
                          xytext=(start_state.x, start_state.y),
                          arrowprops=dict(arrowstyle='->', color='cyan', lw=2.0), zorder=5)

            ax1.add_patch(patches.Circle((start_state.x, start_state.y), _robot_r,
                                          fill=False, edgecolor='orange',
                                          linestyle='--', linewidth=1.8, zorder=4))
            ax1.add_patch(patches.Circle((start_state.x, start_state.y), _robot_r + _safe_dist,
                                          fill=False, edgecolor='gold',
                                          linestyle=':', linewidth=1.5, zorder=4))

            ax1.plot(goal[0], goal[1], 'r*', markersize=16,
                     markeredgecolor='darkred', markeredgewidth=1, zorder=5)

            proxy_invalid  = plt.Line2D([0], [0], color='red',       lw=1.5, alpha=0.6,  label=f'Invalid ({len(invalid_indices)}, showing ≤120)')
            proxy_valid    = plt.Line2D([0], [0], color='limegreen',  lw=1.5, alpha=0.8,  label=f'Valid ({num_valid}, showing ≤30)')
            proxy_nominal  = plt.Line2D([0], [0], color='royalblue',  lw=2.5,             label='Nominal trajectory')
            proxy_robot    = plt.Line2D([0], [0], marker='o', color='cyan', lw=0, markersize=9, label='Robot position')
            proxy_goal     = plt.Line2D([0], [0], marker='*', color='red',  lw=0, markersize=12, label='Goal')
            proxy_hard     = patches.Patch(facecolor=[1.0,0.55,0.10], label=f'Hard zone (r={_robot_r}m)')
            proxy_soft     = patches.Patch(facecolor=[1.0,1.0,0.45], label=f'Soft zone (+{_safe_dist}m)')
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

            categories  = ['Collision\nonly', 'Lin-vel\nonly', 'Ang-vel\nonly',
                           'Collision\n+lin-vel', 'Collision\n+ang-vel',
                           'Control\nonly', 'Other']
            counts      = [collision_only, linear_velocity_only, angular_velocity_only,
                           collision_linearv, collision_angularv, control_only, other_violations]
            bar_colors  = ['#d62728', '#ff7f0e', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
            bars = ax2.bar(categories, counts, color=bar_colors, edgecolor='black', linewidth=0.7)

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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_path = os.path.join(os.getcwd(), f'mppi_no_valid_trajectories_debug_{timestamp}.png')
            plt.savefig(debug_path, dpi=300, bbox_inches='tight')
            plt.close()
            self._log_info(f"MPPI debug plot saved to: {debug_path}")

        except Exception as e:
            self._log_error(f"Error in MPPI debug plot: {e}")

    def simulate_trajectories_vectorized(self, start_state: RobotState, control_sequences: np.ndarray) -> np.ndarray:
        """Vectorized trajectory simulation for all control sequences at once."""
        num_rollouts, horizon, nu = control_sequences.shape
        trajectories = np.zeros((num_rollouts, horizon + 1, 5))

        trajectories[:, 0, 0] = start_state.x
        trajectories[:, 0, 1] = start_state.y
        trajectories[:, 0, 2] = start_state.theta
        trajectories[:, 0, 3] = start_state.v
        trajectories[:, 0, 4] = start_state.w

        for i in range(horizon):
            x = trajectories[:, i, 0]
            y = trajectories[:, i, 1]
            theta = trajectories[:, i, 2]
            v = trajectories[:, i, 3]
            w = trajectories[:, i, 4]
            a = control_sequences[:, i, 0]
            alpha = control_sequences[:, i, 1]

            # Clamp velocities to hardware limits (mirrors real motor behaviour)
            new_v = np.clip(v + a * self.dynamics.dt, self.v_min, self.v_max)
            new_w = np.clip(w + alpha * self.dynamics.dt, self.w_min, self.w_max)

            new_x = x + v * np.cos(theta) * self.dynamics.dt
            new_y = y + v * np.sin(theta) * self.dynamics.dt
            new_theta = theta + w * self.dynamics.dt

            trajectories[:, i + 1, 0] = new_x
            trajectories[:, i + 1, 1] = new_y
            trajectories[:, i + 1, 2] = new_theta
            trajectories[:, i + 1, 3] = new_v
            trajectories[:, i + 1, 4] = new_w

        return trajectories

    def validate_trajectories_vectorized(self, trajectories: np.ndarray, robot_radius: float = 0.22) -> np.ndarray:
        """Vectorized trajectory validation — collision checking only (velocities clamped in simulation)."""
        num_rollouts, num_steps, _ = trajectories.shape
        valid_mask = np.ones(num_rollouts, dtype=bool)

        if np.any(valid_mask):
            valid_trajectories = trajectories[valid_mask]
            all_x = valid_trajectories[:, :, 0].flatten()
            all_y = valid_trajectories[:, :, 1].flatten()
            collision_results = self.occupancy_map.check_collision_new(all_x, all_y, robot_radius=robot_radius)
            collision_matrix = collision_results.reshape(valid_trajectories.shape[0], num_steps)
            trajectory_collisions = np.any(collision_matrix, axis=1)
            valid_indices = np.where(valid_mask)[0]
            valid_mask[valid_indices[trajectory_collisions]] = False

        return valid_mask

    def compute_rewards_vectorized(self, trajectories: np.ndarray, goal: np.ndarray,
                                 safety_dilated_grid: np.ndarray = None,
                                 control_sequences: np.ndarray = None) -> np.ndarray:
        """Vectorized reward computation for all trajectories at once."""
        num_rollouts, num_steps, _ = trajectories.shape
        rewards = np.zeros(num_rollouts)

        x = trajectories[:, :, 0]
        y = trajectories[:, :, 1]
        distances = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        rewards = -np.sum(distances * 10.0, axis=1)

        final_x = trajectories[:, -1, 0]
        final_y = trajectories[:, -1, 1]
        terminal_distances = np.sqrt((final_x - goal[0]) ** 2 + (final_y - goal[1]) ** 2)
        rewards -= terminal_distances * 100.0

        if safety_dilated_grid is not None:
            all_x = x.flatten()
            all_y = y.flatten()
            grid_x = ((all_x - self.occupancy_map.x_min) / self.occupancy_map.resolution).astype(int)
            grid_y = ((all_y - self.occupancy_map.y_min) / self.occupancy_map.resolution).astype(int)
            out_of_bounds = ((grid_x < 0) | (grid_x >= self.occupancy_map.grid_width) |
                            (grid_y < 0) | (grid_y >= self.occupancy_map.grid_height))
            near_obstacle_violations = np.zeros(len(all_x), dtype=bool)
            valid_mask = ~out_of_bounds
            if np.any(valid_mask):
                valid_x = grid_x[valid_mask]
                valid_y = grid_y[valid_mask]
                near_obstacle_violations[valid_mask] = safety_dilated_grid[valid_y, valid_x]
            violation_matrix = near_obstacle_violations.reshape(num_rollouts, num_steps)
            violations_per_trajectory = np.sum(violation_matrix, axis=1)
            near_obstacle_penalty = violations_per_trajectory * 5  # from planner_haa_real_world.py
            rewards -= near_obstacle_penalty

        if control_sequences is not None:
            control_diff = np.diff(control_sequences, axis=1)
            control_smoothness_penalty = np.sum(control_diff**2, axis=(1, 2))
            rewards -= self.smoothing_weight * control_smoothness_penalty

        return rewards

    def plan(self, start_state: RobotState, goal: np.ndarray, debug_plot: bool = False, robot_radius: float = 0.22) -> tuple:
        """Plan using MPPI and return the control sequence and the resulting trajectory."""
        start_time = time.time()

        if self.previous_controls is not None and len(self.previous_controls) >= self.mppi.num_nodes:
            nominal_controls = np.vstack([
                self.previous_controls[1:],
                np.zeros((1, 2))
            ])
        else:
            # Cold start
            goal_x = goal[0]
            goal_y = goal[1]
            dx_world = goal_x - start_state.x
            dy_world = goal_y - start_state.y
            cos_theta = np.cos(start_state.theta)
            sin_theta = np.sin(start_state.theta)
            dx_robot = dx_world * cos_theta + dy_world * sin_theta
            dy_robot = -dx_world * sin_theta + dy_world * cos_theta
            angle_to_goal = np.arctan2(dy_robot, dx_robot)
            nominal_controls = np.zeros((self.mppi.num_nodes, 2))
            nominal_controls[:, 0] = 0.1 * self.a_max
            if angle_to_goal < 0:
                nominal_controls[:, 1] = -0.3 * self.alpha_max  # from planner_haa_real_world.py
            else:
                nominal_controls[:, 1] = 0.3 * self.alpha_max

        for iteration in range(1):
            sampled_controls = self.mppi.sample_control_knots(nominal_controls)
            sampled_controls[:, :, 0] = np.clip(sampled_controls[:, :, 0], self.a_min, self.a_max)
            sampled_controls[:, :, 1] = np.clip(sampled_controls[:, :, 1], self.alpha_min, self.alpha_max)

            all_trajectories = self.simulate_trajectories_vectorized(start_state, sampled_controls)
            valid_mask = self.validate_trajectories_vectorized(all_trajectories, robot_radius=robot_radius)

            if np.any(valid_mask):
                valid_controls = sampled_controls[valid_mask]
                valid_trajectories = all_trajectories[valid_mask]
                safe_distance = 0.050
                total_safety_radius = robot_radius + safe_distance
                safety_radius_cells = int(total_safety_radius / self.occupancy_map.resolution)
                safety_dilated_grid = self.occupancy_map.dilate_grid_new(safety_radius_cells)
                valid_rewards = self.compute_rewards_vectorized(valid_trajectories, goal, safety_dilated_grid, valid_controls)
            else:
                valid_controls = np.array([])
                valid_rewards = np.array([])

            if len(valid_controls) > 0:
                nominal_controls = self.mppi.update_nominal_knots(valid_controls, valid_rewards)
                nominal_controls[:, 0] = np.clip(nominal_controls[:, 0], self.a_min, self.a_max)
                nominal_controls[:, 1] = np.clip(nominal_controls[:, 1], self.alpha_min, self.alpha_max)
                self.previous_controls = nominal_controls.copy()
            else:
                self.previous_controls = None
                self._log_warn("No valid trajectories found!")
                if debug_plot:
                    self.debug_plot_no_valid_trajectories(start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask)
                return None, None

        control_sequence_batch = nominal_controls.reshape(1, -1, 2)
        trajectory_batch = self.simulate_trajectories_vectorized(start_state, control_sequence_batch)
        best_trajectory = trajectory_batch[0]

        return best_trajectory, nominal_controls

    def sample_debug_trajectories(self, start_state: RobotState, goal: np.ndarray, robot_radius: float):
        """Sample a fresh batch of trajectories for debug visualisation."""
        nominal = self.previous_controls.copy() if self.previous_controls is not None \
                  else np.zeros((self.mppi.num_nodes, 2))
        sampled = self.mppi.sample_control_knots(nominal)
        sampled[:, :, 0] = np.clip(sampled[:, :, 0], self.a_min, self.a_max)
        sampled[:, :, 1] = np.clip(sampled[:, :, 1], self.alpha_min, self.alpha_max)
        all_trajs  = self.simulate_trajectories_vectorized(start_state, sampled)
        valid_mask = self.validate_trajectories_vectorized(all_trajs, robot_radius)
        return sampled, nominal, all_trajs, valid_mask


class KanayamaController:
    """Kanayama/Samson-style unicycle tracking controller with integral action.

    Modified for dual-planner switching: clips to HPA or HAA limits based on active planner.
    """

    def __init__(self, kx: float = 1.0, ky: float = 1.0, kth: float = 2.0, kv: float = 1.0, kw: float = 1.0,
                 kix: float = 0.1, kiy: float = 0.1, kith: float = 0.1, max_integral: float = 1.0,
                 v_limit_haa: float = 0.2, omega_limit_haa: float = 0.9,
                 v_limit_hpa: float = 0.26, omega_limit_hpa: float = 1.82):
        self.kx = kx
        self.ky = ky
        self.kth = kth
        self.kv = kv
        self.kw = kw
        self.kix = kix
        self.kiy = kiy
        self.kith = kith
        self.max_integral = max_integral
        self.integral_ex = 0.0
        self.integral_ey = 0.0
        self.integral_eth = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_eth = 0.0
        self.dt = 0.02  # 50Hz control loop
        self.v_limit_haa = v_limit_haa
        self.omega_limit_haa = omega_limit_haa
        self.v_limit_hpa = v_limit_hpa
        self.omega_limit_hpa = omega_limit_hpa
        self.prev_v_ref = 0.0
        self.prev_w_ref = 0.0

    def compute_control(self, current_state, ref_state, current_planner="HAA"):
        """Compute control inputs using Kanayama tracking law with integral action.

        Args:
            current_state: [x, y, theta, v, w]
            ref_state: [x_ref, y_ref, theta_ref, v_ref, w_ref]
            current_planner: "HPA" or "HAA" — determines velocity clipping limits
        """
        x, y, theta, v, w = current_state
        xr, yr, thr, vr, wr = ref_state

        dx, dy = xr - x, yr - y
        ex = np.cos(theta) * dx + np.sin(theta) * dy
        ey = -np.sin(theta) * dx + np.cos(theta) * dy
        eth = thr - theta
        eth = np.arctan2(np.sin(eth), np.cos(eth))

        self.integral_ex += (ex + self.prev_ex) * self.dt / 2.0
        self.integral_ey += (ey + self.prev_ey) * self.dt / 2.0
        self.integral_eth += (eth + self.prev_eth) * self.dt / 2.0

        self.integral_ex = np.clip(self.integral_ex, -self.max_integral, self.max_integral)
        self.integral_ey = np.clip(self.integral_ey, -self.max_integral, self.max_integral)
        self.integral_eth = np.clip(self.integral_eth, -self.max_integral, self.max_integral)

        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_eth = eth

        v_cmd = vr * np.cos(eth) + self.kx * ex + self.kv * (vr - v)
        w_cmd = wr + vr * self.ky * ey + vr * self.kth * np.sin(eth) + self.kw * (wr - w)

        v_cmd += self.kix * self.integral_ex
        w_cmd += self.kiy * self.integral_ey + self.kith * self.integral_eth

        # Clip based on active planner
        if current_planner == "HPA":
            v_cmd = np.clip(v_cmd, 0, self.v_limit_hpa)
            w_cmd = np.clip(w_cmd, -self.omega_limit_hpa, self.omega_limit_hpa)
        else:  # HAA
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


# ---------------------------------------------------------------------------
# ROS2 Node: MPC+NN Switching Controller for Real World
# ---------------------------------------------------------------------------
class SwitchMPCNNRealWorldNode(Node):
    def __init__(self):
        super().__init__('switch_mpc_nn_real_world')

        # ── Parameters ──────────────────────────────────────────────────────
        # MPC/HAA parameters
        self.declare_parameter('horizon_haa', 40)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_limit_haa', 0.2)
        self.declare_parameter('omega_limit_haa', 0.9)
        self.declare_parameter('a_limit', 0.5)
        self.declare_parameter('alpha_limit', 0.5)
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('goal', '[0.0, 0.0, 0.0, 0.0, 0.0]')
        self.declare_parameter('scenario_has_random_obstacles', False)
        # NN/HPA parameters
        self.declare_parameter('v_limit_hpa', 0.26)
        self.declare_parameter('omega_limit_hpa', 1.82)
        self.declare_parameter('lidar_max_range', 2.0)
        self.declare_parameter('model_path', '')
        # Kanayama gains
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

        # ── Load parameters ─────────────────────────────────────────────────
        self.N_haa = self.get_parameter('horizon_haa').value
        self.N_hpa = 20  # braking trajectory horizon
        self.dt = self.get_parameter('dt').value
        self.v_limit_haa = self.get_parameter('v_limit_haa').value
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value
        self.v_limit_hpa = self.get_parameter('v_limit_hpa').value
        self.omega_limit_hpa = self.get_parameter('omega_limit_hpa').value
        self.a_limit = self.get_parameter('a_limit').value
        self.alpha_limit = self.get_parameter('alpha_limit').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value

        # Goal
        raw_goal = self.get_parameter('goal').value
        import ast
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal
        self.scenario_has_random_obstacles = bool(self.get_parameter('scenario_has_random_obstacles').value)
        self.goal_temp = self.goal.copy()
        self.goal_orig_history = []
        self.goal_temp_history = []
        self.state_traj = []

        # ── State flags ─────────────────────────────────────────────────────
        self.HAA_map_ready = False
        self.state_ready = False
        self.lidar_ready = False
        self.goal_received = False
        self._goal_wait_logged = False
        self.target_reached = False
        self.shutdown_requested = False
        self.step = 0
        self.HAA_mppi_planner = None
        self.HAA_occupancy_map = None
        self.HAA_raw_occ = None

        # ── Switching state ─────────────────────────────────────────────────
        self.current_planner = "HPA"  # Start in NN mode
        self.haa_check_failed_first_time = False
        self.braking_loop_count = 0
        self.switch_point = []  # Records (x,y,theta,v,w) at each switch
        self.planner_history = []  # Records planner name at each planning step (synced with state_traj)

        # ── Trajectory tracking ─────────────────────────────────────────────
        self.current_trajectory = None
        self.current_controls = None
        self.trajectory_start_time = None
        self.trajectory_ready = False
        self.first_plan = True

        # ── Failure save path ───────────────────────────────────────────────
        self.failure_path = self.get_parameter('failure_path').value
        self.plot_data = []
        self.plot_path = os.path.join(os.path.expanduser('~'), 'plot_data_lowlvlctrl.npz')
        self.trajectory_path = os.path.join(os.path.expanduser('~'), 'robot_trajectory_switch.png')
        self.Xopt_history = []
        self.control_command_history = []

        # ── Low-pass filter and rate-limiter state ──────────────────────────
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0

        # ── /odom velocity filter constants ─────────────────────────────────
        self._ODOM_V_MAX     = 0.26
        self._ODOM_OMEGA_MAX = 1.82
        self._ODOM_EMA_ALPHA = 0.3

        # ── LiDAR scan ──────────────────────────────────────────────────────
        self.latest_lidar_scan = None

        # ── Kanayama controller ─────────────────────────────────────────────
        self.kx = self.get_parameter('kx').value
        self.ky = self.get_parameter('ky').value
        self.kth = self.get_parameter('kth').value
        self.kv = self.get_parameter('kv').value
        self.kw = self.get_parameter('kw').value
        self.kix = self.get_parameter('kix').value
        self.kiy = self.get_parameter('kiy').value
        self.kith = self.get_parameter('kith').value
        self.max_integral = self.get_parameter('max_integral').value

        self.kanayama_controller = KanayamaController(
            self.kx, self.ky, self.kth, self.kv, self.kw,
            self.kix, self.kiy, self.kith, self.max_integral,
            v_limit_haa=self.v_limit_haa, omega_limit_haa=self.omega_limit_haa,
            v_limit_hpa=self.v_limit_hpa, omega_limit_hpa=self.omega_limit_hpa
        )

        # ── Robot state ─────────────────────────────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.vx_world = 0.0
        self.vy_world = 0.0
        self.w_world  = 0.0

        # ── Stuck detection ─────────────────────────────────────────────────
        self._stuck_timeout_sec   = 10.0
        self._stuck_threshold_m   = 0.1
        self._stuck_last_prog_time = None
        self._stuck_last_prog_pos  = None

        # ── Load NN model ───────────────────────────────────────────────────
        _param_path = self.get_parameter('model_path').value.strip()
        if _param_path:
            model_path = os.path.expanduser(_param_path)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f'NN model not found at {model_path}')
            raise FileNotFoundError(f'NN model not found at {model_path}')

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.nn_model = MLP(input_dim=364, hidden_dims=[256, 128, 64],
                                output_dim=2, dropout=0.1)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.nn_model.load_state_dict(checkpoint['model_state_dict'])
                self.get_logger().info(f'NN model loaded from wrapped checkpoint: {model_path}')
            elif isinstance(checkpoint, dict):
                self.nn_model.load_state_dict(checkpoint)
                self.get_logger().info(f'NN model loaded from state dict: {model_path}')
            else:
                self.nn_model = checkpoint
                self.get_logger().info(f'NN model loaded (full model object): {model_path}')
            self.nn_model.eval()
        except Exception as e:
            self.get_logger().error(f'Failed to load NN model: {e}')
            raise

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS pubs / subs ─────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.inflation_map_pub = self.create_publisher(OccupancyGrid, '/inflation_map', 1)
        self.slam_pose_pub = self.create_publisher(Odometry, '/slam_pose', 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        # Real-world LiDAR with Best-Effort QoS
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, scan_qos)

        # ── Timers ──────────────────────────────────────────────────────────
        self.state_timer = self.create_timer(0.01, self.state_update_cb)   # 100Hz TF2 pose
        self.planning_timer = self.create_timer(0.1, self.planning_loop)   # 10Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)    # 50Hz

        self.get_logger().info("Real-world MPC+NN switching planner node started:")
        self.get_logger().info("  - Mode: HPA (NN) ↔ HAA (MPC) switching")
        self.get_logger().info("  - State source  : TF2 map→base_footprint (pose) + /odom (velocity)")
        self.get_logger().info("  - Map source    : /map (Cartographer SLAM)")
        self.get_logger().info("  - LiDAR source  : /scan (real TurtleBot3 LDS-01)")
        self.get_logger().info("  - Goal source   : /move_base_simple/goal (RViz2 2D Goal Pose)")
        self.get_logger().info(f"  - HAA limits    : v={self.v_limit_haa} m/s, ω={self.omega_limit_haa} rad/s")
        self.get_logger().info(f"  - HPA limits    : v={self.v_limit_hpa} m/s, ω={self.omega_limit_hpa} rad/s")
        self.get_logger().info(f"  - lidar_max_range clipped to {self.lidar_max_range} m (training range)")
        self.get_logger().info("Waiting for /map, TF2, and /scan ...")

    # ── Utility ─────────────────────────────────────────────────────────────

    def _time_to_sec(self, ros_time):
        return float(ros_time.nanoseconds) * 1e-9

    def _stop_timers(self):
        for timer in [self.planning_timer, self.control_timer]:
            if timer is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass

    # ── LiDAR callback (from planner_nn_real_world.py) ──────────────────────

    def lidar_cb(self, msg: LaserScan):
        """Convert real LiDAR → 360-element float32 array in [0, lidar_max_range]."""
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            max_r  = self.lidar_max_range
            bad_mask = ~np.isfinite(ranges) | (ranges <= 0.0)
            ranges[bad_mask] = max_r
            np.clip(ranges, 0.0, max_r, out=ranges)
            n = len(ranges)
            if n != 360:
                indices = np.linspace(0, n - 1, 360)
                ranges  = np.interp(indices, np.arange(n), ranges).astype(np.float32)
            self.latest_lidar_scan = ranges
            self.lidar_ready = True
        except Exception as e:
            self.get_logger().warn(f'lidar_cb error: {e}')

    # ── /odom velocity callback ─────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        """Read velocity from /odom twist with clipping + EMA smoothing."""
        v_raw = max(0.0, min(msg.twist.twist.linear.x, self._ODOM_V_MAX))
        w_raw = max(-self._ODOM_OMEGA_MAX, min(msg.twist.twist.angular.z, self._ODOM_OMEGA_MAX))
        a = self._ODOM_EMA_ALPHA
        self.v     = a * v_raw + (1.0 - a) * self.v
        self.omega = a * w_raw + (1.0 - a) * self.omega
        self.vx_world = self.v * np.cos(self.theta)
        self.vy_world = self.v * np.sin(self.theta)
        self.w_world  = self.omega

    # ── TF2 pose estimation ─────────────────────────────────────────────────

    def state_update_cb(self):
        """Look up map→base_footprint TF at 100 Hz for position and heading."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
            self.x     = t.transform.translation.x
            self.y     = t.transform.translation.y
            q          = t.transform.rotation
            _, _, self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.state_ready = True

            # Publish SLAM pose for recording / validation
            slam_msg = Odometry()
            slam_msg.header.stamp    = self.get_clock().now().to_msg()
            slam_msg.header.frame_id = 'map'
            slam_msg.child_frame_id  = 'base_footprint'
            slam_msg.pose.pose.position.x = self.x
            slam_msg.pose.pose.position.y = self.y
            slam_msg.pose.pose.position.z = 0.0
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.theta)
            slam_msg.pose.pose.orientation.x = qx
            slam_msg.pose.pose.orientation.y = qy
            slam_msg.pose.pose.orientation.z = qz
            slam_msg.pose.pose.orientation.w = qw
            slam_msg.twist.twist.linear.x  = self.v
            slam_msg.twist.twist.angular.z = self.omega
            self.slam_pose_pub.publish(slam_msg)

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    # ── Map callback ────────────────────────────────────────────────────────

    def map_cb(self, msg: OccupancyGrid):
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.HAA_raw_occ = data.copy()
        self.HAA_occupancy_map = OccupancyGridMap(
            width=msg.info.width * msg.info.resolution,
            height=msg.info.height * msg.info.resolution,
            resolution=msg.info.resolution
        )
        self.HAA_occupancy_map.occupancy_grid = data
        self.HAA_occupancy_map.x_min = msg.info.origin.position.x
        self.HAA_occupancy_map.y_min = msg.info.origin.position.y
        self.HAA_occupancy_map.x_max = msg.info.origin.position.x + msg.info.width * msg.info.resolution
        self.HAA_occupancy_map.y_max = msg.info.origin.position.y + msg.info.height * msg.info.resolution
        self.HAA_map_ready = True
        self.get_logger().debug("HAA map received: dynamic occupancy grid map built.")
        self._publish_inflation_map(msg)

    def _publish_inflation_map(self, source_msg: OccupancyGrid):
        """Publish /inflation_map for RViz2 visualisation of obstacle inflation zones."""
        omap = self.HAA_occupancy_map
        raw = omap.occupancy_grid
        res = omap.resolution
        _soft_extra = 0.05
        hard_radius_cells = int(self.robot_radius / res)
        soft_radius_cells = int((self.robot_radius + _soft_extra) / res)
        obstacle_mask = raw > 50
        if np.any(obstacle_mask):
            dist_cells = ndimage.distance_transform_edt(~obstacle_mask)
        else:
            dist_cells = np.full(raw.shape, 9999.0)
        viz = np.zeros(raw.shape, dtype=np.int8)
        viz[dist_cells <= soft_radius_cells] = 20
        viz[dist_cells <= hard_radius_cells] = 60
        viz[obstacle_mask] = 100
        out = OccupancyGrid()
        out.header = source_msg.header
        out.info   = source_msg.info
        out.data   = viz.flatten().tolist()
        self.inflation_map_pub.publish(out)

    # ── Goal callback ───────────────────────────────────────────────────────

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
        self.HAA_mppi_planner = None   # reset warm-start
        self.step = 0
        self.state_traj = []
        self.planner_history = []
        self.switch_point = []
        self._goal_wait_logged = False
        self.kanayama_controller.reset_integral_errors()

        # Reset switching state
        self.current_planner = "HPA"
        self.haa_check_failed_first_time = False
        self.braking_loop_count = 0
        self.current_trajectory = None
        self.current_controls = None

        # Reset stuck tracker
        self._stuck_last_prog_time = self.get_clock().now().nanoseconds * 1e-9
        self._stuck_last_prog_pos  = (self.x, self.y)

        self.get_logger().info(
            f"New goal from RViz2: x={x:.3f} m, y={y:.3f} m, theta={np.degrees(theta):.1f} deg"
        )

    # ── NN helpers ──────────────────────────────────────────────────────────

    def _transform_goal_to_robot_frame(self, gx: float, gy: float):
        """Transform goal from map frame to robot body frame."""
        dx = gx - self.x
        dy = gy - self.y
        cos_t = np.cos(-self.theta)
        sin_t = np.sin(-self.theta)
        return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t

    def get_nn_control(self):
        """Get (v_cmd, w_cmd) from NN for current state and goal. Returns None if lidar not ready."""
        if self.latest_lidar_scan is None:
            return None
        goal_x_robot, goal_y_robot = self._transform_goal_to_robot_frame(
            self.goal_temp[0], self.goal_temp[1]
        )
        inp = np.concatenate(
            [[self.v, self.omega], [goal_x_robot, goal_y_robot], self.latest_lidar_scan],
            dtype=np.float32
        )
        tensor = torch.FloatTensor(inp).unsqueeze(0)
        with torch.no_grad():
            out = self.nn_model(tensor).cpu().numpy().flatten()
        v_cmd = float(np.clip(out[0], 0.0, self.v_limit_hpa))
        w_cmd = float(np.clip(out[1], -self.omega_limit_hpa, self.omega_limit_hpa))
        return v_cmd, w_cmd

    def generate_braking_trajectory(self, start_state: RobotState, horizon: int):
        """Generate a braking trajectory with max braking (a = -a_limit*1.2, alpha = 0)."""
        dynamics = TurtlebotDynamics(dt=self.dt)
        trajectory = np.zeros((horizon + 1, 5))
        controls = np.zeros((horizon, 2))

        trajectory[0] = [start_state.x, start_state.y, start_state.theta, start_state.v, start_state.w]

        max_braking_a = -self.a_limit * 1.2
        controls[:, 0] = max_braking_a
        controls[:, 1] = 0.0

        current_state = RobotState(
            x=start_state.x, y=start_state.y, theta=start_state.theta,
            v=start_state.v, w=start_state.w
        )

        for i in range(horizon):
            current_state = dynamics.step(current_state, np.array([max_braking_a, 0.0]))
            current_state.v = max(0.0, current_state.v)
            trajectory[i + 1] = [current_state.x, current_state.y, current_state.theta,
                                  current_state.v, current_state.w]

        return trajectory, controls

    # ── Planning loop — switching state machine ─────────────────────────────

    def planning_loop(self, event=None):
        """Dual system: HPA = NN (no MPC), HAA = MPC (4s horizon, dynamic map).
        While on HPA: haa_check_state = current + NN output; check HAA feasibility; switch to HAA if not.
        While on HAA: run HAA MPPI; switch back to HPA when in safety margin."""
        if self.shutdown_requested:
            return

        if not (self.HAA_map_ready and self.state_ready):
            if not self.HAA_map_ready:
                self.get_logger().warn("HAA map not ready — waiting for /map from Cartographer")
            if not self.state_ready:
                self.get_logger().warn("State not ready — waiting for TF2 map→base_footprint")
            return

        # Wait for goal from RViz2
        if not self.goal_received:
            if not self._goal_wait_logged:
                self.get_logger().info(
                    "Ready. Set a goal using '2D Goal Pose' button in RViz2 "
                    "(publishes to /move_base_simple/goal)."
                )
                self._goal_wait_logged = True
            return

        # Current state — velocity clamping (Tube MPC pattern from planner_haa_real_world.py)
        x_true = [self.x, self.y, self.theta, self.v, self.omega]
        if self.v > self.v_limit_haa - 0.005:
            self.v = self.v_limit_haa - 0.005
        if self.v < 0:
            self.v = 0.01
        if self.omega > self.omega_limit_haa:
            self.omega = self.omega_limit_haa - 0.005
        if self.omega < -self.omega_limit_haa:
            self.omega = -self.omega_limit_haa + 0.005
        x0 = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(x_true.copy())
        self.planner_history.append(self.current_planner)

        # Goal-reached check
        dist = np.linalg.norm(np.array(x_true[:2]) - np.array(self.goal_temp[:2]))
        if dist < 0.1 and not self.target_reached:
            self.target_reached = True
            self.publish_stop_command()
            self.save_trajectory_plot()
            self.goal_received = False
            self.trajectory_ready = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            self.current_planner = "HPA"
            self.get_logger().info(
                f"Goal reached! (distance={dist:.3f} m). "
                "Robot stopped. Set next goal via '2D Goal Pose' in RViz2."
            )
            return

        # Stuck-timeout check
        if self._stuck_last_prog_time is not None:
            _now = self.get_clock().now().nanoseconds * 1e-9
            _disp = np.hypot(self.x - self._stuck_last_prog_pos[0],
                             self.y - self._stuck_last_prog_pos[1])
            if _disp >= self._stuck_threshold_m:
                self._stuck_last_prog_time = _now
                self._stuck_last_prog_pos  = (self.x, self.y)
            elif (_now - self._stuck_last_prog_time) > self._stuck_timeout_sec:
                _elapsed = _now - self._stuck_last_prog_time
                self.get_logger().warn(
                    f"STUCK TIMEOUT: robot has not moved {self._stuck_threshold_m}m "
                    f"in {_elapsed:.1f}s. Aborting goal and saving debug plot."
                )
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
                self.current_planner = "HPA"
                return

        try:
            self.goal_temp = self.goal.copy()

            # Initialize HAA planner if needed
            if self.HAA_mppi_planner is None:
                dynamics = TurtlebotDynamics(dt=self.dt)
                haa_horizon = self.N_haa
                haa_mppi = MPPI(
                    sigma=5,
                    temperature=1.0,
                    num_nodes=haa_horizon,
                    num_rollouts=1500,
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

            # === SWITCHING LOGIC ===
            if self.current_planner == "HPA":
                # === BRAKING MODE CHECK ===
                if self.haa_check_failed_first_time:
                    self.braking_loop_count += 1
                    self.get_logger().info(f"Braking mode: loop {self.braking_loop_count}/2")

                    braking_start_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2], v=x_true[3], w=x_true[4])
                    braking_trajectory, braking_controls = self.generate_braking_trajectory(
                        braking_start_state, self.N_hpa
                    )
                    self.current_trajectory = braking_trajectory
                    self.current_controls = braking_controls
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    self.current_planner = "HPA"

                    if self.braking_loop_count >= 2:
                        self.get_logger().info("Braking phase complete — switching to HAA")
                        self.current_planner = "HAA"
                        self.haa_check_failed_first_time = False
                        self.braking_loop_count = 0
                        self.kanayama_controller.reset_integral_errors()
                        return
                    else:
                        return

                # HPA = NN: need lidar
                if not self.lidar_ready or self.latest_lidar_scan is None:
                    self.get_logger().warn("HPA (NN) waiting for lidar")
                    return

                nn_out = self.get_nn_control()
                if nn_out is None:
                    return
                v_nn, w_nn = nn_out

                # haa_check_state = one-step prediction with NN output + N_r braking steps
                dynamics = TurtlebotDynamics(dt=self.dt)
                a = (v_nn - x_true[3]) / self.dt
                alpha = (w_nn - x_true[4]) / self.dt
                a = np.clip(a, -self.a_limit, self.a_limit)
                alpha = np.clip(alpha, -self.alpha_limit, self.alpha_limit)
                current_robot_state = RobotState(
                    x=x_true[0], y=x_true[1], theta=x_true[2],
                    v=x_true[3], w=x_true[4]
                )
                haa_check_state = dynamics.step(current_robot_state, np.array([a, alpha]))

                # Apply N_r braking steps for recoverable-set check
                N_r = 3
                max_braking_control = np.array([-self.a_limit, 0])
                for _ in range(N_r):
                    haa_check_state = dynamics.step(haa_check_state, max_braking_control)
                    haa_check_state.v = max(0, haa_check_state.v)

                # HAA feasibility check
                haa_check_start_time = time.time()
                haa_check_Xopt, haa_check_Uopt = self.HAA_mppi_planner.plan(
                    haa_check_state, np.array(self.goal_temp[:2]), robot_radius=self.robot_radius + 0.05
                )
                haa_check_end_time = time.time()
                haa_check_feasible = (
                    haa_check_Xopt is not None and haa_check_Uopt is not None
                    and len(haa_check_Xopt) > 0 and len(haa_check_Uopt) > 0
                )
                self.get_logger().info(f'HAA check feasible: {haa_check_feasible}')
                self.get_logger().info(f'HAA check time: {haa_check_end_time - haa_check_start_time:.6f}s')

                if haa_check_feasible:
                    # Stay on HPA (NN); control loop will use NN output directly
                    self.current_trajectory = None
                    self.current_controls = None
                    self.trajectory_ready = True
                    self.current_planner = "HPA"
                else:
                    # HAA check failed — start 2-loop braking then switch to HAA
                    if not self.haa_check_failed_first_time:
                        self.get_logger().info('HAA check failed — starting 2-loop braking phase')
                        self.get_logger().info(f'HAA check state: x={haa_check_state.x:.3f}, y={haa_check_state.y:.3f}, '
                                               f'theta={haa_check_state.theta:.3f}, v={haa_check_state.v:.3f}, w={haa_check_state.w:.3f}')
                        self.switch_point.append(x_true.copy())
                        self.haa_check_failed_first_time = True
                        self.braking_loop_count = 1
                        braking_start_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2], v=x_true[3], w=x_true[4])
                        braking_trajectory, braking_controls = self.generate_braking_trajectory(
                            braking_start_state, self.N_hpa
                        )
                        self.current_trajectory = braking_trajectory
                        self.current_controls = braking_controls
                        self.trajectory_start_time = self.get_clock().now()
                        self.trajectory_ready = True
                        self.current_planner = "HPA"
                        self.get_logger().info(f"Braking mode: loop {self.braking_loop_count}/2")
                    else:
                        self.get_logger().warn("HAA check failed but already in braking mode — unexpected state")
                        self.trajectory_ready = True
                        self.current_planner = "HPA"

            elif self.current_planner == "HAA":
                haa_current_state = RobotState(
                    x=x0[0], y=x0[1], theta=x0[2],
                    v=np.clip(x0[3], 0, self.v_limit_haa),
                    w=np.clip(x0[4], -self.omega_limit_haa, self.omega_limit_haa)
                )
                haa_start_time = time.time()
                haa_Xopt, haa_Uopt = self.HAA_mppi_planner.plan(
                    haa_current_state, np.array(self.goal_temp[:2]),
                    debug_plot=True, robot_radius=self.robot_radius
                )
                haa_end_time = time.time()

                if haa_Xopt is not None and haa_Uopt is not None and len(haa_Xopt) > 0 and len(haa_Uopt) > 0:
                    self.current_trajectory = haa_Xopt
                    self.current_controls = haa_Uopt
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    self.current_planner = "HAA"
                    self.haa_check_failed_first_time = False
                    self.braking_loop_count = 0
                    self.get_logger().debug(f"HAA planning time: {haa_end_time - haa_start_time:.6f}s")

                    # Safety margin check for switching back to HPA
                    N_s = 1
                    # add log to see if we are close to the velocity threshold for switching back to HPA
                    self.get_logger().debug(f"HAA current velocity: {haa_current_state.v:.3f}, "
                                            f"velocity limit: {self.v_limit_haa - N_s * self.dt * self.a_limit:.3f}")
                    # if haa_current_state.v < self.v_limit_haa - N_s * self.dt * self.a_limit:
                    safety_margin = N_s * self.dt * self.v_limit_haa + self.v_limit_haa**2 / (2 * self.a_limit) + 0.05
                    collisions = self.HAA_mppi_planner.occupancy_map.check_collision_new(
                        np.array([haa_current_state.x]),
                        np.array([haa_current_state.y]),
                        self.robot_radius + safety_margin
                    )
                    
                    if not np.any(collisions):
                        self.get_logger().info("Safety margin satisfied — switching back to HPA (NN)")
                        self.current_planner = "HPA"
                        self.current_trajectory = None
                        self.current_controls = None
                        self.switch_point.append(x_true.copy())
                        return
                else:
                    # HAA planning failed — stop robot and wait for new goal
                    self.get_logger().warn("HAA planning failed after switch — no feasible trajectory")
                    self.get_logger().info(
                        f'State at failure: x={haa_current_state.x:.3f}, y={haa_current_state.y:.3f}, '
                        f'theta={haa_current_state.theta:.3f}, v={haa_current_state.v:.3f}, w={haa_current_state.w:.3f}'
                    )
                    self.trajectory_ready = False
                    self.publish_stop_command()
                    self.goal_received = False
                    self.HAA_mppi_planner = None
                    self._goal_wait_logged = False
                    self.current_planner = "HPA"
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
                    'planner': self.current_planner
                }
                self.Xopt_history.append(xopt_data)

            self.kanayama_controller.reset_integral_errors()

        except Exception as e:
            self.get_logger().error(f"Planning exception at step {self.step}: {e}")
            np.savez(self.failure_path, x0=x0, goal=self.goal_temp, occ=self.HAA_raw_occ)
            self.get_logger().info(f"Failure data saved to {self.failure_path}")
            self.publish_stop_command()
            self.goal_received = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            self.current_planner = "HPA"
            return

        self.step += 1

    # ── Reference state computation ─────────────────────────────────────────

    def get_reference_state(self, current_time):
        """Get reference state from planned trajectory (2-step lookahead with feedback)."""
        if not self.trajectory_ready or self.current_trajectory is None or self.current_controls is None:
            return None
        if len(self.current_trajectory) == 0 or len(self.current_controls) == 0:
            return None

        x_true = np.array([self.x, self.y, self.theta, self.v, self.omega])
        x_nominal = np.array(self.current_trajectory[0])
        u_nominal = np.array(self.current_controls[0])
        error = x_true - x_nominal

        K_feedback = [[0.0, 0.0, 0.0, -2, 0.0], [0.0, 0.0, 0, 0.0, -0.6]]
        u = u_nominal + K_feedback @ error

        if self.HAA_mppi_planner is not None and self.HAA_mppi_planner.dynamics is not None:
            dynamics = self.HAA_mppi_planner.dynamics
        else:
            dynamics = TurtlebotDynamics(dt=self.dt)

        current_robot_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2],
                                        v=x_true[3], w=x_true[4])
        ref_robot_state = dynamics.step(current_robot_state, u)
        ref_robot_state = dynamics.step(ref_robot_state, self.current_controls[1])

        ref_state = self.current_trajectory[2]  # tested — works well
        return ref_state

    # ── Control loop — dual-mode ────────────────────────────────────────────

    def control_loop(self, event=None):
        """Control loop at 50Hz: NN direct (HPA) or Kanayama tracking (HAA/braking)."""
        if self.shutdown_requested:
            return
        if not self.state_ready:
            return
        if not self.trajectory_ready:
            return

        current_time = self.get_clock().now()

        if self.current_planner == "HPA" and self.current_trajectory is not None:
            # HPA braking mode — use Kanayama tracking on braking trajectory
            ref_state = self.get_reference_state(current_time)
            if ref_state is None:
                self.publish_stop_command()
                return
            current_state = [self.x, self.y, self.theta, self.v, self.omega]
            v_cmd, w_cmd = self.kanayama_controller.compute_control(
                current_state, ref_state, current_planner="HPA"
            )
        elif self.current_planner == "HPA" and self.current_trajectory is None:
            # HPA NN mode — run inference directly
            if not self.lidar_ready or self.latest_lidar_scan is None:
                return
            nn_out = self.get_nn_control()
            if nn_out is None:
                return
            v_cmd, w_cmd = nn_out
            # Publish NN output directly without EMA/rate-limiter
            twist = Twist()
            twist.linear.x = v_cmd
            twist.angular.z = w_cmd
            self.cmd_pub.publish(twist)
            self.control_command_history.append({
                'timestamp': self._time_to_sec(current_time),
                'v_cmd': v_cmd,
                'w_cmd': w_cmd,
                'current_state': [self.x, self.y, self.theta, self.v, self.omega],
                'planner': 'HPA_NN'
            })
            # Seed the EMA + rate-limiter state with the command we just
            # published so the next braking/HAA tick picks up from the last
            # NN command (~0.26 m/s) instead of 0. Without this seed, the
            # filter starts from its stale value (0 after init or stop) and
            # the rate-limiter clamps cmd_v to ~0 for several ticks at the
            # HPA→HAA boundary.
            self.v_cmd_filtered = v_cmd
            self.w_cmd_filtered = w_cmd
            self.v_cmd_prev     = v_cmd
            self.w_cmd_prev     = w_cmd
            return
        elif self.current_planner == "HAA":
            # HAA mode — Kanayama tracking on MPC trajectory
            ref_state = self.get_reference_state(current_time)
            if ref_state is None:
                self.publish_stop_command()
                return
            current_state = [self.x, self.y, self.theta, self.v, self.omega]
            v_cmd, w_cmd = self.kanayama_controller.compute_control(
                current_state, ref_state, current_planner="HAA"
            )
        else:
            return

        # EMA low-pass filter (for braking and HAA modes)
        _alpha = 0.6
        v_smooth = _alpha * self.v_cmd_filtered + (1.0 - _alpha) * v_cmd
        w_smooth = _alpha * self.w_cmd_filtered + (1.0 - _alpha) * w_cmd

        # Rate limiter
        _dt_ctrl = 0.02
        _max_dv = self.a_limit     * _dt_ctrl
        _max_dw = self.alpha_limit * _dt_ctrl
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
            f"[{self.current_planner}] cmd_vel: v={self.v_cmd_filtered:.3f} w={self.w_cmd_filtered:.3f} | "
            f"raw: v={v_cmd:.3f} w={w_cmd:.3f} | "
            f"state v={self.v:.3f} w={self.omega:.3f}",
            throttle_duration_sec=0.2,
        )

        self.control_command_history.append({
            'timestamp': self._time_to_sec(current_time),
            'v_cmd': self.v_cmd_filtered,
            'w_cmd': self.w_cmd_filtered,
            'current_state': [self.x, self.y, self.theta, self.v, self.omega],
            'planner': self.current_planner
        })

    # ── Stop command ────────────────────────────────────────────────────────

    def publish_stop_command(self):
        """Publish stop command and reset filter state."""
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ── Save data on exit ───────────────────────────────────────────────────

    def save_data_on_exit(self):
        """Save data when the program exits."""
        if hasattr(self, 'x') and hasattr(self, 'y') and hasattr(self, 'theta'):
            x0 = [self.x, self.y, self.theta, self.v, self.omega]
            haa_occ_data = self.HAA_occupancy_map.occupancy_grid if self.HAA_occupancy_map else None
            np.savez(self.failure_path, x0=x0, goal=self.goal_temp,
                     haa_occ=haa_occ_data,
                     goal_orig_history=self.goal_orig_history, goal_temp_history=self.goal_temp_history,
                     state_traj=self.state_traj, control_command_history=self.control_command_history,
                     switch_points=self.switch_point)

    # ── Trajectory plot ─────────────────────────────────────────────────────

    def save_trajectory_plot(self):
        """Save a plot of the robot trajectory with switch points marked."""
        if len(self.state_traj) < 2:
            self.get_logger().warn("Not enough trajectory data to plot")
            return

        try:
            x_coords = [state[0] for state in self.state_traj]
            y_coords = [state[1] for state in self.state_traj]
            theta_coords = [state[2] for state in self.state_traj]
            v_coords = [state[3] for state in self.state_traj]
            w_coords = [state[4] for state in self.state_traj]
            theta_coords_unwrapped = np.unwrap(np.array(theta_coords))
            time_steps = np.arange(len(self.state_traj)) * self.dt

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

            # Occupancy map background with inflation zones
            _p_soft = _p_hard = _p_obs = None
            if self.HAA_occupancy_map is not None:
                omap = self.HAA_occupancy_map
                raw  = omap.occupancy_grid
                res  = omap.resolution
                _safe_dist = 0.10
                hard_r = int(self.robot_radius / res)
                soft_r = int((self.robot_radius + _safe_dist) / res)
                obstacle_mask = raw > 50
                if np.any(obstacle_mask):
                    dist_c = ndimage.distance_transform_edt(~obstacle_mask)
                else:
                    dist_c = np.full(raw.shape, 9999.0)
                rgba = np.ones((*raw.shape, 4), dtype=float)
                rgba[dist_c <= soft_r] = [1.0, 1.0, 0.45, 1.0]
                rgba[dist_c <= hard_r] = [1.0, 0.55, 0.10, 1.0]
                rgba[obstacle_mask]    = [0.15, 0.15, 0.15, 1.0]
                ax1.imshow(rgba,
                           extent=[omap.x_min, omap.x_max, omap.y_min, omap.y_max],
                           origin='lower', aspect='equal', zorder=0)
                _p_soft = patches.Rectangle((0,0),1,1, facecolor=[1.0,1.0,0.45], label=f'Soft zone (+{_safe_dist}m)')
                _p_hard = patches.Rectangle((0,0),1,1, facecolor=[1.0,0.55,0.10], label=f'Hard zone (robot_r={self.robot_radius}m)')
                _p_obs  = patches.Rectangle((0,0),1,1, facecolor=[0.15,0.15,0.15], label='Obstacle')

            # Position trajectory coloured by active planner (HPA=blue, HAA=red,
            # BRAKING=orange). Mode transitions are visible as colour changes in the
            # line itself instead of separate switch-point markers.
            _traj_color_map = {"HPA": "#2196F3", "HAA": "#E53935", "BRAKING": "#FF9800"}
            _traj_label_map = {"HPA": "HPA (NN)", "HAA": "HAA (MPC)", "BRAKING": "Braking"}

            _N_traj = len(x_coords)
            _planners = list(self.planner_history) if self.planner_history else []
            # Align planner history length with trajectory length
            if len(_planners) < _N_traj:
                _planners = _planners + [_planners[-1] if _planners else "HPA"] * (_N_traj - len(_planners))
            elif len(_planners) > _N_traj:
                _planners = _planners[:_N_traj]

            # Build contiguous segments of identical mode
            _segs = []
            if _N_traj >= 1 and _planners:
                _s = 0
                for _i in range(1, _N_traj):
                    if _planners[_i] != _planners[_s]:
                        _segs.append((_s, _i - 1, _planners[_s]))
                        _s = _i
                _segs.append((_s, _N_traj - 1, _planners[_s]))

                # Re-label the last ~2 steps of any HPA segment that ends in a switch
                # to HAA as BRAKING (2 planning loops of deceleration at 10 Hz)
                for _si in range(len(_segs)):
                    _ss, _se, _sm = _segs[_si]
                    if _sm == "HPA" and _si + 1 < len(_segs) and _segs[_si + 1][2] == "HAA":
                        _brake_start = max(_ss, _se - 1)
                        if _brake_start > _ss:
                            _segs[_si] = (_ss, _brake_start - 1, "HPA")
                            _segs.insert(_si + 1, (_brake_start, _se, "BRAKING"))

            # Plot each segment with its planner colour. Overlap endpoints by +1 so
            # consecutive segments visually connect without gaps.
            _legend_added = set()
            if _segs:
                for _ss, _se, _sm in _segs:
                    _end = min(_se + 2, _N_traj)  # +1 for inclusive, +1 for overlap
                    _lbl = _traj_label_map.get(_sm, _sm) if _sm not in _legend_added else ''
                    ax1.plot(x_coords[_ss:_end], y_coords[_ss:_end],
                             color=_traj_color_map.get(_sm, 'gray'),
                             linewidth=2, zorder=3, label=_lbl)
                    _legend_added.add(_sm)
            else:
                ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Robot Trajectory')

            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start Position', zorder=5)
            ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=14, label='Target Position', zorder=5)

            # Robot radius tube
            positions = np.column_stack([x_coords, y_coords])
            if len(positions) > 1 and _HAS_SHAPELY:
                buffered = LineString(positions).buffer(self.robot_radius)
                if buffered.geom_type == 'Polygon':
                    bx, by = buffered.exterior.xy
                    tube_polygon = np.column_stack([bx, by])
                    tube_patch = patches.Polygon(
                        tube_polygon, facecolor='lightblue', edgecolor='blue',
                        alpha=0.35, linewidth=1, label='Robot Radius'
                    )
                    ax1.add_patch(tube_patch)
                elif buffered.geom_type == 'MultiPolygon':
                    for j, poly in enumerate(buffered.geoms):
                        bx, by = poly.exterior.xy
                        tube_polygon = np.column_stack([bx, by])
                        tube_patch = patches.Polygon(
                            tube_polygon, facecolor='lightblue', edgecolor='blue',
                            alpha=0.35, linewidth=1,
                            label='Robot Radius' if j == 0 else ''
                        )
                        ax1.add_patch(tube_patch)

            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('Robot Position Trajectory (MPC+NN Switching)')
            _extra_handles = [h for h in [_p_soft, _p_hard, _p_obs] if h is not None]
            _leg_handles, _ = ax1.get_legend_handles_labels()
            ax1.legend(handles=_leg_handles + _extra_handles)
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Helper: plot a (time, value) series coloured by planner mode segments.
            # `modes` is a list the same length as `xs`/`ys` giving the mode tag at each
            # sample. Segments are joined by overlapping one index so the colour-change
            # points connect visually without gaps.
            def _plot_by_mode(ax, xs, ys, modes, add_legend=True):
                if len(xs) == 0:
                    return
                _N = len(xs)
                # Build contiguous mode segments
                _ss_list = []
                _s = 0
                for _i in range(1, _N):
                    if modes[_i] != modes[_s]:
                        _ss_list.append((_s, _i - 1, modes[_s]))
                        _s = _i
                _ss_list.append((_s, _N - 1, modes[_s]))
                _seen = set()
                for _a, _b, _m in _ss_list:
                    _end = min(_b + 2, _N)
                    _lbl = _traj_label_map.get(_m, _m) if add_legend and _m not in _seen else ''
                    ax.plot(xs[_a:_end], ys[_a:_end],
                            color=_traj_color_map.get(_m, 'gray'),
                            linewidth=1.8, zorder=3, label=_lbl)
                    _seen.add(_m)

            # Mode tags aligned with state_traj (used by ax2/ax3/ax4). Reuse the
            # BRAKING-relabelled segments from ax1 to produce a per-sample mode list.
            _state_modes = list(_planners) if _planners else ["HPA"] * _N_traj
            for _ss, _se, _sm in _segs:
                for _k in range(_ss, min(_se + 1, _N_traj)):
                    _state_modes[_k] = _sm

            # Orientation — coloured by planner
            _plot_by_mode(ax2, time_steps, theta_coords_unwrapped, _state_modes)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Orientation (rad)')
            ax2.set_title('Robot Orientation vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Linear velocity (odom-measured) — coloured by planner
            _plot_by_mode(ax3, time_steps, v_coords, _state_modes)
            ax3.axhline(y=self.v_limit_haa, color='#E53935', linestyle='--', alpha=0.6,
                        label=f'HAA limit ({self.v_limit_haa} m/s)')
            ax3.axhline(y=self.v_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6,
                        label=f'HPA limit ({self.v_limit_hpa} m/s)')
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Linear Velocity (m/s)')
            ax3.set_title('Linear Velocity vs Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Angular velocity (odom-measured) — coloured by planner
            _plot_by_mode(ax4, time_steps, w_coords, _state_modes)
            ax4.axhline(y=self.omega_limit_haa, color='#E53935', linestyle='--', alpha=0.6,
                        label=f'HAA limit (±{self.omega_limit_haa} rad/s)')
            ax4.axhline(y=-self.omega_limit_haa, color='#E53935', linestyle='--', alpha=0.6)
            ax4.axhline(y=self.omega_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6,
                        label=f'HPA limit (±{self.omega_limit_hpa} rad/s)')
            ax4.axhline(y=-self.omega_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Angular Velocity (rad/s)')
            ax4.set_title('Angular Velocity vs Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Controller commands — coloured by active planner at command time.
            # cmd['planner'] tagging: 'HPA_NN' = NN direct (HPA), 'HPA' = braking
            # (Kanayama-filtered path with HPA active), 'HAA' = MPC tracking.
            if self.control_command_history:
                cmd_timestamps = [cmd['timestamp'] for cmd in self.control_command_history]
                cmd_v = [cmd['v_cmd'] for cmd in self.control_command_history]
                cmd_w = [cmd['w_cmd'] for cmd in self.control_command_history]
                _cmd_tag_map = {'HPA_NN': 'HPA', 'HPA': 'BRAKING', 'HAA': 'HAA'}
                cmd_modes = [_cmd_tag_map.get(cmd.get('planner', 'HPA_NN'), 'HPA')
                             for cmd in self.control_command_history]
                start_time_cmd = cmd_timestamps[0] if cmd_timestamps else 0
                cmd_time_steps = [(t - start_time_cmd) for t in cmd_timestamps]

                _plot_by_mode(ax5, cmd_time_steps, cmd_v, cmd_modes)
                ax5.axhline(y=self.v_limit_haa, color='#E53935', linestyle='--', alpha=0.6,
                            label=f'HAA limit ({self.v_limit_haa} m/s)')
                ax5.axhline(y=self.v_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6,
                            label=f'HPA limit ({self.v_limit_hpa} m/s)')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Linear Velocity Command (m/s)')
                ax5.set_title('Controller Linear Velocity Commands')
                ax5.legend()
                ax5.grid(True, alpha=0.3)

                _plot_by_mode(ax6, cmd_time_steps, cmd_w, cmd_modes)
                ax6.axhline(y=self.omega_limit_haa, color='#E53935', linestyle='--', alpha=0.6,
                            label=f'HAA limit (±{self.omega_limit_haa} rad/s)')
                ax6.axhline(y=-self.omega_limit_haa, color='#E53935', linestyle='--', alpha=0.6)
                ax6.axhline(y=self.omega_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6,
                            label=f'HPA limit (±{self.omega_limit_hpa} rad/s)')
                ax6.axhline(y=-self.omega_limit_hpa, color='#2196F3', linestyle=':', alpha=0.6)
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Angular Velocity Command (rad/s)')
                ax6.set_title('Controller Angular Velocity Commands')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'No Control Command Data', transform=ax5.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
                ax5.set_title('Controller Linear Velocity Commands')
                ax5.grid(True, alpha=0.3)
                ax6.text(0.5, 0.5, 'No Control Command Data', transform=ax6.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
                ax6.set_title('Controller Angular Velocity Commands')
                ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.trajectory_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Trajectory plot saved to {self.trajectory_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory plot: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SwitchMPCNNRealWorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
