#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import os
import atexit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import time
from scipy import ndimage
import torch
import torch.nn as nn

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
                 use_noise_ramp: bool = False, noise_ramp: float = 1.0, nu: int = 2):
        self.sigma = sigma # Control noise standard deviation
        self.temperature = temperature # Temperature for softmax weighting
        self.num_nodes = num_nodes # Planning horizon
        self.num_rollouts = num_rollouts # More samples to ensure valid trajectories
        self.use_noise_ramp = use_noise_ramp # Whether to use noise ramp
        self.noise_ramp = noise_ramp # Noise ramp factor
        self.nu = nu # Number of control inputs
        self.a_limit = 0.25
        self.alpha_limit = 0.5
        self.v_limit_haa = 0.14
        self.omega_limit_haa = 0.9
        self.dt = 0.1
    
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
            ramp = self.noise_ramp * np.linspace(1, 1 / num_nodes, num_nodes, endpoint=True)[:, None]
            sigma = ramp[:, :, None] * adaptive_sigma[None, :, :]
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
        obstacle_mask = self.occupancy_grid > 0.1
        
        # Early exit: no obstacles
        if not np.any(obstacle_mask):
            return np.zeros_like(obstacle_mask, dtype=bool)
        
        # Use distance transform method (faster than binary_dilation for large radii)
        distance_map = ndimage.distance_transform_edt(~obstacle_mask)
        dilated_grid = distance_map <= robot_radius_cells
        
        return dilated_grid
    
    def check_collision_new(self, x: np.ndarray, y: np.ndarray, robot_radius: float = 0.2) -> np.ndarray:
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
        
        # Convert to binary: > 60 = occupied (1), otherwise = free (0)
        binary_grid = (self.occupancy_grid > 60).astype(int)
        
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
    
    def add_obstacle_rectangle(self, x: float, y: float, width: float, height: float):
        """Add rectangular obstacle to occupancy grid."""
        # Convert obstacle bounds to grid coordinates directly
        x_min_world = x - width / 2
        x_max_world = x + width / 2
        y_min_world = y - height / 2
        y_max_world = y + height / 2
        
        # Convert to grid coordinates
        x_min_grid = int((x_min_world - self.x_min) / self.resolution)
        x_max_grid = int((x_max_world - self.x_min) / self.resolution)
        y_min_grid = int((y_min_world - self.y_min) / self.resolution)
        y_max_grid = int((y_max_world - self.y_min) / self.resolution)
        
        # Ensure bounds are within grid
        x_min_grid = max(0, min(x_min_grid, self.grid_width))
        x_max_grid = max(0, min(x_max_grid, self.grid_width))
        y_min_grid = max(0, min(y_min_grid, self.grid_height))
        y_max_grid = max(0, min(y_max_grid, self.grid_height))
        
        # Set obstacle cells to occupied (100 for full occupancy)
        self.occupancy_grid[y_min_grid:y_max_grid, x_min_grid:x_max_grid] = 100
    

class MPPIPlanner:
    """MPPI Planner for Turtlebot navigation."""
    
    def __init__(self, mppi: MPPI, dynamics: TurtlebotDynamics, occupancy_map: OccupancyGridMap,
                 v_min: float = -0.26, v_max: float = 0.26, w_min: float = -1.82, w_max: float = 1.82,
                 a_min: float = -1.0, a_max: float = 1.0, alpha_min: float = -1.0, alpha_max: float = 1.0,
                 smoothing_weight: float = 0.1):
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
        
        # Initialize nominal control sequence
        self.nominal_controls = np.zeros((mppi.num_nodes, mppi.nu))
        
        # Warm start: store previous optimal control sequence
        self.previous_controls = None
    
    def update_occupancy_map(self, new_occupancy_map: OccupancyGridMap):
        """Update the occupancy map while preserving control history."""
        self.occupancy_map = new_occupancy_map
        # Keep all other state (nominal_controls, previous_controls, etc.) intact
    
    def debug_plot_no_valid_trajectories(self, start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask):
        """Debug function to plot trajectories when no valid ones are found."""
        try:
            num_valid = np.sum(valid_mask)
            num_total = len(sampled_controls)
            
            # Detailed analysis of why trajectories are invalid
            collision_only = 0
            velocity_only = 0
            both_violations = 0
            other_violations = 0
            self.get_logger().info(f"Nominal controls: {nominal_controls}")
            self.get_logger().info(f"start state: {start_state}")
            
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
                
                # Print collision states if any found
                # if has_collision:
                #     self.get_logger().warn(f"Trajectory {i} has {len(collision_states)} collision states:")
                #     for step_idx, x, y, theta in collision_states:
                #         self.get_logger().warn(f"  Step {step_idx}: x={x:.3f}, y={y:.3f}, theta={theta:.3f}")
                
                # Check velocity violations
                has_velocity_violation = False
                v_violations = np.any((trajectory[:, 3] < self.v_min) | (trajectory[:, 3] > self.v_max))
                w_violations = np.any((trajectory[:, 4] < self.w_min) | (trajectory[:, 4] > self.w_max))
                if v_violations or w_violations:
                    has_velocity_violation = True
                    # Print detailed velocity violation information
                    # v_min_violation = np.any(trajectory[:, 3] < self.v_min)
                    # v_max_violation = np.any(trajectory[:, 3] > self.v_max)
                    # w_min_violation = np.any(trajectory[:, 4] < self.w_min)
                    # w_max_violation = np.any(trajectory[:, 4] > self.w_max)
                    
                    # if v_min_violation:
                    #     min_v = np.min(trajectory[:, 3])
                    #     self.get_logger().warn(f"❌ Linear velocity MIN violation: {min_v:.3f} < {self.v_min:.3f}")
                    # if v_max_violation:
                    #     max_v = np.max(trajectory[:, 3])
                    #     self.get_logger().warn(f"❌ Linear velocity MAX violation: {max_v:.3f} > {self.v_max:.3f}")
                    # if w_min_violation:
                    #     min_w = np.min(trajectory[:, 4])
                    #     self.get_logger().warn(f"❌ Angular velocity MIN violation: {min_w:.3f} < {self.w_min:.3f}")
                    # if w_max_violation:
                    #     max_w = np.max(trajectory[:, 4])
                    #     self.get_logger().warn(f"❌ Angular velocity MAX violation: {max_w:.3f} > {self.w_max:.3f}")
                
                # Check control violations
                has_control_violation = False
                a_violations = np.any((control_seq[:, 0] < self.a_min) | (control_seq[:, 0] > self.a_max))
                alpha_violations = np.any((control_seq[:, 1] < self.alpha_min) | (control_seq[:, 1] > self.alpha_max))
                if a_violations or alpha_violations:
                    has_control_violation = True
                    # Print detailed control violation information
                    a_min_violation = np.any(control_seq[:, 0] < self.a_min)
                    a_max_violation = np.any(control_seq[:, 0] > self.a_max)
                    alpha_min_violation = np.any(control_seq[:, 1] < self.alpha_min)
                    alpha_max_violation = np.any(control_seq[:, 1] > self.alpha_max)
                    
                    # if a_min_violation:
                    #     min_a = np.min(control_seq[:, 0])
                    #     self.get_logger().warn(f"❌ Linear acceleration MIN violation: {min_a:.3f} < {self.a_min:.3f}")
                    # if a_max_violation:
                    #     max_a = np.max(control_seq[:, 0])
                    #     self.get_logger().warn(f"❌ Linear acceleration MAX violation: {max_a:.3f} > {self.a_max:.3f}")
                    # if alpha_min_violation:
                    #     min_alpha = np.min(control_seq[:, 1])
                    #     self.get_logger().warn(f"❌ Angular acceleration MIN violation: {min_alpha:.3f} < {self.alpha_min:.3f}")
                    # if alpha_max_violation:
                    #     max_alpha = np.max(control_seq[:, 1])
                    #     self.get_logger().warn(f"❌ Angular acceleration MAX violation: {max_alpha:.3f} > {self.alpha_max:.3f}")
                
                # Categorize violations
                if has_collision and has_velocity_violation:
                    both_violations += 1
                elif has_collision and not has_velocity_violation and not has_control_violation:
                    collision_only += 1
                elif not has_collision and (has_velocity_violation):
                    velocity_only += 1
                else:
                    other_violations += 1
            
            self.get_logger().info(f"MPPI Debug Analysis:")
            self.get_logger().info(f"  Total sampled trajectories: {num_total}")
            self.get_logger().info(f"  Valid trajectories: {num_valid}")
            self.get_logger().info(f"  Success rate: {num_valid/num_total*100:.1f}%")
            self.get_logger().info(f"  Invalid trajectory breakdown:")
            self.get_logger().info(f"    Collision only: {collision_only} ({collision_only/num_total*100:.1f}%)")
            self.get_logger().info(f"    Velocity only: {velocity_only} ({velocity_only/num_total*100:.1f}%)")
            self.get_logger().info(f"    Both collision and velocity violations: {both_violations} ({both_violations/num_total*100:.1f}%)")
            self.get_logger().info(f"    Other violations: {other_violations} ({other_violations/num_total*100:.1f}%)")
            
            # Create debug plots - two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot 1: Occupancy grid with trajectories
            self.occupancy_map.visualize(ax1)
            ax1.set_title(f'MPPI Debug - No Valid Trajectories ({num_valid}/{num_total})')
            
            # Plot all sampled trajectories
            for i in range(min(50, num_total)):  # Limit to 50 for visibility
                trajectory = all_trajectories[i]
                is_valid = valid_mask[i]
                
                # Extract x, y coordinates
                traj_x = trajectory[:, 0]
                traj_y = trajectory[:, 1]
                
                # Plot trajectory with different colors for valid/invalid
                color = 'green' if is_valid else 'red'
                alpha = 0.8 if is_valid else 0.3
                linewidth = 1 if is_valid else 0.5
                
                ax1.plot(traj_x, traj_y, color=color, alpha=alpha, linewidth=linewidth)
                
                # Mark start and end points
                ax1.plot(traj_x[0], traj_y[0], 'o', color=color, markersize=4)
                ax1.plot(traj_x[-1], traj_y[-1], 's', color=color, markersize=4)
            
            # Plot start and goal
            ax1.plot(start_state.x, start_state.y, 'go', markersize=10, label='Start')
            ax1.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
            
            # Add legend
            ax1.legend(['Valid trajectories', 'Invalid trajectories', 'Start', 'Goal'])
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Occupancy grid only with robot position
            self.occupancy_map.visualize(ax2)
            ax2.set_title('Occupancy Grid Map with Robot Position')
            
            # Plot robot position with orientation
            ax2.plot(start_state.x, start_state.y, 'go', markersize=15, label='Robot Position', markeredgecolor='black', markeredgewidth=2)
            
            # Draw robot orientation arrow
            arrow_length = 0.2
            dx = arrow_length * np.cos(start_state.theta)
            dy = arrow_length * np.sin(start_state.theta)
            ax2.arrow(start_state.x, start_state.y, dx, dy, 
                     head_width=0.05, head_length=0.05, fc='green', ec='green', linewidth=2)
            
            # Plot goal
            ax2.plot(goal[0], goal[1], 'ro', markersize=15, label='Goal', markeredgecolor='black', markeredgewidth=2)
            
            # Add robot radius circle
            robot_radius = 0.2
            circle = plt.Circle((start_state.x, start_state.y), robot_radius, 
                              fill=False, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax2.add_patch(circle)
            
            # Add legend and grid
            ax2.legend(['Robot Position', 'Goal', 'Robot Radius'])
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot in the specified directory
            debug_path = os.path.join('/home/rant3/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch', 'mppi_no_valid_trajectories_debug.png')
            plt.savefig(debug_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            self.get_logger().info(f"MPPI debug plot saved to: {debug_path}")
            
        except Exception as e:
            self.get_logger().error(f"Error in MPPI debug plot: {e}")
    
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
            
            # Update velocities
            new_v = v + a * self.dynamics.dt
            new_w = w + alpha * self.dynamics.dt
            
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
    
    def validate_trajectories_vectorized(self, trajectories: np.ndarray, robot_radius: float = 0.2) -> np.ndarray:
        """Vectorized trajectory validation for all trajectories at once."""
        num_rollouts, num_steps, _ = trajectories.shape
        valid_mask = np.ones(num_rollouts, dtype=bool)
        
        # Check velocity constraints for all trajectories at once
        v_violations = (trajectories[:, :, 3] < self.v_min) | (trajectories[:, :, 3] > self.v_max)
        w_violations = (trajectories[:, :, 4] < self.w_min) | (trajectories[:, :, 4] > self.w_max)
        velocity_violations = np.any(v_violations | w_violations, axis=1)
        valid_mask &= ~velocity_violations
        
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
            near_obstacle_penalty = violations_per_trajectory * 2.0  # Penalty per violation
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
    
    def plan(self, start_state: RobotState, goal: np.ndarray, debug_plot: bool = False, robot_radius: float = 0.2) -> tuple:
        """Plan using MPPI and return the control sequence and the resulting trajectory.
        
        Args:
            start_state: Initial robot state
            goal: Target position [x, y]
            debug_plot: If True, generate debug plots when no valid trajectories are found
        """
        # Start timing
        start_time = time.time()
        #self.get_logger().info(f"\n=== PLAN TIMING DEBUG ===")
        
        # Warm start: reuse previous solution if available
        warm_start_time = time.time()
        if self.previous_controls is not None and len(self.previous_controls) >= self.mppi.num_nodes:
            # Shift previous solution: remove first control, add zero at end
            nominal_controls = np.vstack([
                self.previous_controls[1:],  # Remove first control
                np.zeros((1, 2))            # Add zero control at end
            ])
            #nominal_controls = np.zeros((self.mppi.num_nodes, 2))
            #self.get_logger().info(f"1. Warm start initialization: {time.time() - warm_start_time:.6f}s")
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
                nominal_controls[:, 1] = -0.5 * self.alpha_max
            else:  # Target in positive horizontal axis (right side)
                nominal_controls[:, 1] = 0.5 * self.alpha_max
            #self.get_logger().info(f"1. Cold start initialization: {time.time() - cold_start_time:.6f}s")
            #nominal_controls = np.zeros((self.mppi.num_nodes, 2))
        # MPPI iterations
        mppi_iterations_time = time.time()
        for iteration in range(1):  # Increased iterations for better convergence
            iteration_start = time.time()
            #self.get_logger().info(f"\n--- MPPI Iteration {iteration + 1} ---")
            
            # Sample control sequences
            sample_time = time.time()
            sampled_controls = self.mppi.sample_control_knots(nominal_controls)
            #self.get_logger().info(f"2. Sample control sequences: {time.time() - sample_time:.6f}s")
            
            # Apply control limits to all sampled controls BEFORE simulation
            clip_time = time.time()
            sampled_controls[:, :, 0] = np.clip(sampled_controls[:, :, 0], self.a_min, self.a_max)
            sampled_controls[:, :, 1] = np.clip(sampled_controls[:, :, 1], self.alpha_min, self.alpha_max)
            #self.get_logger().info(f"3. Apply control limits: {time.time() - clip_time:.6f}s")
            
            # Vectorized trajectory simulation and validation
            simulation_time = time.time()
            all_trajectories = self.simulate_trajectories_vectorized(start_state, sampled_controls)
            #self.get_logger().info(f"4. Trajectory simulation: {time.time() - simulation_time:.6f}s")
            
            validation_time = time.time()
            valid_mask = self.validate_trajectories_vectorized(all_trajectories, robot_radius = robot_radius)
            #self.get_logger().info(f"5. Trajectory validation: {time.time() - validation_time:.6f}s")
            
            reward_time = time.time()
            if np.any(valid_mask):
                valid_controls = sampled_controls[valid_mask]
                valid_trajectories = all_trajectories[valid_mask]
                
                # Pre-compute safety-dilated grid for near-obstacle penalty (compute once per planning iteration)
                safe_distance = 0.05  # meters - additional safety margin
                total_safety_radius = robot_radius + safe_distance
                safety_radius_cells = int(total_safety_radius / self.occupancy_map.resolution)
                safety_dilated_grid = self.occupancy_map.dilate_grid_new(safety_radius_cells)
                
                valid_rewards = self.compute_rewards_vectorized(valid_trajectories, goal, safety_dilated_grid, valid_controls)
                #self.get_logger().info(f"6. Compute rewards: {time.time() - reward_time:.6f}s")
            else:
                valid_controls = np.array([])
                valid_rewards = np.array([])
                #self.get_logger().info(f"6. Compute rewards (no valid): {time.time() - reward_time:.6f}s")
            
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
                #self.get_logger().info(f"7. Update nominal controls: {time.time() - update_time:.6f}s")
                
            else:
                # If no valid trajectories found, debug plot and return None
                self.get_logger().info("No valid trajectories found! Creating debug plot.")
                if debug_plot:
                    self.debug_plot_no_valid_trajectories(start_state, goal, sampled_controls, nominal_controls, all_trajectories, valid_mask)
                end_time = time.time()
                # self.get_logger().info(f"7. Debug plot (no valid): {time.time() - update_time:.6f}s")
                # self.get_logger().info(f"=== TOTAL PLAN TIME: {end_time - start_time:.6f}s ===\n")
                return None, None
            
            # self.get_logger().info(f"--- Iteration {iteration + 1} total: {time.time() - iteration_start:.6f}s ---")
        
        #self.get_logger().info(f"8. MPPI iterations total: {time.time() - mppi_iterations_time:.6f}s")

        # Propagate optimal controls through system dynamics to get consistent trajectory
        final_simulation_time = time.time()
        # Use vectorized method with single control sequence (add batch dimension)
        control_sequence_batch = nominal_controls.reshape(1, -1, 2)
        trajectory_batch = self.simulate_trajectories_vectorized(start_state, control_sequence_batch)
        best_trajectory = trajectory_batch[0]  # Extract single trajectory
        #self.get_logger().info(f"9. Final trajectory simulation: {time.time() - final_simulation_time:.6f}s")
        
        end_time = time.time()
        total_time = end_time - start_time
        #self.get_logger().info(f"=== TOTAL PLAN TIME: {total_time:.6f}s ===\n")
        return best_trajectory, nominal_controls

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
        
        self.v_limit_haa = 0.14
        self.v_limit_hpa = 0.2
        self.omega_limit_haa = 0.9
        self.omega_limit_hpa = 0.9
        
        # Previous reference velocities for feedforward
        self.prev_v_ref = 0.0
        self.prev_w_ref = 0.0
        
    def compute_control(self, current_state, ref_state, current_planner):
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
        
        # Clip the control inputs
        #if current_planner is HPA, we can use the larger speed limit
        #if current_planner is HAA, we can use the smaller speed limit
        if current_planner == "HPA":
            v_cmd = np.clip(v_cmd, 0, self.v_limit_hpa)
            w_cmd = np.clip(w_cmd, -self.omega_limit_hpa, self.omega_limit_hpa)
        else:
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
        self.get_logger().info("Integral errors reset to zero")

class MPPINavigationNode(Node):
    def __init__(self):
        super().__init__('SLES')
        # TODO(migration): All rospy.get_param calls below require declare_parameter() first in ROS2
        # Declare all ROS2 parameters with defaults (converted from rospy.get_param)
        self.declare_parameter('horizon_haa', 40)
        self.declare_parameter('horizon_hpa', 20)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_limit_haa', 0.14)
        self.declare_parameter('v_limit_hpa', 0.2)
        self.declare_parameter('omega_limit_haa', 0.9)
        self.declare_parameter('omega_limit_hpa', 1.0)
        self.declare_parameter('a_limit', 0.25)
        self.declare_parameter('alpha_limit', 0.5)
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('goal', '[-1.5, -1.5, 0, 0, 0]')
        self.declare_parameter('kx', 0.6)
        self.declare_parameter('ky', 8.0)
        self.declare_parameter('kth', 1.6)
        self.declare_parameter('kv', 1.0)
        self.declare_parameter('kw', 1.0)
        self.declare_parameter('kix', 0.1)
        self.declare_parameter('kiy', 0.0)
        self.declare_parameter('kith', 0.1)
        self.declare_parameter('max_integral', 1.0)
        self.declare_parameter('Q_diag', '[10,10,0,0,0]')
        self.declare_parameter('R_diag', '[0,0]')
        self.declare_parameter('failure_path', os.path.join(os.path.expanduser('~'), 'mppi_failure_data.npz'))
        self.declare_parameter('K_feedback', '[]')
        # Random obstacle position parameters (used by _get_obstacles_from_random_world for visualization)
        for _i in range(1, 17):
            self.declare_parameter(f'random_x_{_i}', 0.0)
            self.declare_parameter(f'random_y_{_i}', 0.0)
            self.declare_parameter(f'random_size_{_i}', 0.2)
            self.declare_parameter(f'random_shape_{_i}', 'rectangle')

        # Load params
        self.N_haa             = self.get_parameter('horizon_haa').value  # default: 40
        self.N_hpa             = self.get_parameter('horizon_hpa').value  # default: 20
        self.dt            = self.get_parameter('dt').value  # default: 0.1
        self.v_limit_haa       = self.get_parameter('v_limit_haa').value  # default: 0.26
        self.omega_limit_haa   = self.get_parameter('omega_limit_haa').value  # default: 1.82
        self.a_limit       = self.get_parameter('a_limit').value  # default: 1.0
        self.alpha_limit   = self.get_parameter('alpha_limit').value  # default: 1.0
        self.robot_radius  = self.get_parameter('robot_radius').value  # default: 0.22

        # Goal
        raw_goal = self.get_parameter('goal').value  # default: [-1.5, -1.5, 0, 0, 0]
        import ast
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal
        self.goal_temp = self.goal.copy()
        self.goal_orig_history = []
        self.goal_temp_history = []
        self.state_traj = []

        # State
        self.HAA_map_ready = False
        self.state_ready = False
        self.lidar_ready = False
        self.latest_lidar_scan = None
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
        
        # Dual planner control variables
        self.current_planner = "HPA"  # Current active planner (always start with HPA)
        
        # Braking mode variables (for 2-loop braking when HAA check fails first time)
        self.haa_check_failed_first_time = False  # Flag to track if HAA check has failed for the first time
        self.braking_loop_count = 0  # Counter for how many planning loops we've been in braking mode
        
        self.shutdown_requested = False # Flag to stop loops immediately


        # Failure save path
        default_path = os.path.join(os.path.expanduser('~'), 'mppi_failure_data.npz')
        self.failure_path = self.get_parameter('failure_path').value  # default: default_path
        
        # List to store data for plotting (lower level control)
        self.plot_data = [] 
        self.plot_path = os.path.join(os.path.expanduser('~'), 'plot_data_lowlvlctrl.npz')

        self.trajectory_path = os.path.join(os.path.expanduser('~'), 'robot_trajectory_switch.png')
        self.target_reached = False # Flag to track if target is reached
        # Xopt history storage
        self.Xopt_history = []
        self.switch_point = [] # Switch point index
        
        # Control command history storage
        self.control_command_history = []
        
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
        
        self.v_limit_haa = self.get_parameter('v_limit_haa').value  # default: 0.14
        self.v_limit_hpa = self.get_parameter('v_limit_hpa').value  # default: 0.26
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value  # default: 1.82
        self.omega_limit_hpa = self.get_parameter('omega_limit_hpa').value  # default: 1.82
        self.kanayama_controller = KanayamaController(
            self.kx, self.ky, self.kth, self.kv, self.kw,
            self.kix, self.kiy, self.kith, self.max_integral
        )

        # Load Neural Network for HPA (same architecture as planner_NN)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f"NN model not found at {model_path}")
            raise FileNotFoundError(f"NN model not found at {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            class MLP(nn.Module):
                def __init__(self, input_dim=364, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.1):
                    super(MLP, self).__init__()
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.BatchNorm1d(hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout))
                        prev_dim = hidden_dim
                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            self.nn_model = MLP(input_dim=364, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.1)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.nn_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.nn_model = checkpoint
            self.nn_model.eval()
            self.get_logger().info("HPA NN model loaded from %s", model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load NN model: {e}")
            raise

        # ROS pubs/subs
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.state_cb, 10)
        self.create_subscription(OccupancyGrid, '/lidar_occupancy_map', self.map_cb, 10)
        self.create_subscription(LaserScan, '/simulated_scan', self.lidar_cb, 10)

        # Separate timers for planning (10Hz) and control (20Hz)
        self.planning_timer = self.create_timer(0.1, self.planning_loop)  # 10Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)   # 20Hz

        self.get_logger().info("MPPI planner node started (HPA=NN, HAA=MPC):")
        self.get_logger().info("  - Planning loop: 10Hz")
        self.get_logger().info("  - Control loop: 50Hz")
        self.get_logger().info("  - HPA = trained NN (no planning); HAA = MPPI (4s horizon, dynamic map)")
        # rclpy.spin() called in main()
    
    def predict_robot_state(self, current_state, control_sequence, steps_ahead):
        """Predict robot state steps_ahead into the future using control sequence."""
        if control_sequence is None or len(control_sequence) == 0:
            return current_state
        
        # Use dynamics to propagate state
        dynamics = TurtlebotDynamics(dt=self.dt)
        predicted_state = RobotState(
            x=current_state[0], y=current_state[1], theta=current_state[2],
            v=current_state[3], w=current_state[4]
        )
        
        # Clamp steps_ahead to available control sequence length
        steps_to_use = min(steps_ahead, len(control_sequence))
        
        for i in range(steps_to_use):
            control = control_sequence[i]
            predicted_state = dynamics.step(predicted_state, control)
        
        return [predicted_state.x, predicted_state.y, predicted_state.theta, 
                predicted_state.v, predicted_state.w]
        
    def save_data_on_exit(self):
        """Save the occupancy grid, goal, and state information when the program exits."""
        if hasattr(self, 'x') and hasattr(self, 'y') and hasattr(self, 'theta'):
            print(self.failure_path)
            x0 = [self.x, self.y, self.theta, self.v, self.omega]
            # Save occupancy grid data (HPA = NN has no map; HAA map only)
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

    def _get_obstacles_from_random_world(self):
        """Load obstacle positions/sizes/shapes from ROS params set by random world (set_random_config_node / spawn_random_world).
        Returns list of dicts: [{'x': float, 'y': float, 'size': float, 'shape': str}, ...].
        Tries params random_x_i, random_y_i, random_size_i, random_shape_i for i in 1..16 (with and without leading /).
        """
        obstacles = []
        for i in range(1, 17):
            try:
                # In ROS2, parameters must be declared before retrieval.
                # These are declared in MPPINavigationNode.__init__ via declare_parameter.
                x = self.get_parameter(f'random_x_{i}').value
                y = self.get_parameter(f'random_y_{i}').value
                if x is None or y is None:
                    continue
                size = self.get_parameter(f'random_size_{i}').value or 0.2
                shape_val = self.get_parameter(f'random_shape_{i}').value
                shape = (shape_val or 'rectangle').strip().lower()
                obstacles.append({'x': float(x), 'y': float(y), 'size': float(size), 'shape': shape})
            except Exception:
                continue
        return obstacles

    def _draw_obstacles_on_ax(self, ax):
        """Draw obstacles on matplotlib axis ax. Uses random world params if available; else HAA occupancy grid."""
        obstacle_count = 0
        # 1) Try obstacles from random world ROS params
        obs_list = self._get_obstacles_from_random_world()
        for obs in obs_list:
            x, y, size, shape = obs['x'], obs['y'], obs['size'], obs['shape']
            half = size / 2.0
            bl_x, bl_y = x - half, y - half
            label = 'Obstacles' if obstacle_count == 0 else ''
            if shape == 'rectangle':
                patch = patches.Rectangle(
                    (bl_x, bl_y), size, size,
                    facecolor='gray', edgecolor='black', linewidth=1.5, alpha=0.7, label=label
                )
                ax.add_patch(patch)
            elif shape == 'hexagon':
                hex_radius = size / 2
                vertices = [(x + hex_radius * np.cos(j * np.pi / 3), y + hex_radius * np.sin(j * np.pi / 3)) for j in range(6)]
                patch = patches.Polygon(
                    vertices, facecolor='gray', edgecolor='black', linewidth=1.5, alpha=0.7, label=label
                )
                ax.add_patch(patch)
            elif shape == 'triangle':
                base = size
                half_base = base / 2.0
                height_tri = base * np.sqrt(3) / 2.0
                vertices = [(x - half_base, y), (x + half_base, y), (x, y + height_tri)]
                patch = patches.Polygon(
                    vertices, facecolor='gray', edgecolor='black', linewidth=1.5, alpha=0.7, label=label
                )
                ax.add_patch(patch)
            else:
                patch = patches.Rectangle(
                    (bl_x, bl_y), size, size,
                    facecolor='gray', edgecolor='black', linewidth=1.5, alpha=0.7, label=label
                )
                ax.add_patch(patch)
            obstacle_count += 1
        if obstacle_count > 0:
            return
        # 2) Fallback: draw occupied cells from HAA occupancy grid (e.g. fixed or lidar-built map)
        if self.HAA_occupancy_map is not None and hasattr(self.HAA_occupancy_map, 'occupancy_grid'):
            grid = self.HAA_occupancy_map.occupancy_grid
            res = self.HAA_occupancy_map.resolution
            x_min, y_min = self.HAA_occupancy_map.x_min, self.HAA_occupancy_map.y_min
            occupied = grid > 0.1
            if np.any(occupied):
                rows, cols = np.where(occupied)
                for idx in range(0, len(rows), max(1, len(rows) // 500)):
                    r, c = rows[idx], cols[idx]
                    wx = x_min + (c + 0.5) * res
                    wy = y_min + (r + 0.5) * res
                    rect = patches.Rectangle(
                        (wx, wy), res, res,
                        facecolor='gray', edgecolor='none', alpha=0.6, label='Obstacles' if obstacle_count == 0 else ''
                    )
                    ax.add_patch(rect)
                    obstacle_count += 1

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
            
            # Plot 1: Position trajectory (x, y)
            ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Robot Trajectory')
            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start Position')
            ax1.plot(self.goal[0], self.goal[1], 'ro', markersize=20, label='Target Position')
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)

            # Plot switch points if available
            if self.switch_point:
                switch_x = [point[0] for point in self.switch_point]
                switch_y = [point[1] for point in self.switch_point]
                ax1.scatter(switch_x, switch_y, c='orange', marker='*', s=200, alpha=0.8, 
                           label=f'Switch Points ({len(self.switch_point)})', edgecolors='black', linewidth=1)
                # Draw circles at switch points with robot radius
                for sx, sy in zip(switch_x, switch_y):
                    circle = patches.Circle((sx, sy), self.robot_radius, 
                                      fill=False, color='orange', linestyle='--', 
                                      linewidth=1.5, alpha=0.6)
                    ax1.add_patch(circle)

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

            # Plot obstacles from random world (ROS params) or from HAA occupancy grid
            self._draw_obstacles_on_ax(ax1)

            ax1.set_xlabel('X Position (m)')
            ax1.set_ylabel('Y Position (m)')
            ax1.set_title('Robot Position Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Plot 2: Orientation over time (unwrapped)
            ax2.plot(time_steps, theta_coords_unwrapped, 'g-', linewidth=2, label='Orientation (θ)')
            
            # Add switch points to orientation plot
            if self.switch_point:
                for i, switch_point in enumerate(self.switch_point):
                    # Find the closest time step to the switch point
                    switch_x = switch_point[0]
                    switch_y = switch_point[1]
                    # Find closest trajectory point to switch point
                    distances = [(x_coords[j] - switch_x)**2 + (y_coords[j] - switch_y)**2 for j in range(len(x_coords))]
                    closest_idx = np.argmin(distances)
                    switch_time = time_steps[closest_idx]
                    switch_theta = theta_coords_unwrapped[closest_idx]
                    ax2.scatter(switch_time, switch_theta, c='orange', marker='*', s=200, alpha=0.8, 
                               edgecolors='black', linewidth=1, label='Switch Point' if i == 0 else "")
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Orientation (rad)')
            ax2.set_title('Robot Orientation vs Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Linear velocity over time
            ax3.plot(time_steps, v_coords, 'r-', linewidth=2, label='Linear Velocity (v)')
            ax3.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label=f'HAA Max Velocity')
            #ax3.axhline(y=0.2, color='b', linestyle='--', alpha=0.7, label=f'HPA Max Velocity')
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='HAA Min Velocity')
            
            # Add switch points to linear velocity plot
            if self.switch_point:
                for i, switch_point in enumerate(self.switch_point):
                    # Find the closest time step to the switch point
                    switch_x = switch_point[0]
                    switch_y = switch_point[1]
                    # Find closest trajectory point to switch point
                    distances = [(x_coords[j] - switch_x)**2 + (y_coords[j] - switch_y)**2 for j in range(len(x_coords))]
                    closest_idx = np.argmin(distances)
                    switch_time = time_steps[closest_idx]
                    switch_v = v_coords[closest_idx]
                    ax3.scatter(switch_time, switch_v, c='orange', marker='*', s=200, alpha=0.8, 
                               edgecolors='black', linewidth=1, label='Switch Point' if i == 0 else "")
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Linear Velocity (m/s)')
            ax3.set_title('Linear Velocity vs Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Angular velocity over time
            ax4.plot(time_steps, w_coords, 'm-', linewidth=2, label='Angular Velocity (ω)')
            ax4.axhline(y=1, color='m', linestyle='--', alpha=0.7, label=f'Max Angular Velocity')
            ax4.axhline(y=-1, color='m', linestyle='--', alpha=0.7, label=f'Min Angular Velocity')
            
            # Add switch points to angular velocity plot
            if self.switch_point:
                for i, switch_point in enumerate(self.switch_point):
                    # Find the closest time step to the switch point
                    switch_x = switch_point[0]
                    switch_y = switch_point[1]
                    # Find closest trajectory point to switch point
                    distances = [(x_coords[j] - switch_x)**2 + (y_coords[j] - switch_y)**2 for j in range(len(x_coords))]
                    closest_idx = np.argmin(distances)
                    switch_time = time_steps[closest_idx]
                    switch_w = w_coords[closest_idx]
                    ax4.scatter(switch_time, switch_w, c='orange', marker='*', s=200, alpha=0.8, 
                               edgecolors='black', linewidth=1, label='Switch Point' if i == 0 else "")
            
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
                ax5.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label=f'Max Velocity')
                ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7, label=f'Min Velocity (0)')
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Linear Velocity Command (m/s)')
                ax5.set_title('Controller Linear Velocity Commands')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                
                # Plot 6: Controller Commands - Angular Velocity
                ax6.plot(cmd_time_steps, cmd_w, 'g-', linewidth=2, label='Angular Velocity Command')
                ax6.axhline(y=1, color='m', linestyle='--', alpha=0.7, label=f'Max Angular Velocity')
                ax6.axhline(y=-1, color='m', linestyle='--', alpha=0.7, label=f'Min Angular Velocity')
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

    def save_hpa_first_solve_plot(self, Xopt, Uopt, start_state, goal):
        """Save a plot of the first HPA solve for debugging."""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Trajectory in world coordinates
            ax1.plot(Xopt[:, 0], Xopt[:, 1], 'b-', linewidth=2, label='HPA Trajectory')
            ax1.plot(start_state.x, start_state.y, 'go', markersize=10, label='Start')
            ax1.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('First HPA Solve - Trajectory')
            ax1.legend()
            ax1.grid(True)
            ax1.axis('equal')
            
            # Plot 2: Control inputs over time
            time_steps_u = np.arange(len(Uopt)) * self.dt
            ax2.plot(time_steps_u, Uopt[:, 0], 'b-', linewidth=2, label='Linear Acceleration (m/s^2)')
            ax2.plot(time_steps_u, Uopt[:, 1], 'r-', linewidth=2, label='Angular Acceleration (rad/s^2)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Control Input')
            ax2.set_title('First HPA Solve - Control Sequence')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Robot orientation over time
            time_steps_x = np.arange(len(Xopt)) * self.dt
            ax3.plot(time_steps_x, Xopt[:, 2], 'g-', linewidth=2, label='Theta (rad)')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Orientation (rad)')
            ax3.set_title('First HPA Solve - Robot Orientation')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Robot velocities over time
            ax4.plot(time_steps_x, Xopt[:, 3], 'b-', linewidth=2, label='Linear Vel (m/s)')
            ax4.plot(time_steps_x, Xopt[:, 4], 'r-', linewidth=2, label='Angular Vel (rad/s)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Velocity')
            ax4.set_title('First HPA Solve - Robot Velocities')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join('/home/rant3/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch', 'first_hpa_solve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            self.get_logger().info(f"First HPA solve plot saved to: {plot_path}")
            self.get_logger().info(f"Trajectory shape: {Xopt.shape}, Control shape: {Uopt.shape}")
            self.get_logger().info(f"Start state: x={start_state.x:.3f}, y={start_state.y:.3f}, theta={start_state.theta:.3f}")
            self.get_logger().info(f"Goal: x={goal[0]:.3f}, y={goal[1]:.3f}")
            
        except Exception as e:
            self.get_logger().error(f"Error in first HPA solve plot: {e}")

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
        #self.get_logger().info("HAA map received: dynamic occupancy grid map built.")
        

    def state_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index('turtlebot3_waffle_pi')
        except ValueError:
            self.get_logger().warn("Turtlebot3_waffle_pi not found in model states")
            return
        pose = msg.pose[idx]
        twist = msg.twist[idx]
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.x = pose.position.x
        self.y = pose.position.y
        self.theta = yaw
        self.v = np.hypot(twist.linear.x, twist.linear.y)
        self.omega = twist.angular.z
        self.state_ready = True

    def lidar_cb(self, msg: LaserScan):
        """Callback for lidar scan (required for HPA NN)."""
        try:
            ranges = np.array(msg.ranges)
            ranges[ranges == -1] = 1.0
            if len(ranges) != 360:
                indices = np.linspace(0, len(ranges) - 1, 360)
                ranges = np.interp(indices, np.arange(len(ranges)), ranges)
            self.latest_lidar_scan = ranges
            self.lidar_ready = True
        except Exception as e:
            self.get_logger().warn(f"Error processing lidar: {e}")

    def transform_goal_to_robot_frame(self, goal_world_x, goal_world_y):
        """Transform goal from world to robot frame."""
        dx = goal_world_x - self.x
        dy = goal_world_y - self.y
        cos_theta = np.cos(-self.theta)
        sin_theta = np.sin(-self.theta)
        goal_x_robot = dx * cos_theta - dy * sin_theta
        goal_y_robot = dx * sin_theta + dy * cos_theta
        return goal_x_robot, goal_y_robot

    def preprocess_nn_input(self, lidar_scan, v, w, target_x_robot, target_y_robot):
        """Build NN input [v, w, target_x, target_y, lidar_360] = 364 dims."""
        input_array = np.concatenate([
            [v, w],
            [target_x_robot, target_y_robot],
            lidar_scan
        ])
        return torch.FloatTensor(input_array).unsqueeze(0)

    def get_nn_control(self):
        """Get (v_cmd, w_cmd) from NN for current state and goal. Returns None if lidar not ready."""
        if self.latest_lidar_scan is None:
            return None
        goal_x_robot, goal_y_robot = self.transform_goal_to_robot_frame(
            self.goal_temp[0], self.goal_temp[1]
        )
        input_tensor = self.preprocess_nn_input(
            self.latest_lidar_scan, self.v, self.omega,
            goal_x_robot, goal_y_robot
        )
        with torch.no_grad():
            output = self.nn_model(input_tensor)
            out = output.cpu().numpy().flatten()
        v_cmd = np.clip(out[0], 0.0, self.v_limit_hpa)
        w_cmd = np.clip(out[1], -self.omega_limit_hpa, self.omega_limit_hpa)
        return v_cmd, w_cmd

    def generate_braking_trajectory(self, start_state: RobotState, horizon: int) -> tuple:
        """Generate a braking trajectory with max braking (a = -a_limit, alpha = 0).
        
        Args:
            start_state: Initial robot state
            horizon: Number of time steps in the trajectory
            
        Returns:
            trajectory: Array of shape (horizon+1, 5) with states [x, y, theta, v, w]
            controls: Array of shape (horizon, 2) with controls [a, alpha]
        """
        dynamics = TurtlebotDynamics(dt=self.dt)
        trajectory = np.zeros((horizon + 1, 5))
        controls = np.zeros((horizon, 2))
        
        # Initialize trajectory with start state
        trajectory[0, 0] = start_state.x
        trajectory[0, 1] = start_state.y
        trajectory[0, 2] = start_state.theta
        trajectory[0, 3] = start_state.v
        trajectory[0, 4] = start_state.w
        
        # Max braking control: a = -a_limit, alpha = 0
        max_braking_control = np.array([-self.a_limit*1.2, 0.0])
        
        # Set all controls to max braking
        controls[:, 0] = -self.a_limit*1.2
        controls[:, 1] = 0.0
        
        # Simulate trajectory
        current_state = RobotState(
            x=start_state.x, y=start_state.y, theta=start_state.theta,
            v=start_state.v, w=start_state.w
        )
        
        for i in range(horizon):
            current_state = dynamics.step(current_state, max_braking_control)
            # Ensure velocity doesn't go negative
            current_state.v = max(0.0, current_state.v)
            
            trajectory[i + 1, 0] = current_state.x
            trajectory[i + 1, 1] = current_state.y
            trajectory[i + 1, 2] = current_state.theta
            trajectory[i + 1, 3] = current_state.v
            trajectory[i + 1, 4] = current_state.w
        
        return trajectory, controls

    def planning_loop(self, event):
        """Dual system: HPA = NN (no MPC), HAA = MPC (4s horizon, dynamic map).
        While on HPA: haa_check_state = current + NN output; check HAA feasibility; switch to HAA if not.
        While on HAA: run HAA MPPI; switch back to HPA when in safety margin."""
        # Check if shutdown has been requested
        if self.shutdown_requested:
            return

        # Check if all required data is ready
        if not (self.HAA_map_ready and self.state_ready):
            if not self.HAA_map_ready:
                self.get_logger().warn("HAA map not ready")
            if not self.state_ready:
                self.get_logger().warn("State not ready")
            return

        # current state
        x_true = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(x_true.copy())

        # Check if target is reached
        dist = np.linalg.norm(np.array(x_true[:2]) - np.array(self.goal_temp[:2]))
        if dist < 0.2 and not self.target_reached:
            self.get_logger().info("Target reached in planning loop. Shutting down both loops...")
            self.target_reached = True
            
            # Save trajectory plot
            self.save_trajectory_plot()
            
            # Save all data before shutdown
            # self.save_data_on_exit()
            
            # Stop robot
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            
            # Stop both loops
            self.planning_timer.shutdown()
            self.control_timer.shutdown()
            rclpy.shutdown()  # "Target reached successfully"
            return

        try:
            self.goal_temp = self.goal.copy()
            
            # === HPA is NN (no MPPI); HAA is MPC (4s horizon, dynamic map) ===
            # === HAA PLANNER (4-second horizon, dynamic map) ===
            # Initialize HAA planner if it doesn't exist
            if self.HAA_mppi_planner is None:
                dynamics = TurtlebotDynamics(dt=self.dt)
                # HAA: 4-second horizon (40 nodes at 0.1s dt)
                haa_horizon = self.N_haa  # 4 seconds
                haa_mppi = MPPI(
                    sigma=1,
                    temperature=0.1,
                    num_nodes=haa_horizon,
                    num_rollouts=1000,
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
                    smoothing_weight=0.
                )
                self.get_logger().info("HAA planner initialized (4s horizon, dynamic map).")
            
            # Update HAA planner with latest dynamic map
            if self.HAA_mppi_planner.occupancy_map != self.HAA_occupancy_map:
                self.HAA_mppi_planner.update_occupancy_map(self.HAA_occupancy_map)
               #self.get_logger().info("HAA planner updated with new occupancy map.")
            
            # === SWITCHING LOGIC: HPA = NN, HAA = MPC ===
            # While on HPA (NN): no trajectory from NN; check haa_check_state = current + NN output, then HAA feasibility.
            # While on HAA: run HAA MPPI; optionally switch back to HPA (NN) when in safety margin.

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
                        self.get_logger().info("Braking phase complete - switching to HAA")
                        self.current_planner = "HAA"
                        self.haa_check_failed_first_time = False
                        self.braking_loop_count = 0
                        return
                    else:
                        return

                # HPA = NN: need lidar for NN and for haa_check_state prediction
                if not self.lidar_ready or self.latest_lidar_scan is None:
                    self.get_logger().warn("HPA (NN) waiting for lidar")
                    return

                nn_out = self.get_nn_control()
                if nn_out is None:
                    return
                v_nn, w_nn = nn_out

                # haa_check_state = one-step prediction: current state + NN controller output
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

                # Optional: apply N_r braking steps for recoverable-set check
                N_r = 3
                max_braking_control = np.array([-self.a_limit, 0])
                for _ in range(N_r):
                    haa_check_state = dynamics.step(haa_check_state, max_braking_control)
                    haa_check_state.v = max(0, haa_check_state.v)

                # HAA feasibility check from haa_check_state
                haa_check_start_time = time.time()
                haa_check_Xopt, haa_check_Uopt = self.HAA_mppi_planner.plan(
                    haa_check_state, np.array(self.goal_temp[:2]), robot_radius=self.robot_radius + 0.02
                )
                haa_check_end_time = time.time()
                haa_check_feasible = (
                    haa_check_Xopt is not None and haa_check_Uopt is not None
                    and len(haa_check_Xopt) > 0 and len(haa_check_Uopt) > 0
                )
                self.get_logger().info(f'HAA check feasible: {haa_check_feasible}')
                self.get_logger().info(f'HAA check time: {haa_check_end_time - haa_check_start_time:.6f}s')

                if haa_check_feasible:
                    # Stay on HPA (NN); control loop will use NN output (no trajectory)
                    self.current_trajectory = None
                    self.current_controls = None
                    self.trajectory_ready = True
                    self.current_planner = "HPA"
                else:
                    # HAA check failed - start 2-loop braking then switch to HAA
                    if not self.haa_check_failed_first_time:
                        self.get_logger().info('HAA check failed for the first time - starting 2-loop braking phase')
                        self.get_logger().info(f'HAA check state: x={haa_check_state.x:.3f}, y={haa_check_state.y:.3f}, theta={haa_check_state.theta:.3f}, v={haa_check_state.v:.3f}, w={haa_check_state.w:.3f}')
                        self.switch_point.append(x_true)
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
                        self.get_logger().warn("HAA check failed but already in braking mode - unexpected state")
                        self.trajectory_ready = True
                        self.current_planner = "HPA"
            elif self.current_planner == "HAA":
                haa_current_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2], v=np.clip(x_true[3],0,self.v_limit_haa), w=np.clip(x_true[4],-self.omega_limit_haa,self.omega_limit_haa))
                haa_start_time = time.time()
                haa_Xopt, haa_Uopt = self.HAA_mppi_planner.plan(haa_current_state, np.array(self.goal_temp[:2]), debug_plot=True, robot_radius = self.robot_radius)
                haa_end_time = time.time()
                
                if haa_Xopt is not None and haa_Uopt is not None and len(haa_Xopt) > 0 and len(haa_Uopt) > 0:
                    self.current_trajectory = haa_Xopt
                    self.current_controls = haa_Uopt
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    self.current_planner = "HAA"
                    # Reset braking flags
                    self.haa_check_failed_first_time = False
                    self.braking_loop_count = 0
                    #self.get_logger().info(f"Using HAA trajectory (permanent switch)")
                    self.get_logger().info(f"HAA planning time: {haa_end_time - haa_start_time:.6f}s")

                    N_s = 1  # time steps for forward simulation
                    
                    if haa_current_state.v < self.v_limit_haa - N_s * self.dt * self.a_limit:
                        safety_margin = N_s*self.dt*self.v_limit_haa+self.v_limit_haa**2/(2*self.a_limit)+0.12
                        # check_collision_new expects arrays, so convert scalars to arrays
                        collisions = self.HAA_mppi_planner.occupancy_map.check_collision_new(
                            np.array([haa_current_state.x]), 
                            np.array([haa_current_state.y]), 
                            self.robot_radius + safety_margin
                        )
                        if not np.any(collisions):
                            self.get_logger().info("Current state is inside the safety margin set of HAA safety envelope - switching back to HPA (NN)")
                            self.current_planner = "HPA"
                            self.current_trajectory = None
                            self.current_controls = None
                            self.switch_point.append(x_true)
                            return
                else:
                    self.get_logger().warn("HAA planning failed after switch")
                    self.get_logger().info(f'HAA planning failed state: x={haa_current_state.x:.3f}, y={haa_current_state.y:.3f}, theta={haa_current_state.theta:.3f}, v={haa_current_state.v:.3f}, w={haa_current_state.w:.3f}')
                    self.trajectory_ready = False
                    self.shutdown_requested = True
                    # Save trajectory plot
                    self.save_trajectory_plot()
                    #self.save_data_on_exit()
                    self.planning_timer.shutdown()
                    self.control_timer.shutdown()
                    rclpy.shutdown()  # "No valid trajectories found"
                    return
            
            # Store Xopt history for the active planner
            if self.trajectory_ready and self.current_trajectory is not None:
                xopt_data = {
                    'Xopt': self.current_trajectory.copy(),
                    'Uopt': self.current_controls.copy(),
                    'start_state': [x_true[0], x_true[1], x_true[2], x_true[3], x_true[4]],
                    'goal': self.goal_temp[:2].copy(),
                    'step': self.step,
                    'timestamp': self.get_clock().now().to_sec(),
                    'planner': self.current_planner
                }
                self.Xopt_history.append(xopt_data)
                #self.get_logger().info(f"Stored {self.current_planner} trajectory with {len(self.current_trajectory)} points")

            # Reset integral errors for new trajectory
            self.kanayama_controller.reset_integral_errors()
            
        except Exception as e:
            self.get_logger().error(f"Planning failed at step {self.step}: {e}")
            # Save failure data to disk
            print(self.failure_path)
            np.savez(self.failure_path, x0=x_true, goal=self.goal_temp, occ=self.HAA_raw_occ)
            self.get_logger().info("Saved failure data: x_true, goal, and raw occupancy grid.")
            self.get_logger().info(f"x_true: {x_true}, goal: {self.goal_temp}")
            # Stop robot
            stop_twist = Twist()
            self.cmd_pub.publish(stop_twist)
            # Stop control loop
            self.control_timer.shutdown()
            return

        self.step += 1
        #self.get_logger().info(f'Planner current output track state is {self.current_trajectory[2]}')
    
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
        
        K_feedback = [[0.0, 0.0, 0.0, -1.2, 0.0], [0.0, 0.0, -0.35, 0.0, -0.15]]
        # Feedback control: u = u_nominal + K * error
        u = u_nominal + K_feedback @ error
        
        # Apply control to current state using dynamics
        if self.HAA_mppi_planner is not None and self.HAA_mppi_planner.dynamics is not None:
            dynamics = self.HAA_mppi_planner.dynamics
        else:
            # Fallback: create dynamics instance if planner not ready
            dynamics = TurtlebotDynamics(dt=self.dt)
        
        current_robot_state = RobotState(x=x_true[0], y=x_true[1], theta=x_true[2], 
                                        v=x_true[3], w=x_true[4])
        ref_robot_state = dynamics.step(current_robot_state, u)
        
        # Convert RobotState to array format [x, y, theta, v, w]
        ref_state = np.array([ref_robot_state.x, ref_robot_state.y, ref_robot_state.theta,
                             ref_robot_state.v, ref_robot_state.w])
        
        ref_state = self.current_trajectory[2] #tested that works well

        return ref_state
    
    def control_loop(self, event):
        """Control loop: HPA = NN output directly; HAA = trajectory + Kanayama."""
        if self.shutdown_requested:
            return

        if not self.state_ready:
            return

        current_time = self.get_clock().now()
        current_state = [self.x, self.y, self.theta, self.v, self.omega]

        if self.current_planner == "HPA":
            # HPA braking: trajectory + Kanayama; HPA NN: NN output directly
            if self.current_trajectory is not None and self.current_controls is not None:
                # Braking phase - use braking trajectory + Kanayama
                if not self.trajectory_ready:
                    return
                ref_state = self.get_reference_state(current_time)
                if ref_state is None:
                    self.publish_stop_command()
                    return
                v_cmd, w_cmd = self.kanayama_controller.compute_control(
                    current_state, ref_state, self.current_planner
                )
            else:
                # HPA NN mode - use NN output directly
                if not self.lidar_ready or self.latest_lidar_scan is None:
                    return
                nn_out = self.get_nn_control()
                if nn_out is None:
                    return
                v_cmd, w_cmd = nn_out
                ref_state = None
        else:
            # HAA = MPC: use trajectory + Kanayama
            if not self.trajectory_ready:
                return
            ref_state = self.get_reference_state(current_time)
            if ref_state is None:
                self.publish_stop_command()
                return
            v_cmd, w_cmd = self.kanayama_controller.compute_control(
                current_state, ref_state, self.current_planner
            )

        twist = Twist()
        twist.linear.x = v_cmd
        twist.angular.z = w_cmd
        self.cmd_pub.publish(twist)

        control_data = {
            'timestamp': current_time.to_sec(),
            'v_cmd': v_cmd,
            'w_cmd': w_cmd,
            'current_state': current_state.copy(),
            'ref_state': ref_state.copy() if ref_state is not None else None
        }
        self.control_command_history.append(control_data)

    
    def publish_stop_command(self):
        """Publish stop command."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
    

def main(args=None):
    rclpy.init(args=args)
    node = MPPINavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
