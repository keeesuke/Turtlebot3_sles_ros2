#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, RegularPolygon, Polygon
from shapely.geometry import LineString
import json
import os
import argparse
from pathlib import Path
import math

class RobotDataAnalyzer:
    def __init__(self, data_file):
        """
        Initialize the robot data analyzer
        
        Args:
            data_file: Path to the .npz data file
        """
        self.data_file = data_file
        self.data = None
        self.obstacle_data = {}  # Will store full obstacle data with size and shape
        self.occupancy_grid_data = None  # Will store occupancy grid data
        self.load_data()
        self.load_obstacle_data()
        self.load_occupancy_grid()
    
    def load_data(self):
        """Load data from the .npz file"""
        try:
            self.data = np.load(self.data_file)
            print(f"Data loaded from: {self.data_file}")
            print(f"Available data keys: {list(self.data.keys())}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def load_obstacle_data(self):
        """Load obstacle data (with size and shape) from JSON file in the same directory"""
        try:
            # Get the directory of the npz file
            npz_dir = Path(self.data_file).parent
            # Find robot_states JSON file (contains obstacle_positions with full data)
            json_files = list(npz_dir.glob('robot_states_*.json'))
            if json_files:
                json_file = json_files[0]  # Use the first one found
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if 'obstacle_positions' in json_data:
                        self.obstacle_data = json_data['obstacle_positions']
                        print(f"Loaded obstacle data from: {json_file}")
                        print(f"Found {len(self.obstacle_data)} obstacles with size and shape information")
            else:
                print("Warning: Could not find robot_states JSON file to load obstacle details")
        except Exception as e:
            print(f"Warning: Could not load obstacle data from JSON: {e}")
            # Fallback: use default values if obstacle_positions array exists
            if 'obstacle_positions' in self.data and 'obstacle_names' in self.data:
                obstacle_positions = self.data['obstacle_positions']
                obstacle_names = self.data['obstacle_names']
                for i, (pos, name) in enumerate(zip(obstacle_positions, obstacle_names)):
                    self.obstacle_data[str(name)] = {
                        'position': pos.tolist() if hasattr(pos, 'tolist') else list(pos),
                        'size': 0.2,  # Default size
                        'shape': 'rectangle'  # Default shape
                    }
    
    def load_occupancy_grid(self):
        """Load occupancy grid data from the separate npz file or from JSON"""
        try:
            # Get the directory of the npz file
            npz_dir = Path(self.data_file).parent
            
            # Try to load from separate occupancy_grid_data npz file first
            grid_data_files = list(npz_dir.glob('occupancy_grid_data_*.npz'))
            if grid_data_files:
                grid_data_file = grid_data_files[0]
                grid_data = np.load(grid_data_file)
                self.occupancy_grid_data = {
                    'data': grid_data['occupancy_grid_data'],
                    'resolution': float(grid_data['resolution']),
                    'width': int(grid_data['width']),
                    'height': int(grid_data['height']),
                    'origin_x': float(grid_data['origin_x']),
                    'origin_y': float(grid_data['origin_y']),
                    'timestamp': float(grid_data['timestamp'])
                }
                print(f"Loaded occupancy grid data from: {grid_data_file}")
                return
            
            # Fallback: try to load from JSON file
            json_files = list(npz_dir.glob('occupancy_grid_final_*.json'))
            if json_files:
                json_file = json_files[0]
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if 'occupancy_grid' in json_data:
                        grid_info = json_data['occupancy_grid']
                        self.occupancy_grid_data = {
                            'data': np.array(grid_info['data'], dtype=np.int8),
                            'resolution': grid_info['info']['resolution'],
                            'width': grid_info['info']['width'],
                            'height': grid_info['info']['height'],
                            'origin_x': grid_info['info']['origin']['position'][0],
                            'origin_y': grid_info['info']['origin']['position'][1],
                            'timestamp': grid_info['timestamp']
                        }
                        print(f"Loaded occupancy grid data from: {json_file}")
                        return
            
            # Fallback: try to get from main npz file (if available)
            if 'occupancy_grid_width' in self.data and self.data['occupancy_grid_width'] > 0:
                # Check if there's a separate occupancy_grid_data file
                print("Warning: Occupancy grid metadata found but data array not loaded")
        except Exception as e:
            print(f"Warning: Could not load occupancy grid data: {e}")
    
    def get_data_summary(self):
        """Get summary statistics of the recorded data"""
        if self.data is None:
            return None
        
        summary = {}
        
        # Position data
        positions = self.data['positions']
        summary['total_distance'] = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        summary['start_position'] = positions[0]
        summary['end_position'] = positions[-1]
        summary['position_range'] = {
            'x_min': np.min(positions[:, 0]),
            'x_max': np.max(positions[:, 0]),
            'y_min': np.min(positions[:, 1]),
            'y_max': np.max(positions[:, 1])
        }
        
        # Time data
        robot_timestamps = self.data['robot_state_timestamps']
        control_timestamps = self.data['control_input_timestamps']
        
        summary['robot_duration'] = robot_timestamps[-1] - robot_timestamps[0]
        summary['control_duration'] = control_timestamps[-1] - control_timestamps[0]
        summary['robot_num_samples'] = len(robot_timestamps)
        summary['control_num_samples'] = len(control_timestamps)
        summary['robot_avg_frequency'] = len(robot_timestamps) / summary['robot_duration'] if summary['robot_duration'] > 0 else 0
        summary['control_avg_frequency'] = len(control_timestamps) / summary['control_duration'] if summary['control_duration'] > 0 else 0
        
        # Control data
        control_linear = self.data['control_linear']
        control_angular = self.data['control_angular']
        # Linear velocity is a scalar (forward speed) for TurtleBot3
        summary['max_linear_speed'] = np.max(np.abs(control_linear))
        summary['max_angular_speed'] = np.max(np.abs(control_angular))
        summary['avg_linear_speed'] = np.mean(np.abs(control_linear))
        summary['avg_angular_speed'] = np.mean(np.abs(control_angular))
        
        # Target information
        summary['target_position'] = self.data['target_position']
        summary['tolerance'] = self.data['tolerance']
        
        # Calculate final distance to target
        final_distance = np.linalg.norm(positions[-1] - self.data['target_position'])
        summary['final_distance_to_target'] = final_distance
        summary['target_reached'] = final_distance <= self.data['tolerance']
        
        # Lidar data (if available)
        if 'lidar_timestamps' in self.data:
            lidar_timestamps = self.data['lidar_timestamps']
            if len(lidar_timestamps) > 0:
                summary['lidar_duration'] = lidar_timestamps[-1] - lidar_timestamps[0]
                summary['lidar_num_samples'] = len(lidar_timestamps)
                summary['lidar_avg_frequency'] = len(lidar_timestamps) / summary['lidar_duration'] if summary['lidar_duration'] > 0 else 0
            else:
                summary['lidar_duration'] = 0
                summary['lidar_num_samples'] = 0
                summary['lidar_avg_frequency'] = 0
        else:
            summary['lidar_duration'] = 0
            summary['lidar_num_samples'] = 0
            summary['lidar_avg_frequency'] = 0
        
        # Obstacle positions (if available)
        if 'obstacle_positions' in self.data:
            obstacle_positions = self.data['obstacle_positions']
            if len(obstacle_positions) > 0:
                summary['num_obstacles'] = len(obstacle_positions)
                summary['obstacle_positions'] = obstacle_positions
            else:
                summary['num_obstacles'] = 0
                summary['obstacle_positions'] = []
        else:
            summary['num_obstacles'] = 0
            summary['obstacle_positions'] = []
        
        return summary
    
    def plot_trajectory(self, save_path=None):
        """Plot the robot trajectory"""
        if self.data is None:
            return
        
        positions = self.data['positions']
        target_pos = self.data['target_position']
        tolerance = self.data['tolerance']
        
        plt.figure(figsize=(10, 8))
        
        # Plot obstacles if available - use true size (not enlarged)
        if self.obstacle_data:
            for i, (obs_name, obs_info) in enumerate(self.obstacle_data.items()):
                pos = obs_info.get('position', [0, 0, 0])
                size = obs_info.get('size', 0.2)
                shape = obs_info.get('shape', 'rectangle').lower()
                
                x, y = pos[0], pos[1]
                
                # Draw obstacle based on shape at true size
                if shape == 'rectangle':
                    half_size = size / 2.0
                    rect = Rectangle((x - half_size, y - half_size), size, size,
                                   facecolor='gray', edgecolor='black', alpha=0.7,
                                   label='Obstacle' if i == 0 else '')
                    plt.gca().add_patch(rect)
                
                elif shape == 'hexagon':
                    # Hexagon: size is the diameter across opposite vertices
                    # RegularPolygon by default has a vertex pointing right (0 deg)
                    # We want a flat side up, so rotate by 30 degrees (pi/6)
                    radius = size / 2.0
                    hexagon = RegularPolygon((x, y), numVertices=6, radius=radius,
                                           orientation=math.pi/6, facecolor='gray',
                                           edgecolor='black', alpha=0.7,
                                           label='Obstacle' if i == 0 else '')
                    plt.gca().add_patch(hexagon)
                
                elif shape == 'triangle':
                    # Triangle: size is the side length (base length)
                    # Create equilateral triangle vertices (matching spawn_random_world.py)
                    base = size
                    height_tri = base * math.sqrt(3) / 2.0  # equilateral triangle height
                    half_base = base / 2.0
                    vertices = np.array([
                        [x - half_base, y - height_tri / 3.0],  # bottom left
                        [x + half_base, y - height_tri / 3.0],  # bottom right
                        [x, y + 2.0 * height_tri / 3.0]         # top
                    ])
                    triangle = Polygon(vertices, facecolor='gray', edgecolor='black',
                                     alpha=0.7, label='Obstacle' if i == 0 else '')
                    plt.gca().add_patch(triangle)
                
                else:
                    # Unknown shape - default to rectangle
                    half_size = size / 2.0
                    rect = Rectangle((x - half_size, y - half_size), size, size,
                                   facecolor='gray', edgecolor='black', alpha=0.7,
                                   label='Obstacle' if i == 0 else '')
                    plt.gca().add_patch(rect)
        
        # Fallback: if obstacle_data is empty but obstacle_positions array exists
        elif 'obstacle_positions' in self.data:
            obstacle_positions = self.data['obstacle_positions']
            if len(obstacle_positions) > 0:
                # Plot each obstacle as a rectangle with default size (true size)
                for i, obs_pos in enumerate(obstacle_positions):
                    default_size = 0.2
                    half_size = default_size / 2.0
                    rect = Rectangle((obs_pos[0] - half_size, obs_pos[1] - half_size),
                                   default_size, default_size,
                                   facecolor='gray', edgecolor='black', alpha=0.7,
                                   label='Obstacle' if i == 0 else '')
                    plt.gca().add_patch(rect)
        
        # Draw tube around trajectory (0.2m buffer) using Shapely
        robot_radius = 0.20
        if len(positions) > 1:
            # Create LineString from trajectory points
            line = LineString(positions)
            # Create buffer (tube) around the line
            buffered = line.buffer(robot_radius)
            
            # Convert Shapely polygon to matplotlib patch
            if buffered.geom_type == 'Polygon':
                # Get exterior coordinates
                x, y = buffered.exterior.xy
                tube_polygon = np.column_stack([x, y])
                tube_patch = Polygon(tube_polygon, facecolor='lightblue', edgecolor='blue', 
                                   alpha=0.5, linewidth=1, label='Robot Size')
                plt.gca().add_patch(tube_patch)
            elif buffered.geom_type == 'MultiPolygon':
                # Handle case where buffer creates multiple polygons
                for poly in buffered.geoms:
                    x, y = poly.exterior.xy
                    tube_polygon = np.column_stack([x, y])
                    tube_patch = Polygon(tube_polygon, facecolor='lightblue', edgecolor='blue', 
                                       alpha=0.5, linewidth=1, label='Robot Size' if poly == buffered.geoms[0] else '')
                    plt.gca().add_patch(tube_patch)
        
        # Plot trajectory
        plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Robot Path')
        
        # Mark start and end points
        plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        # Mark target position
        target_circle = plt.Circle(target_pos, tolerance, color='red', alpha=0.3, label='Target Zone')
        plt.gca().add_patch(target_circle)
        plt.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        
        # Only show plot if explicitly requested (default: save only, no display)
        if hasattr(self, '_show_plots') and self._show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_occupancy_grid(self, save_path=None):
        """Plot the final occupancy grid"""
        if self.occupancy_grid_data is None:
            print("No occupancy grid data available to plot")
            return
        
        grid_data = self.occupancy_grid_data['data']
        resolution = self.occupancy_grid_data['resolution']
        width = self.occupancy_grid_data['width']
        height = self.occupancy_grid_data['height']
        origin_x = self.occupancy_grid_data['origin_x']
        origin_y = self.occupancy_grid_data['origin_y']
        
        # Reshape the 1D array into a 2D grid
        # ROS OccupancyGrid stores data in row-major order: data[i] = grid[row][col]
        # where row 0 is at the bottom (y_min) and row (height-1) is at the top (y_max)
        # Reshape directly: first row (index 0) corresponds to y_min, last row to y_max
        grid_2d = grid_data.reshape((height, width))
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Calculate world coordinates
        # The origin (origin_x, origin_y) is the bottom-left corner of the grid in world coordinates
        # Grid cell (0, 0) is at (origin_x, origin_y)
        # Grid cell (height-1, width-1) is at (origin_x + width*resolution, origin_y + height*resolution)
        x_min = origin_x
        y_min = origin_y
        x_max = origin_x + width * resolution
        y_max = origin_y + height * resolution
        
        # Create extent for imshow: [left, right, bottom, top]
        # With origin='lower', row 0 (first row) is at the bottom, which matches ROS convention
        extent = [x_min, x_max, y_min, y_max]
        
        # Plot the occupancy grid
        # Values: -1 = unknown (gray), 0 = free (white), 1-100 = occupied (black to dark gray)
        # We'll map: -1 -> 0.5 (light gray), 0 -> 1.0 (white), 1-100 -> 0.0-0.3 (dark gray to black)
        display_grid = np.where(grid_2d == -1, 0.5,  # Unknown = gray
                                np.where(grid_2d == 0, 1.0,  # Free = white
                                        1.0 - grid_2d / 100.0))  # Occupied = black to dark gray
        
        # Use origin='lower' so that row 0 (y_min) is at the bottom
        im = plt.imshow(display_grid, extent=extent, origin='lower', 
                       cmap='gray', interpolation='nearest', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, label='Occupancy')
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(['Occupied', 'Unknown', 'Free'])
        
        # Overlay robot trajectory if available
        if 'positions' in self.data:
            positions = self.data['positions']
            plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Robot Path')
            plt.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
            plt.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        # Overlay target if available
        if 'target_position' in self.data:
            target_pos = self.data['target_position']
            tolerance = self.data.get('tolerance', 0.3)
            target_circle = plt.Circle(target_pos, tolerance, color='red', alpha=0.3, label='Target Zone')
            plt.gca().add_patch(target_circle)
            plt.plot(target_pos[0], target_pos[1], 'r*', markersize=15, label='Target')
        
        # Overlay obstacles if available
        if self.obstacle_data:
            for obs_name, obs_info in self.obstacle_data.items():
                pos = obs_info.get('position', [0, 0, 0])
                size = obs_info.get('size', 0.2)
                shape = obs_info.get('shape', 'rectangle').lower()
                
                x, y = pos[0], pos[1]
                
                if shape == 'rectangle':
                    half_size = size / 2.0
                    rect = Rectangle((x - half_size, y - half_size), size, size,
                                   facecolor='orange', edgecolor='red', alpha=0.6, linewidth=2)
                    plt.gca().add_patch(rect)
                elif shape == 'hexagon':
                    # Hexagon: size is diameter across opposite vertices
                    # RegularPolygon by default has a vertex pointing right (0 deg)
                    # We want a flat side up, so rotate by 30 degrees (pi/6)
                    radius = size / 2.0
                    hexagon = RegularPolygon((x, y), numVertices=6, radius=radius,
                                           orientation=math.pi/6, facecolor='orange',
                                           edgecolor='red', alpha=0.6, linewidth=2)
                    plt.gca().add_patch(hexagon)
                elif shape == 'triangle':
                    base = size
                    height_tri = base * math.sqrt(3) / 2.0
                    half_base = base / 2.0
                    vertices = np.array([
                        [x - half_base, y - height_tri / 3.0],
                        [x + half_base, y - height_tri / 3.0],
                        [x, y + 2.0 * height_tri / 3.0]
                    ])
                    triangle = Polygon(vertices, facecolor='orange', edgecolor='red',
                                     alpha=0.6, linewidth=2)
                    plt.gca().add_patch(triangle)
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Final Occupancy Grid Map')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Occupancy grid plot saved to: {save_path}")
        
        # Only show plot if explicitly requested (default: save only, no display)
        if hasattr(self, '_show_plots') and self._show_plots:
            plt.show()
        else:
            plt.close()
    
    def _prepare_velocity_data(self):
        """Helper function to prepare velocity data for plotting."""
        if self.data is None:
            return None
        
        control_timestamps = self.data['control_input_timestamps']
        robot_timestamps = self.data['robot_state_timestamps']
        control_linear = self.data['control_linear']
        control_angular = self.data['control_angular']
        orientations = self.data['orientations']
        
        # Normalize timestamps to start from 0, aligned to when control inputs start
        # This ensures all plots start at the same time (when MPC planner/teleop started)
        if len(control_timestamps) > 0:
            control_t0 = control_timestamps[0]
            control_times_normalized = control_timestamps - control_t0
            
            # For robot orientation, only include data from when control inputs started
            if len(robot_timestamps) > 0:
                # Find robot states that occur at or after the first control input
                mask = robot_timestamps >= control_t0
                robot_times_aligned = robot_timestamps[mask] - control_t0
                orientations_aligned = orientations[mask]
            else:
                robot_times_aligned = np.array([])
                orientations_aligned = np.array([])
        else:
            # If no control inputs, use robot timestamps as-is
            control_times_normalized = control_timestamps
            if len(robot_timestamps) > 0:
                robot_t0 = robot_timestamps[0]
                robot_times_aligned = robot_timestamps - robot_t0
                orientations_aligned = orientations
            else:
                robot_times_aligned = np.array([])
                orientations_aligned = np.array([])
        
        # Calculate speed (linear velocity is already a scalar for TurtleBot3)
        speed = np.abs(control_linear)
        
        return {
            'control_times_normalized': control_times_normalized,
            'control_linear': control_linear,
            'control_angular': control_angular,
            'speed': speed,
            'robot_times_aligned': robot_times_aligned,
            'orientations_aligned': orientations_aligned
        }
    
    def plot_velocities_manual(self, save_path=None):
        """Plot velocity profiles for manual control."""
        data = self._prepare_velocity_data()
        if data is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Linear velocity (scalar for differential drive)
        plt.subplot(2, 2, 1)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['control_linear'], 'b-', label='Linear Velocity')
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Linear Velocity Command (Forward Speed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Angular velocity (scalar for differential drive)
        plt.subplot(2, 2, 2)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['control_angular'], 'r-', label='Angular Velocity')
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity Command (Rotation Speed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speed
        plt.subplot(2, 2, 3)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['speed'], 'm-', linewidth=2)
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Speed (m/s)')
        plt.title('Robot Speed')
        plt.grid(True, alpha=0.3)
        
        # Orientation - aligned to start at first velocity command
        plt.subplot(2, 2, 4)
        if len(data['robot_times_aligned']) > 0:
            plt.plot(data['robot_times_aligned'], data['orientations_aligned'], 'c-', linewidth=2)
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Yaw (rad)')
        plt.title('Robot Orientation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Velocity plot saved to: {save_path}")
        
        # Only show plot if explicitly requested (default: save only, no display)
        if hasattr(self, '_show_plots') and self._show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_velocities_MPC(self, save_path=None):
        """Plot velocity profiles for MPC control with speed limit line."""
        data = self._prepare_velocity_data()
        if data is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Linear velocity (scalar for differential drive)
        plt.subplot(2, 2, 1)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['control_linear'], 'b-', label='Linear Velocity')
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Linear Velocity (m/s)')
        plt.title('Linear Velocity Command (Forward Speed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Angular velocity (scalar for differential drive)
        plt.subplot(2, 2, 2)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['control_angular'], 'r-', label='Angular Velocity')
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity Command (Rotation Speed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speed with speed limit line
        plt.subplot(2, 2, 3)
        if len(data['control_times_normalized']) > 0:
            plt.plot(data['control_times_normalized'], data['speed'], 'm-', linewidth=2, label='Robot Speed')
        # Add speed limit line for MPC
        speed_limit = 0.15
        plt.axhline(y=speed_limit, color='r', linestyle='--', linewidth=2, label=f'Speed Limit ({speed_limit} m/s)')
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Speed (m/s)')
        plt.title('Robot Speed')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Orientation - aligned to start at first velocity command
        plt.subplot(2, 2, 4)
        if len(data['robot_times_aligned']) > 0:
            plt.plot(data['robot_times_aligned'], data['orientations_aligned'], 'c-', linewidth=2)
        plt.xlabel('Time (s, from first command)')
        plt.ylabel('Yaw (rad)')
        plt.title('Robot Orientation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Velocity plot saved to: {save_path}")
        
        # Only show plot if explicitly requested (default: save only, no display)
        if hasattr(self, '_show_plots') and self._show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_velocities(self, save_path=None):
        """Plot velocity profiles - automatically detects MPC vs manual based on file path."""
        # Check if this is MPC data by looking at the file path
        is_mpc = 'MPC' in str(self.data_file) or 'mpc' in str(self.data_file).lower()
        
        if is_mpc:
            self.plot_velocities_MPC(save_path)
        else:
            self.plot_velocities_manual(save_path)
    
    def generate_report(self, output_file=None):
        """Generate a comprehensive analysis report"""
        if self.data is None:
            return
        
        summary = self.get_data_summary()
        
        report = f"""
Robot Data Analysis Report
==========================

Data File: {self.data_file}
Generated: {np.datetime64('now')}

Summary Statistics:
------------------
Total Distance Traveled: {summary['total_distance']:.3f} m
Robot State Duration: {summary['robot_duration']:.2f} s
Control Input Duration: {summary['control_duration']:.2f} s
Robot State Samples: {summary['robot_num_samples']}
Control Input Samples: {summary['control_num_samples']}
Robot State Frequency: {summary['robot_avg_frequency']:.2f} Hz
Control Input Frequency: {summary['control_avg_frequency']:.2f} Hz

Start Position: [{summary['start_position'][0]:.3f}, {summary['start_position'][1]:.3f}]
End Position: [{summary['end_position'][0]:.3f}, {summary['end_position'][1]:.3f}]
Target Position: [{summary['target_position'][0]:.3f}, {summary['target_position'][1]:.3f}]

Position Range:
  X: [{summary['position_range']['x_min']:.3f}, {summary['position_range']['x_max']:.3f}]
  Y: [{summary['position_range']['y_min']:.3f}, {summary['position_range']['y_max']:.3f}]

Velocity Statistics:
-------------------
Maximum Linear Speed: {summary['max_linear_speed']:.3f} m/s
Average Linear Speed: {summary['avg_linear_speed']:.3f} m/s
Maximum Angular Speed: {summary['max_angular_speed']:.3f} rad/s
Average Angular Speed: {summary['avg_angular_speed']:.3f} rad/s

Lidar Statistics:
----------------
Lidar Duration: {summary['lidar_duration']:.2f} s
Lidar Samples: {summary['lidar_num_samples']}
Lidar Frequency: {summary['lidar_avg_frequency']:.2f} Hz

Obstacle Information:
-------------------
Number of Obstacles: {summary['num_obstacles']}
"""
        
        # Add obstacle positions if available
        if summary['num_obstacles'] > 0:
            report += "\nObstacle Positions:\n"
            for i, obs_pos in enumerate(summary['obstacle_positions']):
                report += f"  Obstacle {i+1}: [{obs_pos[0]:.3f}, {obs_pos[1]:.3f}, {obs_pos[2]:.3f}]\n"
        
        report += f"""
Target Achievement:
------------------
Final Distance to Target: {summary['final_distance_to_target']:.3f} m
Tolerance: {summary['tolerance']:.3f} m
Target Reached: {'Yes' if summary['target_reached'] else 'No'}
"""
        
        print(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze robot data from recording session')
    parser.add_argument('data_file', help='Path to the .npz data file')
    parser.add_argument('--plot-trajectory', action='store_true', help='Plot robot trajectory')
    parser.add_argument('--plot-velocities', action='store_true', help='Plot velocity profiles')
    parser.add_argument('--plot-all', action='store_true', help='Generate all plots')
    parser.add_argument('--plot-occupancy', action='store_true', help='Plot occupancy grid')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots (only display)')
    parser.add_argument('--show-plots', action='store_true', help='Display plots (default: only save, do not display)')
    parser.add_argument('--report', action='store_true', help='Generate analysis report')
    parser.add_argument('--output-dir', default=None, help='Output directory for saved files (default: same as npz file)')
    
    args = parser.parse_args()
    
    # Get the directory of the npz file - plots will be saved here by default
    data_file_path = Path(args.data_file).resolve()
    npz_file_dir = data_file_path.parent
    
    # Use output directory if specified, otherwise use the npz file's directory
    output_dir = args.output_dir if args.output_dir else str(npz_file_dir)
    
    # Create output directory if saving plots (which is now the default)
    if not args.no_save:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}")
    
    # Initialize analyzer
    analyzer = RobotDataAnalyzer(args.data_file)
    
    # Set show_plots flag based on argument
    analyzer._show_plots = args.show_plots
    
    if analyzer.data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Generate plots - always save unless --no-save is specified
    if args.plot_all or args.plot_trajectory:
        save_path = os.path.join(output_dir, 'trajectory.png') if not args.no_save else None
        analyzer.plot_trajectory(save_path)
    
    if args.plot_all or args.plot_velocities:
        save_path = os.path.join(output_dir, 'velocities.png') if not args.no_save else None
        analyzer.plot_velocities(save_path)
    
    # Plot occupancy grid
    if args.plot_all or args.plot_occupancy:
        save_path = os.path.join(output_dir, 'occupancy_grid.png') if not args.no_save else None
        analyzer.plot_occupancy_grid(save_path)
    
    # Generate report - always save to the same directory
    if args.report:
        report_path = os.path.join(output_dir, 'analysis_report.txt') if not args.no_save else None
        analyzer.generate_report(report_path)
    
    # If no specific options, show summary
    if not any([args.plot_trajectory, args.plot_velocities, args.plot_all, args.report]):
        analyzer.generate_report()

if __name__ == '__main__':
    main()
