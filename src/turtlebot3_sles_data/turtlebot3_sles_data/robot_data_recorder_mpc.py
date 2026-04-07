#!/usr/bin/env python3
"""
Refactored Robot Data Recorder with Incremental Writing
- Uses JSON Lines format for incremental writing (one JSON object per line)
- Low memory usage (constant, not proportional to recording time)
- Crash-safe (data persists even if process crashes)
- Clean separation of concerns
"""

import rclpy
from rclpy.node import Node
import signal
import sys
import os
import time
import numpy as np
import json
from datetime import datetime
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
# ROS2: GetModelState service no longer used (removed)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion
import subprocess
import glob


class TargetDetector:
    """Handles target detection logic"""
    
    def __init__(self, target_position, tolerance):
        self.target_position = target_position
        self.tolerance = tolerance
        self.target_reached = False
    
    def check_reached(self, current_x, current_y):
        """Check if target is reached"""
        if self.target_reached:
            return True
        
        distance = np.sqrt((current_x - self.target_position[0])**2 + 
                          (current_y - self.target_position[1])**2)
        
        if distance <= self.tolerance:
            self.target_reached = True
            return True
        
        return False
    
    def get_distance(self, current_x, current_y):
        """Get current distance to target"""
        return np.sqrt((current_x - self.target_position[0])**2 + 
                      (current_y - self.target_position[1])**2)


class DataWriter:
    """Handles incremental data writing to JSON Lines format"""
    
    def __init__(self, data_dir, timestamp):
        self.data_dir = data_dir
        self.timestamp = timestamp
        
        # File handles for JSON Lines format (one JSON object per line)
        self.robot_states_file = None
        self.control_inputs_file = None
        self.lidar_scans_file = None
        
        # Metadata to save at the end
        self.obstacle_positions = {}
        self.final_occupancy_grid = None
        self.target_position = None
        self.tolerance = None
        
        # Counters for statistics
        self.robot_state_count = 0
        self.control_input_count = 0
        self.lidar_scan_count = 0
        
        # Rate limiting (downsample to avoid too many writes)
        self.robot_state_last_write = 0
        self.robot_state_write_interval = 0.02  # 50 Hz max (down from 1000 Hz)
    
    def start_recording(self, target_position, tolerance, obstacle_positions):
        """Open files and start recording"""
        self.target_position = target_position
        self.tolerance = tolerance
        self.obstacle_positions = obstacle_positions
        
        # Open files for writing (JSON Lines format)
        self.robot_states_file = open(
            os.path.join(self.data_dir, f"robot_states_{self.timestamp}.jsonl"), 'w'
        )
        self.control_inputs_file = open(
            os.path.join(self.data_dir, f"control_inputs_{self.timestamp}.jsonl"), 'w'
        )
        self.lidar_scans_file = open(
            os.path.join(self.data_dir, f"lidar_scans_{self.timestamp}.jsonl"), 'w'
        )
        
        self.get_logger().info("Data files opened for incremental writing")
    
    def write_robot_state(self, robot_state, timestamp):
        """Write robot state immediately to file"""
        # Rate limiting: only write every N seconds
        current_time = time.time()
        if current_time - self.robot_state_last_write < self.robot_state_write_interval:
            return
        
        self.robot_state_last_write = current_time
        
        try:
            data = {
                'position': robot_state['position'],
                'orientation': robot_state['orientation'],
                'yaw': robot_state['yaw'],
                'linear_velocity': robot_state['linear_velocity'],
                'angular_velocity': robot_state['angular_velocity'],
                'timestamp': timestamp
            }
            json.dump(data, self.robot_states_file)
            self.robot_states_file.write('\n')
            self.robot_states_file.flush()  # Ensure written to disk
            self.robot_state_count += 1
        except Exception as e:
            self.get_logger().error(f"Error writing robot state: {e}")
    
    def write_control_input(self, control_input, timestamp):
        """Write control input immediately to file"""
        try:
            data = {
                'linear_x': control_input['linear_x'],
                'linear_y': control_input['linear_y'],
                'linear_z': control_input['linear_z'],
                'angular_x': control_input['angular_x'],
                'angular_y': control_input['angular_y'],
                'angular_z': control_input['angular_z'],
                'timestamp': timestamp
            }
            json.dump(data, self.control_inputs_file)
            self.control_inputs_file.write('\n')
            self.control_inputs_file.flush()
            self.control_input_count += 1
        except Exception as e:
            self.get_logger().error(f"Error writing control input: {e}")
    
    def write_lidar_scan(self, lidar_scan, timestamp):
        """Write lidar scan immediately to file"""
        try:
            data = {
                'ranges': lidar_scan['ranges'],
                'angle_min': lidar_scan['angle_min'],
                'angle_max': lidar_scan['angle_max'],
                'angle_increment': lidar_scan['angle_increment'],
                'range_min': lidar_scan['range_min'],
                'range_max': lidar_scan['range_max'],
                'timestamp': timestamp
            }
            json.dump(data, self.lidar_scans_file)
            self.lidar_scans_file.write('\n')
            self.lidar_scans_file.flush()
            self.lidar_scan_count += 1
        except Exception as e:
            self.get_logger().error(f"Error writing lidar scan: {e}")
    
    def set_final_occupancy_grid(self, occupancy_grid):
        """Store final occupancy grid (saved at end)"""
        self.final_occupancy_grid = occupancy_grid
    
    def finalize(self):
        """Close files and convert to final formats (JSON arrays and NPZ)"""
        self.get_logger().info("Finalizing data recording...")
        
        # Close JSON Lines files
        try:
            if self.robot_states_file:
                self.robot_states_file.close()
            if self.control_inputs_file:
                self.control_inputs_file.close()
            if self.lidar_scans_file:
                self.lidar_scans_file.close()
        except Exception as e:
            self.get_logger().error(f"Error closing JSON Lines files: {e}")
        
        self.get_logger().info(f"Recorded {self.robot_state_count} robot states")
        self.get_logger().info(f"Recorded {self.control_input_count} control inputs")
        self.get_logger().info(f"Recorded {self.lidar_scan_count} lidar scans")
        
        # Convert JSON Lines to JSON arrays and NPZ
        # Use time.sleep instead of rospy.sleep to avoid ROS shutdown interruption
        try:
            self._convert_to_final_formats()
        except Exception as e:
            self.get_logger().error(f"Error in finalize() during conversion: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            import sys
            sys.stderr.flush()
            raise  # Re-raise to be handled by caller
    
    def _convert_to_final_formats(self):
        """Convert JSON Lines files to JSON arrays and NPZ format"""
        self.get_logger().info("Converting JSON Lines to final formats...")
        
        try:
            # Read JSON Lines and convert to arrays
            robot_states = []
            robot_state_timestamps = []
            control_inputs = []
            control_input_timestamps = []
            lidar_scans = []
            lidar_timestamps = []
            
            # Read robot states
            states_file_path = os.path.join(self.data_dir, f"robot_states_{self.timestamp}.jsonl")
            if os.path.exists(states_file_path):
                file_size = os.path.getsize(states_file_path)
                self.get_logger().info(f"Reading robot states from {states_file_path} ({file_size} bytes)")
                with open(states_file_path, 'r') as f:
                    line_count = 0
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                robot_states.append(data)
                                robot_state_timestamps.append(data['timestamp'])
                                line_count += 1
                            except json.JSONDecodeError as e:
                                self.get_logger().warn(f"Error parsing JSON line in robot states: {e}")
                    self.get_logger().info(f"Loaded {line_count} robot states from JSON Lines")
            else:
                self.get_logger().warn(f"Robot states JSON Lines file not found: {states_file_path}")
            
            # Read control inputs
            controls_file_path = os.path.join(self.data_dir, f"control_inputs_{self.timestamp}.jsonl")
            if os.path.exists(controls_file_path):
                file_size = os.path.getsize(controls_file_path)
                self.get_logger().info(f"Reading control inputs from {controls_file_path} ({file_size} bytes)")
                with open(controls_file_path, 'r') as f:
                    line_count = 0
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                control_inputs.append(data)
                                control_input_timestamps.append(data['timestamp'])
                                line_count += 1
                            except json.JSONDecodeError as e:
                                self.get_logger().warn(f"Error parsing JSON line in control inputs: {e}")
                    self.get_logger().info(f"Loaded {line_count} control inputs from JSON Lines")
            else:
                self.get_logger().warn(f"Control inputs JSON Lines file not found: {controls_file_path}")
            
            # Read lidar scans
            lidar_file_path = os.path.join(self.data_dir, f"lidar_scans_{self.timestamp}.jsonl")
            if os.path.exists(lidar_file_path):
                file_size = os.path.getsize(lidar_file_path)
                self.get_logger().info(f"Reading lidar scans from {lidar_file_path} ({file_size} bytes)")
                with open(lidar_file_path, 'r') as f:
                    line_count = 0
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                lidar_scans.append(data)
                                lidar_timestamps.append(data['timestamp'])
                                line_count += 1
                            except json.JSONDecodeError as e:
                                self.get_logger().warn(f"Error parsing JSON line in lidar scans: {e}")
                    self.get_logger().info(f"Loaded {line_count} lidar scans from JSON Lines")
            else:
                self.get_logger().warn(f"Lidar scans JSON Lines file not found: {lidar_file_path}")
            
            self.get_logger().info(f"Data loaded: {len(robot_states)} states, {len(control_inputs)} controls, {len(lidar_scans)} scans")
            
            # Save as JSON arrays (for compatibility with existing tools)
            self.get_logger().info("Saving JSON array files...")
            self._save_json_arrays(robot_states, robot_state_timestamps,
                                 control_inputs, control_input_timestamps,
                                 lidar_scans, lidar_timestamps)
            
            # Save as NPZ (using existing conversion logic)
            self.get_logger().info("Saving NPZ files...")
            self._save_npz_files(robot_states, robot_state_timestamps,
                               control_inputs, control_input_timestamps,
                               lidar_scans, lidar_timestamps)
            
            self.get_logger().info("✓ Conversion to final formats complete")
            
        except Exception as e:
            self.get_logger().error("=" * 60)
            self.get_logger().error(f"✗✗✗ ERROR converting to final formats: {e}")
            self.get_logger().error(f"Error type: {type(e).__name__}")
            import traceback
            self.get_logger().error("Full traceback:")
            self.get_logger().error(traceback.format_exc())
            self.get_logger().error("=" * 60)
            # Flush stderr to ensure error is visible
            import sys
            sys.stderr.flush()
            # Try fallback conversion script
            self.get_logger().warn("Attempting fallback conversion...")
            self._fallback_conversion()
    
    def _save_json_arrays(self, robot_states, robot_state_timestamps,
                         control_inputs, control_input_timestamps,
                         lidar_scans, lidar_timestamps):
        """Save data as JSON arrays (for compatibility)"""
        # Robot states
        states_file = os.path.join(self.data_dir, f"robot_states_{self.timestamp}.json")
        with open(states_file, 'w') as f:
            json.dump({
                'robot_states': robot_states,
                'robot_state_timestamps': robot_state_timestamps,
                'target_position': self.target_position,
                'tolerance': self.tolerance,
                'obstacle_positions': self.obstacle_positions
            }, f, indent=2)
        
        # Control inputs
        controls_file = os.path.join(self.data_dir, f"control_inputs_{self.timestamp}.json")
        with open(controls_file, 'w') as f:
            json.dump({
                'control_inputs': control_inputs,
                'control_input_timestamps': control_input_timestamps,
                'target_position': self.target_position,
                'tolerance': self.tolerance,
                'obstacle_positions': self.obstacle_positions
            }, f, indent=2)
        
        # Lidar scans
        lidar_file = os.path.join(self.data_dir, f"lidar_scans_{self.timestamp}.json")
        with open(lidar_file, 'w') as f:
            json.dump({
                'lidar_scans': lidar_scans,
                'lidar_timestamps': lidar_timestamps,
                'target_position': self.target_position,
                'tolerance': self.tolerance,
                'obstacle_positions': self.obstacle_positions
            }, f, indent=2)
        
        # Final occupancy grid
        if self.final_occupancy_grid is not None:
            occupancy_grid_file = os.path.join(self.data_dir, f"occupancy_grid_final_{self.timestamp}.json")
            with open(occupancy_grid_file, 'w') as f:
                json.dump({
                    'occupancy_grid': self.final_occupancy_grid,
                    'target_position': self.target_position,
                    'tolerance': self.tolerance,
                    'obstacle_positions': self.obstacle_positions
                }, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        
        self.get_logger().info("✓ JSON array files saved")
    
    def _save_npz_files(self, robot_states, robot_state_timestamps,
                       control_inputs, control_input_timestamps,
                       lidar_scans, lidar_timestamps):
        """Save data as NPZ files"""
        self.get_logger().info("Starting NPZ file creation...")
        
        try:
            # Convert to numpy arrays
            if len(robot_states) > 0:
                positions = np.array([[state['position'][0], state['position'][1]] 
                                    for state in robot_states])
                orientations = np.array([state['yaw'] for state in robot_states])
                linear_vels = np.array([state['linear_velocity'][0] for state in robot_states])
                angular_vels = np.array([state['angular_velocity'][2] for state in robot_states])
            else:
                positions = np.array([]).reshape(0, 2)
                orientations = np.array([])
                linear_vels = np.array([])
                angular_vels = np.array([])
            
            if len(control_inputs) > 0:
                control_linear = np.array([ctrl['linear_x'] for ctrl in control_inputs])
                control_angular = np.array([ctrl['angular_z'] for ctrl in control_inputs])
            else:
                control_linear = np.array([])
                control_angular = np.array([])
            
            if len(lidar_scans) > 0:
                lidar_ranges_list = [np.array(scan['ranges']) for scan in lidar_scans]
                lidar_angle_mins = np.array([scan['angle_min'] for scan in lidar_scans])
                lidar_angle_maxs = np.array([scan['angle_max'] for scan in lidar_scans])
                lidar_angle_increments = np.array([scan['angle_increment'] for scan in lidar_scans])
                lidar_range_mins = np.array([scan['range_min'] for scan in lidar_scans])
                lidar_range_maxs = np.array([scan['range_max'] for scan in lidar_scans])
            else:
                lidar_ranges_list = []
                lidar_angle_mins = np.array([])
                lidar_angle_maxs = np.array([])
                lidar_angle_increments = np.array([])
                lidar_range_mins = np.array([])
                lidar_range_maxs = np.array([])
            
            # Obstacle positions
            obstacle_names = []
            obstacle_positions_array = []
            for obs_name, obs_data in self.obstacle_positions.items():
                obstacle_names.append(obs_name)
                obstacle_positions_array.append(obs_data['position'])
            
            if len(obstacle_positions_array) > 0:
                obstacle_positions_array = np.array(obstacle_positions_array)
                obstacle_names_array = np.array(obstacle_names, dtype=object)
            else:
                obstacle_positions_array = np.array([]).reshape(0, 3)
                obstacle_names_array = np.array([], dtype=object)
            
            # Occupancy grid
            if self.final_occupancy_grid is not None:
                occupancy_grid_resolution = self.final_occupancy_grid['info']['resolution']
                occupancy_grid_width = self.final_occupancy_grid['info']['width']
                occupancy_grid_height = self.final_occupancy_grid['info']['height']
                occupancy_grid_origin_x = self.final_occupancy_grid['info']['origin']['position']['x']
                occupancy_grid_origin_y = self.final_occupancy_grid['info']['origin']['position']['y']
                occupancy_grid_timestamp = self.final_occupancy_grid.get('timestamp', 0.0)
                occupancy_grid_data = np.array(self.final_occupancy_grid['data'], dtype=np.int8)
            else:
                occupancy_grid_resolution = 0.0
                occupancy_grid_width = 0
                occupancy_grid_height = 0
                occupancy_grid_origin_x = 0.0
                occupancy_grid_origin_y = 0.0
                occupancy_grid_timestamp = 0.0
                occupancy_grid_data = np.array([], dtype=np.int8)
            
            # Save main NPZ file
            self.get_logger().info(f"Creating main NPZ file: robot_data_{self.timestamp}.npz")
            np_file = os.path.join(self.data_dir, f"robot_data_{self.timestamp}.npz")
            np.savez(np_file,
                positions=positions,
                orientations=orientations,
                linear_velocities=linear_vels,
                angular_velocities=angular_vels,
                control_linear=control_linear,
                control_angular=control_angular,
                robot_state_timestamps=np.array(robot_state_timestamps),
                control_input_timestamps=np.array(control_input_timestamps),
                lidar_timestamps=np.array(lidar_timestamps),
                lidar_angle_mins=lidar_angle_mins,
                lidar_angle_maxs=lidar_angle_maxs,
                lidar_angle_increments=lidar_angle_increments,
                lidar_range_mins=lidar_range_mins,
                lidar_range_maxs=lidar_range_maxs,
                occupancy_grid_timestamp=occupancy_grid_timestamp,
                occupancy_grid_resolution=occupancy_grid_resolution,
                occupancy_grid_width=occupancy_grid_width,
                occupancy_grid_height=occupancy_grid_height,
                occupancy_grid_origin_x=occupancy_grid_origin_x,
                occupancy_grid_origin_y=occupancy_grid_origin_y,
                target_position=np.array(self.target_position),
                tolerance=self.tolerance,
                obstacle_positions=obstacle_positions_array,
                obstacle_names=obstacle_names_array)
            
            # Sync to disk
            with open(np_file, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            
            # Verify file was created
            if os.path.exists(np_file) and os.path.getsize(np_file) > 0:
                self.get_logger().info(f"✓ Main NPZ file saved: {np_file} ({os.path.getsize(np_file)} bytes)")
            else:
                raise Exception(f"NPZ file was not created or is empty: {np_file}")
            
            # Save lidar ranges separately
            if len(lidar_ranges_list) > 0:
                self.get_logger().info(f"Creating lidar ranges NPZ file...")
                lidar_ranges_file = os.path.join(self.data_dir, f"lidar_ranges_{self.timestamp}.npz")
                lidar_data_dict = {f'scan_{i}': ranges for i, ranges in enumerate(lidar_ranges_list)}
                np.savez(lidar_ranges_file, **lidar_data_dict, num_scans=len(lidar_ranges_list))
                with open(lidar_ranges_file, 'r+b') as f:
                    f.flush()
                    os.fsync(f.fileno())
                self.get_logger().info(f"✓ Lidar ranges NPZ file saved: {lidar_ranges_file}")
            
            # Save occupancy grid data separately
            if self.final_occupancy_grid is not None and len(occupancy_grid_data) > 0:
                self.get_logger().info(f"Creating occupancy grid NPZ file...")
                occupancy_grid_data_file = os.path.join(self.data_dir, f"occupancy_grid_data_{self.timestamp}.npz")
                np.savez(occupancy_grid_data_file,
                        occupancy_grid_data=occupancy_grid_data,
                        resolution=occupancy_grid_resolution,
                        width=occupancy_grid_width,
                        height=occupancy_grid_height,
                        origin_x=occupancy_grid_origin_x,
                        origin_y=occupancy_grid_origin_y,
                        timestamp=occupancy_grid_timestamp)
                with open(occupancy_grid_data_file, 'r+b') as f:
                    f.flush()
                    os.fsync(f.fileno())
                self.get_logger().info(f"✓ Occupancy grid NPZ file saved: {occupancy_grid_data_file}")
            
            self.get_logger().info("✓ All NPZ files saved successfully")
        
        except Exception as e:
            self.get_logger().error(f"✗✗✗ ERROR saving NPZ files: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            raise  # Re-raise to be caught by _convert_to_final_formats
    
    def _fallback_conversion(self):
        """Fallback: Use convert_json_to_npz.py script"""
        self.get_logger().warn("Attempting fallback conversion using convert_json_to_npz.py...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            convert_script = os.path.join(script_dir, "convert_json_to_npz.py")
            
            if os.path.exists(convert_script):
                result = subprocess.run(
                    ['python3', convert_script, self.data_dir, '--no-analysis'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.get_logger().info("✓ Fallback conversion successful")
                else:
                    self.get_logger().error(f"Fallback conversion failed: {result.stderr}")
            else:
                self.get_logger().error(f"Conversion script not found: {convert_script}")
        except Exception as e:
            self.get_logger().error(f"Fallback conversion error: {e}")


class RobotDataRecorder(Node):
    """Main recorder class with incremental writing"""
    
    def __init__(self, target_position=None, tolerance=0.3, v_limit_haa=None):
        super().__init__('robot_data_recorder')
        
        # Configuration
        self.target_position = target_position if target_position else [-1.5, -1.5]
        self.tolerance = tolerance
        self.robot_name = "turtlebot3_waffle_pi"
        self.v_limit_haa = v_limit_haa
        
        # State
        self.recording = True
        self.data_saved = False
        self.finalization_in_progress = False
        self.finalization_complete = False
        
        # Components
        self.target_detector = TargetDetector(self.target_position, self.tolerance)
        
        # Create session directory
        base_data_dir = os.path.join(os.path.expanduser('~'), 'robot_data')
        os.makedirs(base_data_dir, exist_ok=True)
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include v_limit_haa in folder name if provided
        if self.v_limit_haa is not None:
            # Format v_limit_haa value (remove leading zeros, handle decimals)
            v_str = str(self.v_limit_haa).replace('.', 'p')
            session_folder = f"session_{session_timestamp}_MPC_v{v_str}"
        else:
            session_folder = f"session_{session_timestamp}_MPC"
        self.data_dir = os.path.join(base_data_dir, session_folder)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load obstacle positions
        self.obstacle_positions = {}
        self.load_obstacle_positions()
        
        # Initialize data writer
        self.data_writer = DataWriter(self.data_dir, session_timestamp)
        self.data_writer.start_recording(
            self.target_position, 
            self.tolerance, 
            self.obstacle_positions
        )
        
        # ROS subscribers
        self.model_states_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/simulated_scan', self.lidar_callback, 10)
        self.occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/lidar_occupancy_map', self.occupancy_grid_callback, 10)
        
        self.get_logger().info(f"Robot Data Recorder initialized (incremental writing mode)")
        self.get_logger().info(f"Target position: {self.target_position}")
        self.get_logger().info(f"Tolerance: {self.tolerance}")
        self.get_logger().info(f"Data directory: {self.data_dir}")
    
    def load_obstacle_positions(self):
        """Load obstacle positions from ROS parameters or Gazebo"""
        obstacles_loaded = False
        
        # Try ROS parameters first
        for i in range(1, 17):
            try:
                x = rospy.get_param(f'/random_x_{i}')
                y = rospy.get_param(f'/random_y_{i}')
                size = rospy.get_param(f'/random_size_{i}', 0.2)
                shape = rospy.get_param(f'/random_shape_{i}', 'rectangle')
                name = f"obstacle_{i}"
                self.obstacle_positions[name] = {
                    'position': [x, y, 0.0],
                    'size': size,
                    'shape': shape
                }
                obstacles_loaded = True
            except KeyError:
                pass
        
        # Fallback: Gazebo model states
        if not obstacles_loaded:
            try:
                get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
                for i in range(1, 17):
                    name = f"obstacle_{i}"
                    try:
                        resp = get_model_state(name, 'world')
                        if resp.success:
                            self.obstacle_positions[name] = {
                                'position': [resp.pose.position.x, resp.pose.position.y, resp.pose.position.z],
                                'size': 0.2,
                                'shape': 'rectangle'
                            }
                            obstacles_loaded = True
                    except:
                        pass
            except Exception as e:
                self.get_logger().warn(f"Could not get obstacles from Gazebo: {e}")
        
        self.get_logger().info(f"Loaded {len(self.obstacle_positions)} obstacles")
    
    def model_states_callback(self, msg):
        """Callback for robot state - writes immediately"""
        if not self.recording or self.target_detector.target_reached:
            return
        
        try:
            # Find robot
            robot_index = None
            for i, name in enumerate(msg.name):
                if self.robot_name in name:
                    robot_index = i
                    break
            
            if robot_index is None:
                return
            
            pose = msg.pose[robot_index]
            twist = msg.twist[robot_index]
            
            # Extract data
            x = pose.position.x
            y = pose.position.y
            z = pose.position.z
            
            orientation = [pose.orientation.x, pose.orientation.y, 
                          pose.orientation.z, pose.orientation.w]
            _, _, yaw = euler_from_quaternion(orientation)
            
            current_time = self.get_clock().now().to_sec()
            
            robot_state = {
                'position': [x, y, z],
                'orientation': orientation,
                'yaw': yaw,
                'linear_velocity': [twist.linear.x, twist.linear.y, twist.linear.z],
                'angular_velocity': [twist.angular.x, twist.angular.y, twist.angular.z]
            }
            
            # Write immediately (with rate limiting)
            self.data_writer.write_robot_state(robot_state, current_time)
            
            # Check target (non-blocking)
            if self.target_detector.check_reached(x, y) and not self.finalization_in_progress:
                distance = self.target_detector.get_distance(x, y)
                self.get_logger().info(f"🎯 Target reached! Distance: {distance:.3f}m")
                self.finalization_in_progress = True
                self._handle_target_reached()
        
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")
    
    def cmd_vel_callback(self, msg):
        """Callback for control input - writes immediately"""
        if not self.recording or self.target_detector.target_reached:
            return
        
        try:
            current_time = self.get_clock().now().to_sec()
            control_input = {
                'linear_x': msg.linear.x,
                'linear_y': msg.linear.y,
                'linear_z': msg.linear.z,
                'angular_x': msg.angular.x,
                'angular_y': msg.angular.y,
                'angular_z': msg.angular.z
            }
            
            self.data_writer.write_control_input(control_input, current_time)
        
        except Exception as e:
            self.get_logger().error(f"Error in cmd_vel_callback: {e}")
    
    def lidar_callback(self, msg):
        """Callback for lidar scan - writes immediately"""
        if not self.recording or self.target_detector.target_reached:
            return
        
        try:
            current_time = self.get_clock().now().to_sec()
            lidar_scan = {
                'ranges': list(msg.ranges),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'range_min': msg.range_min,
                'range_max': msg.range_max
            }
            
            self.data_writer.write_lidar_scan(lidar_scan, current_time)
        
        except Exception as e:
            self.get_logger().error(f"Error in lidar_callback: {e}")
    
    def occupancy_grid_callback(self, msg):
        """Callback for occupancy grid - stores final one"""
        if not self.recording or self.target_detector.target_reached:
            return
        
        try:
            # Store as dict for JSON serialization
            occupancy_grid = {
                'info': {
                    'resolution': msg.info.resolution,
                    'width': msg.info.width,
                    'height': msg.info.height,
                    'origin': {
                        'position': {
                            'x': msg.info.origin.position.x,
                            'y': msg.info.origin.position.y,
                            'z': msg.info.origin.position.z
                        }
                    }
                },
                'data': list(msg.data),
                'timestamp': self.get_clock().now().to_sec()
            }
            
            self.data_writer.set_final_occupancy_grid(occupancy_grid)
        
        except Exception as e:
            self.get_logger().error(f"Error in occupancy_grid_callback: {e}")
    
    def _handle_target_reached(self):
        """Handle target reached - mark for finalization (don't shutdown from callback)"""
        self.recording = False
        
        self.get_logger().info("🎯 Target reached! Will finalize in main loop...")
        # Don't call finalize() here - let the main loop handle it
        # This prevents ROS shutdown from interrupting the conversion process
        # The main run() loop will detect target_reached and call finalize()
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        self.get_logger().info("Recording stopped")
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.get_logger().info("🛑 Received interrupt signal. Finalizing...")
        self.recording = False
        
        try:
            self.data_writer.finalize()
            self.get_logger().info("✅ Data saved")
        except Exception as e:
            self.get_logger().error(f"Error saving data: {e}")
        finally:
            sys.exit(0)
    
    def check_simulation_running(self):
        """Check if simulation is running"""
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5)
            self.get_logger().info("✓ Gazebo simulation detected")
            return True
        except Exception:
            self.get_logger().error("✗ Gazebo simulation not detected")
            return False
    
    def run(self):
        """Main run function"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        if not self.check_simulation_running():
            self.get_logger().error("Please launch simulation first")
            return
        
        self.get_logger().info("✓ Data recorder ready (incremental writing mode)")
        self.get_logger().info("Press Ctrl+C to stop recording")
        
        # Wait for target or shutdown
        while not not rclpy.ok() and not self.target_detector.target_reached:
            time.sleep(0.1)
        
        # Target reached - finalize in main thread (not callback)
        if self.target_detector.target_reached:
            self.get_logger().info("Target reached in main loop. Finalizing data...")
            self.recording = False
            
            try:
                # Finalize writes and convert to final formats (in main thread, not callback)
                self.data_writer.finalize()
                
                # Verify NPZ file exists
                # Extract timestamp from folder name (handle both old and new formats)
                folder_name = os.path.basename(self.data_dir)
                timestamp = folder_name.replace("session_", "").replace("_MPC", "").split("_v")[0]
                np_file = os.path.join(self.data_dir, f"robot_data_{timestamp}.npz")
                
                # Wait for file system sync
                self.get_logger().info("Waiting for file system sync...")
                time.sleep(1.0)
                
                # Check if NPZ file exists
                if os.path.exists(np_file) and os.path.getsize(np_file) > 0:
                    self.data_saved = True
                    self.get_logger().info(f"✅ Recording complete! NPZ file: {np_file} ({os.path.getsize(np_file)} bytes)")
                else:
                    self.get_logger().warn(f"NPZ file not found at: {np_file}")
                    self.get_logger().warn("Attempting fallback conversion...")
                    try:
                        self.data_writer._fallback_conversion()
                        time.sleep(1.0)
                        # Check again after fallback
                        if os.path.exists(np_file) and os.path.getsize(np_file) > 0:
                            self.data_saved = True
                            self.get_logger().info(f"✅ Fallback conversion successful! NPZ file: {np_file}")
                        else:
                            self.get_logger().error("✗ Fallback conversion also failed - NPZ file still missing")
                            self.data_saved = False
                    except Exception as fallback_err:
                        self.get_logger().error(f"✗ Fallback conversion error: {fallback_err}")
                        self.data_saved = False
                
                # Additional wait to ensure all files are synced
                time.sleep(0.5)
                self.get_logger().info("Initiating shutdown...")
                rclpy.shutdown()
            
            except Exception as e:
                self.get_logger().error(f"✗✗✗ Error finalizing data: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                # Try fallback
                try:
                    self.get_logger().warn("Attempting fallback conversion due to error...")
                    self.data_writer._fallback_conversion()
                    time.sleep(1.0)
                    # Extract timestamp from folder name (handle both old and new formats)
                    folder_name = os.path.basename(self.data_dir)
                    timestamp = folder_name.replace("session_", "").replace("_MPC", "").split("_v")[0]
                    np_file = os.path.join(self.data_dir, f"robot_data_{timestamp}.npz")
                    if os.path.exists(np_file) and os.path.getsize(np_file) > 0:
                        self.get_logger().info("✓ Fallback conversion succeeded after error")
                        self.data_saved = True
                except Exception as fallback_err:
                    self.get_logger().error(f"Fallback also failed: {fallback_err}")
                rclpy.shutdown()")
        
        # If we get here and target wasn't reached, it was a manual stop
        else:
            # Manual stop
            self.get_logger().info("Manual stop requested, finalizing...")
            self.recording = False
            try:
                self.data_writer.finalize()
                # Verify NPZ file
                # Extract timestamp from folder name (handle both old and new formats)
                folder_name = os.path.basename(self.data_dir)
                timestamp = folder_name.replace("session_", "").replace("_MPC", "").split("_v")[0]
                np_file = os.path.join(self.data_dir, f"robot_data_{timestamp}.npz")
                time.sleep(1.0)
                if os.path.exists(np_file) and os.path.getsize(np_file) > 0:
                    self.get_logger().info(f"✅ Manual stop complete! NPZ file: {np_file}")
                else:
                    self.get_logger().warn("NPZ file missing after manual stop, attempting fallback...")
                    self.data_writer._fallback_conversion()
            except Exception as e:
                self.get_logger().error(f"Error finalizing: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
            rclpy.shutdown()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot Data Recorder (Incremental Writing)')
    parser.add_argument('--target-x', type=float, default=-1.5, help='Target X position')
    parser.add_argument('--target-y', type=float, default=-1.5, help='Target Y position')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Distance tolerance')
    parser.add_argument('--v-limit-haa', type=float, default=None, help='v_limit_haa value to include in folder name')
    
    args = parser.parse_args()
    
    target_position = [args.target_x, args.target_y]
    recorder = RobotDataRecorder(target_position=target_position, tolerance=args.tolerance, v_limit_haa=args.v_limit_haa)
    recorder.run()


def ros2_main(args=None):
    rclpy.init(args=args)
    # Note: RobotDataRecorder is both a ROS2 node AND uses argparse.
    # Parse args first, then create the node.
    import argparse
    parser = argparse.ArgumentParser(description='Robot Data Recorder (ROS2)')
    parser.add_argument('--target-x', type=float, default=-1.5)
    parser.add_argument('--target-y', type=float, default=-1.5)
    parser.add_argument('--tolerance', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='waffle_pi')
    parsed_args, _ = parser.parse_known_args()

    node = RobotDataRecorder(
        target_x=parsed_args.target_x,
        target_y=parsed_args.target_y,
        tolerance=parsed_args.tolerance,
        model=parsed_args.model,
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    ros2_main()


# ROS2 entry point
main = ros2_main
