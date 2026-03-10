#!/usr/bin/env python3
"""
Convert JSON files from a robot data session folder to NPZ format
and then run analysis on the generated NPZ file.

Usage:
    python3 convert_json_to_npz.py <session_folder>
    
Example:
    python3 convert_json_to_npz.py ~/robot_data/session_20260126_190311_MPC
"""

import numpy as np
import json
import os
import sys
import argparse
import glob
from pathlib import Path

def load_json_file(file_path):
    """Load JSON file and return its contents"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_timestamp_from_folder(folder_path):
    """Extract timestamp from folder name (e.g., session_20260126_190311_MPC -> 20260126_190311)
       Also handles new format with v_limit_haa suffix (e.g., session_20260126_190311_MPC_v0p14)"""
    folder_name = os.path.basename(folder_path)
    if folder_name.startswith("session_"):
        # Remove session_ prefix and _MPC suffix, also handle _v* suffix if present
        timestamp = folder_name.replace("session_", "").replace("_MPC", "").split("_v")[0]
        return timestamp
    return None

def convert_json_to_npz(session_folder):
    """Convert JSON files in session folder to NPZ format"""
    session_folder = os.path.abspath(session_folder)
    
    if not os.path.isdir(session_folder):
        print(f"Error: {session_folder} is not a directory")
        return None
    
    # Extract timestamp from folder name
    timestamp = extract_timestamp_from_folder(session_folder)
    if timestamp is None:
        print(f"Warning: Could not extract timestamp from folder name. Using current time.")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Processing session folder: {session_folder}")
    print(f"Using timestamp: {timestamp}")
    
    # Load JSON files
    robot_states_file = os.path.join(session_folder, f"robot_states_{timestamp}.json")
    control_inputs_file = os.path.join(session_folder, f"control_inputs_{timestamp}.json")
    lidar_scans_file = os.path.join(session_folder, f"lidar_scans_{timestamp}.json")
    occupancy_grid_file = os.path.join(session_folder, f"occupancy_grid_final_{timestamp}.json")
    
    print("\nLoading JSON files...")
    
    # Load robot states
    robot_states_data = load_json_file(robot_states_file)
    if robot_states_data is None:
        print(f"Error: Could not load {robot_states_file}")
        return None
    
    robot_states = robot_states_data.get('robot_states', [])
    robot_state_timestamps = robot_states_data.get('robot_state_timestamps', [])
    target_position = robot_states_data.get('target_position', [0, 0, 0])
    tolerance = robot_states_data.get('tolerance', 0.1)
    obstacle_positions = robot_states_data.get('obstacle_positions', {})
    
    print(f"  ✓ Loaded {len(robot_states)} robot states")
    
    # Load control inputs
    control_inputs_data = load_json_file(control_inputs_file)
    if control_inputs_data is None:
        print(f"Warning: Could not load {control_inputs_file}, using empty control inputs")
        control_inputs = []
        control_input_timestamps = []
    else:
        control_inputs = control_inputs_data.get('control_inputs', [])
        control_input_timestamps = control_inputs_data.get('control_input_timestamps', [])
        print(f"  ✓ Loaded {len(control_inputs)} control inputs")
    
    # Load lidar scans
    lidar_scans_data = load_json_file(lidar_scans_file)
    if lidar_scans_data is None:
        print(f"Warning: Could not load {lidar_scans_file}, using empty lidar scans")
        lidar_scans = []
        lidar_timestamps = []
    else:
        lidar_scans = lidar_scans_data.get('lidar_scans', [])
        lidar_timestamps = lidar_scans_data.get('lidar_timestamps', [])
        print(f"  ✓ Loaded {len(lidar_scans)} lidar scans")
    
    # Load occupancy grid
    occupancy_grid_data_json = load_json_file(occupancy_grid_file)
    if occupancy_grid_data_json is None:
        print(f"Warning: Could not load {occupancy_grid_file}, using empty occupancy grid")
        final_occupancy_grid = None
    else:
        final_occupancy_grid = occupancy_grid_data_json.get('occupancy_grid', None)
        if final_occupancy_grid:
            print(f"  ✓ Loaded occupancy grid")
        else:
            print(f"  ⚠ Occupancy grid file exists but is empty")
    
    # Convert to numpy arrays
    print("\nConverting to numpy arrays...")
    
    # Robot states
    if len(robot_states) > 0:
        positions = np.array([[state['position'][0], state['position'][1]] for state in robot_states])
        orientations = np.array([state['yaw'] for state in robot_states])
        linear_vels = np.array([state['linear_velocity'][0] for state in robot_states])
        angular_vels = np.array([state['angular_velocity'][2] for state in robot_states])
    else:
        positions = np.array([]).reshape(0, 2)
        orientations = np.array([])
        linear_vels = np.array([])
        angular_vels = np.array([])
    
    # Control inputs
    if len(control_inputs) > 0:
        control_linear = np.array([ctrl['linear_x'] for ctrl in control_inputs])
        control_angular = np.array([ctrl['angular_z'] for ctrl in control_inputs])
    else:
        control_linear = np.array([])
        control_angular = np.array([])
    
    # Lidar data
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
    for obs_name, obs_data in obstacle_positions.items():
        obstacle_names.append(obs_name)
        obstacle_positions_array.append(obs_data['position'])
    
    if len(obstacle_positions_array) > 0:
        obstacle_positions_array = np.array(obstacle_positions_array)
        obstacle_names_array = np.array(obstacle_names, dtype=object)
    else:
        obstacle_positions_array = np.array([]).reshape(0, 3)
        obstacle_names_array = np.array([], dtype=object)
    
    # Occupancy grid
    if final_occupancy_grid is not None:
        occupancy_grid_resolution = final_occupancy_grid['info']['resolution']
        occupancy_grid_width = final_occupancy_grid['info']['width']
        occupancy_grid_height = final_occupancy_grid['info']['height']
        occupancy_grid_origin_x = final_occupancy_grid['info']['origin']['position']['x']
        occupancy_grid_origin_y = final_occupancy_grid['info']['origin']['position']['y']
        occupancy_grid_timestamp = final_occupancy_grid.get('timestamp', 0.0)
        occupancy_grid_data = np.array(final_occupancy_grid['data'], dtype=np.int8)
    else:
        occupancy_grid_resolution = 0.0
        occupancy_grid_width = 0
        occupancy_grid_height = 0
        occupancy_grid_origin_x = 0.0
        occupancy_grid_origin_y = 0.0
        occupancy_grid_timestamp = 0.0
        occupancy_grid_data = np.array([], dtype=np.int8)
    
    # Save main NPZ file
    print("\nSaving NPZ files...")
    np_file = os.path.join(session_folder, f"robot_data_{timestamp}.npz")
    
    try:
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
                target_position=np.array(target_position),
                tolerance=tolerance,
                obstacle_positions=obstacle_positions_array,
                obstacle_names=obstacle_names_array)
        
        # Sync file to disk
        with open(np_file, 'r+b') as f:
            f.flush()
            os.fsync(f.fileno())
        
        print(f"  ✓ Saved main NPZ file: {np_file}")
    except Exception as e:
        print(f"  ✗ Error saving main NPZ file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save lidar ranges separately
    if len(lidar_ranges_list) > 0:
        lidar_ranges_file = os.path.join(session_folder, f"lidar_ranges_{timestamp}.npz")
        try:
            lidar_data_dict = {f'scan_{i}': ranges for i, ranges in enumerate(lidar_ranges_list)}
            np.savez(lidar_ranges_file, **lidar_data_dict, num_scans=len(lidar_ranges_list))
            with open(lidar_ranges_file, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            print(f"  ✓ Saved lidar ranges NPZ file: {lidar_ranges_file}")
        except Exception as e:
            print(f"  ✗ Error saving lidar ranges NPZ file: {e}")
    
    # Save occupancy grid data separately
    if final_occupancy_grid is not None and len(occupancy_grid_data) > 0:
        occupancy_grid_data_file = os.path.join(session_folder, f"occupancy_grid_data_{timestamp}.npz")
        try:
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
            print(f"  ✓ Saved occupancy grid data NPZ file: {occupancy_grid_data_file}")
        except Exception as e:
            print(f"  ✗ Error saving occupancy grid data NPZ file: {e}")
    
    print(f"\n✓ Conversion complete! NPZ file saved: {np_file}")
    return np_file

def main():
    parser = argparse.ArgumentParser(description='Convert JSON files to NPZ and run analysis')
    parser.add_argument('session_folder', help='Path to session folder (e.g., ~/robot_data/session_20260126_190311_MPC)')
    parser.add_argument('--no-analysis', action='store_true', help='Skip running analysis after conversion')
    
    args = parser.parse_args()
    
    # Expand user path
    session_folder = os.path.expanduser(args.session_folder)
    
    # Convert JSON to NPZ
    npz_file = convert_json_to_npz(session_folder)
    
    if npz_file is None:
        print("\n✗ Conversion failed. Exiting.")
        sys.exit(1)
    
    # Run analysis if requested
    if not args.no_analysis:
        print("\n" + "="*60)
        print("Running analysis on generated NPZ file...")
        print("="*60)
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        analyze_script = os.path.join(script_dir, "analyze_robot_data.py")
        
        if not os.path.exists(analyze_script):
            print(f"Error: Analysis script not found: {analyze_script}")
            sys.exit(1)
        
        # Run analysis
        import subprocess
        try:
            result = subprocess.run(
                ['python3', analyze_script, npz_file, '--plot-all', '--report', '--output-dir', session_folder],
                cwd=script_dir,
                check=True
            )
            print("\n✓ Analysis complete!")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Analysis failed with exit code {e.returncode}")
            sys.exit(1)
    else:
        print("\nSkipping analysis (--no-analysis flag set)")

if __name__ == "__main__":
    main()
