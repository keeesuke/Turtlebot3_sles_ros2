#!/usr/bin/env python3
"""
ROS node that loads robot initial pose and obstacle positions from a session directory.
This is used to recreate the same world configuration from a previous simulation.
"""

import rospy
import json
import os
import sys
from pathlib import Path

def load_robot_state_from_session(session_path):
    """Load initial robot state from session JSON file."""
    session_path = Path(session_path)
    
    # Find robot_states JSON file
    json_files = list(session_path.glob('robot_states_*.json'))
    
    if not json_files:
        rospy.logwarn(f"No robot_states JSON file found in {session_path}, using default values")
        return None
    
    json_file = json_files[0]
    rospy.loginfo(f"Reading robot initial state from: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'robot_states' not in data or len(data['robot_states']) == 0:
            rospy.logwarn("No robot states found in JSON file, using default values")
            return None
        
        # Get the first robot state (initial state)
        first_state = data['robot_states'][0]
        return first_state
    except Exception as e:
        rospy.logwarn(f"Error reading robot state from {json_file}: {e}, using default values")
        return None

def load_obstacles_from_session(session_path):
    """Load obstacle positions from session JSON file."""
    session_path = Path(session_path)
    
    # Find robot_states JSON file (obstacles are stored there too)
    json_files = list(session_path.glob('robot_states_*.json'))
    
    if not json_files:
        rospy.logwarn(f"No robot_states JSON file found in {session_path}")
        return None
    
    json_file = json_files[0]
    rospy.loginfo(f"Reading obstacle positions from: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'obstacle_positions' not in data:
            rospy.logwarn("No obstacle_positions found in JSON file")
            return None
        
        return data['obstacle_positions']
    except Exception as e:
        rospy.logwarn(f"Error reading obstacle positions from {json_file}: {e}")
        return None

def main():
    rospy.init_node('load_config_from_session', anonymous=True)
    
    # Get session path from ROS parameter, command line argument, or environment variable
    session_path = rospy.get_param('~session_path', '')
    
    if not session_path:
        # Try command line argument
        if len(sys.argv) > 1:
            session_path = sys.argv[1]
        else:
            # Try to get from environment variable
            session_path = os.environ.get('SESSION_PATH', '')
            if not session_path:
                rospy.logerr("No session path provided. Usage: load_config_from_session.py <session_path> or set ~session_path parameter")
                rospy.signal_shutdown("No session path provided")
                return
    
    session_path = Path(os.path.expanduser(session_path))
    
    if not session_path.exists():
        rospy.logerr(f"Session path does not exist: {session_path}")
        rospy.signal_shutdown("Session path does not exist")
        return
    
    # Load robot initial state
    robot_state = load_robot_state_from_session(session_path)
    
    if robot_state:
        robot_x = robot_state['position'][0]
        robot_y = robot_state['position'][1]
        robot_z = robot_state['position'][2]
        robot_yaw = robot_state['yaw']
        rospy.loginfo(f"Loaded robot initial state: x={robot_x:.4f}, y={robot_y:.4f}, z={robot_z:.4f}, yaw={robot_yaw:.4f}")
    else:
        # Fallback to default values
        robot_x = 1.5
        robot_y = 1.5
        robot_z = 0.0
        robot_yaw = -2.356
        rospy.logwarn(f"Using default robot initial state: x={robot_x:.4f}, y={robot_y:.4f}, z={robot_z:.4f}, yaw={robot_yaw:.4f}")
    
    # Load obstacle positions
    obstacle_positions = load_obstacles_from_session(session_path)
    
    if obstacle_positions:
        # Set obstacle parameters
        obstacle_list = []
        for name, obs_data in sorted(obstacle_positions.items()):
            pos = obs_data.get('position', [0, 0, 0])
            size = obs_data.get('size', 0.2)
            shape = obs_data.get('shape', 'rectangle')
            obstacle_list.append((name, pos, size, shape))
        
        # Sort obstacles by name to get consistent numbering
        obstacle_list.sort(key=lambda x: x[0])
        
        # Set ROS parameters for obstacles
        for i, (name, pos, size, shape) in enumerate(obstacle_list, 1):
            rospy.set_param(f'random_x_{i}', pos[0])
            rospy.set_param(f'random_y_{i}', pos[1])
            rospy.set_param(f'random_size_{i}', size)
            rospy.set_param(f'random_shape_{i}', shape)
            rospy.loginfo(f"Set obstacle {i} ({name}): x={pos[0]:.4f}, y={pos[1]:.4f}, size={size:.4f}, shape={shape}")
        
        rospy.loginfo(f"Loaded {len(obstacle_list)} obstacles from session")
    else:
        rospy.logwarn("No obstacles loaded from session, using defaults")
    
    # Set robot pose parameters
    rospy.set_param('random_robot_x', robot_x)
    rospy.set_param('random_robot_y', robot_y)
    rospy.set_param('random_robot_z', robot_z)
    rospy.set_param('random_robot_yaw', robot_yaw)
    
    rospy.loginfo(f"Configuration loaded from session: {session_path}")
    
    # Exit after setting parameters
    rospy.signal_shutdown("Configuration loaded successfully")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
