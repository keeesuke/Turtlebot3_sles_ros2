#!/usr/bin/env python3
"""
Script to extract obstacle positions from a session directory and create a fixed configuration node.
"""

import json
import os
import sys
import math
from pathlib import Path

def extract_obstacles_from_session(session_path):
    """Extract obstacle positions from a session directory."""
    session_path = Path(session_path)
    
    # Find robot_states JSON file
    json_files = list(session_path.glob('robot_states_*.json'))
    
    if not json_files:
        print(f"Error: No robot_states JSON file found in {session_path}")
        return None, None
    
    json_file = json_files[0]
    print(f"Reading obstacle positions from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'obstacle_positions' not in data:
        print(f"Error: No obstacle_positions found in {json_file}")
        return None, None
    
    obstacle_positions = data['obstacle_positions']
    print(f"Found {len(obstacle_positions)} obstacles")
    
    return obstacle_positions, json_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 extract_obstacles_from_session.py <session_path>")
        print("Example: python3 extract_obstacles_from_session.py ~/robot_data/session_20260122-165737_MPC")
        sys.exit(1)
    
    session_path = sys.argv[1]
    obstacles, json_file = extract_obstacles_from_session(session_path)
    
    if obstacles:
        print("\nObstacle positions:")
        for name, obs_data in sorted(obstacles.items()):
            pos = obs_data.get('position', [0, 0, 0])
            size = obs_data.get('size', 0.2)
            shape = obs_data.get('shape', 'rectangle')
            print(f"  {name}: position={pos}, size={size}, shape={shape}")
    
    # Also extract robot initial state
    if json_file:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'robot_states' in data and len(data['robot_states']) > 0:
                first_state = data['robot_states'][0]
                pos = first_state.get('position', [0, 0, 0])
                yaw = first_state.get('yaw', 0.0)
                print(f"\nRobot initial state:")
                print(f"  Position: {pos}")
                print(f"  Yaw: {yaw:.4f} radians ({math.degrees(yaw):.2f} degrees)")
        except Exception as e:
            print(f"\nCould not extract robot initial state: {e}")
