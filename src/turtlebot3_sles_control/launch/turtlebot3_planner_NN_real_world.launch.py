#!/usr/bin/env python3
# Real-world launch file for the Neural Network planner.
# Differences from turtlebot3_planner_NN.launch.py (simulation):
#   - Runs planner_nn_real_world.py  (TF2 pose, /scan, /move_base_simple/goal)
#   - No goal parameter at startup — goal is set interactively via RViz2 at runtime
#
# Usage:
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py
#
# Prerequisites (must already be running before this launch):
#   1. turtlebot3_node         (hw driver — publishes /scan, /tf odom→base_footprint)
#   2. turtlebot3_cartographer (SLAM — publishes /tf map→odom)
#   3. rviz2  (optional but recommended for '2D Goal Pose' goal input)
#
# Tunable parameters (pass on command line or via a YAML):
#   v_limit_haa     : max forward speed   [default 0.2 m/s]
#   omega_limit_haa : max angular speed   [default 0.9 rad/s]
#   robot_radius    : footprint radius    [default 0.15 m]
#   lidar_max_range : clip LiDAR to this  [default 2.0 m — must match training]

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path
import os


def _src_config_from_colcon_install():
    """Resolve workspace/src/Config from an install-tree launch file path."""
    parts = Path(__file__).resolve().parts
    if 'install' not in parts:
        return None
    i = parts.index('install')
    cfg = Path(*parts[:i]) / 'src' / 'Config'
    return str(cfg) if cfg.is_dir() else None


def _default_config_dir():
    sles_ws = os.environ.get('SLES_WS', '').strip()
    if sles_ws:
        return os.path.join(sles_ws, 'src', 'Config')
    found = _src_config_from_colcon_install()
    if found:
        return found
    return os.path.join(os.getcwd(), 'src', 'Config')


def generate_launch_description():
    goal_file = os.path.join(_default_config_dir(), 'goal.yaml')

    # ── Launch argument: path to the trained model (.pth file) ──────────────
    # Pass on the command line:
    #   ros2 launch ... model_path:=/path/to/best_model.pth
    # If omitted, the node falls back to best_model.pth next to the script
    # in the install tree (original behaviour).
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='',
        description=(
            'Absolute path to the trained best_model.pth file. '
            'If empty, falls back to the model installed alongside the node script. '
            'Example: model_path:=~/robot_data/real_world_models/best_model_real.pth'
        ),
    )

    planner_params = {
        # Velocity limits — match training-time constraints used in simulation.
        # Waffle Pi hardware max is 0.26 m/s / 1.82 rad/s; leave safety margin.
        'v_limit_haa':     0.26,   # m/s
        'omega_limit_haa': 1.82,   # rad/s
        'robot_radius':    0.15,  # m  (waffle_pi footprint + margin)
        # LiDAR clipping: must match training max-range. (original: 1.0)
        'lidar_max_range': 2.0,   # m
    }

    # Only load goal.yaml if it exists (goal is normally set at runtime via RViz2)
    yaml_params = [goal_file] if os.path.isfile(goal_file) else []

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_nn_real_world.py',
        name='nn_planner_node',
        parameters=yaml_params + [planner_params, {
            'model_path': LaunchConfiguration('model_path'),
        }],
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        planner_node,
    ])
