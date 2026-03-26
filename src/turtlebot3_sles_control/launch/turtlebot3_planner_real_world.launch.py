#!/usr/bin/env python3
# Real-world launch file for the HAA/MPPI planner.
# Differences from turtlebot3_planner.launch.py (simulation):
#   - Runs planner_haa_real_world.py (TF2 pose, /map, /move_base_simple/goal)
#   - No scenario.yaml (no simulated random obstacles)
#   - Goal is set interactively via RViz2 '2D Goal Pose' button at runtime
#
# Usage:
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_real_world.launch.py
#
# Prerequisites (must already be running before this launch):
#   1. turtlebot3_node  (hardware driver — publishes /odom, /scan, /tf odom→base_footprint)
#   2. turtlebot3_cartographer  (SLAM — publishes /map and /tf map→odom)
#   3. rviz2  (optional but required for '2D Goal Pose' goal input)

from launch import LaunchDescription
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

    # MPC parameters — kept identical to simulation for first real-world test.
    # Tune after confirming basic navigation works.
    planner_params = {
        'horizon_haa':     40,    # 4 s prediction horizon (40 steps × 0.1 s)
        'dt':              0.1,
        'v_limit_haa':     0.20,  # m/s  (Waffle Pi max ≈ 0.26 m/s; leave margin)
        'omega_limit_haa': 1.22,  # rad/s (Waffle Pi hardware max)
        'a_limit':         0.5,   # m/s²  (raised from 0.25 for quicker acceleration)
        'alpha_limit':     1.0,   # rad/s² (raised from 0.5)
        'robot_radius':    0.12,  # m (waffle_pi footprint with safety margin)
        'kx':              0.6,
        'ky':              8.0,
        'kth':             1.6,
        'kv':              1.0,
        'kw':              1.0,
        'kix':             0.1,
        'kiy':             0.0,
        'kith':            0.1,
        'max_integral':    1.0,
        # goal param is ignored at startup — node waits for RViz2 '2D Goal Pose'
        'goal':            '[0.0, 0.0, 0.0, 0.0, 0.0]',
    }

    # Only load goal.yaml if it exists (contains 'goal' param, overridden by runtime RViz2 anyway)
    yaml_params = [goal_file] if os.path.isfile(goal_file) else []

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_haa_real_world.py',
        name='planner_node',
        parameters=yaml_params + [planner_params],
        output='screen',
    )

    return LaunchDescription([planner_node])
