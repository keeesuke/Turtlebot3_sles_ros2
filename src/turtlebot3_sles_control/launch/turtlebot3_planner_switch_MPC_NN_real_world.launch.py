#!/usr/bin/env python3
# Real-world launch file for the MPC+NN switching planner.
# Dual-planner simplex architecture: HPA (NN) ↔ HAA (MPC) switching.
#
# Usage:
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py model_path:=/path/to/best_model.pth
#
# Prerequisites (must already be running before this launch):
#   1. turtlebot3_node         (hw driver — publishes /odom, /scan, /tf odom→base_footprint)
#   2. turtlebot3_cartographer (SLAM — publishes /map and /tf map→odom)
#   3. rviz2                   (optional but recommended for '2D Goal Pose' goal input)

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
        # MPC/HAA limits (conservative — used during safety-assured mode)
        'horizon_haa':     40,    # 4 s prediction horizon (40 steps × 0.1 s)
        'dt':              0.1,
        'v_limit_haa':     0.15,  # m/s  (conservative; reduced from 0.2 to widen HPA-HAA gap. original: 0.2)
        'omega_limit_haa': 0.9,   # rad/s
        'a_limit':         0.5,   # m/s²
        'alpha_limit':     0.5,   # rad/s²
        'robot_radius':    0.15,  # m  (waffle_pi footprint with safety margin)

        # NN/HPA limits (wider — used during high-performance NN mode)
        'v_limit_hpa':     0.21,  # m/s  (lowered from 0.26 to match measured hardware cap; training odom was 0.21-0.23 when cmd=0.26. original: 0.26)
        'omega_limit_hpa': 1.82,  # rad/s (NN trained limit)

        # LiDAR clipping: NN trained with this max-range. (original: 1.0)
        'lidar_max_range': 2.0,   # m

        # Kanayama tracking controller gains
        'kx':              0.6,
        'ky':              8.0,
        'kth':             1.6,
        'kv':              1.0,
        'kw':              1.0,
        'kix':             0.1,
        'kiy':             0.0,
        'kith':            0.1,
        'max_integral':    1.0,

        # Goal is set interactively via RViz2 '2D Goal Pose' button at runtime
        'goal':            '[0.0, 0.0, 0.0, 0.0, 0.0]',
    }

    # Only load goal.yaml if it exists (goal is normally set at runtime via RViz2)
    yaml_params = [goal_file] if os.path.isfile(goal_file) else []

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_switch_mpc_nn_real_world.py',
        name='switch_planner_node',
        parameters=yaml_params + [planner_params, {
            'model_path': LaunchConfiguration('model_path'),
        }],
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        planner_node,
    ])
