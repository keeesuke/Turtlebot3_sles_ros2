#!/usr/bin/env python3
# Real-world launch file for the MPC + learned policy switching planner.
# Dual-planner simplex architecture: HPA (learned policy) ↔ HAA (MPPI MPC) switching.
#
# Supports diffusion, flow-matching, and deterministic checkpoints — model type
# is auto-detected from the filename by PolicyRunner.
#
# Usage (diffusion):
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_diffusion_real_world.launch.py
#
# Usage (flow matching):
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_diffusion_real_world.launch.py \
#       model_path:=/home/acrl/robot_data/real_world_models/flow_h50/best_model_flow_matching.pth
#
# Policy visualizations (lidar + predicted trajectory in robot frame, saved every vis_every_n
# planning cycles) are written to ~/policy_vis/ by default.
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

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/acrl/robot_data/real_world_models/flow_h50/best_model_flow_matching.pth',
        description='Path to .pth checkpoint. Model type auto-detected from filename '
                    '("flow" → flow matching, "deterministic" → MLP, else → diffusion).',
    )
    num_inference_steps_arg = DeclareLaunchArgument(
        'num_inference_steps',
        default_value='10',
        description='Denoising/ODE steps (diffusion: 10–100, flow: 10–50).',
    )
    num_samples_arg = DeclareLaunchArgument(
        'num_samples',
        default_value='1',
        description='Trajectory samples per call (1 = fastest; >1 picks median sample).',
    )
    vis_dir_arg = DeclareLaunchArgument(
        'vis_dir',
        default_value=os.path.join(os.path.expanduser('~'), 'policy_vis'),
        description='Directory for periodic policy visualization PNGs.',
    )
    vis_every_n_arg = DeclareLaunchArgument(
        'vis_every_n',
        default_value='10',
        description='Save a visualization every N planning cycles (10 Hz → every 1 s at default).',
    )

    planner_params = {
        # MPC/HAA limits (conservative)
        'horizon_haa':     40,
        'dt':              0.1,
        'v_limit_haa':     0.15,   # m/s
        'omega_limit_haa': 0.9,    # rad/s
        'a_limit':         0.5,    # m/s²
        'alpha_limit':     0.5,    # rad/s²
        'robot_radius':    0.15,   # m

        # HPA policy limits (match training v_max/w_max)
        'v_limit_hpa':     0.21,   # m/s
        'omega_limit_hpa': 1.82,   # rad/s

        # Pure pursuit (HPA control)
        'pure_pursuit_lookahead': 0.3,   # m
        'pure_pursuit_v_ref':     0.15,  # m/s

        # LiDAR: must match lidar_max_range used during dataset creation
        'lidar_max_range': 3.5,    # m  (overridden from checkpoint meta at runtime)

        # Kanayama gains (HAA/braking only)
        'kx':              0.6,
        'ky':              8.0,
        'kth':             1.6,
        'kv':              1.0,
        'kw':              1.0,
        'kix':             0.1,
        'kiy':             0.0,
        'kith':            0.1,
        'max_integral':    1.0,

        # Goal set at runtime via RViz2 '2D Goal Pose'
        'goal':            '[0.0, 0.0, 0.0, 0.0, 0.0]',
    }

    yaml_params = [goal_file] if os.path.isfile(goal_file) else []

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_switch_mpc_diffusion_real_world.py',
        name='switch_mpc_diffusion_node',
        parameters=yaml_params + [planner_params, {
            'model_path':          LaunchConfiguration('model_path'),
            'num_inference_steps': LaunchConfiguration('num_inference_steps'),
            'num_samples':         LaunchConfiguration('num_samples'),
            'vis_dir':             LaunchConfiguration('vis_dir'),
            'vis_every_n':         LaunchConfiguration('vis_every_n'),
        }],
        output='screen',
    )

    return LaunchDescription([
        model_path_arg,
        num_inference_steps_arg,
        num_samples_arg,
        vis_dir_arg,
        vis_every_n_arg,
        planner_node,
    ])
