#!/usr/bin/env python3
# Standalone diffusion / flow-matching policy planner (no MPC safety wrapper).
# Use this for debugging the policy in isolation before enabling the switcher.
#
# Model type is auto-detected from the checkpoint filename:
#   "flow"          → flow-matching policy
#   "deterministic" → deterministic MLP
#   anything else   → diffusion policy (DDPM)
#
# Usage (flow matching — default):
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_diffusion_real_world.launch.py
#
# Usage (diffusion):
#   ros2 launch turtlebot3_sles_control turtlebot3_planner_diffusion_real_world.launch.py \
#       model_path:=/home/acrl/robot_data/real_world_models/diffusion_h50/last_model_diffusion_policy.pth
#
# Visualizations (lidar + predicted trajectory, robot frame) saved to ~/policy_vis/ every
# vis_every_n planning cycles.  View with:
#   eog ~/policy_vis/vis_step_*.png
#
# Prerequisites:
#   1. turtlebot3_node          (/odom, /scan, /tf odom→base_footprint)
#   2. turtlebot3_cartographer  (/tf map→odom)
#   3. rviz2                    (optional — for '2D Goal Pose')

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/acrl/robot_data/real_world_models/flow_h50/best_model_flow_matching.pth',
        description='Path to .pth checkpoint. Model type auto-detected from filename.',
    )
    num_inference_steps_arg = DeclareLaunchArgument(
        'num_inference_steps',
        default_value='50',
        description='ODE/denoising steps (flow: 10–50, diffusion: 10–100).',
    )
    num_samples_arg = DeclareLaunchArgument(
        'num_samples',
        default_value='1',
        description='Trajectory samples per inference call.',
    )
    vis_dir_arg = DeclareLaunchArgument(
        'vis_dir',
        default_value=os.path.join(os.path.expanduser('~'), 'policy_vis'),
        description='Directory for periodic visualization PNGs.',
    )
    vis_every_n_arg = DeclareLaunchArgument(
        'vis_every_n',
        default_value='5',
        description='Save visualization every N planning cycles (10 Hz → every 0.5 s at default).',
    )

    planner_params = {
        'v_limit':                0.21,   # m/s
        'omega_limit':            1.82,   # rad/s
        'lidar_max_range':        3.5,    # m  (overridden from checkpoint meta at runtime)
        'pure_pursuit_lookahead': 0.3,    # m
        'pure_pursuit_v_ref':     0.15,   # m/s
    }

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_diffusion_real_world.py',
        name='policy_planner_node',
        parameters=[planner_params, {
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
