#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_planner_diffusion (Diffusion Policy planner)
# Migrated from turtlebot3_planner_diffusion.launch (ROS1)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    goal = LaunchConfiguration('goal').perform(context)
    inference_steps = int(LaunchConfiguration('inference_steps').perform(context))
    use_warm_start_str = LaunchConfiguration('use_warm_start').perform(context).lower()
    use_warm_start = use_warm_start_str in ('true', '1', 'yes')

    planner_diffusion_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_diffusion.py',
        name='planner_node',
        parameters=[{
            'v_limit_haa':     0.2,
            'omega_limit_haa': 0.9,
            'inference_steps': inference_steps,
            'use_warm_start':  use_warm_start,
            'goal':            goal,
        }],
        output='screen',
    )
    return [planner_diffusion_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('goal', default_value='[-1.5, -1.5, 0, 0, 0]',
                              description='Goal state [x, y, theta, v, w]'),
        DeclareLaunchArgument('inference_steps', default_value='5',
                              description='Number of DDIM inference steps'),
        DeclareLaunchArgument('use_warm_start', default_value='true',
                              description='Use warm-start initialization for DDIM'),
        OpaqueFunction(function=_launch_setup),
    ])
