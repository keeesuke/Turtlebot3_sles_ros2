#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_planner (HAA/MPPI planner)
# Migrated from turtlebot3_planner.launch (ROS1)

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    goal = LaunchConfiguration('goal').perform(context)

    planner_node = Node(
        package='turtlebot3_sles_control',
        executable='planner_haa_only.py',
        name='planner_node',
        parameters=[{
            'horizon_hpa':     20,
            'horizon_haa':     40,
            'dt':              0.1,
            'v_limit_haa':     0.14,
            'v_limit_hpa':     0.2,
            'omega_limit_haa': 0.9,
            'omega_limit_hpa': 1.0,
            'a_limit':         0.25,
            'alpha_limit':     0.5,
            'kx':              0.6,
            'ky':              8.0,
            'kth':             1.6,
            'kv':              1.0,
            'kw':              1.0,
            'kix':             0.1,
            'kiy':             0.0,
            'kith':            0.1,
            'max_integral':    1.0,
            'Q_diag':          '[10, 10, 0, 0, 0]',
            'R_diag':          '[0, 0]',
            'goal':            goal,
        }],
        output='screen',
    )
    return [planner_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'goal',
            default_value='[-1.5, -1.5, 0, 0, 0]',
            description='Goal state [x, y, theta, v, w]'
        ),
        OpaqueFunction(function=_launch_setup),
    ])
