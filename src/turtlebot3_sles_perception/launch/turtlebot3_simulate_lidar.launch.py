#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_simulate_lidar
# Migrated from turtlebot3_simulate_lidar.launch (ROS1)
# Runs the fixed 8-obstacle simulated LiDAR publisher.
#
# Usage:
#   ros2 launch turtlebot3_sles_perception turtlebot3_simulate_lidar.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Obstacle positions (fixed defaults matching original ROS1 launch)
    obstacle_defaults = {
        'random_x_1': 0.7,  'random_y_1': 0.3,
        'random_x_2': -0.8, 'random_y_2': 0.7,
        'random_x_3': -1.0, 'random_y_3': 1.0,
        'random_x_4': 0.5,  'random_y_4': 0.5,
        'random_x_5': -0.5, 'random_y_5': -0.5,
        'random_x_6': 0.5,  'random_y_6': -1.0,
        'random_x_7': -1.5, 'random_y_7': -1.5,
        'random_x_8': 1.5,  'random_y_8': -1.5,
    }

    simulate_lidar_node = Node(
        package='turtlebot3_sles_perception',
        executable='simulate_lidar_publisher.py',
        name='simulate_lidar_publisher',
        parameters=[{
            'lidar_publish_frequency': 20.0,
            'num_lidar_rays': 360,
            'lidar_max_distance': 1.0,
            'gaussian_probability': 0.91,
            'exponential_probability': 0.03,
            'uniform_probability': 0.03,
            'max_probability': 0.03,
            **obstacle_defaults,
        }],
        output='screen',
    )

    return LaunchDescription([simulate_lidar_node])
