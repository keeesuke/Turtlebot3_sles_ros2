#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_simulate_mapping
# Migrated from turtlebot3_simulate_mapping.launch (ROS1)
# Runs the LiDAR occupancy grid mapping node.
#
# Prerequisites: /simulated_scan topic must be published
#   (run turtlebot3_simulate_lidar or turtlebot3_simulate_lidar_random first)
#
# Usage:
#   ros2 launch turtlebot3_sles_perception turtlebot3_simulate_mapping.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    lidar_mapping_node = Node(
        package='turtlebot3_sles_perception',
        executable='sim_mapping.py',
        name='lidar_mapping_node',
        output='screen',
    )

    return LaunchDescription([lidar_mapping_node])
