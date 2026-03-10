#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_simulate_lidar_random
# Migrated from turtlebot3_simulate_lidar_random.launch (ROS1)
# Runs the multi-shape (16 obstacle) simulated LiDAR publisher.
# Obstacle positions should be set via parameters or overridden from the
# turtlebot3_custom_world_random launch (which also starts this node).
#
# Usage (standalone, with default obstacle positions):
#   ros2 launch turtlebot3_sles_perception turtlebot3_simulate_lidar_random.launch.py
#
# Usage (override positions):
#   ros2 launch turtlebot3_sles_perception turtlebot3_simulate_lidar_random.launch.py \
#       random_x_1:=0.5 random_y_1:=0.5 ...

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    # Resolve all 16 positions from launch arguments
    params = {}
    for i in range(1, 17):
        params[f'random_x_{i}'] = float(LaunchConfiguration(f'random_x_{i}').perform(context))
        params[f'random_y_{i}'] = float(LaunchConfiguration(f'random_y_{i}').perform(context))

    params['lidar_publish_frequency'] = 20.0
    params['num_lidar_rays'] = 360
    params['lidar_max_distance'] = 1.0
    params['gaussian_probability'] = 0.91
    params['exponential_probability'] = 0.03
    params['uniform_probability'] = 0.03
    params['max_probability'] = 0.03

    simulate_lidar_new_node = Node(
        package='turtlebot3_sles_perception',
        executable='simulate_lidar_publisher_new.py',
        name='simulate_lidar_publisher',
        parameters=[params],
        output='screen',
    )
    return [simulate_lidar_new_node]


# Default 16 obstacle positions (rectangle layout)
_DEFAULT_POSITIONS = [
    (0.7, 0.3), (-0.8, 0.7), (-1.0, 1.0), (0.5, 0.5),
    (-0.5, -0.5), (0.5, -1.0), (-1.5, -1.5), (1.5, -1.5),
    (1.2, 1.2), (-1.2, 1.2), (0.0, 1.5), (0.0, -1.5),
    (1.5, 0.0), (-1.5, 0.0), (1.0, 0.0), (-1.0, 0.0),
]


def generate_launch_description():
    ld = LaunchDescription()

    for i, (dx, dy) in enumerate(_DEFAULT_POSITIONS, start=1):
        ld.add_action(DeclareLaunchArgument(f'random_x_{i}', default_value=str(dx),
                                             description=f'X position of obstacle {i}'))
        ld.add_action(DeclareLaunchArgument(f'random_y_{i}', default_value=str(dy),
                                             description=f'Y position of obstacle {i}'))

    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
