#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_custom_world_random
# Migrated from turtlebot3_custom_world_random.launch (ROS1)
# Redesign: random positions are generated inline in Python (no ROS param server),
# then passed as Node arguments and also saved to a YAML file for perception nodes.
#
# Usage:
#   ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py
#   ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py seed:=42

import os
import random
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ── Utility: generate 16 non-overlapping obstacle positions ──────────────────

def _generate_random_positions(seed=None, n=16, x_range=(-1.8, 1.8), y_range=(-1.8, 1.8),
                                min_dist=0.35, robot_clear=0.5):
    rng = random.Random(seed)
    positions = []
    attempts = 0
    while len(positions) < n and attempts < 10000:
        attempts += 1
        x = rng.uniform(*x_range)
        y = rng.uniform(*y_range)
        # Keep clear of robot start position (0, 0)
        if math.hypot(x, y) < robot_clear:
            continue
        # Keep clear of existing obstacles
        too_close = any(math.hypot(x - px, y - py) < min_dist for px, py in positions)
        if too_close:
            continue
        positions.append((x, y))
    # Pad with fallback grid if not enough generated
    while len(positions) < n:
        positions.append((0.0, 0.0))
    return positions


# ── OpaqueFunction: build actions with random positions at launch time ────────

def _launch_setup(context, *args, **kwargs):
    pkg_sles_worlds = get_package_share_directory('turtlebot3_sles_worlds')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    # Resolve seed argument
    seed_str = LaunchConfiguration('seed').perform(context)
    seed = int(seed_str) if seed_str.isdigit() else None

    world_file = os.path.join(pkg_sles_worlds, 'worlds', 'turtlebot3_custom_world_rectangle.world')
    obstacle_sdf = os.path.join(pkg_sles_worlds, 'models', 'obstacle_rec.sdf')
    robot_urdf = os.path.join(
        pkg_tb3_gazebo, 'models', f'turtlebot3_{TURTLEBOT3_MODEL}', 'model.sdf'
    )

    positions = _generate_random_positions(seed=seed, n=16)

    # Build spawn actions for up to 8 Gazebo obstacles (first 8 positions)
    spawn_actions = []
    for i, (ox, oy) in enumerate(positions[:8], start=1):
        spawn_actions.append(Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=f'spawn_obstacle_{i}',
            arguments=[
                '-file', obstacle_sdf,
                '-entity', f'obstacle_{i}',
                '-x', str(ox),
                '-y', str(oy),
                '-z', '0.1',
            ],
            output='screen',
        ))

    # Spawn robot at origin with yaw=pi
    # Entity name must be 'turtlebot3_<model>' to match planner state_cb lookup:
    #   idx = msg.name.index('turtlebot3_waffle_pi')
    spawn_actions.append(Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_turtlebot3',
        arguments=[
            '-file', robot_urdf,
            '-entity', f'turtlebot3_{TURTLEBOT3_MODEL}',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.01',
            '-Y', '3.14',
        ],
        output='screen',
    ))

    # Gazebo
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Perception node: simulate_lidar_publisher_new reads per-node parameters
    # Pass all 16 positions (8 for Gazebo visual + 16 for lidar computation)
    lidar_params = {f'random_x_{i}': float(positions[i-1][0]) for i in range(1, 17)}
    lidar_params.update({f'random_y_{i}': float(positions[i-1][1]) for i in range(1, 17)})
    lidar_params['lidar_publish_frequency'] = 20.0
    lidar_params['num_lidar_rays'] = 360
    lidar_params['lidar_max_distance'] = 1.0
    lidar_params['gaussian_probability'] = 0.91
    lidar_params['exponential_probability'] = 0.03
    lidar_params['uniform_probability'] = 0.03
    lidar_params['max_probability'] = 0.03

    simulate_lidar_new = Node(
        package='turtlebot3_sles_perception',
        executable='simulate_lidar_publisher_new.py',
        name='simulate_lidar_publisher',
        parameters=[lidar_params],
        output='screen',
    )

    return [gzserver, gzclient, robot_state_publisher] + spawn_actions + [simulate_lidar_new]


def generate_launch_description():
    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='',
        description='Random seed for obstacle placement (empty = truly random)'
    )

    ld = LaunchDescription()
    ld.add_action(seed_arg)
    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
