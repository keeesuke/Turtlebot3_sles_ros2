#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_custom_world_test
# Migrated from turtlebot3_custom_world_test.launch (ROS1)
# Launches Gazebo with custom rectangle world and spawns 8 obstacles + robot

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sles_worlds = get_package_share_directory('turtlebot3_sles_worlds')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    world_file = os.path.join(pkg_sles_worlds, 'worlds', 'turtlebot3_custom_world_rectangle.world')
    obstacle_sdf = os.path.join(pkg_sles_worlds, 'models', 'obstacle_rec.sdf')

    # Robot model SDF (from official turtlebot3_gazebo package)
    robot_model_folder = 'turtlebot3_' + TURTLEBOT3_MODEL
    robot_urdf = os.path.join(
        pkg_tb3_gazebo,
        'models',
        robot_model_folder,
        'model.sdf'
    )

    # Launch arguments
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    x_pos = DeclareLaunchArgument('x_pos', default_value='0.0')
    y_pos = DeclareLaunchArgument('y_pos', default_value='0.0')
    z_pos = DeclareLaunchArgument('z_pos', default_value='0.0')
    yaw   = DeclareLaunchArgument('yaw',   default_value='3.14')

    # Fixed obstacle positions (defaults matching original ROS1 launch)
    obstacle_positions = [
        ( '0.7',  '0.3'),
        ('-0.8',  '0.7'),
        ('-1.0',  '1.0'),
        ( '0.5',  '0.5'),
        ('-0.5', '-0.5'),
        ( '0.5', '-1.0'),
        ('-1.5', '-1.5'),
        ( '1.5', '-1.5'),
    ]

    # Gazebo server + client
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

    # Robot state publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tb3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    )

    # Spawn 8 obstacles
    spawn_obstacles = []
    for i, (ox, oy) in enumerate(obstacle_positions, start=1):
        spawn_obstacles.append(Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=f'spawn_obstacle_{i}',
            arguments=[
                '-file', obstacle_sdf,
                '-entity', f'obstacle_{i}',
                '-x', ox,
                '-y', oy,
                '-z', '0.1',
            ],
            output='screen',
        ))

    # Spawn robot
    # Entity name must be 'turtlebot3_<model>' to match planner state_cb lookup:
    #   idx = msg.name.index('turtlebot3_waffle_pi')
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_turtlebot3',
        arguments=[
            '-file', robot_urdf,
            '-entity', f'turtlebot3_{TURTLEBOT3_MODEL}',
            '-x', LaunchConfiguration('x_pos'),
            '-y', LaunchConfiguration('y_pos'),
            '-z', LaunchConfiguration('z_pos'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(use_sim_time)
    ld.add_action(x_pos)
    ld.add_action(y_pos)
    ld.add_action(z_pos)
    ld.add_action(yaw)
    ld.add_action(gzserver)
    ld.add_action(gzclient)
    ld.add_action(robot_state_publisher)
    for node in spawn_obstacles:
        ld.add_action(node)
    ld.add_action(spawn_robot)

    return ld
