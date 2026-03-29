#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    map_topic_arg = DeclareLaunchArgument(
        "map_topic",
        default_value="/map",
        description="OccupancyGrid topic to visualize.",
    )
    update_rate_arg = DeclareLaunchArgument(
        "update_rate_hz",
        default_value="5.0",
        description="Plot refresh rate in Hz.",
    )
    map_frame_arg = DeclareLaunchArgument(
        "map_frame",
        default_value="map",
        description="TF target frame used as the map frame.",
    )
    base_frame_arg = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="TF source frame used for the robot pose overlay.",
    )
    robot_arrow_length_arg = DeclareLaunchArgument(
        "robot_arrow_length_m",
        default_value="0.25",
        description="Robot heading arrow length in meters.",
    )
    save_maps_arg = DeclareLaunchArgument(
        "save_maps",
        default_value="false",
        description="If true, save raw OccupancyGrid data to disk.",
    )
    save_dir_arg = DeclareLaunchArgument(
        "save_dir",
        default_value="/tmp/map_occupancy_logs",
        description="Directory to save OccupancyGrid logs.",
    )
    save_rate_arg = DeclareLaunchArgument(
        "save_rate_hz",
        default_value="1.0",
        description="Maximum save frequency in Hz.",
    )

    plotter_node = Node(
        package="turtlebot3_sles_control",
        executable="map_occupancy_plotter.py",
        name="map_occupancy_plotter",
        output="screen",
        parameters=[
            {
                "map_topic": LaunchConfiguration("map_topic"),
                "update_rate_hz": ParameterValue(
                    LaunchConfiguration("update_rate_hz"), value_type=float
                ),
                "map_frame": LaunchConfiguration("map_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
                "robot_arrow_length_m": ParameterValue(
                    LaunchConfiguration("robot_arrow_length_m"), value_type=float
                ),
                "save_maps": ParameterValue(
                    LaunchConfiguration("save_maps"), value_type=bool
                ),
                "save_dir": LaunchConfiguration("save_dir"),
                "save_rate_hz": ParameterValue(
                    LaunchConfiguration("save_rate_hz"), value_type=float
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            map_topic_arg,
            update_rate_arg,
            map_frame_arg,
            base_frame_arg,
            robot_arrow_length_arg,
            save_maps_arg,
            save_dir_arg,
            save_rate_arg,
            plotter_node,
        ]
    )
