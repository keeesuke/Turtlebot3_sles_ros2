#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_custom_world_random
# Migrated from turtlebot3_custom_world_random.launch (ROS1)
# Redesign: random positions/sizes/shapes are generated inline in Python,
# then passed as Node parameters to spawn_random_world.py and simulate_lidar_publisher_new.py
#
# Usage:
#   ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py
#   ros2 launch turtlebot3_sles_worlds turtlebot3_custom_world_random.launch.py seed:=42

import os
import random
import math
import ast

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


OBSTACLE_SIZES        = [0.15, 0.2, 0.25]
OBSTACLE_SHAPES_LIST  = ['rectangle', 'hexagon', 'triangle']
GOAL_DEFAULT_STATE    = '[-1.5, -1.5, 0, 0, 0]'

def _distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _is_valid_robot_position(x, y, target_x, target_y, min_distance_target=3.5):
    return _distance(x, y, target_x, target_y) >= min_distance_target


def _is_far_from_obstacles(x, y, existing_obstacles, min_distance=0.78):
    return all(_distance(x, y, obs_x, obs_y) >= min_distance for obs_x, obs_y in existing_obstacles)


def _is_between_robot_and_target(x, y, robot_x, robot_y, target_x, target_y, tolerance=0.5):
    path_length = _distance(robot_x, robot_y, target_x, target_y)
    if path_length < 0.01:
        return True

    dx_path = target_x - robot_x
    dy_path = target_y - robot_y
    dx_point = x - robot_x
    dy_point = y - robot_y

    projection = (dx_point * dx_path + dy_point * dy_path) / path_length
    if projection < -0.2 or projection > path_length + 0.2:
        return False

    perp_x = dx_point - projection * (dx_path / path_length)
    perp_y = dy_point - projection * (dy_path / path_length)
    perp_dist = math.hypot(perp_x, perp_y)
    return perp_dist <= tolerance


def _generate_obstacle_on_path(rng, robot_x, robot_y, target_x, target_y,
                               min_distance_robot=0.5, min_distance_target=0.5,
                               max_perpendicular_offset=0.3):
    max_attempts = 1000
    path_length = _distance(robot_x, robot_y, target_x, target_y)
    if path_length < 0.01:
        return None

    dx = (target_x - robot_x) / path_length
    dy = (target_y - robot_y) / path_length
    perp_dx = -dy
    perp_dy = dx

    for _ in range(max_attempts):
        t = rng.uniform(0.2, 0.8)
        base_x = robot_x + t * (target_x - robot_x)
        base_y = robot_y + t * (target_y - robot_y)
        offset = rng.uniform(-max_perpendicular_offset, max_perpendicular_offset)
        x = base_x + offset * perp_dx
        y = base_y + offset * perp_dy

        dist_to_robot = _distance(x, y, robot_x, robot_y)
        dist_to_target = _distance(x, y, target_x, target_y)
        if (
            dist_to_robot >= min_distance_robot
            and dist_to_target >= min_distance_target
            and -2.0 <= x <= 2.0
            and -2.0 <= y <= 2.0
            and _is_between_robot_and_target(x, y, robot_x, robot_y, target_x, target_y, tolerance=0.5)
        ):
            return (x, y)
    return None


def _generate_normal_size(rng, mean=0.2, sigma=0.02, min_size=0.1, max_size=0.3):
    size = rng.gauss(mean, sigma)
    return max(min_size, min(max_size, size))


def _parse_goal_xy(goal_str, fallback_goal_str=GOAL_DEFAULT_STATE):
    fallback = ast.literal_eval(fallback_goal_str)
    try:
        parsed = ast.literal_eval(goal_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
            return float(parsed[0]), float(parsed[1])
    except Exception:
        pass
    return float(fallback[0]), float(fallback[1])


def _generate_ros1_style_config(seed=None, target_x=None, target_y=None):
    """
    ROS1-like generation:
      - random robot start (far from goal)
      - 4 obstacles around robot->goal path
      - 3 obstacles per quadrant (12 total)
      - distance constraints among all obstacles and from robot/goal
    Returns:
      robot_pose: (x, y, z, yaw)
      obstacle_configs: list[(x, y, size, shape)] length=16
    """
    rng = random.Random(seed)

    # Robot pose generation (ROS1-compatible range/constraint)
    robot_x, robot_y = None, None
    for _ in range(1000):
        cand_x = rng.uniform(0.0, 1.7)
        cand_y = rng.uniform(0.0, 1.7)
        if _is_valid_robot_position(cand_x, cand_y, target_x, target_y, min_distance_target=3.5):
            robot_x, robot_y = cand_x, cand_y
            break

    if robot_x is None or robot_y is None:
        robot_x = rng.uniform(0.0, 1.7)
        robot_y = rng.uniform(0.0, 1.7)

    robot_z = 0.01
    angle_to_goal = math.atan2(target_y - robot_y, target_x - robot_x)
    robot_yaw = (angle_to_goal + rng.gauss(0.785, 0.1)) % (2.0 * math.pi)

    obstacle_configs = []
    all_obstacles = []
    min_inter_obstacle = 0.78

    # 4 path obstacles
    for _ in range(4):
        obstacle_pos = None
        for _ in range(1000):
            candidate = _generate_obstacle_on_path(
                rng,
                robot_x,
                robot_y,
                target_x=target_x,
                target_y=target_y,
                min_distance_robot=0.5,
                min_distance_target=0.5,
                max_perpendicular_offset=0.3,
            )
            if candidate is None:
                continue
            x, y = candidate
            if _is_far_from_obstacles(x, y, all_obstacles, min_distance=min_inter_obstacle):
                obstacle_pos = (x, y)
                break

        if obstacle_pos is None:
            for _ in range(1000):
                x = rng.uniform(-1.8, 1.8)
                y = rng.uniform(-1.8, 1.8)
                if (
                    _distance(x, y, robot_x, robot_y) >= 0.6
                    and _distance(x, y, target_x, target_y) >= 0.6
                    and _is_far_from_obstacles(x, y, all_obstacles, min_distance=min_inter_obstacle)
                ):
                    obstacle_pos = (x, y)
                    break

        if obstacle_pos is None:
            obstacle_pos = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))

        x, y = obstacle_pos
        all_obstacles.append((x, y))
        size = _generate_normal_size(rng, mean=0.25, sigma=0.02, min_size=0.2, max_size=0.3)
        shape = rng.choice(['rectangle', 'hexagon'])
        obstacle_configs.append((x, y, size, shape))

    quadrants = [
        {'x_range': (0.0, 2.0), 'y_range': (0.0, 2.0)},
        {'x_range': (-2.0, 0.0), 'y_range': (0.0, 2.0)},
        {'x_range': (-2.0, 0.0), 'y_range': (-2.0, 0.0)},
        {'x_range': (0.0, 2.0), 'y_range': (-2.0, 0.0)},
    ]

    for quadrant in quadrants:
        quadrant_obstacles = []
        for _ in range(3):
            placed = None
            for _ in range(1000):
                x = rng.uniform(*quadrant['x_range'])
                y = rng.uniform(*quadrant['y_range'])
                if (
                    _distance(x, y, robot_x, robot_y) >= 0.6
                    and _distance(x, y, target_x, target_y) >= 0.6
                    and _is_far_from_obstacles(x, y, all_obstacles, min_distance=min_inter_obstacle)
                    and _is_far_from_obstacles(x, y, quadrant_obstacles, min_distance=min_inter_obstacle)
                ):
                    placed = (x, y)
                    break

            if placed is None:
                placed = (
                    rng.uniform(*quadrant['x_range']),
                    rng.uniform(*quadrant['y_range']),
                )

            x, y = placed
            all_obstacles.append((x, y))
            quadrant_obstacles.append((x, y))
            size = _generate_normal_size(rng, mean=0.2, sigma=0.02, min_size=0.15, max_size=0.3)
            shape = rng.choice(['rectangle', 'hexagon'])
            obstacle_configs.append((x, y, size, shape))

    if len(obstacle_configs) > 16:
        obstacle_configs = obstacle_configs[:16]
    while len(obstacle_configs) < 16:
        obstacle_configs.append((0.0, 0.0, 0.2, 'rectangle'))

    return (robot_x, robot_y, robot_z, robot_yaw), obstacle_configs


def _launch_setup(context, *args, **kwargs):
    pkg_sles_worlds = get_package_share_directory('turtlebot3_sles_worlds')
    pkg_gazebo_ros  = get_package_share_directory('gazebo_ros')
    pkg_tb3_gazebo  = get_package_share_directory('turtlebot3_gazebo')

    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    # Resolve seed argument
    seed_str = LaunchConfiguration('seed').perform(context)
    seed = int(seed_str) if seed_str.isdigit() else None
    goal_str = LaunchConfiguration('goal').perform(context)
    goal_x, goal_y = _parse_goal_xy(goal_str, fallback_goal_str=GOAL_DEFAULT_STATE)

    world_file = os.path.join(
        pkg_sles_worlds, 'worlds', 'turtlebot3_custom_world_rectangle.world'
    )

    # Generate ROS1-style random robot pose + obstacle configs
    robot_pose, configs = _generate_ros1_style_config(seed=seed, target_x=goal_x, target_y=goal_y)
    robot_x, robot_y, robot_z, robot_yaw = robot_pose

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

    spawn_params = {'model': TURTLEBOT3_MODEL}
    for i, (x, y, size, shape) in enumerate(configs, start=1):
        spawn_params[f'random_x_{i}']     = float(x)
        spawn_params[f'random_y_{i}']     = float(y)
        spawn_params[f'random_size_{i}']  = float(size)
        spawn_params[f'random_shape_{i}'] = shape
    spawn_params['random_robot_x']   = float(robot_x)
    spawn_params['random_robot_y']   = float(robot_y)
    spawn_params['random_robot_z']   = float(robot_z)
    spawn_params['random_robot_yaw'] = float(robot_yaw)

    spawn_random_world = Node(
        package='turtlebot3_sles_worlds',
        executable='spawn_random_world.py',
        name='spawn_random_world',
        parameters=[spawn_params],
        output='screen',
    )

    lidar_params = {}
    for i, (x, y, size, shape) in enumerate(configs, start=1):
        lidar_params[f'random_x_{i}']     = float(x)
        lidar_params[f'random_y_{i}']     = float(y)
        lidar_params[f'random_size_{i}']  = float(size)
        lidar_params[f'random_shape_{i}'] = shape
    lidar_params['lidar_publish_frequency'] = 20.0
    lidar_params['num_lidar_rays']          = 360
    lidar_params['lidar_max_distance']      = 1.0
    lidar_params['gaussian_probability']    = 0.91
    lidar_params['exponential_probability'] = 0.03
    lidar_params['uniform_probability']     = 0.03
    lidar_params['max_probability']         = 0.03

    simulate_lidar_new = Node(
        package='turtlebot3_sles_perception',
        executable='simulate_lidar_publisher_new.py',
        name='simulate_lidar_publisher',
        parameters=[lidar_params],
        output='screen',
    )

    return [
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_random_world,
        simulate_lidar_new,
    ]

def generate_launch_description():
    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='',
        description='Random seed for obstacle placement (empty = truly random)'
    )
    goal_arg = DeclareLaunchArgument(
        'goal',
        default_value=GOAL_DEFAULT_STATE,
        description='Goal state [x, y, theta, v, w] used by world randomizer'
    )

    ld = LaunchDescription()
    ld.add_action(seed_arg)
    ld.add_action(goal_arg)
    ld.add_action(OpaqueFunction(function=_launch_setup))
    return ld
