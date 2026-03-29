#!/usr/bin/env python3
# ROS2 Humble launch file for turtlebot3_planner (HAA/MPPI planner)
# Migrated from turtlebot3_planner.launch (ROS1)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from pathlib import Path


def _find_src_config_near_launch_file():
    """Find workspace .../src/Config by walking up from this launch file (works from any cwd)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        cfg = os.path.join(d, 'Config')
        if os.path.isdir(cfg):
            return cfg
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _src_config_from_colcon_install_launch():
    """If launch file lives under workspace/install/..., use <ws>/src/Config."""
    parts = Path(__file__).resolve().parts
    if 'install' not in parts:
        return None
    i = parts.index('install')
    ws_root = Path(*parts[:i])
    cfg = ws_root / 'src' / 'Config'
    if cfg.is_dir():
        return str(cfg)
    return None


def _default_shared_config_file():
    sles_ws = os.environ.get('SLES_WS', '').strip()
    if sles_ws:
        return os.path.join(sles_ws, 'src', 'Config')
    found = _find_src_config_near_launch_file()
    if found:
        return found
    found = _src_config_from_colcon_install_launch()
    if found:
        return found
    return os.path.join(os.getcwd(), 'src', 'Config')


def _default_goal_file():
    return os.path.join(_default_shared_config_file(), 'goal.yaml')


def _default_scenario_file():
    return os.path.join(_default_shared_config_file(), 'scenario.yaml')


def _launch_setup(context, *args, **kwargs):
    goal = LaunchConfiguration('goal').perform(context)
    goal_file = LaunchConfiguration('goal_file').perform(context)
    scenario_file = LaunchConfiguration('scenario_file').perform(context)

    planner_params = {
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
    }
    if goal:
        planner_params['goal'] = goal

    # Only pass YAML paths that exist — missing files break ros2 param loading when cwd is wrong.
    yaml_params = [p for p in (scenario_file, goal_file) if os.path.isfile(p)]

    planner_node = Node(
        package='turtlebot3_sles_control',
        #executable='planner_haa_only.py',
        executable='planner_haa_real_world.py',
        name='planner_node',
        parameters=yaml_params + [planner_params],
        output='screen',
    )
    return [planner_node]


def generate_launch_description():
    default_goal_file = _default_goal_file()
    default_scenario_file = _default_scenario_file()

    return LaunchDescription([
        DeclareLaunchArgument(
            'goal_file',
            default_value=default_goal_file,
            description='Shared goal YAML path under workspace src/Config'
        ),
        DeclareLaunchArgument(
            'scenario_file',
            default_value=default_scenario_file,
            description='Shared runtime scenario YAML path under workspace src/Config'
        ),
        DeclareLaunchArgument(
            'goal',
            default_value='',
            description='Optional goal override [x, y, theta, v, w] (empty = read from goal_file)'
        ),
        OpaqueFunction(function=_launch_setup),
    ])
