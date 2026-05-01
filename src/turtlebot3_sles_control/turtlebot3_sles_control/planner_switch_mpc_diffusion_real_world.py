#!/usr/bin/env python3
"""
ROS2 MPC+Diffusion Policy Switching Planner for TurtleBot3 — Real World

Dual-planner simplex architecture:
  HPA (High-Performance Autonomous) = Diffusion Policy (fast, reactive)
  HAA (High-Assurance Autonomous)   = MPPI MPC (conservative, safe)

While in HPA: Diffusion policy predicts a horizon-length trajectory.
              Each cycle checks HAA feasibility from the predicted next state.
              If infeasible → 2-loop braking → switch to HAA.
While in HAA: MPPI plans; checks safety margin → switch back to HPA when safe.

Identical to planner_switch_mpc_nn_real_world.py except:
  - MLP replaced with PolicyRunner (diffusion / flow / deterministic checkpoint)
  - HPA mode tracks the predicted trajectory via Kanayama (same as HAA)
  - HAA feasibility check uses the first predicted waypoint as the one-step-ahead state

Prerequisites (must already be running):
  1. turtlebot3_node       (hw driver — /scan, /tf odom→base_footprint)
  2. turtlebot3_cartographer  (SLAM — /tf map→odom, /map)
  3. rviz2                 (optional — for interactive goal input)
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import os
import sys
import atexit
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import time
from datetime import datetime
from scipy import ndimage

import torch
import torch.nn as nn

# PolicyRunner lives alongside this file in the diffusion_policy package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from policy_runner import PolicyRunner, Observation

try:
    from shapely.geometry import LineString
    _HAS_SHAPELY = True
except Exception:
    LineString = None
    _HAS_SHAPELY = False


# ---------------------------------------------------------------------------
# MPPI Implementation (unchanged from planner_switch_mpc_nn_real_world.py)
# ---------------------------------------------------------------------------
class MPPI:
    def __init__(self, sigma: float = 0.1, temperature: float = 0.05,
                 num_nodes: int = 20, num_rollouts: int = 100,
                 use_noise_ramp: bool = False, noise_ramp: float = 1.0, nu: int = 2,
                 a_limit: float = 0.25, alpha_limit: float = 0.5,
                 v_limit_haa: float = 0.14, v_limit_hpa: float = 0.2,
                 omega_limit_haa: float = 0.9, omega_limit_hpa: float = 1.0,
                 dt: float = 0.1):
        self.sigma = sigma
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.num_rollouts = num_rollouts
        self.use_noise_ramp = use_noise_ramp
        self.noise_ramp = noise_ramp
        self.nu = nu
        self.a_limit = a_limit
        self.alpha_limit = alpha_limit
        self.v_limit_haa = v_limit_haa
        self.v_limit_hpa = v_limit_hpa
        self.omega_limit_haa = omega_limit_haa
        self.omega_limit_hpa = omega_limit_hpa
        self.dt = dt

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        num_nodes = self.num_nodes
        num_rollouts = self.num_rollouts
        _sigma = self.sigma
        control_magnitude = np.abs(nominal_knots)
        control_magnitude_2 = np.ones_like(control_magnitude)
        control_magnitude_2[:, 0] = 0.2
        control_magnitude_2[:, 1] = 0.4
        min_noise_scale = 0.1 * _sigma
        adaptive_sigma = np.maximum(control_magnitude_2 * _sigma, min_noise_scale)
        if self.use_noise_ramp:
            ramp = self.noise_ramp * np.linspace(1 / num_nodes, 1, num_nodes, endpoint=True)[:, None]
            sigma = (ramp * adaptive_sigma)[None, :, :]
        else:
            sigma = adaptive_sigma[None, :, :]
        sigma = np.broadcast_to(sigma, (num_rollouts - 1, num_nodes, self.nu))
        noise = np.random.randn(num_rollouts - 1, num_nodes, self.nu)
        noised_knots = nominal_knots[None, :, :] + sigma * noise
        return np.concatenate([nominal_knots[None], noised_knots])

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        costs = -rewards
        beta = np.min(costs)
        _weights = np.exp(-(costs - beta) / self.temperature)
        weights = _weights / np.sum(_weights)
        return np.sum(weights[:, None, None] * sampled_knots, axis=0)


@dataclass
class RobotState:
    x: float
    y: float
    theta: float
    v: float
    w: float


class TurtlebotDynamics:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.nu = 2
        self.nx = 5

    def step(self, state: RobotState, control: np.ndarray) -> RobotState:
        a, alpha = control
        new_v = state.v + a * self.dt
        new_w = state.w + alpha * self.dt
        new_x = state.x + state.v * np.cos(state.theta) * self.dt
        new_y = state.y + state.v * np.sin(state.theta) * self.dt
        new_theta = state.theta + state.w * self.dt
        return RobotState(new_x, new_y, new_theta, new_v, new_w)


class OccupancyGridMap:
    def __init__(self, width: float = 4.0, height: float = 4.0, resolution: float = 0.02):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        self.x_min = -width / 2
        self.y_min = -height / 2
        self.x_max = width / 2
        self.y_max = height / 2
        self.goal = None
        self.start = None

    def world_to_grid(self, x, y):
        return int((x - self.x_min) / self.resolution), int((y - self.y_min) / self.resolution)

    def grid_to_world(self, gx, gy):
        return self.x_min + (gx + 0.5) * self.resolution, self.y_min + (gy + 0.5) * self.resolution

    def dilate_grid_new(self, robot_radius_cells):
        obstacle_mask = self.occupancy_grid > 50
        if not np.any(obstacle_mask):
            return np.zeros_like(obstacle_mask, dtype=bool)
        distance_map = ndimage.distance_transform_edt(~obstacle_mask)
        return distance_map <= robot_radius_cells

    def check_collision_new(self, x: np.ndarray, y: np.ndarray, robot_radius: float = 0.22) -> np.ndarray:
        grid_x = ((x - self.x_min) / self.resolution).astype(int)
        grid_y = ((y - self.y_min) / self.resolution).astype(int)
        out_of_bounds = ((grid_x < 0) | (grid_x >= self.grid_width) |
                         (grid_y < 0) | (grid_y >= self.grid_height))
        collisions = np.zeros(len(x), dtype=bool)
        collisions[out_of_bounds] = True
        valid_mask = ~out_of_bounds
        if not np.any(valid_mask):
            return collisions
        robot_radius_cells = int(robot_radius / self.resolution)
        dilate_grid = self.dilate_grid_new(robot_radius_cells)
        collisions[valid_mask] = dilate_grid[grid_y[valid_mask], grid_x[valid_mask]]
        return collisions

    def get_distance_to_goal(self, x, y):
        if self.goal is None:
            return 0.0
        return np.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)

    def set_goal(self, x, y):
        self.goal = np.array([x, y])

    def set_start(self, x, y):
        self.start = np.array([x, y])


class MPPIPlanner:
    def __init__(self, mppi: MPPI, dynamics: TurtlebotDynamics, occupancy_map: OccupancyGridMap,
                 v_min=-0.26, v_max=0.26, w_min=-1.82, w_max=1.82,
                 a_min=-1.0, a_max=1.0, alpha_min=-1.0, alpha_max=1.0,
                 smoothing_weight=0.1, logger=None):
        self.mppi = mppi
        self.dynamics = dynamics
        self.occupancy_map = occupancy_map
        self.v_min = v_min
        self.v_max = v_max
        self.w_min = w_min
        self.w_max = w_max
        self.a_min = a_min
        self.a_max = a_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.smoothing_weight = smoothing_weight
        self._logger = logger
        self.nominal_controls = np.zeros((mppi.num_nodes, mppi.nu))
        self.previous_controls = None

    def _log_info(self, msg):
        if self._logger: self._logger.info(msg)
        else: print(f'[MPPIPlanner/INFO] {msg}')

    def _log_warn(self, msg):
        if self._logger: self._logger.warn(msg)
        else: print(f'[MPPIPlanner/WARN] {msg}')

    def _log_error(self, msg):
        if self._logger: self._logger.error(msg)
        else: print(f'[MPPIPlanner/ERROR] {msg}')

    def update_occupancy_map(self, new_occupancy_map):
        self.occupancy_map = new_occupancy_map

    def simulate_trajectories_vectorized(self, start_state, control_sequences):
        num_rollouts, horizon, nu = control_sequences.shape
        trajectories = np.zeros((num_rollouts, horizon + 1, 5))
        trajectories[:, 0, 0] = start_state.x
        trajectories[:, 0, 1] = start_state.y
        trajectories[:, 0, 2] = start_state.theta
        trajectories[:, 0, 3] = start_state.v
        trajectories[:, 0, 4] = start_state.w
        for i in range(horizon):
            x = trajectories[:, i, 0]
            y = trajectories[:, i, 1]
            theta = trajectories[:, i, 2]
            v = trajectories[:, i, 3]
            w = trajectories[:, i, 4]
            a = control_sequences[:, i, 0]
            alpha = control_sequences[:, i, 1]
            new_v = np.clip(v + a * self.dynamics.dt, self.v_min, self.v_max)
            new_w = np.clip(w + alpha * self.dynamics.dt, self.w_min, self.w_max)
            trajectories[:, i+1, 0] = x + v * np.cos(theta) * self.dynamics.dt
            trajectories[:, i+1, 1] = y + v * np.sin(theta) * self.dynamics.dt
            trajectories[:, i+1, 2] = theta + w * self.dynamics.dt
            trajectories[:, i+1, 3] = new_v
            trajectories[:, i+1, 4] = new_w
        return trajectories

    def validate_trajectories_vectorized(self, trajectories, robot_radius=0.22):
        num_rollouts, num_steps, _ = trajectories.shape
        valid_mask = np.ones(num_rollouts, dtype=bool)
        if np.any(valid_mask):
            valid_trajectories = trajectories[valid_mask]
            all_x = valid_trajectories[:, :, 0].flatten()
            all_y = valid_trajectories[:, :, 1].flatten()
            collision_results = self.occupancy_map.check_collision_new(all_x, all_y, robot_radius)
            collision_matrix = collision_results.reshape(valid_trajectories.shape[0], num_steps)
            trajectory_collisions = np.any(collision_matrix, axis=1)
            valid_indices = np.where(valid_mask)[0]
            valid_mask[valid_indices[trajectory_collisions]] = False
        return valid_mask

    def compute_rewards_vectorized(self, trajectories, goal, safety_dilated_grid=None, control_sequences=None):
        num_rollouts, num_steps, _ = trajectories.shape
        x = trajectories[:, :, 0]
        y = trajectories[:, :, 1]
        distances = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        rewards = -np.sum(distances * 10.0, axis=1)
        final_x = trajectories[:, -1, 0]
        final_y = trajectories[:, -1, 1]
        terminal_distances = np.sqrt((final_x - goal[0])**2 + (final_y - goal[1])**2)
        rewards -= terminal_distances * 100.0
        if safety_dilated_grid is not None:
            all_x = x.flatten()
            all_y = y.flatten()
            grid_x = ((all_x - self.occupancy_map.x_min) / self.occupancy_map.resolution).astype(int)
            grid_y = ((all_y - self.occupancy_map.y_min) / self.occupancy_map.resolution).astype(int)
            out_of_bounds = ((grid_x < 0) | (grid_x >= self.occupancy_map.grid_width) |
                             (grid_y < 0) | (grid_y >= self.occupancy_map.grid_height))
            near_obstacle_violations = np.zeros(len(all_x), dtype=bool)
            valid_mask = ~out_of_bounds
            if np.any(valid_mask):
                near_obstacle_violations[valid_mask] = safety_dilated_grid[grid_y[valid_mask], grid_x[valid_mask]]
            violation_matrix = near_obstacle_violations.reshape(num_rollouts, num_steps)
            rewards -= np.sum(violation_matrix, axis=1) * 5
        if control_sequences is not None:
            control_diff = np.diff(control_sequences, axis=1)
            rewards -= self.smoothing_weight * np.sum(control_diff**2, axis=(1, 2))
        return rewards

    def plan(self, start_state, goal, debug_plot=False, robot_radius=0.22):
        if self.previous_controls is not None and len(self.previous_controls) >= self.mppi.num_nodes:
            nominal_controls = np.vstack([self.previous_controls[1:], np.zeros((1, 2))])
        else:
            dx_world = goal[0] - start_state.x
            dy_world = goal[1] - start_state.y
            cos_theta = np.cos(start_state.theta)
            sin_theta = np.sin(start_state.theta)
            dx_robot = dx_world * cos_theta + dy_world * sin_theta
            dy_robot = -dx_world * sin_theta + dy_world * cos_theta
            angle_to_goal = np.arctan2(dy_robot, dx_robot)
            nominal_controls = np.zeros((self.mppi.num_nodes, 2))
            nominal_controls[:, 0] = 0.1 * self.a_max
            nominal_controls[:, 1] = (0.3 if angle_to_goal >= 0 else -0.3) * self.alpha_max

        for _ in range(1):
            sampled_controls = self.mppi.sample_control_knots(nominal_controls)
            sampled_controls[:, :, 0] = np.clip(sampled_controls[:, :, 0], self.a_min, self.a_max)
            sampled_controls[:, :, 1] = np.clip(sampled_controls[:, :, 1], self.alpha_min, self.alpha_max)
            all_trajectories = self.simulate_trajectories_vectorized(start_state, sampled_controls)
            valid_mask = self.validate_trajectories_vectorized(all_trajectories, robot_radius)
            if np.any(valid_mask):
                valid_controls = sampled_controls[valid_mask]
                valid_trajectories = all_trajectories[valid_mask]
                safety_radius_cells = int((robot_radius + 0.05) / self.occupancy_map.resolution)
                safety_dilated_grid = self.occupancy_map.dilate_grid_new(safety_radius_cells)
                valid_rewards = self.compute_rewards_vectorized(
                    valid_trajectories, goal, safety_dilated_grid, valid_controls)
            else:
                valid_controls = np.array([])

            if len(valid_controls) > 0:
                nominal_controls = self.mppi.update_nominal_knots(valid_controls, valid_rewards)
                nominal_controls[:, 0] = np.clip(nominal_controls[:, 0], self.a_min, self.a_max)
                nominal_controls[:, 1] = np.clip(nominal_controls[:, 1], self.alpha_min, self.alpha_max)
                self.previous_controls = nominal_controls.copy()
            else:
                self.previous_controls = None
                self._log_warn("No valid trajectories found!")
                return None, None

        control_sequence_batch = nominal_controls.reshape(1, -1, 2)
        trajectory_batch = self.simulate_trajectories_vectorized(start_state, control_sequence_batch)
        return trajectory_batch[0], nominal_controls

    def sample_debug_trajectories(self, start_state, goal, robot_radius):
        nominal = self.previous_controls.copy() if self.previous_controls is not None \
                  else np.zeros((self.mppi.num_nodes, 2))
        sampled = self.mppi.sample_control_knots(nominal)
        sampled[:, :, 0] = np.clip(sampled[:, :, 0], self.a_min, self.a_max)
        sampled[:, :, 1] = np.clip(sampled[:, :, 1], self.alpha_min, self.alpha_max)
        all_trajs = self.simulate_trajectories_vectorized(start_state, sampled)
        valid_mask = self.validate_trajectories_vectorized(all_trajs, robot_radius)
        return sampled, nominal, all_trajs, valid_mask


class KanayamaController:
    def __init__(self, kx=1.0, ky=1.0, kth=2.0, kv=1.0, kw=1.0,
                 kix=0.1, kiy=0.1, kith=0.1, max_integral=1.0,
                 v_limit_haa=0.2, omega_limit_haa=0.9,
                 v_limit_hpa=0.26, omega_limit_hpa=1.82):
        self.kx = kx
        self.ky = ky
        self.kth = kth
        self.kv = kv
        self.kw = kw
        self.kix = kix
        self.kiy = kiy
        self.kith = kith
        self.max_integral = max_integral
        self.integral_ex = 0.0
        self.integral_ey = 0.0
        self.integral_eth = 0.0
        self.prev_ex = 0.0
        self.prev_ey = 0.0
        self.prev_eth = 0.0
        self.dt = 0.02
        self.v_limit_haa = v_limit_haa
        self.omega_limit_haa = omega_limit_haa
        self.v_limit_hpa = v_limit_hpa
        self.omega_limit_hpa = omega_limit_hpa
        self.prev_v_ref = 0.0
        self.prev_w_ref = 0.0

    def compute_control(self, current_state, ref_state, current_planner="HAA"):
        x, y, theta, v, w = current_state
        xr, yr, thr, vr, wr = ref_state
        dx, dy = xr - x, yr - y
        ex = np.cos(theta) * dx + np.sin(theta) * dy
        ey = -np.sin(theta) * dx + np.cos(theta) * dy
        eth = np.arctan2(np.sin(thr - theta), np.cos(thr - theta))
        self.integral_ex = np.clip(self.integral_ex + (ex + self.prev_ex) * self.dt / 2, -self.max_integral, self.max_integral)
        self.integral_ey = np.clip(self.integral_ey + (ey + self.prev_ey) * self.dt / 2, -self.max_integral, self.max_integral)
        self.integral_eth = np.clip(self.integral_eth + (eth + self.prev_eth) * self.dt / 2, -self.max_integral, self.max_integral)
        self.prev_ex = ex
        self.prev_ey = ey
        self.prev_eth = eth
        v_cmd = vr * np.cos(eth) + self.kx * ex + self.kv * (vr - v) + self.kix * self.integral_ex
        w_cmd = wr + vr * self.ky * ey + vr * self.kth * np.sin(eth) + self.kw * (wr - w) \
                + self.kiy * self.integral_ey + self.kith * self.integral_eth
        if current_planner == "HPA":
            v_cmd = np.clip(v_cmd, 0, self.v_limit_hpa)
            w_cmd = np.clip(w_cmd, -self.omega_limit_hpa, self.omega_limit_hpa)
        else:
            v_cmd = np.clip(v_cmd, 0, self.v_limit_haa)
            w_cmd = np.clip(w_cmd, -self.omega_limit_haa, self.omega_limit_haa)
        return v_cmd, w_cmd

    def reset_integral_errors(self):
        self.integral_ex = self.integral_ey = self.integral_eth = 0.0
        self.prev_ex = self.prev_ey = self.prev_eth = 0.0


# ---------------------------------------------------------------------------
# ROS2 Node: MPC + Diffusion Policy Switching Controller
# ---------------------------------------------------------------------------
class SwitchMPCDiffusionRealWorldNode(Node):
    def __init__(self):
        super().__init__('switch_mpc_diffusion_real_world')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('horizon_haa', 40)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_limit_haa', 0.2)
        self.declare_parameter('omega_limit_haa', 0.9)
        self.declare_parameter('a_limit', 0.5)
        self.declare_parameter('alpha_limit', 0.5)
        self.declare_parameter('robot_radius', 0.15)
        self.declare_parameter('goal', '[0.0, 0.0, 0.0, 0.0, 0.0]')
        # HPA / diffusion parameters
        self.declare_parameter('v_limit_hpa', 0.26)
        self.declare_parameter('omega_limit_hpa', 1.82)
        self.declare_parameter('lidar_max_range', 3.5)
        self.declare_parameter('model_path', '')
        self.declare_parameter('num_inference_steps', 10)
        self.declare_parameter('num_samples', 1)
        self.declare_parameter('pure_pursuit_lookahead', 0.3)
        self.declare_parameter('pure_pursuit_v_ref', 0.15)
        self.declare_parameter('vis_dir', os.path.join(os.path.expanduser('~'), 'policy_vis'))
        self.declare_parameter('vis_every_n', 5)   # save viz every N planning cycles
        # Kanayama gains
        self.declare_parameter('kx', 1.0)
        self.declare_parameter('ky', 1.0)
        self.declare_parameter('kth', 1.0)
        self.declare_parameter('kv', 1.0)
        self.declare_parameter('kw', 1.0)
        self.declare_parameter('kix', 0.1)
        self.declare_parameter('kiy', 0.1)
        self.declare_parameter('kith', 0.1)
        self.declare_parameter('max_integral', 1.0)
        self.declare_parameter('failure_path', os.path.join(os.path.expanduser('~'), 'mppi_failure_data.npz'))

        # ── Load parameters ─────────────────────────────────────────────────
        self.N_haa = self.get_parameter('horizon_haa').value
        self.N_hpa = 20
        self.dt = self.get_parameter('dt').value
        self.v_limit_haa = self.get_parameter('v_limit_haa').value
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value
        self.v_limit_hpa = self.get_parameter('v_limit_hpa').value
        self.omega_limit_hpa = self.get_parameter('omega_limit_hpa').value
        self.a_limit = self.get_parameter('a_limit').value
        self.alpha_limit = self.get_parameter('alpha_limit').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.pure_pursuit_L = self.get_parameter('pure_pursuit_lookahead').value
        self.pure_pursuit_v_ref = self.get_parameter('pure_pursuit_v_ref').value
        self.vis_dir = self.get_parameter('vis_dir').value
        self.vis_every_n = int(self.get_parameter('vis_every_n').value)
        self._vis_counter = 0
        os.makedirs(self.vis_dir, exist_ok=True)

        import ast
        raw_goal = self.get_parameter('goal').value
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal
        self.goal_temp = self.goal.copy()
        self.goal_orig_history = []
        self.goal_temp_history = []
        self.state_traj = []

        # ── State flags ─────────────────────────────────────────────────────
        self.HAA_map_ready = False
        self.state_ready = False
        self.lidar_ready = False
        self.goal_received = False
        self._goal_wait_logged = False
        self.target_reached = False
        self.shutdown_requested = False
        self.step = 0
        self.HAA_mppi_planner = None
        self.HAA_occupancy_map = None
        self.HAA_raw_occ = None

        # ── Switching state ─────────────────────────────────────────────────
        self.current_planner = "HPA"
        self.haa_check_failed_first_time = False
        self.braking_loop_count = 0
        self.switch_point = []
        self.planner_history = []

        # ── Trajectory tracking ─────────────────────────────────────────────
        self.current_trajectory = None
        self.current_controls = None
        self.trajectory_start_time = None
        self.trajectory_ready = False
        self.first_plan = True

        # ── Failure save path ───────────────────────────────────────────────
        self.failure_path = self.get_parameter('failure_path').value
        self.plot_data = []
        self.plot_path = os.path.join(os.path.expanduser('~'), 'plot_data_lowlvlctrl.npz')
        self.trajectory_path = os.path.join(os.path.expanduser('~'), 'robot_trajectory_switch_diffusion.png')
        self.Xopt_history = []
        self.control_command_history = []

        # ── Low-pass filter / rate-limiter ───────────────────────────────────
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0

        # ── /odom filter constants ───────────────────────────────────────────
        self._ODOM_V_MAX = 0.26
        self._ODOM_OMEGA_MAX = 1.82
        self._ODOM_EMA_ALPHA = 0.3

        # ── LiDAR scan ───────────────────────────────────────────────────────
        self.latest_lidar_scan = None   # raw metres, 360 elements

        # ── Kanayama controller ──────────────────────────────────────────────
        self.kanayama_controller = KanayamaController(
            self.get_parameter('kx').value, self.get_parameter('ky').value,
            self.get_parameter('kth').value, self.get_parameter('kv').value,
            self.get_parameter('kw').value,
            self.get_parameter('kix').value, self.get_parameter('kiy').value,
            self.get_parameter('kith').value, self.get_parameter('max_integral').value,
            v_limit_haa=self.v_limit_haa, omega_limit_haa=self.omega_limit_haa,
            v_limit_hpa=self.v_limit_hpa, omega_limit_hpa=self.omega_limit_hpa,
        )

        # ── Robot state ──────────────────────────────────────────────────────
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0

        # ── Stuck detection ──────────────────────────────────────────────────
        self._stuck_timeout_sec = 10.0
        self._stuck_threshold_m = 0.1
        self._stuck_last_prog_time = None
        self._stuck_last_prog_pos = None

        # ── Load diffusion policy ─────────────────────────────────────────────
        _param_path = self.get_parameter('model_path').value.strip()
        if _param_path:
            model_path = os.path.expanduser(_param_path)
        else:
            model_path = '/home/acrl/robot_data/real_world_models/diffusion_h50/last_model_diffusion_policy.pth'
        if not os.path.exists(model_path):
            self.get_logger().error(f'Diffusion model not found at {model_path}')
            raise FileNotFoundError(f'Diffusion model not found at {model_path}')

        self.policy_runner = PolicyRunner(
            ckpt_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_inference_steps=self.get_parameter('num_inference_steps').value,
            num_samples=self.get_parameter('num_samples').value,
        )
        self.policy_runner.reset()

        # Read lidar_max_range and v_max/w_max from the loaded checkpoint meta
        _meta = self.policy_runner.meta
        if _meta:
            lr = float(_meta.get('lidar_max_range', self.lidar_max_range))
            self.lidar_max_range = lr
            self._policy_v_max = float(_meta.get('v_max', 0.26))
            self._policy_w_max = float(_meta.get('w_max', 1.82))
        else:
            self._policy_v_max = 0.26
            self._policy_w_max = 1.82

        self.get_logger().info(
            f'Diffusion policy loaded from {model_path} '
            f'(type={self.policy_runner.model_type}, '
            f'horizon={self.policy_runner.horizon}, '
            f'history_len={self.policy_runner.history_len})'
        )

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS pubs / subs ─────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.inflation_map_pub = self.create_publisher(OccupancyGrid, '/inflation_map', 1)
        self.slam_pose_pub = self.create_publisher(Odometry, '/slam_pose', 10)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, scan_qos)

        # ── Timers ───────────────────────────────────────────────────────────
        self.state_timer   = self.create_timer(0.01, self.state_update_cb)   # 100 Hz
        self.planning_timer = self.create_timer(0.1, self.planning_loop)     # 10 Hz
        self.control_timer  = self.create_timer(0.02, self.control_loop)     # 50 Hz

        self.get_logger().info("MPC+Diffusion switching planner started.")
        self.get_logger().info(f"  HPA: diffusion policy (horizon={self.policy_runner.horizon})")
        self.get_logger().info(f"  HAA: MPPI MPC")
        self.get_logger().info(f"  lidar_max_range={self.lidar_max_range} m")

    # ── Utility ─────────────────────────────────────────────────────────────

    def _time_to_sec(self, ros_time):
        return float(ros_time.nanoseconds) * 1e-9

    def _stop_timers(self):
        for timer in [self.planning_timer, self.control_timer]:
            if timer is not None:
                try: timer.cancel()
                except Exception: pass

    # ── LiDAR callback ───────────────────────────────────────────────────────

    def lidar_cb(self, msg: LaserScan):
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            max_r = self.lidar_max_range
            bad_mask = ~np.isfinite(ranges) | (ranges <= 0.0)
            ranges[bad_mask] = max_r
            np.clip(ranges, 0.0, max_r, out=ranges)
            n = len(ranges)
            if n != 360:
                indices = np.linspace(0, n - 1, 360)
                ranges = np.interp(indices, np.arange(n), ranges).astype(np.float32)
            self.latest_lidar_scan = ranges  # raw metres [0, lidar_max_range]
            self.lidar_ready = True
        except Exception as e:
            self.get_logger().warn(f'lidar_cb error: {e}')

    # ── /odom callback ───────────────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        v_raw = max(0.0, min(msg.twist.twist.linear.x, self._ODOM_V_MAX))
        w_raw = max(-self._ODOM_OMEGA_MAX, min(msg.twist.twist.angular.z, self._ODOM_OMEGA_MAX))
        a = self._ODOM_EMA_ALPHA
        self.v     = a * v_raw + (1.0 - a) * self.v
        self.omega = a * w_raw + (1.0 - a) * self.omega

    # ── TF2 pose estimation ──────────────────────────────────────────────────

    def state_update_cb(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            self.x = t.transform.translation.x
            self.y = t.transform.translation.y
            q = t.transform.rotation
            _, _, self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.state_ready = True

            slam_msg = Odometry()
            slam_msg.header.stamp = self.get_clock().now().to_msg()
            slam_msg.header.frame_id = 'map'
            slam_msg.child_frame_id = 'base_footprint'
            slam_msg.pose.pose.position.x = self.x
            slam_msg.pose.pose.position.y = self.y
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, self.theta)
            slam_msg.pose.pose.orientation.x = qx
            slam_msg.pose.pose.orientation.y = qy
            slam_msg.pose.pose.orientation.z = qz
            slam_msg.pose.pose.orientation.w = qw
            slam_msg.twist.twist.linear.x = self.v
            slam_msg.twist.twist.angular.z = self.omega
            self.slam_pose_pub.publish(slam_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    # ── Map callback ─────────────────────────────────────────────────────────

    def map_cb(self, msg: OccupancyGrid):
        data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.HAA_raw_occ = data.copy()
        self.HAA_occupancy_map = OccupancyGridMap(
            width=msg.info.width * msg.info.resolution,
            height=msg.info.height * msg.info.resolution,
            resolution=msg.info.resolution,
        )
        self.HAA_occupancy_map.occupancy_grid = data
        self.HAA_occupancy_map.x_min = msg.info.origin.position.x
        self.HAA_occupancy_map.y_min = msg.info.origin.position.y
        self.HAA_occupancy_map.x_max = msg.info.origin.position.x + msg.info.width * msg.info.resolution
        self.HAA_occupancy_map.y_max = msg.info.origin.position.y + msg.info.height * msg.info.resolution
        self.HAA_map_ready = True
        self._publish_inflation_map(msg)

    def _publish_inflation_map(self, source_msg: OccupancyGrid):
        omap = self.HAA_occupancy_map
        raw = omap.occupancy_grid
        res = omap.resolution
        hard_radius_cells = int(self.robot_radius / res)
        soft_radius_cells = int((self.robot_radius + 0.05) / res)
        obstacle_mask = raw > 50
        dist_cells = ndimage.distance_transform_edt(~obstacle_mask) if np.any(obstacle_mask) \
                     else np.full(raw.shape, 9999.0)
        viz = np.zeros(raw.shape, dtype=np.int8)
        viz[dist_cells <= soft_radius_cells] = 20
        viz[dist_cells <= hard_radius_cells] = 60
        viz[obstacle_mask] = 100
        out = OccupancyGrid()
        out.header = source_msg.header
        out.info = source_msg.info
        out.data = viz.flatten().tolist()
        self.inflation_map_pub.publish(out)

    # ── Goal callback ─────────────────────────────────────────────────────────

    def goal_cb(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.goal = [x, y, theta, 0.0, 0.0]
        self.goal_temp = self.goal.copy()
        self.goal_received = True
        self.target_reached = False
        self.trajectory_ready = False
        self.HAA_mppi_planner = None
        self.step = 0
        self.state_traj = []
        self.planner_history = []
        self.switch_point = []
        self._goal_wait_logged = False
        self.kanayama_controller.reset_integral_errors()
        self.current_planner = "HPA"
        self.haa_check_failed_first_time = False
        self.braking_loop_count = 0
        self.current_trajectory = None
        self.current_controls = None
        self.policy_runner.reset()
        self._stuck_last_prog_time = self.get_clock().now().nanoseconds * 1e-9
        self._stuck_last_prog_pos = (self.x, self.y)
        self.get_logger().info(f"New goal: x={x:.3f} m, y={y:.3f} m, theta={np.degrees(theta):.1f}°")

    # ── Diffusion policy inference ───────────────────────────────────────────

    def _get_diffusion_trajectory(self):
        """
        Run the policy for the current state and return
        (world_traj, controls, traj_local) where:
          world_traj  : (H+1, 5) [x, y, theta, v, w] — traj[0]=now
          controls    : (H, 2)   [a, alpha]
          traj_local  : (H, action_dim) raw model output in robot frame at t

        Returns (None, None, None) if lidar or state is not ready.
        """
        if self.latest_lidar_scan is None:
            return None, None, None

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        obs = Observation(
            x=self.x,
            y=self.y,
            theta=self.theta,
            v=self.v,
            w=self.omega,
            v_cmd=self._last_v_cmd,
            w_cmd=self._last_w_cmd,
            target_x=self.goal_temp[0],
            target_y=self.goal_temp[1],
            lidar_ranges=self.latest_lidar_scan,   # raw metres
            lidar_max_range=self.lidar_max_range,
            timestamp=now_sec,
        )

        traj_local = self.policy_runner.get_trajectory(obs)   # (H, action_dim)

        # Convert to world-frame poses (H, 3): [x, y, theta]
        world_poses = PolicyRunner.traj_to_world(
            traj_local[:, :4], self.x, self.y, self.theta
        )  # (H, 3)

        H = world_poses.shape[0]
        action_dim = traj_local.shape[1]
        dt = self.policy_runner.dt

        # Use model-predicted v, w (channels 4,5) — training targets include
        # (dx, dy, sin_dθ, cos_dθ, v/v_max, w/w_max), so these are supervised.
        if action_dim >= 6:
            v_traj = traj_local[:, 4] * self._policy_v_max
            w_traj = traj_local[:, 5] * self._policy_w_max
        else:
            dx = np.diff(world_poses[:, 0], prepend=self.x)
            dy = np.diff(world_poses[:, 1], prepend=self.y)
            v_traj = np.sqrt(dx**2 + dy**2) / dt
            raw_dtheta = np.diff(world_poses[:, 2], prepend=self.theta)
            w_traj = np.array([((d + math.pi) % (2 * math.pi) - math.pi) for d in raw_dtheta]) / dt

        # Prepend current state so traj[0]=now, traj[k]=t+k*dt — same convention
        # as HAA (simulate_trajectories_vectorized) and braking trajectories.
        # This ensures get_reference_state()[traj[2]] is a 0.2s lookahead in all modes.
        v_traj = np.clip(v_traj, 0.0, self.v_limit_hpa)
        w_traj = np.clip(w_traj, -self.omega_limit_hpa, self.omega_limit_hpa)
        current_wp = np.array([[self.x, self.y, self.theta, self.v, self.omega]])
        world_traj = np.vstack(
            [current_wp, np.column_stack([world_poses, v_traj, w_traj])]
        )  # (H+1, 5)

        # Compute acceleration controls [a, alpha] including the 0→1 step.
        # controls[k] takes the robot from traj[k] to traj[k+1].
        v_full     = np.concatenate([[self.v],     v_traj])   # (H+1,)
        w_full     = np.concatenate([[self.omega], w_traj])   # (H+1,)
        a_seq      = np.diff(v_full) / dt                      # (H,)
        alpha_seq  = np.diff(w_full) / dt                      # (H,)
        controls   = np.column_stack([a_seq, alpha_seq])       # (H, 2)

        return world_traj, controls, traj_local

    # ── Policy visualization ──────────────────────────────────────────────────

    def _save_policy_vis(self, traj_local: np.ndarray) -> None:
        """
        Save a training-style top-down visualization in robot frame.
        Matches render_sample() from visualize_trajectories.py:
          - lidar point cloud (gray/blue dots)
          - predicted XY trajectory (red)
          - goal (green star)
        The lidar and trajectory are both in robot frame, so no coordinate
        conversion is needed — same as training-time visualization.
        """
        try:
            n_beams = len(self.latest_lidar_scan)
            lidar_max = self.lidar_max_range

            # Sanitize lidar to metres (same as _process_lidar, normalize=False)
            lidar_raw = np.asarray(self.latest_lidar_scan, dtype=np.float32).copy()
            bad = ~np.isfinite(lidar_raw) | (lidar_raw <= 0.0)
            lidar_raw[bad] = lidar_max
            lidar_m = np.clip(lidar_raw, 0.0, lidar_max)

            # Convert to Cartesian (robot frame): beam 0 = forward (+x), CCW
            angles = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
            valid  = lidar_m < lidar_max
            hit_x  = lidar_m[valid]  * np.cos(angles[valid])
            hit_y  = lidar_m[valid]  * np.sin(angles[valid])
            miss_x = np.cos(angles[~valid]) * lidar_max
            miss_y = np.sin(angles[~valid]) * lidar_max

            # Goal in robot frame
            c, s = math.cos(self.theta), math.sin(self.theta)
            dx_w = self.goal_temp[0] - self.x
            dy_w = self.goal_temp[1] - self.y
            goal_x_r =  dx_w * c + dy_w * s
            goal_y_r = -dx_w * s + dy_w * c

            fig, ax = plt.subplots(figsize=(6, 6))

            # Lidar hits (solid) and no-returns (faint ring)
            ax.scatter(hit_x,  hit_y,  s=4, c='dimgray',      alpha=0.8,  label='lidar hit',  zorder=1)
            ax.scatter(miss_x, miss_y, s=1, c='lightsteelblue', alpha=0.25, label='no return', zorder=1)

            # Predicted trajectory (channels 0,1 = dx, dy in robot frame)
            traj_xy = traj_local[:, :2]
            xs = np.concatenate([[0.0], traj_xy[:, 0]])
            ys = np.concatenate([[0.0], traj_xy[:, 1]])
            ax.plot(xs, ys, 'r-', linewidth=2.0, label='predicted', zorder=3)
            ax.scatter(traj_xy[:, 0], traj_xy[:, 1], s=18, c='red', zorder=4)

            # Robot origin and heading arrow
            ax.scatter([0], [0], s=80, c='black', marker='^', zorder=5, label='robot')
            ax.annotate('', xy=(0.4, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

            # Goal
            ax.scatter([goal_x_r], [goal_y_r], s=120, c='lime', marker='*',
                       zorder=5, label='goal', edgecolors='darkgreen', linewidths=0.8)

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x (m, forward)')
            ax.set_ylabel('y (m, left)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
            model_type = getattr(self.policy_runner, 'model_type', 'policy')
            ax.set_title(
                f'{model_type}  step={self.step}  '
                f'v={self.v:.2f}m/s  w={self.omega:.2f}rad/s',
                fontsize=9,
            )

            fig.tight_layout()
            fname = os.path.join(self.vis_dir, f'vis_step_{self.step:06d}.png')
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            self.get_logger().info(f'[vis] saved → {fname}')
        except Exception as e:
            self.get_logger().warn(f'_save_policy_vis failed: {e}')

    # ── Braking trajectory ────────────────────────────────────────────────────

    def generate_braking_trajectory(self, start_state: RobotState, horizon: int):
        dynamics = TurtlebotDynamics(dt=self.dt)
        trajectory = np.zeros((horizon + 1, 5))
        controls = np.zeros((horizon, 2))
        trajectory[0] = [start_state.x, start_state.y, start_state.theta, start_state.v, start_state.w]
        max_braking_a = -self.a_limit * 1.2
        controls[:, 0] = max_braking_a
        current_state = RobotState(start_state.x, start_state.y, start_state.theta, start_state.v, start_state.w)
        for i in range(horizon):
            current_state = dynamics.step(current_state, np.array([max_braking_a, 0.0]))
            current_state.v = max(0.0, current_state.v)
            trajectory[i+1] = [current_state.x, current_state.y, current_state.theta,
                                current_state.v, current_state.w]
        return trajectory, controls

    # ── Planning loop ─────────────────────────────────────────────────────────

    def planning_loop(self, event=None):
        if self.shutdown_requested:
            return
        if not (self.HAA_map_ready and self.state_ready):
            return
        if not self.goal_received:
            if not self._goal_wait_logged:
                self.get_logger().info(
                    "Ready. Set a goal using '2D Goal Pose' button in RViz2.")
                self._goal_wait_logged = True
            return

        x_true = [self.x, self.y, self.theta, self.v, self.omega]
        x0 = x_true.copy()
        self.state_traj.append(x_true.copy())
        self.planner_history.append(self.current_planner)

        # Goal-reached check
        dist = np.linalg.norm(np.array(x_true[:2]) - np.array(self.goal_temp[:2]))
        if dist < 0.1 and not self.target_reached:
            self.target_reached = True
            self.publish_stop_command()
            self.save_trajectory_plot()
            self.goal_received = False
            self.trajectory_ready = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            self.current_planner = "HPA"
            self.get_logger().info(f"Goal reached! (dist={dist:.3f} m). Robot stopped.")
            return

        # Stuck-timeout check
        if self._stuck_last_prog_time is not None:
            _now = self.get_clock().now().nanoseconds * 1e-9
            _disp = np.hypot(self.x - self._stuck_last_prog_pos[0],
                             self.y - self._stuck_last_prog_pos[1])
            if _disp >= self._stuck_threshold_m:
                self._stuck_last_prog_time = _now
                self._stuck_last_prog_pos = (self.x, self.y)
            elif (_now - self._stuck_last_prog_time) > self._stuck_timeout_sec:
                self.get_logger().warn("STUCK TIMEOUT — aborting goal.")
                self.publish_stop_command()
                self.goal_received = False
                self.trajectory_ready = False
                self.HAA_mppi_planner = None
                self._goal_wait_logged = False
                self._stuck_last_prog_time = None
                self.current_planner = "HPA"
                return

        try:
            self.goal_temp = self.goal.copy()

            # Initialize HAA planner if needed
            if self.HAA_mppi_planner is None:
                dynamics = TurtlebotDynamics(dt=self.dt)
                haa_mppi = MPPI(
                    sigma=5, temperature=1.0,
                    num_nodes=self.N_haa, num_rollouts=1500,
                    use_noise_ramp=False, nu=2,
                )
                self.HAA_mppi_planner = MPPIPlanner(
                    mppi=haa_mppi, dynamics=dynamics,
                    occupancy_map=self.HAA_occupancy_map,
                    v_min=-0.05, v_max=self.v_limit_haa,
                    w_min=-self.omega_limit_haa, w_max=self.omega_limit_haa,
                    a_min=-self.a_limit, a_max=self.a_limit,
                    alpha_min=-self.alpha_limit, alpha_max=self.alpha_limit,
                    smoothing_weight=0.1, logger=self.get_logger(),
                )
                self.get_logger().info("HAA MPPI planner initialized.")

            if self.HAA_mppi_planner.occupancy_map != self.HAA_occupancy_map:
                self.HAA_mppi_planner.update_occupancy_map(self.HAA_occupancy_map)

            # ═══════════════════════════════════════════════════════════════
            # SWITCHING LOGIC
            # ═══════════════════════════════════════════════════════════════
            if self.current_planner == "HPA":

                # --- Braking phase (after HAA check failure) ---
                if self.haa_check_failed_first_time:
                    self.braking_loop_count += 1
                    self.get_logger().info(f"Braking mode: loop {self.braking_loop_count}/2")
                    braking_start = RobotState(x_true[0], x_true[1], x_true[2], x_true[3], x_true[4])
                    traj, ctrl = self.generate_braking_trajectory(braking_start, self.N_hpa)
                    self.current_trajectory = traj
                    self.current_controls = ctrl
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    if self.braking_loop_count >= 2:
                        self.get_logger().info("Braking complete — switching to HAA")
                        self.current_planner = "HAA"
                        self.haa_check_failed_first_time = False
                        self.braking_loop_count = 0
                        self.kanayama_controller.reset_integral_errors()
                    return

                # --- HPA normal: run diffusion policy ---
                if not self.lidar_ready or self.latest_lidar_scan is None:
                    self.get_logger().warn("HPA waiting for lidar")
                    return

                world_traj, controls, traj_local = self._get_diffusion_trajectory()
                if world_traj is None:
                    return

                # Periodic visualization: lidar + predicted trajectory in robot frame
                self._vis_counter += 1
                if self._vis_counter % self.vis_every_n == 0:
                    self._save_policy_vis(traj_local)

                # Diagnostic: log predicted waypoints each HPA cycle.
                # "local" values should change every cycle if policy is responsive.
                self.get_logger().info(
                    f'[HPA#{self._vis_counter:04d}] '
                    f'local wp[0]=({traj_local[0,0]:.3f},{traj_local[0,1]:.3f}) '
                    f'local wp[4]=({traj_local[4,0]:.3f},{traj_local[4,1]:.3f}) '
                    f'world wp[1]=({world_traj[1,0]:.3f},{world_traj[1,1]:.3f}) '
                    f'robot=({self.x:.3f},{self.y:.3f},{math.degrees(self.theta):.1f}°)'
                )

                # HAA feasibility check: start from the first predicted next state
                # (traj[1] = t+0.1s; traj[0] is the prepended current state).
                dynamics = TurtlebotDynamics(dt=self.dt)
                haa_check_state = RobotState(
                    x=world_traj[1, 0],
                    y=world_traj[1, 1],
                    theta=world_traj[1, 2],
                    v=float(np.clip(world_traj[1, 3], 0, self.v_limit_hpa)),
                    w=float(np.clip(world_traj[1, 4], -self.omega_limit_hpa, self.omega_limit_hpa)),
                )
                # Apply N_r braking steps for recoverable-set check
                N_r = 3
                for _ in range(N_r):
                    haa_check_state = dynamics.step(haa_check_state, np.array([-self.a_limit, 0]))
                    haa_check_state.v = max(0, haa_check_state.v)

                t0 = time.time()
                haa_Xopt, haa_Uopt = self.HAA_mppi_planner.plan(
                    haa_check_state, np.array(self.goal_temp[:2]),
                    robot_radius=self.robot_radius + 0.05,
                )
                haa_feasible = haa_Xopt is not None and len(haa_Xopt) > 0
                self.get_logger().info(
                    f'HAA check: feasible={haa_feasible}  ({time.time()-t0:.3f}s)')

                if haa_feasible:
                    # Stay on HPA — track diffusion trajectory with Kanayama
                    self.current_trajectory = world_traj
                    self.current_controls = controls
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    self.current_planner = "HPA"
                else:
                    if not self.haa_check_failed_first_time:
                        self.get_logger().info("HAA check failed — starting 2-loop braking")
                        self.switch_point.append(x_true.copy())
                        self.haa_check_failed_first_time = True
                        self.braking_loop_count = 1
                        braking_start = RobotState(x_true[0], x_true[1], x_true[2], x_true[3], x_true[4])
                        traj, ctrl = self.generate_braking_trajectory(braking_start, self.N_hpa)
                        self.current_trajectory = traj
                        self.current_controls = ctrl
                        self.trajectory_start_time = self.get_clock().now()
                        self.trajectory_ready = True

            elif self.current_planner == "HAA":
                haa_current_state = RobotState(
                    x=x0[0], y=x0[1], theta=x0[2],
                    v=np.clip(x0[3], 0, self.v_limit_haa),
                    w=np.clip(x0[4], -self.omega_limit_haa, self.omega_limit_haa),
                )
                t0 = time.time()
                haa_Xopt, haa_Uopt = self.HAA_mppi_planner.plan(
                    haa_current_state, np.array(self.goal_temp[:2]),
                    debug_plot=True, robot_radius=self.robot_radius,
                )
                self.get_logger().debug(f"HAA planning: {time.time()-t0:.3f}s")

                if haa_Xopt is not None and len(haa_Xopt) > 0:
                    self.current_trajectory = haa_Xopt
                    self.current_controls = haa_Uopt
                    self.trajectory_start_time = self.get_clock().now()
                    self.trajectory_ready = True
                    self.haa_check_failed_first_time = False
                    self.braking_loop_count = 0

                    safety_margin = (self.dt * self.v_limit_haa
                                     + self.v_limit_haa**2 / (2 * self.a_limit) + 0.05)
                    collisions = self.HAA_mppi_planner.occupancy_map.check_collision_new(
                        np.array([haa_current_state.x]),
                        np.array([haa_current_state.y]),
                        self.robot_radius + safety_margin,
                    )
                    if not np.any(collisions):
                        self.get_logger().info("Safety margin OK — switching back to HPA")
                        self.current_planner = "HPA"
                        self.current_trajectory = None
                        self.current_controls = None
                        self.switch_point.append(x_true.copy())
                        self.policy_runner.reset()
                        return
                else:
                    self.get_logger().warn("HAA planning failed — stopping robot.")
                    self.publish_stop_command()
                    self.goal_received = False
                    self.HAA_mppi_planner = None
                    self._goal_wait_logged = False
                    self.current_planner = "HPA"
                    return

            if self.trajectory_ready and self.current_trajectory is not None:
                self.Xopt_history.append({
                    'Xopt': self.current_trajectory.copy(),
                    'start_state': x0,
                    'goal': self.goal_temp[:2].copy(),
                    'step': self.step,
                    'planner': self.current_planner,
                })

            self.kanayama_controller.reset_integral_errors()

        except Exception as e:
            self.get_logger().error(f"Planning exception: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.publish_stop_command()
            self.goal_received = False
            self.HAA_mppi_planner = None
            self._goal_wait_logged = False
            self.current_planner = "HPA"
            return

        self.step += 1

    # ── Reference state ──────────────────────────────────────────────────────

    def _pure_pursuit_hpa(self):
        """
        Pure-pursuit controller for HPA mode.

        Uses the stored world-frame trajectory (current_trajectory), transforms
        each waypoint into the current robot frame at call time, then finds the
        first waypoint beyond the lookahead distance L and steers toward it.

        Returns (v_cmd, w_cmd) already clipped to HPA limits.
        """
        traj = self.current_trajectory   # (H+1, 5): [x, y, theta, v, w]
        L = self.pure_pursuit_L
        v_ref = self.pure_pursuit_v_ref

        c = math.cos(self.theta)
        s = math.sin(self.theta)

        # Transform world waypoints into current robot frame (skip traj[0] = stored current)
        waypoint = None
        for k in range(1, len(traj)):
            dx_w = traj[k, 0] - self.x
            dy_w = traj[k, 1] - self.y
            dx_l =  dx_w * c + dy_w * s
            dy_l = -dx_w * s + dy_w * c
            dist = math.sqrt(dx_l * dx_l + dy_l * dy_l)
            if dist >= L:
                waypoint = (dx_l, dy_l, dist)
                break

        # Fall back to the farthest waypoint if none reached lookahead
        if waypoint is None:
            dx_w = traj[-1, 0] - self.x
            dy_w = traj[-1, 1] - self.y
            dx_l =  dx_w * c + dy_w * s
            dy_l = -dx_w * s + dy_w * c
            dist = math.sqrt(dx_l * dx_l + dy_l * dy_l)
            waypoint = (dx_l, dy_l, max(dist, 1e-3))

        dx_l, dy_l, L_act = waypoint

        # Pure-pursuit curvature: κ = 2·y / L²
        kappa = 2.0 * dy_l / (L_act * L_act)

        # Scale forward speed down for sharp turns (keeps curvature trackable)
        angle_to_wp = math.atan2(dy_l, dx_l)
        v_cmd = float(np.clip(v_ref * max(0.0, math.cos(angle_to_wp)),
                              0.0, self.v_limit_hpa))
        w_cmd = float(np.clip(v_cmd * kappa,
                              -self.omega_limit_hpa, self.omega_limit_hpa))
        return v_cmd, w_cmd

    def get_reference_state(self, current_time):
        if not self.trajectory_ready or self.current_trajectory is None or self.current_controls is None:
            return None
        if len(self.current_trajectory) < 3 or len(self.current_controls) < 2:
            return None
        x_true = np.array([self.x, self.y, self.theta, self.v, self.omega])
        x_nominal = np.array(self.current_trajectory[0])
        u_nominal = np.array(self.current_controls[0])
        error = x_true - x_nominal
        K_feedback = [[0.0, 0.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.6]]
        u = u_nominal + np.array(K_feedback) @ error
        if self.HAA_mppi_planner is not None and self.HAA_mppi_planner.dynamics is not None:
            dynamics = self.HAA_mppi_planner.dynamics
        else:
            dynamics = TurtlebotDynamics(dt=self.dt)
        current_robot_state = RobotState(x_true[0], x_true[1], x_true[2], x_true[3], x_true[4])
        ref_robot_state = dynamics.step(current_robot_state, u)
        ref_robot_state = dynamics.step(ref_robot_state, self.current_controls[1])
        return self.current_trajectory[2]

    # ── Control loop ──────────────────────────────────────────────────────────

    def control_loop(self, event=None):
        if self.shutdown_requested:
            return
        if not self.state_ready or not self.trajectory_ready:
            return

        current_time = self.get_clock().now()

        if self.current_trajectory is None:
            return

        if self.current_planner == "HPA":
            # Pure pursuit: only needs XY path, no velocity reference required
            v_cmd, w_cmd = self._pure_pursuit_hpa()
        else:
            # HAA / braking: Kanayama with full state reference from MPPI trajectory
            ref_state = self.get_reference_state(current_time)
            if ref_state is None:
                self.publish_stop_command()
                return
            current_state = [self.x, self.y, self.theta, self.v, self.omega]
            v_cmd, w_cmd = self.kanayama_controller.compute_control(
                current_state, ref_state, current_planner=self.current_planner
            )

        # EMA low-pass filter
        _alpha = 0.6
        v_smooth = _alpha * self.v_cmd_filtered + (1.0 - _alpha) * v_cmd
        w_smooth = _alpha * self.w_cmd_filtered + (1.0 - _alpha) * w_cmd

        # Rate limiter
        _dt_ctrl = 0.02
        _max_dv = self.a_limit * _dt_ctrl
        _max_dw = self.alpha_limit * _dt_ctrl
        self.v_cmd_filtered = float(np.clip(v_smooth,
                                            self.v_cmd_prev - _max_dv,
                                            self.v_cmd_prev + _max_dv))
        self.w_cmd_filtered = float(np.clip(w_smooth,
                                            self.w_cmd_prev - _max_dw,
                                            self.w_cmd_prev + _max_dw))
        self.v_cmd_prev = self.v_cmd_filtered
        self.w_cmd_prev = self.w_cmd_filtered
        self._last_v_cmd = self.v_cmd_filtered
        self._last_w_cmd = self.w_cmd_filtered

        twist = Twist()
        twist.linear.x = self.v_cmd_filtered
        twist.angular.z = self.w_cmd_filtered
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"[{self.current_planner}] v={self.v_cmd_filtered:.3f} w={self.w_cmd_filtered:.3f} | "
            f"raw v={v_cmd:.3f} w={w_cmd:.3f}",
            throttle_duration_sec=0.2,
        )

        self.control_command_history.append({
            'timestamp': self._time_to_sec(current_time),
            'v_cmd': self.v_cmd_filtered,
            'w_cmd': self.w_cmd_filtered,
            'planner': self.current_planner,
        })

    # ── Stop command ──────────────────────────────────────────────────────────

    def publish_stop_command(self):
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ── Save data / plots ─────────────────────────────────────────────────────

    def save_trajectory_plot(self):
        if len(self.state_traj) < 2:
            return
        try:
            x_coords = [s[0] for s in self.state_traj]
            y_coords = [s[1] for s in self.state_traj]
            v_coords = [s[3] for s in self.state_traj]
            w_coords = [s[4] for s in self.state_traj]
            time_steps = np.arange(len(self.state_traj)) * self.dt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            if self.HAA_occupancy_map is not None:
                omap = self.HAA_occupancy_map
                raw = omap.occupancy_grid
                res = omap.resolution
                hard_r = int(self.robot_radius / res)
                soft_r = int((self.robot_radius + 0.1) / res)
                obstacle_mask = raw > 50
                dist_c = ndimage.distance_transform_edt(~obstacle_mask) if np.any(obstacle_mask) \
                         else np.full(raw.shape, 9999.0)
                rgba = np.ones((*raw.shape, 4), dtype=float)
                rgba[dist_c <= soft_r] = [1.0, 1.0, 0.45, 1.0]
                rgba[dist_c <= hard_r] = [1.0, 0.55, 0.10, 1.0]
                rgba[obstacle_mask] = [0.15, 0.15, 0.15, 1.0]
                ax1.imshow(rgba, extent=[omap.x_min, omap.x_max, omap.y_min, omap.y_max],
                           origin='lower', aspect='equal', zorder=0)

            _colors = {"HPA": "#2196F3", "HAA": "#E53935"}
            planners = list(self.planner_history)
            for i in range(len(x_coords) - 1):
                c = _colors.get(planners[i] if i < len(planners) else "HPA", "gray")
                ax1.plot(x_coords[i:i+2], y_coords[i:i+2], color=c, linewidth=2, zorder=3)
            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start', zorder=5)
            ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=14, label='Goal', zorder=5)
            from matplotlib.lines import Line2D
            ax1.legend(handles=[
                Line2D([0], [0], color='#2196F3', lw=2, label='HPA (diffusion)'),
                Line2D([0], [0], color='#E53935', lw=2, label='HAA (MPPI)'),
                Line2D([0], [0], marker='o', color='g', lw=0, markersize=8, label='Start'),
                Line2D([0], [0], marker='*', color='r', lw=0, markersize=12, label='Goal'),
            ])
            ax1.set_title('Robot Trajectory (Diffusion + MPC Switching)')
            ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3); ax1.axis('equal')

            ax2.plot(time_steps, v_coords)
            ax2.axhline(y=self.v_limit_haa, color='r', linestyle='--', alpha=0.6, label=f'HAA limit ({self.v_limit_haa})')
            ax2.axhline(y=self.v_limit_hpa, color='b', linestyle=':', alpha=0.6, label=f'HPA limit ({self.v_limit_hpa})')
            ax2.set_title('Linear Velocity'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('v (m/s)')
            ax2.legend(); ax2.grid(True, alpha=0.3)

            ax3.plot(time_steps, w_coords)
            ax3.set_title('Angular Velocity'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('w (rad/s)')
            ax3.grid(True, alpha=0.3)

            if self.control_command_history:
                cmd_t = [c['timestamp'] for c in self.control_command_history]
                cmd_v = [c['v_cmd'] for c in self.control_command_history]
                t0 = cmd_t[0]
                ax4.plot([t - t0 for t in cmd_t], cmd_v, label='v_cmd')
                ax4.set_title('cmd_vel v'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('v (m/s)')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.trajectory_path, dpi=200, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f"Trajectory plot saved to {self.trajectory_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save trajectory plot: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SwitchMPCDiffusionRealWorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
