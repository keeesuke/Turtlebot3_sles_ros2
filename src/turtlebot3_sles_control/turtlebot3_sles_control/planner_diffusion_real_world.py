#!/usr/bin/env python3
"""
ROS2 Diffusion / Flow-Matching Policy Planner for TurtleBot3 — Real World
Standalone policy-only node (no MPC switching).

Architecture
------------
  100 Hz  state_update_cb  — TF2 map→base_footprint pose lookup
   50 Hz  odom_cb          — /odom EMA-filtered velocity
   10 Hz  planning_loop    — PolicyRunner inference → world-frame trajectory
   50 Hz  control_loop     — Pure-pursuit tracking of the latest trajectory

Model type is auto-detected from the checkpoint filename:
  "flow"          → flow-matching policy
  "deterministic" → deterministic MLP
  anything else   → diffusion policy (DDPM)

Prerequisites (must already be running):
  1. turtlebot3_node          (hw driver — /odom, /scan, /tf odom→base_footprint)
  2. turtlebot3_cartographer  (SLAM — /tf map→odom)
  3. rviz2                    (optional — for '2D Goal Pose' goal input)
"""

import math
import os
import sys
import time

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import tf2_ros
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from policy_runner import PolicyRunner, Observation


class PolicyPlannerRealWorld(Node):
    def __init__(self):
        super().__init__('policy_planner_real_world')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('model_path', '')
        self.declare_parameter('num_inference_steps', 10)
        self.declare_parameter('num_samples', 1)
        self.declare_parameter('v_limit', 0.21)          # m/s
        self.declare_parameter('omega_limit', 1.82)      # rad/s
        self.declare_parameter('lidar_max_range', 3.5)   # m
        self.declare_parameter('pure_pursuit_lookahead', 0.3)   # m
        self.declare_parameter('pure_pursuit_v_ref', 0.15)      # m/s
        self.declare_parameter('vis_dir',
            os.path.join(os.path.expanduser('~'), 'policy_vis'))
        self.declare_parameter('vis_every_n', 10)

        # ── Load parameters ─────────────────────────────────────────────────
        self.v_limit       = self.get_parameter('v_limit').value
        self.omega_limit   = self.get_parameter('omega_limit').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.pure_pursuit_L   = self.get_parameter('pure_pursuit_lookahead').value
        self.pure_pursuit_v_ref = self.get_parameter('pure_pursuit_v_ref').value
        self.vis_dir       = self.get_parameter('vis_dir').value
        self.vis_every_n   = int(self.get_parameter('vis_every_n').value)
        os.makedirs(self.vis_dir, exist_ok=True)

        # ── State flags ─────────────────────────────────────────────────────
        self.state_ready   = False
        self.lidar_ready   = False
        self.goal_received = False
        self._goal_wait_logged = False
        self.target_reached = False

        # ── Robot state ──────────────────────────────────────────────────────
        self.x = self.y = self.theta = self.v = self.omega = 0.0
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0

        self._ODOM_V_MAX     = 0.26
        self._ODOM_OMEGA_MAX = 1.82
        self._ODOM_EMA_ALPHA = 0.3

        # ── Sensor / goal state ──────────────────────────────────────────────
        self.latest_lidar_scan = None
        self.goal = [0.0, 0.0, 0.0, 0.0, 0.0]

        # ── Trajectory (written by planning_loop, read by control_loop) ──────
        self.current_trajectory = None   # (H+1, 5) world frame [x,y,theta,v,w]
        self.trajectory_ready   = False
        self._vis_counter       = 0

        # ── Logging ──────────────────────────────────────────────────────────
        self.state_traj = []
        self.control_command_history = []
        self.latest_map = None

        # ── Stuck detection ──────────────────────────────────────────────────
        self._stuck_timeout_sec  = 10.0
        self._stuck_threshold_m  = 0.10
        self._stuck_last_prog_time = None
        self._stuck_last_prog_pos  = None

        # ── Low-pass filter ──────────────────────────────────────────────────
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.v_cmd_prev = 0.0
        self.w_cmd_prev = 0.0

        # ── Load policy ──────────────────────────────────────────────────────
        _param_path = self.get_parameter('model_path').value.strip()
        if _param_path:
            model_path = os.path.expanduser(_param_path)
        else:
            model_path = os.path.join(
                _THIS_DIR,
                'best_model_flow_matching.pth',
            )
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at {model_path}')
            raise FileNotFoundError(f'Model not found at {model_path}')

        self.policy_runner = PolicyRunner(
            ckpt_path=model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_inference_steps=self.get_parameter('num_inference_steps').value,
            num_samples=self.get_parameter('num_samples').value,
        )
        self.policy_runner.reset()

        # Read lidar_max_range and v/w limits from checkpoint meta
        _meta = self.policy_runner.meta
        if _meta:
            self.lidar_max_range = float(_meta.get('lidar_max_range', self.lidar_max_range))
            self._policy_v_max   = float(_meta.get('v_max', 0.26))
            self._policy_w_max   = float(_meta.get('w_max', 1.82))
        else:
            self._policy_v_max = 0.26
            self._policy_w_max = 1.82

        self.get_logger().info(
            f'Policy loaded: type={self.policy_runner.model_type}  '
            f'horizon={self.policy_runner.horizon}  '
            f'history_len={self.policy_runner.history_len}  '
            f'lidar_max_range={self.lidar_max_range} m'
        )

        # ── TF2 ─────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS pubs / subs ─────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, scan_qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_cb, 10)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 10)

        # ── Timers ───────────────────────────────────────────────────────────
        self.state_timer    = self.create_timer(0.01, self.state_update_cb)   # 100 Hz
        self.planning_timer = self.create_timer(0.10, self.planning_loop)     #  10 Hz
        self.control_timer  = self.create_timer(0.02, self.control_loop)      #  50 Hz

        self.get_logger().info('Policy planner started.')
        self.get_logger().info("  Set a goal with '2D Goal Pose' in RViz2.")

    # ── LiDAR callback ───────────────────────────────────────────────────────

    def lidar_cb(self, msg: LaserScan):
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            max_r  = self.lidar_max_range
            bad_mask = ~np.isfinite(ranges) | (ranges <= 0.0)
            ranges[bad_mask] = max_r
            np.clip(ranges, 0.0, max_r, out=ranges)
            n = len(ranges)
            if n != 360:
                indices = np.linspace(0, n - 1, 360)
                ranges  = np.interp(indices, np.arange(n), ranges).astype(np.float32)
            self.latest_lidar_scan = ranges
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
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    # ── Map callback ─────────────────────────────────────────────────────────

    def _map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    # ── Goal callback ─────────────────────────────────────────────────────────

    def goal_cb(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.goal = [x, y, theta, 0.0, 0.0]
        self.goal_received  = True
        self.target_reached = False
        self.trajectory_ready = False
        self.current_trajectory = None
        self.state_traj = []
        self._goal_wait_logged = False
        self.policy_runner.reset()
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._stuck_last_prog_time = now_sec
        self._stuck_last_prog_pos  = (self.x, self.y)
        self.get_logger().info(f'New goal: x={x:.3f} y={y:.3f} theta={math.degrees(theta):.1f}°')

    # ── Planning loop (10 Hz) ─────────────────────────────────────────────────

    def planning_loop(self):
        if not (self.state_ready and self.lidar_ready and self.goal_received):
            if self.goal_received and not self._goal_wait_logged:
                self.get_logger().info("Waiting for TF2 / lidar ...")
                self._goal_wait_logged = True
            return
        if self.target_reached:
            return
        if not self._goal_wait_logged:
            self.get_logger().info("Ready. Running policy.")
            self._goal_wait_logged = True

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        obs = Observation(
            x=self.x, y=self.y, theta=self.theta,
            v=self.v, w=self.omega,
            v_cmd=self._last_v_cmd, w_cmd=self._last_w_cmd,
            target_x=self.goal[0], target_y=self.goal[1],
            lidar_ranges=self.latest_lidar_scan,
            lidar_max_range=self.lidar_max_range,
            timestamp=now_sec,
        )

        try:
            traj_local = self.policy_runner.get_trajectory(obs)   # (H, action_dim)
        except Exception as e:
            self.get_logger().error(f'Policy inference error: {e}')
            return

        # Convert local-frame trajectory to world frame
        world_poses = PolicyRunner.traj_to_world(
            traj_local[:, :4], self.x, self.y, self.theta
        )  # (H, 3): [x, y, theta]

        action_dim = traj_local.shape[1]
        if action_dim >= 6:
            v_traj = np.clip(traj_local[:, 4] * self._policy_v_max, 0.0, self.v_limit)
            w_traj = np.clip(traj_local[:, 5] * self._policy_w_max,
                             -self.omega_limit, self.omega_limit)
        else:
            dt = self.policy_runner.dt
            dx = np.diff(world_poses[:, 0], prepend=self.x)
            dy = np.diff(world_poses[:, 1], prepend=self.y)
            v_traj = np.clip(np.sqrt(dx**2 + dy**2) / dt, 0.0, self.v_limit)
            raw_dth = np.diff(world_poses[:, 2], prepend=self.theta)
            w_traj = np.clip(
                np.array([((d + math.pi) % (2 * math.pi) - math.pi) for d in raw_dth])
                / dt, -self.omega_limit, self.omega_limit
            )

        # Prepend current state so traj[0]=now (matches control convention)
        current_wp = np.array([[self.x, self.y, self.theta, self.v, self.omega]])
        self.current_trajectory = np.vstack(
            [current_wp, np.column_stack([world_poses, v_traj, w_traj])]
        )  # (H+1, 5)
        self.trajectory_ready = True

        # Periodic visualization
        self._vis_counter += 1
        if self._vis_counter % self.vis_every_n == 0:
            self._save_policy_vis(traj_local)

        # Diagnostic: log predicted waypoints + lidar stats each planning cycle.
        _lidar_norm = self.latest_lidar_scan / self.lidar_max_range
        _min_lidar  = float(_lidar_norm.min())
        _fwd_lidar  = float(_lidar_norm[0])   # beam 0 = forward (+x)
        self.get_logger().info(
            f'[plan#{self._vis_counter:04d}] '
            f'local wp[0]=({traj_local[0,0]:.3f},{traj_local[0,1]:.3f}) '
            f'local wp[4]=({traj_local[4,0]:.3f},{traj_local[4,1]:.3f}) '
            f'lidar_min={_min_lidar:.3f}({_min_lidar*self.lidar_max_range:.2f}m) '
            f'lidar_fwd={_fwd_lidar:.3f}({_fwd_lidar*self.lidar_max_range:.2f}m) '
            f'robot=({self.x:.3f},{self.y:.3f},{math.degrees(self.theta):.1f}°)'
        )

    # ── Control loop (50 Hz) — pure pursuit ───────────────────────────────────

    def control_loop(self):
        if self.target_reached:
            return
        if not (self.state_ready and self.goal_received):
            return

        # Log current state
        self.state_traj.append([self.x, self.y, self.theta, self.v, self.omega])

        # Goal-reached check
        dist = math.hypot(self.x - self.goal[0], self.y - self.goal[1])
        if dist < 0.10:
            self.get_logger().info(f'Goal reached! (dist={dist:.3f} m)')
            self.target_reached = True
            self._publish_stop()
            self._save_trajectory_plot()
            return

        # Stuck detection
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._stuck_last_prog_pos is not None:
            disp = math.hypot(self.x - self._stuck_last_prog_pos[0],
                              self.y - self._stuck_last_prog_pos[1])
            if disp >= self._stuck_threshold_m:
                self._stuck_last_prog_time = now_sec
                self._stuck_last_prog_pos  = (self.x, self.y)
            elif (self._stuck_last_prog_time is not None and
                  now_sec - self._stuck_last_prog_time > self._stuck_timeout_sec):
                self.get_logger().warn('Stuck — waiting for new goal.')
                self._publish_stop()
                self.goal_received     = False
                self._goal_wait_logged = False
                return

        if not self.trajectory_ready or self.current_trajectory is None:
            return

        v_cmd, w_cmd = self._pure_pursuit()

        # EMA low-pass + rate limiter
        _alpha   = 0.6
        _dt_ctrl = 0.02
        _max_dv  = 0.5 * _dt_ctrl
        _max_dw  = 0.5 * _dt_ctrl
        v_smooth = _alpha * self.v_cmd_filtered + (1.0 - _alpha) * v_cmd
        w_smooth = _alpha * self.w_cmd_filtered + (1.0 - _alpha) * w_cmd
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
        twist.linear.x  = self.v_cmd_filtered
        twist.angular.z = self.w_cmd_filtered
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f'v={self.v_cmd_filtered:.3f}  w={self.w_cmd_filtered:.3f}  dist={dist:.2f}m',
            throttle_duration_sec=0.2,
        )
        self.control_command_history.append({
            'timestamp': now_sec,
            'v_cmd': self.v_cmd_filtered,
            'w_cmd': self.w_cmd_filtered,
        })

    # ── Pure pursuit ─────────────────────────────────────────────────────────

    def _pure_pursuit(self):
        traj = self.current_trajectory   # (H+1, 5)
        L    = self.pure_pursuit_L
        v_ref = self.pure_pursuit_v_ref
        c = math.cos(self.theta)
        s = math.sin(self.theta)

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

        if waypoint is None:
            dx_w = traj[-1, 0] - self.x
            dy_w = traj[-1, 1] - self.y
            dx_l =  dx_w * c + dy_w * s
            dy_l = -dx_w * s + dy_w * c
            dist = max(math.sqrt(dx_l * dx_l + dy_l * dy_l), 1e-3)
            waypoint = (dx_l, dy_l, dist)

        dx_l, dy_l, L_act = waypoint
        kappa  = 2.0 * dy_l / (L_act * L_act)
        angle  = math.atan2(dy_l, dx_l)
        v_cmd  = float(np.clip(v_ref * max(0.0, math.cos(angle)), 0.0, self.v_limit))
        w_cmd  = float(np.clip(v_cmd * kappa, -self.omega_limit, self.omega_limit))
        return v_cmd, w_cmd

    # ── Visualization ─────────────────────────────────────────────────────────

    def _save_policy_vis(self, traj_local: np.ndarray):
        """Training-style top-down: lidar + predicted trajectory in robot frame."""
        try:
            lidar_max = self.lidar_max_range
            lidar_raw = np.asarray(self.latest_lidar_scan, dtype=np.float32).copy()
            bad = ~np.isfinite(lidar_raw) | (lidar_raw <= 0.0)
            lidar_raw[bad] = lidar_max
            lidar_m = np.clip(lidar_raw, 0.0, lidar_max)   # metres, not normalised

            n_beams = len(lidar_m)
            angles  = np.linspace(0, 2 * math.pi, n_beams, endpoint=False)
            valid   = lidar_m < lidar_max
            hit_x   = lidar_m[valid]  * np.cos(angles[valid])
            hit_y   = lidar_m[valid]  * np.sin(angles[valid])
            miss_x  = np.cos(angles[~valid]) * lidar_max
            miss_y  = np.sin(angles[~valid]) * lidar_max

            # Goal in robot frame
            c, s = math.cos(self.theta), math.sin(self.theta)
            dx_w, dy_w = self.goal[0] - self.x, self.goal[1] - self.y
            gx_r =  dx_w * c + dy_w * s
            gy_r = -dx_w * s + dy_w * c

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(hit_x,  hit_y,  s=4, c='dimgray',       alpha=0.8,  label='lidar hit',  zorder=1)
            ax.scatter(miss_x, miss_y, s=1, c='lightsteelblue', alpha=0.25, label='no return',  zorder=1)

            traj_xy = traj_local[:, :2]
            xs = np.concatenate([[0.0], traj_xy[:, 0]])
            ys = np.concatenate([[0.0], traj_xy[:, 1]])
            ax.plot(xs, ys, 'r-', linewidth=2.0, label='predicted', zorder=3)
            ax.scatter(traj_xy[:, 0], traj_xy[:, 1], s=18, c='red', zorder=4)

            ax.scatter([0], [0], s=80, c='black', marker='^', zorder=5, label='robot')
            ax.annotate('', xy=(0.4, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.scatter([gx_r], [gy_r], s=120, c='lime', marker='*',
                       zorder=5, label='goal', edgecolors='darkgreen', linewidths=0.8)

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x (m, forward)')
            ax.set_ylabel('y (m, left)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=7)
            ax.set_title(
                f'{self.policy_runner.model_type}  step={self._vis_counter}  '
                f'v={self.v:.2f}m/s  w={self.omega:.2f}rad/s',
                fontsize=9,
            )
            fig.tight_layout()
            fname = os.path.join(self.vis_dir, f'vis_step_{self._vis_counter:06d}.png')
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            self.get_logger().info(f'[vis] {fname}')
        except Exception as e:
            self.get_logger().warn(f'_save_policy_vis failed: {e}')

    # ── Stop / save ───────────────────────────────────────────────────────────

    def _publish_stop(self):
        self.v_cmd_filtered = self.w_cmd_filtered = 0.0
        self.v_cmd_prev     = self.w_cmd_prev     = 0.0
        self.cmd_pub.publish(Twist())

    def _save_trajectory_plot(self):
        if len(self.state_traj) < 2:
            return
        try:
            xs = [s[0] for s in self.state_traj]
            ys = [s[1] for s in self.state_traj]
            vs = [s[3] for s in self.state_traj]
            ws = [s[4] for s in self.state_traj]
            ts = np.arange(len(self.state_traj)) * 0.02

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            if self.latest_map is not None:
                m = self.latest_map
                w_cells = m.info.width
                h_cells = m.info.height
                res = m.info.resolution
                ox, oy = m.info.origin.position.x, m.info.origin.position.y
                grid = np.array(m.data, dtype=np.int8).reshape(h_cells, w_cells)
                rgba = np.ones((h_cells, w_cells, 4), dtype=float)
                rgba[grid < 0]  = [0.85, 0.85, 0.85, 1.0]
                rgba[(grid >= 0) & (grid < 50)] = [1.0, 1.0, 1.0, 1.0]
                rgba[grid >= 50] = [0.15, 0.15, 0.15, 1.0]
                ax1.imshow(rgba, extent=[ox, ox + w_cells * res, oy, oy + h_cells * res],
                           origin='lower', aspect='equal', zorder=0)

            ax1.plot(xs, ys, color='#2196F3', linewidth=2, label='Trajectory', zorder=3)
            ax1.plot(xs[0], ys[0], 'go', markersize=10, label='Start', zorder=5)
            ax1.plot(self.goal[0], self.goal[1], 'r*', markersize=14, label='Goal', zorder=5)
            ax1.set_title(f'Robot Trajectory ({self.policy_runner.model_type})')
            ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
            ax1.grid(True, alpha=0.3); ax1.axis('equal'); ax1.legend()

            ax2.plot(ts, vs); ax2.axhline(self.v_limit, color='r', linestyle='--', alpha=0.6)
            ax2.set_title('Linear Velocity'); ax2.set_xlabel('Time (s)'); ax2.set_ylabel('v (m/s)')
            ax2.grid(True, alpha=0.3)

            ax3.plot(ts, ws); ax3.axhline(self.omega_limit, color='r', linestyle='--', alpha=0.6)
            ax3.axhline(-self.omega_limit, color='r', linestyle='--', alpha=0.6)
            ax3.set_title('Angular Velocity'); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('w (rad/s)')
            ax3.grid(True, alpha=0.3)

            if self.control_command_history:
                cmd_t = [c['timestamp'] for c in self.control_command_history]
                cmd_v = [c['v_cmd']     for c in self.control_command_history]
                t0 = cmd_t[0]
                ax4.plot([t - t0 for t in cmd_t], cmd_v, label='v_cmd')
                ax4.set_title('cmd_vel v'); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('v (m/s)')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(
                os.path.expanduser('~'),
                f'robot_trajectory_{self.policy_runner.model_type}_rw.png',
            )
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            self.get_logger().info(f'Trajectory plot saved to {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save trajectory plot: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PolicyPlannerRealWorld()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
