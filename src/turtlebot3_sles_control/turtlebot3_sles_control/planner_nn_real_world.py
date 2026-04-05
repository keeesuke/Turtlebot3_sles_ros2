#!/usr/bin/env python3
"""
ROS2 Neural Network Planner for TurtleBot3 — Real World
Companion to planner_haa_real_world.py (HAA/MPPI).

State estimation : TF2 map → base_footprint  (100 Hz sliding-window)
LiDAR input      : /scan  (real TurtleBot3 LDS-01, clipped to training range)
Goal input       : /move_base_simple/goal  (RViz2 '2D Goal Pose')
Control output   : /cmd_vel  (50 Hz NN inference)

Key differences from simulation planner (planner_nn.py):
  - /scan  instead of /simulated_scan
    Real LiDAR uses float('inf') for no-return; simulation uses -1.
    Both are remapped to lidar_max_range before feeding the network.
  - TF2 map→base_footprint  instead of /gazebo/model_states
    Velocity is estimated from a 300 ms sliding pose window (same as MPC).
  - /move_base_simple/goal subscription for live goal input from RViz2,
    rather than a fixed goal ROS parameter.
  - Stuck-detection: resets goal if robot makes no progress for 10 s.

Prerequisites (must already be running):
  1. turtlebot3_node       (hw driver — /scan, /tf odom→base_footprint)
  2. turtlebot3_cartographer  (SLAM — /tf map→odom)
  3. rviz2                 (optional — for interactive goal input)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# NN architecture — must match the architecture used during training
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Multi-Layer Perceptron matching training architecture."""
    def __init__(self, input_dim=364, hidden_dims=None, output_dim=2, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------
class NNNavigationNodeRealWorld(Node):
    def __init__(self):
        super().__init__('nn_planner_real_world')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('v_limit_haa',      0.2)   # m/s  (leave margin below 0.26)
        self.declare_parameter('omega_limit_haa',  0.9)   # rad/s
        self.declare_parameter('robot_radius',     0.15)  # m  (real-world footprint)
        # NN was trained with 1.0 m max-range simulated LiDAR.
        # Real LiDAR readings beyond this distance carry no useful information
        # for the model, so clip to this value before inference.
        self.declare_parameter('lidar_max_range',  1.0)   # m
        self.declare_parameter('trajectory_path',
                               os.path.join(os.path.expanduser('~'),
                                            'robot_trajectory_nn_rw.png'))

        self.v_limit_haa     = self.get_parameter('v_limit_haa').value
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value
        self.robot_radius    = self.get_parameter('robot_radius').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.trajectory_path = self.get_parameter('trajectory_path').value

        # ── State flags ─────────────────────────────────────────────────────
        self.state_ready   = False
        self.lidar_ready   = False
        self.goal_received = False
        self._goal_wait_logged = False
        self.target_reached    = False
        self.shutdown_requested = False

        # ── Robot state (updated by state_update_cb at 100 Hz) ──────────────
        self.x = self.y = self.theta = self.v = self.omega = 0.0
        self._x_prev = self._y_prev = self._th_prev = None
        self._pose_history = []          # [(t_sec, x, y, theta), ...]

        # ── Latest LiDAR scan (360 rays, 0…lidar_max_range m) ───────────────
        self.latest_lidar_scan: np.ndarray | None = None

        # ── Goal state [x, y, theta, v, w] ──────────────────────────────────
        self.goal = [0.0, 0.0, 0.0, 0.0, 0.0]

        # ── Stuck detection ──────────────────────────────────────────────────
        self._stuck_timeout_sec   = 10.0   # s without progress → abort goal
        self._stuck_threshold_m   = 0.10   # m — minimum movement = "progress"
        self._stuck_last_prog_time = None
        self._stuck_last_prog_pos  = None

        # ── Trajectory / timing logs ─────────────────────────────────────────
        self.state_traj             = []
        self.control_command_history = []
        self.command_timings        = []
        self.command_count          = 0
        self.timing_log_interval    = 100

        # ── Low-pass filter state ────────────────────────────────────────────
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0

        # ── Load NN model ────────────────────────────────────────────────────
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f'NN model not found at {model_path}')
            raise FileNotFoundError(f'NN model not found at {model_path}')

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.nn_model = MLP(input_dim=364, hidden_dims=[256, 128, 64],
                            output_dim=2, dropout=0.1)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Wrapped checkpoint: {'model_state_dict': OrderedDict, ...}
            self.nn_model.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info(f'NN model loaded from wrapped checkpoint: {model_path}')
        elif isinstance(checkpoint, dict):
            # Raw state dict saved with torch.save(model.state_dict(), path)
            self.nn_model.load_state_dict(checkpoint)
            self.get_logger().info(f'NN model loaded from state dict: {model_path}')
        else:
            # Full model saved with torch.save(model, path)
            self.nn_model = checkpoint
            self.get_logger().info(f'NN model loaded (full model object): {model_path}')
        self.nn_model.eval()

        # ── TF2 ──────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── ROS pubs / subs ──────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(LaserScan,    '/scan',                   self.lidar_cb,        10)
        self.create_subscription(PoseStamped,  '/move_base_simple/goal',  self.goal_cb,         10)

        # ── Timers ───────────────────────────────────────────────────────────
        self.state_timer   = self.create_timer(0.01, self.state_update_cb)  # 100 Hz
        self.control_timer = self.create_timer(0.02, self.control_loop)     #  50 Hz

        self.get_logger().info('Real-world NN planner node started:')
        self.get_logger().info('  - State source  : TF2 map→base_footprint (100 Hz sliding window)')
        self.get_logger().info('  - LiDAR source  : /scan (real TurtleBot3 LDS-01)')
        self.get_logger().info('  - Goal source   : /move_base_simple/goal (RViz2 2D Goal Pose)')
        self.get_logger().info('  - Control rate  : 50 Hz NN inference')
        self.get_logger().info(f'  - lidar_max_range clipped to {self.lidar_max_range} m (training range)')
        self.get_logger().info('Waiting for TF2 (map→base_footprint) and /scan ...')

    # ── LiDAR callback ──────────────────────────────────────────────────────

    def lidar_cb(self, msg: LaserScan):
        """Convert real LiDAR → 360-element float32 array in [0, lidar_max_range]."""
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            max_r  = self.lidar_max_range

            # Real LiDAR uses float('inf') / nan for no-return or out-of-range.
            # Simulated scanner used -1 for the same condition.
            # Both map to max_r before feeding the network.
            bad_mask = ~np.isfinite(ranges) | (ranges <= 0.0)
            ranges[bad_mask] = max_r

            # Clip anything beyond the training max range.
            np.clip(ranges, 0.0, max_r, out=ranges)

            # Resample to exactly 360 rays if the sensor has a different count.
            n = len(ranges)
            if n != 360:
                indices = np.linspace(0, n - 1, 360)
                ranges  = np.interp(indices, np.arange(n), ranges).astype(np.float32)

            self.latest_lidar_scan = ranges
            self.lidar_ready = True
        except Exception as e:
            self.get_logger().warn(f'lidar_cb error: {e}')

    # ── TF2 state estimation ─────────────────────────────────────────────────

    def state_update_cb(self):
        """Look up map→base_footprint TF at 100 Hz; estimate v, omega from 300 ms window."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
            x_new  = t.transform.translation.x
            y_new  = t.transform.translation.y
            q      = t.transform.rotation
            _, _, th_new = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # Only add a new entry when TF2 returned a genuinely new transform.
            # Cartographer publishes map→odom at ~10-50 Hz; our timer is 100 Hz.
            # Treating a stale (duplicate) pose as a fresh "v=0" measurement
            # would incorrectly drive the EMA velocity toward zero.
            TF2_CHANGE_THRESHOLD = 0.0003   # 0.3 mm
            pos_changed = (
                self._x_prev is None
                or abs(x_new  - self._x_prev)  > TF2_CHANGE_THRESHOLD
                or abs(y_new  - self._y_prev)  > TF2_CHANGE_THRESHOLD
                or abs(((th_new - self._th_prev) + np.pi) % (2 * np.pi) - np.pi) > 0.0003
            )

            if pos_changed:
                now_sec = self.get_clock().now().nanoseconds * 1e-9
                self._pose_history.append((now_sec, x_new, y_new, th_new))

                # Drop entries older than 300 ms
                cutoff = now_sec - 0.30
                while self._pose_history and self._pose_history[0][0] < cutoff:
                    self._pose_history.pop(0)

                # Need ≥80 ms of data to compute a stable velocity estimate
                if len(self._pose_history) >= 2:
                    t0h, x0h, y0h, th0h = self._pose_history[0]
                    dt_win = now_sec - t0h
                    if dt_win >= 0.08:
                        vx_w  = (x_new  - x0h)  / dt_win
                        vy_w  = (y_new  - y0h)  / dt_win
                        dth   = ((th_new - th0h) + np.pi) % (2 * np.pi) - np.pi
                        w_w   = dth / dt_win
                        # Project world-frame velocity onto robot heading → body-frame forward speed
                        v_body = vx_w * np.cos(th_new) + vy_w * np.sin(th_new)
                        # EMA filter (only on fresh TF2 data — no artificial decay)
                        _a = 0.5
                        self.v     = _a * v_body + (1.0 - _a) * self.v
                        self.omega = _a * w_w    + (1.0 - _a) * self.omega

            self.x = x_new
            self.y = y_new
            self.theta = th_new
            self._x_prev  = x_new
            self._y_prev  = y_new
            self._th_prev = th_new
            self.state_ready = True

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass   # TF not available yet — keep waiting silently

    # ── Goal callback ─────────────────────────────────────────────────────────

    def goal_cb(self, msg: PoseStamped):
        """Receive a new goal from RViz2 '2D Goal Pose' button."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        q = msg.pose.orientation
        _, _, theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.goal           = [x, y, theta, 0.0, 0.0]
        self.goal_received  = True
        self.target_reached = False
        self.shutdown_requested = False
        self.state_traj     = []
        self._goal_wait_logged = False

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._stuck_last_prog_time = now_sec
        self._stuck_last_prog_pos  = (self.x, self.y)

        self.get_logger().info(
            f'New goal from RViz2: x={x:.3f} m, y={y:.3f} m, '
            f'theta={np.degrees(theta):.1f} deg'
        )

    # ── Helper ───────────────────────────────────────────────────────────────

    def _transform_goal_to_robot_frame(self, gx: float, gy: float):
        """Transform goal from map frame to robot body frame."""
        dx = gx - self.x
        dy = gy - self.y
        cos_t = np.cos(-self.theta)
        sin_t = np.sin(-self.theta)
        return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t

    # ── 50 Hz control loop ───────────────────────────────────────────────────

    def control_loop(self):
        """Run NN inference at 50 Hz and publish /cmd_vel."""
        if self.shutdown_requested or self.target_reached:
            return

        # Wait for sensors
        if not (self.state_ready and self.lidar_ready) or self.latest_lidar_scan is None:
            return

        # Wait for goal
        if not self.goal_received:
            if not self._goal_wait_logged:
                self.get_logger().info(
                    "Ready. Set a goal using '2D Goal Pose' button in RViz2 "
                    "(publishes to /move_base_simple/goal)."
                )
                self._goal_wait_logged = True
            return

        # ── Current state ────────────────────────────────────────────────────
        current_state = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(current_state.copy())

        # ── Goal-reached check ───────────────────────────────────────────────
        dist = np.linalg.norm(
            np.array(current_state[:2]) - np.array(self.goal[:2])
        )
        if dist < 0.10:
            self.get_logger().info('Target reached! Stopping.')
            self.target_reached = True
            self._log_timing_stats()
            self._save_trajectory_plot()
            self._publish_stop()
            self.control_timer.cancel()
            rclpy.shutdown()
            return

        # ── Stuck detection ──────────────────────────────────────────────────
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._stuck_last_prog_pos is not None:
            disp = np.hypot(self.x - self._stuck_last_prog_pos[0],
                            self.y - self._stuck_last_prog_pos[1])
            if disp >= self._stuck_threshold_m:
                self._stuck_last_prog_time = now_sec
                self._stuck_last_prog_pos  = (self.x, self.y)
            elif (self._stuck_last_prog_time is not None
                  and now_sec - self._stuck_last_prog_time > self._stuck_timeout_sec):
                self.get_logger().warn(
                    f'Robot stuck for >{self._stuck_timeout_sec:.0f} s at '
                    f'({self.x:.2f}, {self.y:.2f}). Waiting for new goal.'
                )
                self._publish_stop()
                self.goal_received      = False
                self._goal_wait_logged  = False
                return

        # ── NN inference ─────────────────────────────────────────────────────
        goal_x_r, goal_y_r = self._transform_goal_to_robot_frame(
            self.goal[0], self.goal[1]
        )

        try:
            t0  = time.time()
            inp = np.concatenate(
                [[self.v, self.omega], [goal_x_r, goal_y_r], self.latest_lidar_scan],
                dtype=np.float32
            )
            tensor = torch.FloatTensor(inp).unsqueeze(0)
            with torch.no_grad():
                out = self.nn_model(tensor).cpu().numpy().flatten()

            v_cmd = float(np.clip(out[0], 0.0,                  self.v_limit_haa))
            w_cmd = float(np.clip(out[1], -self.omega_limit_haa, self.omega_limit_haa))
            self.v_cmd_filtered = v_cmd
            self.w_cmd_filtered = w_cmd

            dt_ms = (time.time() - t0) * 1000.0
            self.command_timings.append(dt_ms)
            self.command_count += 1
            if self.command_count % self.timing_log_interval == 0:
                recent = self.command_timings[-self.timing_log_interval:]
                self.get_logger().info(
                    f'NN timing (last {self.timing_log_interval}): '
                    f'avg={np.mean(recent):.2f} ms  '
                    f'min={np.min(recent):.2f} ms  '
                    f'max={np.max(recent):.2f} ms'
                )
        except Exception as e:
            self.get_logger().warn(f'NN inference error: {e}')
            self._publish_stop()
            return

        # ── Publish ──────────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = self.v_cmd_filtered
        twist.angular.z = self.w_cmd_filtered
        self.cmd_pub.publish(twist)

        self.control_command_history.append({
            'timestamp': now_sec,
            'v_cmd':     v_cmd,
            'w_cmd':     w_cmd,
            'state':     current_state.copy(),
        })

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _publish_stop(self):
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.cmd_pub.publish(Twist())

    def _log_timing_stats(self):
        if self.command_timings:
            self.get_logger().info(
                f'Final NN timing: '
                f'avg={np.mean(self.command_timings):.2f} ms  '
                f'min={np.min(self.command_timings):.2f} ms  '
                f'max={np.max(self.command_timings):.2f} ms'
            )

    def _save_trajectory_plot(self):
        if len(self.state_traj) < 2:
            return
        try:
            xs = np.array([s[0] for s in self.state_traj])
            ys = np.array([s[1] for s in self.state_traj])
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(xs,       ys,       'b-', linewidth=2,  label='Trajectory')
            ax.plot(xs[0],    ys[0],    'go', markersize=10, label='Start')
            ax.plot(self.goal[0], self.goal[1], 'r*', markersize=16, label='Goal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('NN Real-World Planner Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            plt.tight_layout()
            plt.savefig(self.trajectory_path, dpi=150)
            plt.close(fig)
            self.get_logger().info(f'Trajectory plot saved to {self.trajectory_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save trajectory plot: {e}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = NNNavigationNodeRealWorld()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
