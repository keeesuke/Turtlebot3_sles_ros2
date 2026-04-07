#!/usr/bin/env python3
"""
Real-World Robot Data Recorder for NN Imitation Learning.

Differences from robot_data_recorder_mpc.py (simulation):
  - State source : TF2  map → base_footprint  (100 Hz sliding-window velocity)
  - LiDAR source : /scan  (real TurtleBot3 LDS-01)
  - No Gazebo, no obstacle positions, no occupancy grid
  - No automatic target detection — user stops with Ctrl+C
  - Session folder naming: session_YYYYMMDD_HHMMSS_REAL

Prerequisites (must be running before this node):
  1. turtlebot3_node         (/scan, TF odom→base_footprint)
  2. turtlebot3_cartographer (TF map→odom)
  3. MPPI planner or teleoperation (publishes /cmd_vel)

Usage:
  ros2 run turtlebot3_sles_data robot_data_recorder_real_world.py

Press Ctrl+C to stop recording and save all data.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import signal
import sys
import os
import time
import json
import numpy as np
from datetime import datetime

import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# State recording rate (max). TF2 timer fires at 100 Hz; rate-limit writes.
STATE_WRITE_INTERVAL_SEC  = 0.02   # 50 Hz max
# LiDAR recording rate (max). Full scan at every write would be ~100 KB/s.
LIDAR_WRITE_INTERVAL_SEC  = 0.10   # 10 Hz max
# Velocity estimation
TF2_SLIDING_WINDOW_SEC    = 0.30   # 300 ms window
TF2_EMA_ALPHA             = 0.5    # EMA smoothing factor
TF2_CHANGE_THRESHOLD      = 0.0003 # 0.3 mm — ignore stale transforms


# ---------------------------------------------------------------------------
# Incremental JSONL writer (same pattern as robot_data_recorder_mpc.py)
# ---------------------------------------------------------------------------
class DataWriter:
    """Writes each sample to a separate line in a JSONL file (crash-safe)."""

    def __init__(self, data_dir: str, timestamp: str, logger):
        self.data_dir  = data_dir
        self.timestamp = timestamp
        self.logger    = logger

        self._states_f   = None
        self._controls_f = None
        self._lidar_f    = None

        self.state_count   = 0
        self.control_count = 0
        self.lidar_count   = 0

        self._state_last_write = 0.0
        self._lidar_last_write = 0.0

    def open(self):
        """Open JSONL files for writing."""
        self._states_f   = open(os.path.join(self.data_dir, f'robot_states_{self.timestamp}.jsonl'),   'w')
        self._controls_f = open(os.path.join(self.data_dir, f'control_inputs_{self.timestamp}.jsonl'), 'w')
        self._lidar_f    = open(os.path.join(self.data_dir, f'lidar_scans_{self.timestamp}.jsonl'),    'w')
        self.logger.info(f'JSONL files opened in {self.data_dir}')

    def write_state(self, x, y, yaw, v, omega, ts):
        now = time.time()
        if now - self._state_last_write < STATE_WRITE_INTERVAL_SEC:
            return
        self._state_last_write = now
        entry = {
            'timestamp':        ts,
            'position':         [x, y, 0.0],
            'orientation':      [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
            'yaw':              yaw,
            'linear_velocity':  [v,     0.0, 0.0],
            'angular_velocity': [0.0,   0.0, omega],
        }
        self._states_f.write(json.dumps(entry) + '\n')
        self._states_f.flush()
        self.state_count += 1

    def write_control(self, v_cmd, w_cmd, ts):
        entry = {
            'timestamp':  ts,
            'linear_x':   v_cmd,
            'linear_y':   0.0,
            'linear_z':   0.0,
            'angular_x':  0.0,
            'angular_y':  0.0,
            'angular_z':  w_cmd,
        }
        self._controls_f.write(json.dumps(entry) + '\n')
        self._controls_f.flush()
        self.control_count += 1

    def write_lidar(self, ranges, angle_min, angle_max, angle_increment,
                    range_min, range_max, ts):
        now = time.time()
        if now - self._lidar_last_write < LIDAR_WRITE_INTERVAL_SEC:
            return
        self._lidar_last_write = now
        entry = {
            'timestamp':       ts,
            'ranges':          list(ranges),
            'angle_min':       angle_min,
            'angle_max':       angle_max,
            'angle_increment': angle_increment,
            'range_min':       range_min,
            'range_max':       range_max,
        }
        self._lidar_f.write(json.dumps(entry) + '\n')
        self._lidar_f.flush()
        self.lidar_count += 1

    def close(self):
        for f in (self._states_f, self._controls_f, self._lidar_f):
            try:
                if f:
                    f.flush()
                    f.close()
            except Exception:
                pass

    def finalize(self):
        """Close JSONL files, then convert to JSON-array + NPZ format."""
        self.close()
        self.logger.info(
            f'Recorded — states: {self.state_count}, '
            f'controls: {self.control_count}, lidar: {self.lidar_count}'
        )
        self._convert_to_final_formats()

    # ── Conversion helpers ──────────────────────────────────────────────────

    def _load_jsonl(self, filename):
        path = os.path.join(self.data_dir, filename)
        records = []
        if not os.path.exists(path):
            self.logger.warn(f'File not found: {path}')
            return records
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warn(f'JSON parse error: {e}')
        return records

    def _convert_to_final_formats(self):
        self.logger.info('Converting JSONL → JSON arrays + NPZ …')
        ts = self.timestamp

        states   = self._load_jsonl(f'robot_states_{ts}.jsonl')
        controls = self._load_jsonl(f'control_inputs_{ts}.jsonl')
        lidars   = self._load_jsonl(f'lidar_scans_{ts}.jsonl')

        self.logger.info(
            f'Loaded — states: {len(states)}, controls: {len(controls)}, lidar: {len(lidars)}'
        )

        # ── Save JSON arrays (compatible with existing tools) ─────────────
        def _save_json(name, data_dict):
            path = os.path.join(self.data_dir, name)
            with open(path, 'w') as f:
                json.dump(data_dict, f, indent=2)
            self.logger.info(f'  ✓ {name}')

        state_ts   = [s['timestamp'] for s in states]
        control_ts = [c['timestamp'] for c in controls]
        lidar_ts   = [l['timestamp'] for l in lidars]

        _save_json(f'robot_states_{ts}.json',   {
            'robot_states': states,
            'robot_state_timestamps': state_ts,
            'target_position': [0.0, 0.0, 0.0],   # post-hoc goal; filled by prepare script
            'tolerance': 0.15,
            'obstacle_positions': {},
        })
        _save_json(f'control_inputs_{ts}.json', {
            'control_inputs': controls,
            'control_input_timestamps': control_ts,
            'target_position': [0.0, 0.0, 0.0],
            'tolerance': 0.15,
            'obstacle_positions': {},
        })
        _save_json(f'lidar_scans_{ts}.json', {
            'lidar_scans': lidars,
            'lidar_timestamps': lidar_ts,
            'target_position': [0.0, 0.0, 0.0],
            'tolerance': 0.15,
            'obstacle_positions': {},
        })

        # ── Save main NPZ (raw; mirrors robot_data_recorder_mpc.py format) ─
        positions    = np.array([[s['position'][0], s['position'][1]] for s in states]) \
                       if states else np.zeros((0, 2))
        orientations = np.array([s['yaw'] for s in states]) \
                       if states else np.zeros(0)
        lin_vels     = np.array([s['linear_velocity'][0]  for s in states]) \
                       if states else np.zeros(0)
        ang_vels     = np.array([s['angular_velocity'][2] for s in states]) \
                       if states else np.zeros(0)
        ctrl_lin     = np.array([c['linear_x']  for c in controls]) \
                       if controls else np.zeros(0)
        ctrl_ang     = np.array([c['angular_z'] for c in controls]) \
                       if controls else np.zeros(0)

        if lidars:
            lidar_ranges_list      = [np.array(l['ranges'])         for l in lidars]
            lidar_angle_mins       = np.array([l['angle_min']       for l in lidars])
            lidar_angle_maxs       = np.array([l['angle_max']       for l in lidars])
            lidar_angle_increments = np.array([l['angle_increment'] for l in lidars])
            lidar_range_mins       = np.array([l['range_min']       for l in lidars])
            lidar_range_maxs       = np.array([l['range_max']       for l in lidars])
        else:
            lidar_ranges_list      = []
            lidar_angle_mins       = np.zeros(0)
            lidar_angle_maxs       = np.zeros(0)
            lidar_angle_increments = np.zeros(0)
            lidar_range_mins       = np.zeros(0)
            lidar_range_maxs       = np.zeros(0)

        npz_path = os.path.join(self.data_dir, f'robot_data_{ts}.npz')
        np.savez(npz_path,
                 positions=positions,
                 orientations=orientations,
                 linear_velocities=lin_vels,
                 angular_velocities=ang_vels,
                 control_linear=ctrl_lin,
                 control_angular=ctrl_ang,
                 robot_state_timestamps=np.array(state_ts),
                 control_input_timestamps=np.array(control_ts),
                 lidar_timestamps=np.array(lidar_ts),
                 lidar_angle_mins=lidar_angle_mins,
                 lidar_angle_maxs=lidar_angle_maxs,
                 lidar_angle_increments=lidar_angle_increments,
                 lidar_range_mins=lidar_range_mins,
                 lidar_range_maxs=lidar_range_maxs,
                 target_position=np.array([0.0, 0.0, 0.0]),
                 tolerance=0.15,
                 obstacle_positions=np.zeros((0, 3)),
                 obstacle_names=np.array([], dtype=object))
        with open(npz_path, 'r+b') as f:
            f.flush()
            os.fsync(f.fileno())
        self.logger.info(f'  ✓ robot_data_{ts}.npz  ({os.path.getsize(npz_path)} bytes)')

        # ── Save lidar ranges separately ──────────────────────────────────
        if lidar_ranges_list:
            lr_path = os.path.join(self.data_dir, f'lidar_ranges_{ts}.npz')
            lr_dict = {f'scan_{i}': r for i, r in enumerate(lidar_ranges_list)}
            np.savez(lr_path, **lr_dict, num_scans=len(lidar_ranges_list))
            with open(lr_path, 'r+b') as f:
                f.flush()
                os.fsync(f.fileno())
            self.logger.info(f'  ✓ lidar_ranges_{ts}.npz  ({os.path.getsize(lr_path)} bytes)')

        # ── Write session info ────────────────────────────────────────────
        info = {
            'session_type':         'REAL',
            'timestamp':            ts,
            'total_robot_states':   len(states),
            'total_control_inputs': len(controls),
            'total_lidar_scans':    len(lidars),
            'start_position':       positions[0].tolist()  if len(positions) > 0 else [0, 0],
            'end_position':         positions[-1].tolist() if len(positions) > 0 else [0, 0],
            'goal_position':        positions[-1].tolist() if len(positions) > 0 else [0, 0],
            'note':                 'goal_position is post-hoc; set from final frame by prepare_training_data_real_world.py',
        }
        info_path = os.path.join(self.data_dir, f'session_info_{ts}.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        self.logger.info(f'  ✓ session_info_{ts}.json')

        self.logger.info('✅ Conversion complete.')


# ---------------------------------------------------------------------------
# Main recorder node
# ---------------------------------------------------------------------------
class RealWorldDataRecorder(Node):

    def __init__(self):
        super().__init__('real_world_data_recorder')

        # ── Session folder ────────────────────────────────────────────────
        base_dir = os.path.join(os.path.expanduser('~'), 'robot_data')
        os.makedirs(base_dir, exist_ok=True)
        self._ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.data_dir = os.path.join(base_dir, f'session_{self._ts}_REAL')
        os.makedirs(self.data_dir, exist_ok=True)

        # ── TF2 ──────────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Velocity estimation state (mirror planner_nn_real_world.py) ──
        self._pose_history = []           # [(t_sec, x, y, theta), ...]
        self._x_prev = self._y_prev = self._th_prev = None
        self._v   = 0.0
        self._omega = 0.0
        self._x = self._y = self._theta = 0.0

        # ── Recording state ───────────────────────────────────────────────
        self._recording  = True
        self._finalizing = False

        # ── Data writer ───────────────────────────────────────────────────
        self._writer = DataWriter(self.data_dir, self._ts, self.get_logger())
        self._writer.open()

        # ── Subscriptions ─────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, '/scan', self._lidar_cb, scan_qos)
        self.create_subscription(Twist,     '/cmd_vel', self._cmdvel_cb,  10)

        # ── Timer: poll TF2 at 100 Hz for state ──────────────────────────
        self.create_timer(0.01, self._state_timer_cb)

        # ── Signal handler ────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Real-World Data Recorder started')
        self.get_logger().info(f'  Session folder : {self.data_dir}')
        self.get_logger().info('  State source   : TF2 map→base_footprint')
        self.get_logger().info('  LiDAR source   : /scan')
        self.get_logger().info('  Control source : /cmd_vel')
        self.get_logger().info('Press Ctrl+C to stop recording and save data.')
        self.get_logger().info('=' * 60)

    # ── TF2 state timer ─────────────────────────────────────────────────────

    def _state_timer_cb(self):
        if not self._recording:
            return
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time()
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return

        x_new  = t.transform.translation.x
        y_new  = t.transform.translation.y
        q      = t.transform.rotation
        _, _, th_new = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Ignore stale (duplicate) transforms
        if self._x_prev is not None:
            pos_changed = (
                abs(x_new  - self._x_prev)  > TF2_CHANGE_THRESHOLD
                or abs(y_new  - self._y_prev)  > TF2_CHANGE_THRESHOLD
                or abs(((th_new - self._th_prev) + np.pi) % (2 * np.pi) - np.pi) > TF2_CHANGE_THRESHOLD
            )
            if not pos_changed:
                return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self._pose_history.append((now_sec, x_new, y_new, th_new))

        # Drop entries older than sliding window
        cutoff = now_sec - TF2_SLIDING_WINDOW_SEC
        while self._pose_history and self._pose_history[0][0] < cutoff:
            self._pose_history.pop(0)

        # Estimate velocity from window
        if len(self._pose_history) >= 2:
            t0h, x0h, y0h, th0h = self._pose_history[0]
            dt_win = now_sec - t0h
            if dt_win >= 0.08:
                vx_w  = (x_new  - x0h) / dt_win
                vy_w  = (y_new  - y0h) / dt_win
                dth   = ((th_new - th0h) + np.pi) % (2 * np.pi) - np.pi
                w_w   = dth / dt_win
                v_body = vx_w * np.cos(th_new) + vy_w * np.sin(th_new)
                self._v     = TF2_EMA_ALPHA * v_body + (1.0 - TF2_EMA_ALPHA) * self._v
                self._omega = TF2_EMA_ALPHA * w_w    + (1.0 - TF2_EMA_ALPHA) * self._omega

        self._x, self._y, self._theta = x_new, y_new, th_new
        self._x_prev, self._y_prev, self._th_prev = x_new, y_new, th_new

        self._writer.write_state(
            self._x, self._y, self._theta,
            self._v, self._omega,
            now_sec
        )

    # ── LiDAR callback ──────────────────────────────────────────────────────

    def _lidar_cb(self, msg: LaserScan):
        if not self._recording:
            return
        ts = self.get_clock().now().nanoseconds * 1e-9
        self._writer.write_lidar(
            list(msg.ranges),
            msg.angle_min, msg.angle_max,
            msg.angle_increment,
            msg.range_min, msg.range_max,
            ts,
        )

    # ── cmd_vel callback ─────────────────────────────────────────────────────

    def _cmdvel_cb(self, msg: Twist):
        if not self._recording:
            return
        ts = self.get_clock().now().nanoseconds * 1e-9
        self._writer.write_control(msg.linear.x, msg.angular.z, ts)

    # ── Signal handler ────────────────────────────────────────────────────────

    def _signal_handler(self, signum, frame):
        if self._finalizing:
            return
        self._finalizing = True
        self._recording  = False

        self.get_logger().info('')
        self.get_logger().info('🛑 Received stop signal. Saving data …')
        try:
            self._writer.finalize()
            self.get_logger().info(f'✅ Data saved to: {self.data_dir}')
        except Exception as e:
            self.get_logger().error(f'Error saving data: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = RealWorldDataRecorder()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
