#!/usr/bin/env python3
"""
Experiment Recorder for the three control logics (MPC / NN / Switching).

Records the same data streams as robot_data_recorder_real_world.py PLUS a
per-run summary of parameters and performance metrics. Saves everything to
~/robot_data/experiments/<TIMESTAMP>_<LOGIC>/.

Usage:
    ros2 run turtlebot3_sles_data experiment_recorder.py --logic mpc
    ros2 run turtlebot3_sles_data experiment_recorder.py --logic nn
    ros2 run turtlebot3_sles_data experiment_recorder.py --logic switch

Workflow (one run, one terminal each):
    Term A: this script  (creates the run folder, writes pointer file)
    Term B: ros2 launch turtlebot3_sles_control turtlebot3_planner_*.launch.py
    Drive / set goal in RViz2.  When goal reached or you want to stop,
    Ctrl+C THIS recorder.  Plot images saved by the planner are
    automatically directed into the run folder via the pointer file.

Pointer file (so planners know where to save their plots):
    ~/robot_data/experiments/.current_run  → contains absolute path of
    current run folder.  Created on start, removed on Ctrl+C.

Saved files (in <RUN_FOLDER>/):
    robot_states_<ts>.{jsonl,json}            pose + twist time-series
    control_inputs_<ts>.{jsonl,json}          /cmd_vel time-series
    lidar_scans_<ts>.{jsonl,json}             /scan time-series
    robot_data_<ts>.npz                       merged numpy arrays
    lidar_ranges_<ts>.npz                     raw scan arrays
    session_info_<ts>.json                    same as recorder
    experiment_summary.json                   ← extra: metrics + params
    robot_trajectory*.png                     written by the planner (if any)
"""

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


# ---------------------------------------------------------------------------
# Constants — match robot_data_recorder_real_world.py
# ---------------------------------------------------------------------------
STATE_WRITE_INTERVAL_SEC = 0.02   # 50 Hz max
LIDAR_WRITE_INTERVAL_SEC = 0.10   # 10 Hz max

EXPERIMENTS_ROOT = os.path.expanduser('~/robot_data/experiments')
POINTER_FILE     = os.path.join(EXPERIMENTS_ROOT, '.current_run')

VALID_LOGICS = {'mpc', 'nn', 'switch'}


# ---------------------------------------------------------------------------
# JSONL writer — identical to data recorder
# ---------------------------------------------------------------------------
class DataWriter:
    def __init__(self, data_dir, timestamp, logger):
        self.data_dir  = data_dir
        self.timestamp = timestamp
        self.logger    = logger
        self._states_f = self._controls_f = self._lidar_f = None
        self.state_count = self.control_count = self.lidar_count = 0
        self._state_last_write = self._lidar_last_write = 0.0

    def open(self):
        ts = self.timestamp
        self._states_f   = open(os.path.join(self.data_dir, f'robot_states_{ts}.jsonl'),   'w')
        self._controls_f = open(os.path.join(self.data_dir, f'control_inputs_{ts}.jsonl'), 'w')
        self._lidar_f    = open(os.path.join(self.data_dir, f'lidar_scans_{ts}.jsonl'),    'w')

    def write_state(self, x, y, yaw, v, omega, ts):
        now = time.time()
        if now - self._state_last_write < STATE_WRITE_INTERVAL_SEC:
            return
        self._state_last_write = now
        entry = {
            'timestamp':       ts,
            'position':        [x, y, 0.0],
            'orientation':     [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
            'yaw':             yaw,
            'linear_velocity': [v, 0.0, 0.0],
            'angular_velocity':[0.0, 0.0, omega],
        }
        self._states_f.write(json.dumps(entry) + '\n')
        self._states_f.flush()
        self.state_count += 1

    def write_control(self, v_cmd, w_cmd, ts):
        entry = {
            'timestamp': ts,
            'linear_x': v_cmd, 'linear_y': 0.0, 'linear_z': 0.0,
            'angular_x': 0.0,  'angular_y': 0.0, 'angular_z': w_cmd,
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

    def _load_jsonl(self, filename):
        path = os.path.join(self.data_dir, filename)
        records = []
        if not os.path.exists(path):
            return records
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def finalize(self, window: tuple | None = None):
        """Convert JSONL → JSON + NPZ.

        `window`: optional (t_start, t_end) seconds. If given, only records with
        timestamps in [t_start, t_end] are written to the JSON + NPZ outputs.
        Raw JSONL files always contain the full stream for debugging.

        Returns dict of windowed arrays for summary computation.
        """
        self.close()
        ts = self.timestamp
        all_states   = self._load_jsonl(f'robot_states_{ts}.jsonl')
        all_controls = self._load_jsonl(f'control_inputs_{ts}.jsonl')
        all_lidars   = self._load_jsonl(f'lidar_scans_{ts}.jsonl')

        # ── Apply window filter ──────────────────────────────────────────
        if window is not None:
            t0, t1 = window
            states   = [s for s in all_states   if t0 <= s['timestamp'] <= t1]
            controls = [c for c in all_controls if t0 <= c['timestamp'] <= t1]
            lidars   = [l for l in all_lidars   if t0 <= l['timestamp'] <= t1]
        else:
            states, controls, lidars = all_states, all_controls, all_lidars

        # JSON arrays (windowed)
        for name, payload in [
            (f'robot_states_{ts}.json',   {'robot_states': states,
                                           'robot_state_timestamps': [s['timestamp'] for s in states],
                                           'target_position': [0.0, 0.0, 0.0],
                                           'tolerance': 0.15, 'obstacle_positions': {}}),
            (f'control_inputs_{ts}.json', {'control_inputs': controls,
                                           'control_input_timestamps': [c['timestamp'] for c in controls],
                                           'target_position': [0.0, 0.0, 0.0],
                                           'tolerance': 0.15, 'obstacle_positions': {}}),
            (f'lidar_scans_{ts}.json',    {'lidar_scans': lidars,
                                           'lidar_timestamps': [l['timestamp'] for l in lidars],
                                           'target_position': [0.0, 0.0, 0.0],
                                           'tolerance': 0.15, 'obstacle_positions': {}}),
        ]:
            with open(os.path.join(self.data_dir, name), 'w') as f:
                json.dump(payload, f, indent=2)

        # Main NPZ
        positions    = np.array([[s['position'][0], s['position'][1]] for s in states]) if states else np.zeros((0, 2))
        orientations = np.array([s['yaw']                              for s in states]) if states else np.zeros(0)
        lin_vels     = np.array([s['linear_velocity'][0]               for s in states]) if states else np.zeros(0)
        ang_vels     = np.array([s['angular_velocity'][2]              for s in states]) if states else np.zeros(0)
        ctrl_lin     = np.array([c['linear_x']                         for c in controls]) if controls else np.zeros(0)
        ctrl_ang     = np.array([c['angular_z']                        for c in controls]) if controls else np.zeros(0)

        if lidars:
            lidar_ranges_list      = [np.array(l['ranges']) for l in lidars]
            lidar_angle_mins       = np.array([l['angle_min']       for l in lidars])
            lidar_angle_maxs       = np.array([l['angle_max']       for l in lidars])
            lidar_angle_increments = np.array([l['angle_increment'] for l in lidars])
            lidar_range_mins       = np.array([l['range_min']       for l in lidars])
            lidar_range_maxs       = np.array([l['range_max']       for l in lidars])
        else:
            lidar_ranges_list = []
            lidar_angle_mins = lidar_angle_maxs = np.zeros(0)
            lidar_angle_increments = lidar_range_mins = lidar_range_maxs = np.zeros(0)

        npz_path = os.path.join(self.data_dir, f'robot_data_{ts}.npz')
        np.savez(npz_path,
                 positions=positions, orientations=orientations,
                 linear_velocities=lin_vels, angular_velocities=ang_vels,
                 control_linear=ctrl_lin, control_angular=ctrl_ang,
                 robot_state_timestamps=np.array([s['timestamp'] for s in states]),
                 control_input_timestamps=np.array([c['timestamp'] for c in controls]),
                 lidar_timestamps=np.array([l['timestamp'] for l in lidars]),
                 lidar_angle_mins=lidar_angle_mins,
                 lidar_angle_maxs=lidar_angle_maxs,
                 lidar_angle_increments=lidar_angle_increments,
                 lidar_range_mins=lidar_range_mins,
                 lidar_range_maxs=lidar_range_maxs,
                 target_position=np.array([0.0, 0.0, 0.0]),
                 tolerance=0.15,
                 obstacle_positions=np.zeros((0, 3)),
                 obstacle_names=np.array([], dtype=object))

        if lidar_ranges_list:
            lr_path = os.path.join(self.data_dir, f'lidar_ranges_{ts}.npz')
            lr_dict = {f'scan_{i}': r for i, r in enumerate(lidar_ranges_list)}
            np.savez(lr_path, **lr_dict, num_scans=len(lidar_ranges_list))

        # Session info (compatible with prepare script)
        session_info = {
            'session_type':         'EXPERIMENT',
            'timestamp':            ts,
            'total_robot_states':   len(states),
            'total_control_inputs': len(controls),
            'total_lidar_scans':    len(lidars),
            'start_position':       positions[0].tolist()  if len(positions) > 0 else [0, 0],
            'end_position':         positions[-1].tolist() if len(positions) > 0 else [0, 0],
            'goal_position':        positions[-1].tolist() if len(positions) > 0 else [0, 0],
        }
        with open(os.path.join(self.data_dir, f'session_info_{ts}.json'), 'w') as f:
            json.dump(session_info, f, indent=2)

        return {
            'positions':    positions,
            'lin_vels':     lin_vels,
            'ang_vels':     ang_vels,
            'ctrl_lin':     ctrl_lin,
            'ctrl_ang':     ctrl_ang,
            'state_ts':     np.array([s['timestamp'] for s in states]),
            'control_ts':   np.array([c['timestamp'] for c in controls]),
            'lidar_count':  len(lidars),
        }


# ---------------------------------------------------------------------------
# Experiment recorder node
# ---------------------------------------------------------------------------
class ExperimentRecorder(Node):
    def __init__(self, logic: str, goal_xy: tuple | None = None):
        super().__init__('experiment_recorder')

        self.logic   = logic
        self.goal_xy = goal_xy

        # ── Run folder ────────────────────────────────────────────────────
        os.makedirs(EXPERIMENTS_ROOT, exist_ok=True)
        self._ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(EXPERIMENTS_ROOT, f'{self._ts}_{logic}')
        os.makedirs(self.run_dir, exist_ok=True)

        # Write pointer file so planners can redirect plots here
        with open(POINTER_FILE, 'w') as f:
            f.write(self.run_dir)

        # ── TF2 + state ────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._x = self._y = self._theta = 0.0
        self._v = self._omega = 0.0
        self._ODOM_V_MAX     = 0.26
        self._ODOM_OMEGA_MAX = 1.82
        self._ODOM_EMA_ALPHA = 0.3

        self._goal_received = False
        self._goal_received_at = None
        self._first_state_at   = None
        self._first_cmd_at     = None     # first /cmd_vel received → window start
        self._goal_reach_at    = None     # first time inside tolerance → window end
        self._goal_tolerance   = 0.10     # m (matches planner goal-reach threshold)
        self._recording  = True
        self._finalizing = False

        # ── Writer ────────────────────────────────────────────────────────
        self._writer = DataWriter(self.run_dir, self._ts, self.get_logger())
        self._writer.open()

        # ── Subscriptions ─────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan,   '/scan',                  self._lidar_cb,  scan_qos)
        self.create_subscription(Odometry,    '/odom',                  self._odom_cb,   10)
        self.create_subscription(Twist,       '/cmd_vel',               self._cmdvel_cb, 10)
        self.create_subscription(PoseStamped, '/move_base_simple/goal', self._goal_cb,   10)

        self.create_timer(0.01, self._state_timer_cb)

        # ── Signal handler ────────────────────────────────────────────────
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Experiment Recorder started — logic={logic}')
        self.get_logger().info(f'  Run folder    : {self.run_dir}')
        self.get_logger().info(f'  Pointer file  : {POINTER_FILE}')
        self.get_logger().info('  Set a goal in RViz2 (2D Goal Pose) and run the planner.')
        self.get_logger().info('  Press Ctrl+C to stop and save summary.')
        self.get_logger().info('=' * 60)

    # ── callbacks ────────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        v_raw = max(0.0, min(msg.twist.twist.linear.x, self._ODOM_V_MAX))
        w_raw = max(-self._ODOM_OMEGA_MAX, min(msg.twist.twist.angular.z, self._ODOM_OMEGA_MAX))
        a = self._ODOM_EMA_ALPHA
        self._v     = a * v_raw + (1.0 - a) * self._v
        self._omega = a * w_raw + (1.0 - a) * self._omega

    def _state_timer_cb(self):
        if not self._recording:
            return
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        self._x = t.transform.translation.x
        self._y = t.transform.translation.y
        q = t.transform.rotation
        _, _, self._theta = euler_from_quaternion([q.x, q.y, q.z, q.w])

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._first_state_at is None:
            self._first_state_at = now_sec
        self._writer.write_state(self._x, self._y, self._theta,
                                 self._v, self._omega, now_sec)

        # Goal-reach detection: only count after the first /cmd_vel has fired
        # (avoids false positives if the robot starts inside tolerance).
        if (self.goal_xy is not None
                and self._first_cmd_at is not None
                and self._goal_reach_at is None):
            dist = np.hypot(self._x - self.goal_xy[0], self._y - self.goal_xy[1])
            if dist < self._goal_tolerance:
                self._goal_reach_at = now_sec
                self.get_logger().info(
                    f'Goal reached at t+{now_sec - self._first_cmd_at:.2f}s '
                    f'(dist={dist:.3f} m)'
                )

    def _lidar_cb(self, msg: LaserScan):
        if not self._recording:
            return
        ts = self.get_clock().now().nanoseconds * 1e-9
        self._writer.write_lidar(list(msg.ranges),
                                 msg.angle_min, msg.angle_max, msg.angle_increment,
                                 msg.range_min, msg.range_max, ts)

    def _cmdvel_cb(self, msg: Twist):
        if not self._recording:
            return
        ts = self.get_clock().now().nanoseconds * 1e-9
        if self._first_cmd_at is None:
            self._first_cmd_at = ts
            self.get_logger().info(f'First /cmd_vel received — start of analysis window')
        self._writer.write_control(msg.linear.x, msg.angular.z, ts)

    def _goal_cb(self, msg: PoseStamped):
        if self._goal_received:
            return  # only record the first goal of the run
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self._goal_received = True
        self._goal_received_at = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            f'Goal recorded: ({self.goal_xy[0]:.3f}, {self.goal_xy[1]:.3f}) m')

    # ── Summary computation ─────────────────────────────────────────────
    def _compute_summary(self, finalized: dict, window: tuple | None) -> dict:
        """Compute run summary on data that is ALREADY windowed by `finalize`.

        The window is [first /cmd_vel ts, goal-reach ts] when both events
        occurred. If goal was not reached, the window ends at the last
        recorded state timestamp before Ctrl+C.
        """
        positions = finalized['positions']
        lin_vels  = finalized['lin_vels']
        ang_vels  = finalized['ang_vels']
        ctrl_lin  = finalized['ctrl_lin']
        ctrl_ang  = finalized['ctrl_ang']
        state_ts  = finalized['state_ts']

        # Window-based duration: prefer the explicit window we computed before
        # finalize, fall back to state-stream span if window is empty.
        if window is not None:
            duration = float(window[1] - window[0])
        elif len(state_ts) >= 2:
            duration = float(state_ts[-1] - state_ts[0])
        else:
            duration = 0.0

        # Travel distance — only along the windowed positions
        if len(positions) >= 2:
            diffs = np.diff(positions, axis=0)
            dist = float(np.sum(np.linalg.norm(diffs, axis=1)))
        else:
            dist = 0.0

        # Goal-reach is determined during recording (self._goal_reach_at).
        goal_reached = self._goal_reach_at is not None
        final_dist_to_goal = None
        if self.goal_xy is not None and len(positions) > 0:
            gx, gy = self.goal_xy
            final_dist_to_goal = float(np.hypot(positions[-1, 0] - gx,
                                                positions[-1, 1] - gy))

        def _stats(arr):
            if len(arr) == 0:
                return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
            return {
                'mean': float(np.mean(arr)),
                'min':  float(np.min(arr)),
                'max':  float(np.max(arr)),
                'std':  float(np.std(arr)),
            }

        summary = {
            'logic':                 self.logic,
            'timestamp':             self._ts,
            'run_folder':            self.run_dir,
            'analysis_window': {
                'first_cmd_vel_ts':  self._first_cmd_at,
                'goal_reach_ts':     self._goal_reach_at,
                'window_start_ts':   window[0] if window else None,
                'window_end_ts':     window[1] if window else None,
                'note': ('Metrics & saved JSON/NPZ cover this window only. '
                         'Raw .jsonl files contain the full unwindowed stream.'),
            },
            'duration_sec':          duration,
            'travel_distance_m':     dist,
            'avg_speed_m_s':         dist / duration if duration > 0 else 0.0,
            'goal_position':         list(self.goal_xy) if self.goal_xy else None,
            'goal_reached':          goal_reached,
            'final_distance_to_goal_m': final_dist_to_goal,
            'samples': {
                'states':          int(len(state_ts)),
                'control_cmds':    int(len(ctrl_lin)),
                'lidar_scans':     int(finalized['lidar_count']),
            },
            'odom_linear_velocity':  _stats(lin_vels),
            'odom_angular_velocity': _stats(ang_vels),
            'cmd_linear_velocity':   _stats(ctrl_lin),
            'cmd_angular_velocity':  _stats(ctrl_ang),
        }
        return summary

    # ── Signal handler ──────────────────────────────────────────────────
    def _signal_handler(self, signum, frame):
        if self._finalizing:
            return
        self._finalizing = True
        self._recording  = False

        self.get_logger().info('')
        self.get_logger().info('🛑 Stop signal received. Finalizing…')

        # Remove pointer file first so any late save attempts don't redirect
        try:
            if os.path.exists(POINTER_FILE):
                os.remove(POINTER_FILE)
        except Exception:
            pass

        try:
            # Determine analysis window: [first /cmd_vel, goal-reach]
            #   - If no /cmd_vel ever arrived, fall back to None (no window
            #     filtering, summary uses the raw state-stream span).
            #   - If goal was not reached, end the window at "now" (last
            #     observed state time approximated as get_clock now).
            window = None
            if self._first_cmd_at is not None:
                end_ts = (self._goal_reach_at
                          if self._goal_reach_at is not None
                          else self.get_clock().now().nanoseconds * 1e-9)
                window = (self._first_cmd_at, end_ts)
                self.get_logger().info(
                    f'Analysis window: {end_ts - self._first_cmd_at:.2f} s '
                    f'(first cmd → {"goal" if self._goal_reach_at else "stop"})'
                )
            else:
                self.get_logger().warn(
                    'No /cmd_vel was received — saving full stream without windowing.'
                )

            finalized = self._writer.finalize(window=window)
            summary = self._compute_summary(finalized, window)
            with open(os.path.join(self.run_dir, 'experiment_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            self.get_logger().info('─' * 60)
            self.get_logger().info(f'✅ Run saved to: {self.run_dir}')
            self.get_logger().info(f'   Logic         : {summary["logic"]}')
            self.get_logger().info(f'   Duration      : {summary["duration_sec"]:.2f} s '
                                   f'(first cmd → {"goal" if summary["goal_reached"] else "stop"})')
            self.get_logger().info(f'   Distance      : {summary["travel_distance_m"]:.2f} m')
            self.get_logger().info(f'   Avg speed     : {summary["avg_speed_m_s"]:.3f} m/s')
            self.get_logger().info(f'   Goal reached  : {summary["goal_reached"]}')
            self.get_logger().info(f'   Cmd v max     : {summary["cmd_linear_velocity"]["max"]:.3f} m/s')
            self.get_logger().info(f'   Odom v max    : {summary["odom_linear_velocity"]["max"]:.3f} m/s')
            self.get_logger().info('─' * 60)
        except Exception as e:
            self.get_logger().error(f'Error finalizing: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Record one experiment run.')
    parser.add_argument('--logic', required=True, choices=sorted(VALID_LOGICS),
                        help='Which control logic is being tested in this run.')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ExperimentRecorder(logic=args.logic)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Ensure pointer file is gone
        try:
            if os.path.exists(POINTER_FILE):
                os.remove(POINTER_FILE)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
