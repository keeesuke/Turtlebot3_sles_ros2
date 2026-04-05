#!/usr/bin/env python3
"""
ROS2 Neural Network Planner for TurtleBot3
Converted from: planner_NN (ROS1)

Uses a trained MLP to generate control commands from:
  - 360 LiDAR scan (from /simulated_scan)
  - Robot velocities (v, w)
  - Target position in robot frame
"""

import os
import ast
import time
import threading

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
import torch
import torch.nn as nn

try:
    from shapely.geometry import LineString
    _HAS_SHAPELY = True
except Exception:
    LineString = None
    _HAS_SHAPELY = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


class NNNavigationNode(Node):
    def __init__(self):
        super().__init__('nn_planner')

        # Parameters
        self.declare_parameter('v_limit_haa', 0.26)
        self.declare_parameter('omega_limit_haa', 1.82)
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('goal', '[0, 0, 0, 0, 0]')
        self.declare_parameter('horizon_hpa', 20)
        self.declare_parameter('horizon_haa', 40)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_limit_hpa', 0.2)
        self.declare_parameter('omega_limit_hpa', 1.0)
        self.declare_parameter('trajectory_path',
                               os.path.join(os.path.expanduser('~'), 'robot_trajectory_nn.png'))

        self.v_limit_haa = self.get_parameter('v_limit_haa').value
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.control_dt = self.get_parameter('control_dt').value
        self.trajectory_path = self.get_parameter('trajectory_path').value

        raw_goal = self.get_parameter('goal').value
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal

        # State flags
        self.state_ready = False
        self.lidar_ready = False
        self.latest_lidar_scan = None
        self.target_reached = False
        self.x = self.y = self.theta = self.v = self.omega = 0.0

        # Trajectory logging
        self.state_traj = []
        self.control_command_history = []
        self.command_timings = []
        self.command_count = 0
        self.timing_log_interval = 100

        # Low-pass filter
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0

        # Load NN model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f'NN model not found at {model_path}')
            raise FileNotFoundError(f'NN model not found at {model_path}')

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.nn_model = MLP(input_dim=364, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.1)
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

        # ROS pub/sub
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.state_cb, 10)
        self.create_subscription(LaserScan, '/simulated_scan', self.lidar_cb, 10)

        # Control timer (50 Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info('NN planner node started (50 Hz control, goal: %s)' % str(self.goal))

    def lidar_cb(self, msg: LaserScan):
        try:
            ranges = np.array(msg.ranges)
            ranges[ranges == -1] = 1.0
            if len(ranges) != 360:
                indices = np.linspace(0, len(ranges) - 1, 360)
                ranges = np.interp(indices, np.arange(len(ranges)), ranges)
            self.latest_lidar_scan = ranges
            self.lidar_ready = True
        except Exception as e:
            self.get_logger().warn(f'Error processing lidar scan: {e}')

    def state_cb(self, msg: ModelStates):
        try:
            idx = msg.name.index('turtlebot3_waffle_pi')
        except ValueError:
            return
        pose = msg.pose[idx]
        twist = msg.twist[idx]
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.x = pose.position.x
        self.y = pose.position.y
        self.theta = yaw
        self.v = np.hypot(twist.linear.x, twist.linear.y)
        self.omega = twist.angular.z
        self.state_ready = True

    def _transform_goal_to_robot_frame(self, gx, gy):
        dx = gx - self.x
        dy = gy - self.y
        cos_t = np.cos(-self.theta)
        sin_t = np.sin(-self.theta)
        return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t

    def control_loop(self):
        if not (self.state_ready and self.lidar_ready) or self.latest_lidar_scan is None:
            return
        if self.target_reached:
            return

        current_state = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(current_state.copy())

        dist = np.linalg.norm(np.array(current_state[:2]) - np.array(self.goal[:2]))
        if dist < 0.1:
            self.get_logger().info('Target reached! Shutting down control loop.')
            self.target_reached = True
            self._log_timing_stats()
            self._save_trajectory_plot()
            self._publish_stop()
            self.control_timer.cancel()
            rclpy.shutdown()
            return

        goal_x_r, goal_y_r = self._transform_goal_to_robot_frame(self.goal[0], self.goal[1])

        try:
            t0 = time.time()
            inp = np.concatenate([[self.v, self.omega], [goal_x_r, goal_y_r], self.latest_lidar_scan])
            tensor = torch.FloatTensor(inp).unsqueeze(0)
            with torch.no_grad():
                out = self.nn_model(tensor).cpu().numpy().flatten()
            v_cmd = float(np.clip(out[0], 0.0, self.v_limit_haa))
            w_cmd = float(np.clip(out[1], -self.omega_limit_haa, self.omega_limit_haa))
            self.v_cmd_filtered = v_cmd
            self.w_cmd_filtered = w_cmd

            dt_ms = (time.time() - t0) * 1000
            self.command_timings.append(dt_ms)
            self.command_count += 1
            if self.command_count % self.timing_log_interval == 0:
                recent = self.command_timings[-self.timing_log_interval:]
                self.get_logger().info(
                    f'NN timing (last {self.timing_log_interval}): '
                    f'avg={np.mean(recent):.2f}ms min={np.min(recent):.2f}ms max={np.max(recent):.2f}ms')
        except Exception as e:
            self.get_logger().warn(f'Error in NN control: {e}')
            self._publish_stop()
            return

        twist = Twist()
        twist.linear.x = self.v_cmd_filtered
        twist.angular.z = self.w_cmd_filtered
        self.cmd_pub.publish(twist)

    def _publish_stop(self):
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.cmd_pub.publish(Twist())

    def _log_timing_stats(self):
        if self.command_timings:
            self.get_logger().info(
                f'Final timing: avg={np.mean(self.command_timings):.2f}ms '
                f'min={np.min(self.command_timings):.2f}ms '
                f'max={np.max(self.command_timings):.2f}ms')

    def _save_trajectory_plot(self):
        if len(self.state_traj) < 2:
            return
        try:
            xs = np.array([s[0] for s in self.state_traj])
            ys = np.array([s[1] for s in self.state_traj])
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(xs, ys, 'b-', linewidth=2, label='Trajectory')
            ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
            ax.plot(self.goal[0], self.goal[1], 'r*', markersize=16, label='Goal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('NN Planner Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            plt.tight_layout()
            plt.savefig(self.trajectory_path, dpi=150)
            plt.close(fig)
            self.get_logger().info(f'Trajectory saved to {self.trajectory_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save trajectory plot: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = NNNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
