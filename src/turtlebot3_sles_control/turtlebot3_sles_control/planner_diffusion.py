#!/usr/bin/env python3
"""
ROS2 Diffusion Policy Planner for TurtleBot3
Converted from: planner_diffusion (ROS1)

Uses a trained diffusion model (DDIM-like sampling) to generate control commands.
"""

import os
import ast
import math
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Model architecture (must match training) ──────────────────────────────────

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionDenoisingNetwork(nn.Module):
    def __init__(self, action_dim=2, condition_dim=364, timestep_embed_dim=128,
                 hidden_dims=None, dropout=0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.timestep_embed = SinusoidalPositionEmbeddings(timestep_embed_dim)
        self.timestep_mlp = nn.Sequential(nn.Linear(timestep_embed_dim, timestep_embed_dim), nn.ReLU())
        input_dim = action_dim + timestep_embed_dim + condition_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, noisy_action, timestep, condition):
        t_emb = self.timestep_mlp(self.timestep_embed(timestep))
        x = torch.cat([noisy_action, t_emb, condition], dim=1)
        return self.network(x)


class DiffusionPolicy:
    def __init__(self, model, num_timesteps=100, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alpha_cumprod[:-1]])
        self.beta = beta
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    def fast_sample(self, condition, num_steps=20, warm_start_action=None):
        self.model.eval()
        batch_size = condition.shape[0]
        step_indices = np.linspace(self.num_timesteps - 1, 0, num_steps).astype(int)
        t_start = step_indices[0]

        if warm_start_action is not None:
            ws = torch.FloatTensor(warm_start_action).unsqueeze(0).to(self.device)
            if ws.shape[0] != batch_size:
                ws = ws.repeat(batch_size, 1)
            sqrt_a = self.sqrt_alpha_cumprod[t_start].unsqueeze(0).unsqueeze(1)
            sqrt_1ma = self.sqrt_one_minus_alpha_cumprod[t_start].unsqueeze(0).unsqueeze(1)
            actions = sqrt_a * ws + sqrt_1ma * torch.randn_like(ws)
        else:
            actions = torch.randn((batch_size, 2)).to(self.device)

        with torch.no_grad():
            for i, t in enumerate(step_indices):
                timesteps = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                pred_noise = self.model(actions, timesteps, condition)
                acs = self.alpha_cumprod[t]
                if i < len(step_indices) - 1:
                    t_next = step_indices[i + 1]
                    acs_next = self.alpha_cumprod[t_next]
                    pred0 = (actions - torch.sqrt(1.0 - acs) * pred_noise) / torch.sqrt(acs)
                    actions = torch.sqrt(acs_next) * pred0 + torch.sqrt(1.0 - acs_next) * pred_noise
                else:
                    actions = (actions - torch.sqrt(1.0 - acs) * pred_noise) / torch.sqrt(acs)
        return actions


# ── ROS2 Node ─────────────────────────────────────────────────────────────────

class DiffusionNavigationNode(Node):
    def __init__(self):
        super().__init__('diffusion_planner')

        self.declare_parameter('v_limit_haa', 0.26)
        self.declare_parameter('omega_limit_haa', 1.82)
        self.declare_parameter('inference_steps', 5)
        self.declare_parameter('use_warm_start', True)
        self.declare_parameter('goal', '[0, 0, 0, 0, 0]')
        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('control_dt', 0.02)
        self.declare_parameter('trajectory_path',
                               os.path.join(os.path.expanduser('~'), 'robot_trajectory_diffusion.png'))

        self.v_limit_haa = self.get_parameter('v_limit_haa').value
        self.omega_limit_haa = self.get_parameter('omega_limit_haa').value
        self.inference_steps = self.get_parameter('inference_steps').value
        self.use_warm_start = self.get_parameter('use_warm_start').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.control_dt = self.get_parameter('control_dt').value
        self.trajectory_path = self.get_parameter('trajectory_path').value

        raw_goal = self.get_parameter('goal').value
        self.goal = ast.literal_eval(raw_goal) if isinstance(raw_goal, str) else raw_goal

        self.state_ready = False
        self.lidar_ready = False
        self.latest_lidar_scan = None
        self.target_reached = False
        self.prev_action = None
        self.x = self.y = self.theta = self.v = self.omega = 0.0
        self.v_cmd_filtered = 0.0
        self.w_cmd_filtered = 0.0
        self.state_traj = []
        self.command_timings = []
        self.command_count = 0
        self.timing_log_interval = 100

        # Load diffusion model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best_model_diff.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f'Diffusion model not found at {model_path}')
            raise FileNotFoundError(f'Diffusion model not found at {model_path}')

        checkpoint = torch.load(model_path, map_location='cpu')
        cfg = checkpoint.get('config', {})
        denoiser = DiffusionDenoisingNetwork(
            action_dim=2, condition_dim=364,
            timestep_embed_dim=cfg.get('timestep_embed_dim', 128),
            hidden_dims=cfg.get('hidden_dims', [256, 128, 64]),
            dropout=cfg.get('dropout', 0.1),
        )
        denoiser.load_state_dict(checkpoint['model_state_dict'])
        self.diffusion_policy = DiffusionPolicy(
            model=denoiser,
            num_timesteps=cfg.get('num_timesteps', 100),
            beta_start=cfg.get('beta_start', 0.0001),
            beta_end=cfg.get('beta_end', 0.02),
            device='cpu',
        )
        self.get_logger().info(f'Diffusion model loaded. Steps={self.inference_steps}, warm_start={self.use_warm_start}')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.state_cb, 10)
        self.create_subscription(LaserScan, '/simulated_scan', self.lidar_cb, 10)
        self.control_timer = self.create_timer(0.02, self.control_loop)

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
            self.get_logger().warn(f'Lidar cb error: {e}')

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

    def control_loop(self):
        if not (self.state_ready and self.lidar_ready) or self.latest_lidar_scan is None:
            return
        if self.target_reached:
            return

        current_state = [self.x, self.y, self.theta, self.v, self.omega]
        self.state_traj.append(current_state.copy())

        dist = np.linalg.norm(np.array(current_state[:2]) - np.array(self.goal[:2]))
        if dist < 0.1:
            self.get_logger().info('Target reached!')
            self.target_reached = True
            if self.command_timings:
                self.get_logger().info(f'Avg timing: {np.mean(self.command_timings):.2f}ms')
            self._save_trajectory_plot()
            self._publish_stop()
            self.control_timer.cancel()
            rclpy.shutdown()
            return

        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        cos_t = np.cos(-self.theta)
        sin_t = np.sin(-self.theta)
        goal_x_r = dx * cos_t - dy * sin_t
        goal_y_r = dx * sin_t + dy * cos_t

        try:
            t0 = time.time()
            inp = np.concatenate([[self.v, self.omega], [goal_x_r, goal_y_r], self.latest_lidar_scan])
            cond = torch.FloatTensor(inp).unsqueeze(0)
            warm = self.prev_action if self.use_warm_start else None
            with torch.no_grad():
                actions = self.diffusion_policy.fast_sample(cond, num_steps=self.inference_steps, warm_start_action=warm)
                action_np = actions.cpu().numpy().flatten()
            v_cmd = float(np.clip(action_np[0], 0.0, self.v_limit_haa))
            w_cmd = float(np.clip(action_np[1], -self.omega_limit_haa, self.omega_limit_haa))
            self.v_cmd_filtered = v_cmd
            self.w_cmd_filtered = w_cmd
            self.prev_action = np.array([v_cmd, w_cmd], dtype=np.float32)

            dt_ms = (time.time() - t0) * 1000
            self.command_timings.append(dt_ms)
            self.command_count += 1
            if self.command_count % self.timing_log_interval == 0:
                recent = self.command_timings[-self.timing_log_interval:]
                self.get_logger().info(
                    f'Diffusion timing: avg={np.mean(recent):.2f}ms max={np.max(recent):.2f}ms')
        except Exception as e:
            self.get_logger().warn(f'Diffusion control error: {e}')
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
            ax.set_title('Diffusion Policy Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            plt.tight_layout()
            plt.savefig(self.trajectory_path, dpi=150)
            plt.close(fig)
            self.get_logger().info(f'Trajectory saved to {self.trajectory_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save plot: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
