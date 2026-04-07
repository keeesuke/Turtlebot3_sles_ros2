#!/usr/bin/env python3
"""
Real-time plotter for /cmd_vel and /odom

Subscribes to a Twist topic (default: /cmd_vel) and an Odometry topic (default: /odom)
and plots:
 - cmd_vel linear.x (v) and angular.z (w) history
 - odom linear.x (v) and angular.z (w) history
 - odom position history (x, y trajectory)

Usage:
  python3 cmdvel_odom_plotter.py [--cmd_topic /cmd_vel] [--odom_topic /odom] [--history 300]

Run this with your ROS 2 environment sourced. Matplotlib GUI runs in the main thread,
so the ROS 2 node is spun in a background thread.
"""

import argparse
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class CmdOdomPlotNode(Node):
    def __init__(self, cmd_topic: str, odom_topic: str, maxlen: int = 1000):
        super().__init__('cmd_odom_plotter')
        self.create_subscription(Twist, cmd_topic, self._cmd_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self.maxlen = int(maxlen)

        # cmd_vel history
        self.cmd_t = deque(maxlen=self.maxlen)
        self.cmd_v = deque(maxlen=self.maxlen)
        self.cmd_w = deque(maxlen=self.maxlen)

        # odom history
        self.odom_t = deque(maxlen=self.maxlen)
        self.odom_x = deque(maxlen=self.maxlen)
        self.odom_y = deque(maxlen=self.maxlen)
        self.odom_v = deque(maxlen=self.maxlen)
        self.odom_w = deque(maxlen=self.maxlen)

    def _cmd_cb(self, msg: Twist):
        now = time.time()
        self.cmd_t.append(now)
        self.cmd_v.append(msg.linear.x)
        self.cmd_w.append(msg.angular.z)

    def _odom_cb(self, msg: Odometry):
        now = time.time()
        self.odom_t.append(now)
        p = msg.pose.pose.position
        self.odom_x.append(p.x)
        self.odom_y.append(p.y)
        self.odom_v.append(msg.twist.twist.linear.x)
        self.odom_w.append(msg.twist.twist.angular.z)


def _start_rclpy_spin(node: CmdOdomPlotNode):
    # Run rclpy.spin in background thread so matplotlib can run in main thread
    rclpy.spin(node)


def run_plot(node: CmdOdomPlotNode, interval_ms: int = 100):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 8))

    ax_cmd = fig.add_subplot(3, 1, 1)
    ax_odom_tw = fig.add_subplot(3, 1, 2)
    ax_xy = fig.add_subplot(3, 1, 3)

    def _format_time(xs):
        if not xs:
            return []
        t0 = xs[-1]
        return [x - t0 for x in xs]

    def update(frame):
        # cmd_vel
        ax_cmd.cla()
        cmd_t = list(node.cmd_t)
        cmd_v = list(node.cmd_v)
        cmd_w = list(node.cmd_w)
        if cmd_t:
            tt = _format_time(cmd_t)
            ax_cmd.plot(tt, cmd_v, label='cmd linear.x (v)', color='C0')
            ax_cmd.plot(tt, cmd_w, label='cmd angular.z (w)', color='C1')
        ax_cmd.set_ylabel('cmd_vel')
        ax_cmd.legend()
        ax_cmd.grid(True)

        # odom linear/angular
        ax_odom_tw.cla()
        odom_t = list(node.odom_t)
        odom_v = list(node.odom_v)
        odom_w = list(node.odom_w)
        if odom_t:
            tt2 = _format_time(odom_t)
            ax_odom_tw.plot(tt2, odom_v, label='odom linear.x (v)', color='C2')
            ax_odom_tw.plot(tt2, odom_w, label='odom angular.z (w)', color='C3')
        ax_odom_tw.set_ylabel('odom v / w')
        ax_odom_tw.legend()
        ax_odom_tw.grid(True)

        # odom x,y trajectory
        ax_xy.cla()
        ox = list(node.odom_x)
        oy = list(node.odom_y)
        if ox and oy:
            ax_xy.plot(ox, oy, '-o', markersize=2, label='odom path', color='C4')
            # mark latest position
            ax_xy.plot(ox[-1], oy[-1], 's', color='red', label='latest')
        ax_xy.set_xlabel('x (m)')
        ax_xy.set_ylabel('y (m)')
        ax_xy.axis('equal')
        ax_xy.legend()
        ax_xy.grid(True)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=interval_ms)
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd_topic', type=str, default='/cmd_vel', help='Twist topic (cmd_vel)')
    parser.add_argument('--odom_topic', type=str, default='/odom', help='Odometry topic')
    parser.add_argument('--history', type=int, default=1000, help='Max samples to keep')
    parser.add_argument('--interval', type=int, default=100, help='Plot update interval [ms]')
    args = parser.parse_args(argv)

    rclpy.init()
    node = CmdOdomPlotNode(args.cmd_topic, args.odom_topic, maxlen=args.history)

    spin_thread = threading.Thread(target=_start_rclpy_spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        run_plot(node, interval_ms=args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        # shutdown ROS and exit
        try:
            node.get_logger().info('Shutting down...')
        except Exception:
            pass
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
