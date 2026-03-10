#!/usr/bin/env python3
"""
ROS2 LiDAR Mapping Node — Occupancy Grid via Log-Odds
Converted from: sim_mapping_vector (ROS1)
"""

import os
import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class LidarMappingNode(Node):
    def __init__(self):
        super().__init__('lidar_mapping_node')

        # Subscribers / Publishers
        self.create_subscription(LaserScan, '/simulated_scan', self.lidar_callback, 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/lidar_occupancy_map', 10)

        # Robot state (encoded in LaserScan header fields)
        self.current_scan_robot_state = np.zeros(3)
        self.lidar_ranges = None
        self.lidar_angles = None

        # Grid parameters
        self.delta_x = 0.02
        self.map_x_min = -2.0
        self.map_y_min = -2.0
        self.map_x_max = 2.0
        self.map_y_max = 2.0
        self.grid_width = int((self.map_x_max - self.map_x_min) / self.delta_x)
        self.grid_height = int((self.map_y_max - self.map_y_min) / self.delta_x)

        self.log_odds = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        self.current_occupancy_grid = np.zeros_like(self.log_odds)

        x_coords = self.map_x_min + (np.arange(self.grid_width) + 0.5) * self.delta_x
        y_coords = self.map_y_min + (np.arange(self.grid_height) + 0.5) * self.delta_x
        self.X, self.Y = np.meshgrid(x_coords, y_coords, indexing='xy')

        # LiDAR model params
        self.d_max = 1.0
        self.w_miss = 0.03
        self.truncated_dis = 0.1

        # Load lookup tables from package share directory
        pkg_share = get_package_share_directory('turtlebot3_sles_perception')
        resource_dir = os.path.join(pkg_share, 'resource')
        self.distance_list = np.load(os.path.join(resource_dir, 'di_list.npy'))
        self.measurement_list = np.load(os.path.join(resource_dir, 'ym_list.npy'))
        self.I_table = np.load(os.path.join(resource_dir, 'I_table.npy'))
        self._interpolator = RegularGridInterpolator(
            (self.distance_list, self.measurement_list),
            self.I_table,
            bounds_error=False,
            fill_value=0.0
        )
        self.get_logger().info('LiDAR mapping node ready (lookup tables loaded).')

        self.create_timer(0.05, self.update_map)   # 20 Hz
        self.create_timer(0.05, self.publish_map)  # 20 Hz

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        n = len(ranges)
        angles = np.linspace(0, msg.angle_max, n, endpoint=False, dtype=np.float64)
        self.lidar_ranges = ranges
        self.lidar_angles = angles
        # Robot state is encoded in angle_min / range_min / range_max (legacy convention)
        self.current_scan_robot_state = np.array([msg.range_min, msg.range_max, msg.angle_min])

    def _occupancy_to_log_odds(self, p, epsilon=1e-9):
        p = np.clip(p, epsilon, 1 - epsilon)
        return np.log(p / (1.0 - p))

    def update_map(self):
        if self.lidar_ranges is None:
            return
        start_time = time.time()

        x_r, y_r, theta_r = self.current_scan_robot_state
        ranges = self.lidar_ranges
        angles = self.lidar_angles
        d_max = self.d_max
        td = self.truncated_dis
        w_miss = self.w_miss
        interp = self._interpolator
        logodds = self.log_odds

        pts_den = np.stack([np.full_like(ranges, d_max), ranges], axis=-1)
        denom_beams = interp(pts_den)

        dx = self.X - x_r
        dy = self.Y - y_r
        d_xy = np.hypot(dx, dy)
        in_range_mask = d_xy <= d_max

        theta_xy_world = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        theta_xy_robot = np.mod(theta_xy_world - theta_r, 2 * np.pi)

        beam_idx = (theta_xy_robot / (2 * np.pi) * len(angles)).astype(int)
        beam_idx = np.clip(beam_idx, 0, len(angles) - 1)
        z_i = ranges[beam_idx]

        update_log = np.zeros_like(logodds)
        prob = np.zeros_like(d_xy)

        valid_mask = in_range_mask & (z_i != -1)
        missed_mask = in_range_mask & (z_i == -1)
        behind_mask = in_range_mask & (d_xy > (z_i + td)) & (z_i != -1)

        if valid_mask.any():
            di = d_xy[valid_mask]
            zi = z_i[valid_mask]
            bi = beam_idx[valid_mask]
            num = interp(np.stack([di, zi], axis=-1))
            den = denom_beams[bi]
            prob[valid_mask] = num / den

        prob[missed_mask] = w_miss
        prob[behind_mask] = 0.5

        np.clip(prob, 1e-9, 1 - 1e-9, out=prob)
        update_log[in_range_mask] = self._occupancy_to_log_odds(prob[in_range_mask])

        logodds[in_range_mask] += update_log[in_range_mask]
        np.clip(logodds, -20, 20, out=logodds)
        self.log_odds = logodds
        self.current_occupancy_grid = 1.0 / (1.0 + np.exp(-logodds))

    def publish_map(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.delta_x
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        grid_msg.info.origin.position.x = self.map_x_min
        grid_msg.info.origin.position.y = self.map_y_min
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        occ_data = (100.0 * self.current_occupancy_grid).astype(np.int8)
        grid_msg.data = occ_data.ravel().tolist()
        self.map_publisher.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
