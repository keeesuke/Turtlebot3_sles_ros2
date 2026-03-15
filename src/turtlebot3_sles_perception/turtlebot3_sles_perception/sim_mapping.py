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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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
        self.grid_width  = int((self.map_x_max - self.map_x_min) / self.delta_x)
        self.grid_height = int((self.map_y_max - self.map_y_min) / self.delta_x)

        self.log_odds = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        self.current_occupancy_grid = np.zeros_like(self.log_odds)

        x_coords = self.map_x_min + (np.arange(self.grid_width)  + 0.5) * self.delta_x
        y_coords = self.map_y_min + (np.arange(self.grid_height) + 0.5) * self.delta_x
        self.X, self.Y = np.meshgrid(x_coords, y_coords, indexing='xy')

        # LiDAR model params
        self.d_max        = 1.0
        self.w_miss       = 0.03
        self.truncated_dis = 0.1

        # Load lookup tables from package share directory
        pkg_share    = get_package_share_directory('turtlebot3_sles_perception')
        resource_dir = os.path.join(pkg_share, 'resource')
        self.distance_list    = np.load(os.path.join(resource_dir, 'di_list.npy'))
        self.measurement_list = np.load(os.path.join(resource_dir, 'ym_list.npy'))
        self.I_table          = np.load(os.path.join(resource_dir, 'I_table.npy'))
        self._interpolator = RegularGridInterpolator(
            (self.distance_list, self.measurement_list),
            self.I_table,
            bounds_error=False,
            fill_value=0.0
        )

        plt.ion()
        self.fig  = None
        self.cbar = None

        self.get_logger().info('LiDAR mapping node ready (lookup tables loaded).')

        self.create_timer(0.05, self.update_map)          # 20 Hz
        self.create_timer(0.05, self.publish_map)         # 20 Hz
        # self.create_timer(1.0,  self.plot_occupancy_grid) # 1 Hz

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        n      = len(ranges)
        angles = np.linspace(0, msg.angle_max, n, endpoint=False, dtype=np.float64)
        self.lidar_ranges = ranges
        self.lidar_angles = angles
        # Robot state is encoded in angle_min / range_min / range_max (legacy convention)
        self.current_scan_robot_state = np.array([msg.range_min, msg.range_max, msg.angle_min])

    def _occupancy_to_log_odds(self, p, epsilon=1e-9):
        p = np.clip(p, epsilon, 1 - epsilon)
        return np.log(p / (1.0 - p))

    # alias for backward-compatibility with update_map call style in ROS1 version
    def occupancy_to_log_odds(self, occupancy_prob, epsilon=1e-9):
        return self._occupancy_to_log_odds(occupancy_prob, epsilon)

    def lookup_integral(self, distance, measurement):
        """
        Return the interpolated value of
            ∫₀ᵈⁱ p_mix(y_m | x) dx
        for scalar or array inputs.
        """
        d_arr = np.atleast_1d(distance)
        z_arr = np.atleast_1d(measurement)
        d_flat, y_flat = np.broadcast_arrays(d_arr, z_arr)
        pts    = np.stack([d_flat.ravel(), y_flat.ravel()], axis=-1)
        I_flat = self._interpolator(pts)
        I      = I_flat.reshape(d_flat.shape)
        if I.size == 1:
            return I.item()
        return I

    def update_map(self):
        if self.lidar_ranges is None:
            return
        start_time = time.time()

        x_r, y_r, theta_r = self.current_scan_robot_state
        ranges   = self.lidar_ranges
        angles   = self.lidar_angles
        d_max    = self.d_max
        td       = self.truncated_dis
        w_miss   = self.w_miss
        interp   = self._interpolator
        logodds  = self.log_odds
        prob2log = self.occupancy_to_log_odds

        # 1) Per-beam denominators: ∫₀ᵈₘₐₓ p_mix(ranges[b] | x) dx
        pts_den      = np.stack([np.full_like(ranges, d_max), ranges], axis=-1)
        denom_beams  = interp(pts_den)

        # 2) Cell ranges & bearings
        dx   = self.X - x_r
        dy   = self.Y - y_r
        d_xy = np.hypot(dx, dy)
        in_range_mask = d_xy <= d_max

        theta_xy_world = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        theta_xy_robot = np.mod(theta_xy_world - theta_r, 2 * np.pi)

        # 3) Closest beam index for each cell
        beam_idx = (theta_xy_robot / (2 * np.pi) * len(angles)).astype(int)
        beam_idx = np.clip(beam_idx, 0, len(angles) - 1)
        z_i      = ranges[beam_idx]

        # 4) Masks
        update_log  = np.zeros_like(logodds)
        prob        = np.zeros_like(d_xy)

        valid_mask  = in_range_mask & (z_i != -1)
        missed_mask = in_range_mask & (z_i == -1)
        behind_mask = in_range_mask & (d_xy > (z_i + td)) & (z_i != -1)

        # 5) Valid returns
        if valid_mask.any():
            di  = d_xy[valid_mask]
            zi  = z_i[valid_mask]
            bi  = beam_idx[valid_mask]
            num = interp(np.stack([di, zi], axis=-1))
            den = denom_beams[bi]
            prob[valid_mask] = num / den

        # 6) Missed returns
        prob[missed_mask] = w_miss

        # 7) Behind obstacle → unknown
        prob[behind_mask] = 0.5

        # 8) Clamp & convert to log-odds
        np.clip(prob, 1e-9, 1 - 1e-9, out=prob)
        update_log[in_range_mask] = prob2log(prob[in_range_mask])

        # 9) Accumulate
        logodds[in_range_mask] += update_log[in_range_mask]
        np.clip(logodds, -20, 20, out=logodds)
        self.log_odds = logodds
        self.current_occupancy_grid = 1.0 / (1.0 + np.exp(-logodds))

        total_time = time.time() - start_time
        self.get_logger().debug(f'update_map time: {total_time:.6f}s')

    def plot_occupancy_grid(self):
        """
        Plots the 2D occupancy grid in a single, updating figure.
        Called periodically via timer (1 Hz).
        ROS1版の plot_occupancy_grid を ROS2 向けに移植。
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.cbar = None

        self.ax.clear()

        im = self.ax.imshow(
            self.current_occupancy_grid,
            origin='lower',
            cmap='hot',
            extent=(self.map_x_min, self.map_x_max, self.map_y_min, self.map_y_max),
            vmin=0,
            vmax=1
        )

        if self.cbar is not None:
            self.cbar.remove()
        self.cbar = self.fig.colorbar(im, ax=self.ax)
        self.cbar.set_label('Occupancy Probability')

        self.ax.set_title('Occupancy Grid Map')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')

        plt.draw()
        plt.pause(0.001)

    def publish_map(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp    = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = self.delta_x
        grid_msg.info.width      = self.grid_width
        grid_msg.info.height     = self.grid_height
        grid_msg.info.origin.position.x = self.map_x_min
        grid_msg.info.origin.position.y = self.map_y_min
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        occ_data       = (100.0 * self.current_occupancy_grid).astype(np.int8)
        grid_msg.data  = occ_data.ravel().tolist()
        self.map_publisher.publish(grid_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
