#!/usr/bin/env python3
"""
ROS2 Simulated LiDAR Publisher (fixed 8-obstacle configuration)
Converted from: simulate_lidar_publisher_vector (ROS1)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from tf_transformations import euler_from_quaternion
import numpy as np
from scipy.stats import truncnorm, truncexpon
import time
import copy


class SimulatedLidarPublisher(Node):
    def __init__(self):
        super().__init__('simulate_lidar_publisher')

        # Parameters
        self.declare_parameter('lidar_publish_frequency', 20.0)
        self.declare_parameter('num_lidar_rays', 360)
        self.declare_parameter('lidar_max_distance', 1.0)
        self.declare_parameter('gaussian_probability', 0.91)
        self.declare_parameter('exponential_probability', 0.03)
        self.declare_parameter('uniform_probability', 0.03)
        self.declare_parameter('max_probability', 0.03)

        # Fixed obstacle positions (8 obstacles)
        for i in range(1, 9):
            self.declare_parameter(f'random_x_{i}', 0.0)
            self.declare_parameter(f'random_y_{i}', 0.0)

        self.publish_frequency = self.get_parameter('lidar_publish_frequency').value
        self.num_rays = self.get_parameter('num_lidar_rays').value
        self.max_distance = self.get_parameter('lidar_max_distance').value
        self.gaussian_probability = self.get_parameter('gaussian_probability').value
        self.exponential_probability = self.get_parameter('exponential_probability').value
        self.uniform_probability = self.get_parameter('uniform_probability').value
        self.max_probability = self.get_parameter('max_probability').value

        self.sigma = 0.02
        self.lambda_exp = 5

        # State
        self.true_robot_state = None
        self.scan_time_i = None

        # Publisher / Subscriber
        self.lidar_publisher = self.create_publisher(LaserScan, '/simulated_scan', 10)
        self.create_subscription(ModelStates, '/gazebo/model_states', self.model_states_callback, 10)

        # Build obstacle vertices from parameters
        obstacle_positions = []
        for i in range(1, 9):
            x = self.get_parameter(f'random_x_{i}').value
            y = self.get_parameter(f'random_y_{i}').value
            obstacle_positions.append((x, y))
            self.get_logger().info(f'Obstacle {i}: ({x:.3f}, {y:.3f})')

        self.true_vertices = self._create_obstacles(obstacle_positions)
        self._precalculate_edges()

        # Timer
        self.create_timer(1.0 / self.publish_frequency, self.publish_lidar_data)

    def _create_obstacles(self, obstacle_positions):
        obstacles = []
        for x, y in obstacle_positions:
            half_size = 0.1
            vertices = [
                [x - half_size, y - half_size],
                [x + half_size, y - half_size],
                [x + half_size, y + half_size],
                [x - half_size, y + half_size],
            ]
            obstacles.append(vertices)
        return obstacles

    def _precalculate_edges(self):
        edges_start_list = []
        edges_end_list = []
        for poly in self.true_vertices:
            poly = np.array(poly)
            edges_start_list.append(poly)
            edges_end_list.append(np.roll(poly, -1, axis=0))
        self.edges_start = np.concatenate(edges_start_list, axis=0)
        self.edges_end = np.concatenate(edges_end_list, axis=0)

    def model_states_callback(self, msg):
        try:
            index = msg.name.index('turtlebot3_waffle_pi')
        except ValueError:
            return
        pose = msg.pose[index]
        self.scan_time_i = self.get_clock().now().to_msg()
        true_quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        _, _, true_orientation = euler_from_quaternion(true_quaternion)
        true_orientation = true_orientation % (2 * np.pi)
        self.true_robot_state = np.array([pose.position.x, pose.position.y, true_orientation])

    def simulate_lidar(self):
        weights = {
            'gaussian': self.gaussian_probability,
            'exponential': self.exponential_probability,
            'uniform': self.uniform_probability,
            'max': self.max_probability,
        }

        scan_time = self.scan_time_i
        x_r, y_r, theta_r = self.true_robot_state
        scan_robot_state = copy.deepcopy(self.true_robot_state)

        angles = (np.linspace(0.0, 2 * np.pi, self.num_rays, endpoint=False, dtype=np.float64)
                  + np.float64(theta_r))
        r_x = np.cos(angles).reshape(-1, 1)
        r_y = np.sin(angles).reshape(-1, 1)

        edges_start = self.edges_start
        edges_end = self.edges_end
        edge_dx = (edges_end[:, 0] - edges_start[:, 0]).reshape(1, -1)
        edge_dy = (edges_end[:, 1] - edges_start[:, 1]).reshape(1, -1)
        qp_x = (edges_start[:, 0] - x_r).reshape(1, -1)
        qp_y = (edges_start[:, 1] - y_r).reshape(1, -1)

        den = r_x * edge_dy - r_y * edge_dx
        valid_den_mask = np.abs(den) > 1e-8
        numerator_t = qp_x * edge_dy - qp_y * edge_dx
        t_vals = np.where(valid_den_mask, numerator_t / den, np.inf)
        numerator_u = qp_x * r_y - qp_y * r_x
        u_vals = np.where(valid_den_mask, numerator_u / den, np.inf)

        valid_intersections = (t_vals >= 0) & (u_vals >= 0) & (u_vals <= 1) & valid_den_mask
        t_vals_valid = np.where(valid_intersections, t_vals, np.inf)
        t_min = np.min(t_vals_valid, axis=1)
        distances = np.where(t_min > self.max_distance, -1, t_min)

        choices = np.random.choice(
            ['gaussian', 'exponential', 'uniform', 'max'],
            size=self.num_rays,
            p=list(weights.values())
        )
        noisy_distances = distances.copy()
        gaussian_mask = choices == 'gaussian'
        exponential_mask = choices == 'exponential'
        uniform_mask = choices == 'uniform'
        miss_mask = choices == 'max'

        if np.any(gaussian_mask):
            noisy_distances[gaussian_mask] = self._gaussian_dis(distances[gaussian_mask])
        if np.any(exponential_mask):
            noisy_distances[exponential_mask] = self._exponential_dis(distances[exponential_mask])
        if np.any(uniform_mask):
            noisy_distances[uniform_mask] = np.random.uniform(0, self.max_distance, size=uniform_mask.sum())
        if np.any(miss_mask):
            noisy_distances[miss_mask] = -1.0

        return noisy_distances.tolist(), scan_robot_state, scan_time

    def _gaussian_dis(self, arr):
        arr = np.asarray(arr)
        result = np.full_like(arr, -1.0)
        valid_mask = arr != -1.0
        if np.any(valid_mask):
            d_valid = arr[valid_mask]
            a = (0.0 - d_valid) / self.sigma
            b = (self.max_distance - d_valid) / self.sigma
            result[valid_mask] = truncnorm.rvs(a, b, loc=d_valid, scale=self.sigma)
        return result

    def _exponential_dis(self, arr):
        arr = np.asarray(arr)
        result = np.full_like(arr, -1.0)
        valid_mask = arr != -1.0
        if np.any(valid_mask):
            d_valid = arr[valid_mask]
            scale_exp = 1.0 / self.lambda_exp
            b_exp = d_valid / scale_exp
            result[valid_mask] = truncexpon.rvs(b_exp, loc=0, scale=scale_exp)
        return result

    def publish_lidar_data(self):
        if self.true_robot_state is None:
            return
        lidar_readings, scan_state, scan_time = self.simulate_lidar()

        scan_msg = LaserScan()
        scan_msg.header.stamp = scan_time
        scan_msg.header.frame_id = 'base_link'
        scan_msg.angle_min = float(scan_state[2])
        scan_msg.angle_max = 2 * np.pi
        scan_msg.angle_increment = 2 * np.pi / len(lidar_readings)
        scan_msg.range_min = float(scan_state[0])
        scan_msg.range_max = float(scan_state[1])
        scan_msg.ranges = lidar_readings
        self.lidar_publisher.publish(scan_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedLidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
