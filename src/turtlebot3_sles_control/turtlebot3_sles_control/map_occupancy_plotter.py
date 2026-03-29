#!/usr/bin/env python3

import json
import math
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.time import Time
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class MapOccupancyPlotter(Node):
    def __init__(self) -> None:
        super().__init__("map_occupancy_plotter")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("update_rate_hz", 5.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("robot_arrow_length_m", 0.25)

        # Optional: save raw OccupancyGrid data to disk.
        self.declare_parameter("save_maps", False)
        self.declare_parameter("save_dir", "/tmp/map_occupancy_logs")
        self.declare_parameter("save_rate_hz", 1.0)

        map_topic = self.get_parameter("map_topic").value
        update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._robot_arrow_length_m = float(
            self.get_parameter("robot_arrow_length_m").value
        )

        save_maps_param = self.get_parameter("save_maps").value
        if isinstance(save_maps_param, str):
            self._save_maps = save_maps_param.strip().lower() in ("true", "1", "yes", "on")
        else:
            self._save_maps = bool(save_maps_param)
        self._save_dir = str(self.get_parameter("save_dir").value)
        self._save_rate_hz = float(self.get_parameter("save_rate_hz").value)
        self._last_save_time = None

        if self._save_maps:
            os.makedirs(self._save_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_binary_image = None
        self._latest_extent = None
        self._map_received = False

        self.create_subscription(OccupancyGrid, map_topic, self._map_callback, 10)

        self._fig, self._ax = plt.subplots()
        self._ax.set_title("Occupancy Grid Map (/map)")
        self._ax.set_xlabel("x (cells)")
        self._ax.set_ylabel("y (cells)")
        self._image_artist = self._ax.imshow(
            np.ones((10, 10, 3), dtype=np.uint8) * 255,
            origin="lower",
            interpolation="nearest",
        )

        # Robot pose overlay (updated from TF).
        self._robot_point_artist = self._ax.plot([], [], "ro", markersize=6)[0]
        self._robot_arrow_line_artist = self._ax.plot([], [], "r-", linewidth=2)[0]

        self._fig_binary, self._ax_binary = plt.subplots()
        self._ax_binary.set_title("Binary Occupancy Map (>1 occupied, -1 free)")
        self._ax_binary.set_xlabel("x (cells)")
        self._ax_binary.set_ylabel("y (cells)")
        self._binary_image_artist = self._ax_binary.imshow(
            np.ones((10, 10, 3), dtype=np.uint8) * 255,
            origin="lower",
            interpolation="nearest",
        )
        self._robot_point_artist_binary = self._ax_binary.plot(
            [], [], "ro", markersize=6
        )[0]
        self._robot_arrow_line_artist_binary = self._ax_binary.plot(
            [], [], "r-", linewidth=2
        )[0]

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        period = 1.0 / max(update_rate_hz, 0.1)
        self.create_timer(period, self._plot_timer_callback)
        self.get_logger().info(
            f"Subscribed to '{map_topic}'. Plot colors: occupied=black, free=white, unknown=green."
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _map_callback(self, msg: OccupancyGrid) -> None:
        width = msg.info.width
        height = msg.info.height
        if width == 0 or height == 0:
            return

        grid = np.asarray(msg.data, dtype=np.int16).reshape((height, width))
        rgb = np.full((height, width, 3), 255, dtype=np.uint8)  # free defaults to white

        # Render rules:
        #   - free (0)    => white
        #   - occupied(>0) => black
        #   - unknown (-1) => green
        occupied_mask = grid > 50
        rgb[occupied_mask] = np.array([0, 0, 0], dtype=np.uint8)

        unknown_mask = grid == -1
        rgb[unknown_mask] = np.array([0, 255, 0], dtype=np.uint8)  # unknown in green

        binary_rgb = np.full((height, width, 3), 255, dtype=np.uint8)  # default free=white
        # Your rule: treat anything > 1 as occupied; treat -1 (unknown) as free (white).
        binary_occupied_mask = grid > 50
        binary_rgb[binary_occupied_mask] = np.array([0, 0, 0], dtype=np.uint8)  # occupied=black
        # unknown (-1) is intentionally treated as free => white

        resolution = float(msg.info.resolution)
        origin_x = float(msg.info.origin.position.x)
        origin_y = float(msg.info.origin.position.y)
        extent = (
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution,
        )

        with self._lock:
            self._latest_image = rgb
            self._latest_binary_image = binary_rgb
            self._latest_extent = extent
            self._map_received = True

        if self._save_maps:
            now = self.get_clock().now()
            min_interval_s = 1.0 / max(self._save_rate_hz, 1e-6)
            if self._last_save_time is None:
                do_save = True
            else:
                elapsed_s = (now - self._last_save_time).nanoseconds * 1e-9
                do_save = elapsed_s >= min_interval_s

            if do_save:
                stamp = msg.header.stamp
                stamp_base = f"{stamp.sec}_{stamp.nanosec}"

                grid_i8 = np.asarray(msg.data, dtype=np.int8).reshape((height, width))
                npy_path = os.path.join(self._save_dir, f"map_{stamp_base}.npy")
                meta_path = os.path.join(
                    self._save_dir, f"map_{stamp_base}.json"
                )

                origin = msg.info.origin
                meta = {
                    "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
                    "frame_id": msg.header.frame_id,
                    "width": width,
                    "height": height,
                    "resolution": resolution,
                    "origin": {
                        "x": float(origin.position.x),
                        "y": float(origin.position.y),
                        "z": float(origin.position.z),
                        "orientation": {
                            "x": float(origin.orientation.x),
                            "y": float(origin.orientation.y),
                            "z": float(origin.orientation.z),
                            "w": float(origin.orientation.w),
                        },
                    },
                }

                np.save(npy_path, grid_i8)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f)

                self._last_save_time = now

    def _plot_timer_callback(self) -> None:
        with self._lock:
            image = None if self._latest_image is None else self._latest_image.copy()
            binary_image = (
                None if self._latest_binary_image is None else self._latest_binary_image.copy()
            )
            extent = self._latest_extent
            map_received = self._map_received

        if not map_received:
            self.get_logger().info("no map ready", throttle_duration_sec=5.0)
            return

        if image is None or binary_image is None or extent is None:
            return

        self._image_artist.set_data(image)
        self._image_artist.set_extent(extent)
        self._ax.set_xlim(extent[0], extent[1])
        self._ax.set_ylim(extent[2], extent[3])

        self._binary_image_artist.set_data(binary_image)
        self._binary_image_artist.set_extent(extent)
        self._ax_binary.set_xlim(extent[0], extent[1])
        self._ax_binary.set_ylim(extent[2], extent[3])

        # Update robot pose overlay using TF.
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, Time()
            )
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation

            # Convert quaternion to yaw (rotation around Z).
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )

            x = float(t.x)
            y = float(t.y)
            L = self._robot_arrow_length_m
            x2 = x + L * math.cos(yaw)
            y2 = y + L * math.sin(yaw)

            self._robot_point_artist.set_data([x], [y])
            self._robot_arrow_line_artist.set_data([x, x2], [y, y2])

            self._robot_point_artist_binary.set_data([x], [y])
            self._robot_arrow_line_artist_binary.set_data([x, x2], [y, y2])
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            self.get_logger().info("no tf for robot pose yet", throttle_duration_sec=2.0)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._fig_binary.canvas.draw_idle()
        self._fig_binary.canvas.flush_events()
        plt.pause(0.001)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapOccupancyPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
