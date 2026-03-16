#!/usr/bin/env python3
"""
ROS2 node: spawns robot and obstacles in Gazebo using random positions
passed as node parameters from turtlebot3_custom_world_random.launch.py.
Converted from ROS1 spawn_random_world.py.
"""

import os
import math
import tempfile
import subprocess

import rclpy
from rclpy.node import Node

NUM_OBSTACLES = 16
SPAWN_DELAY = 0.1

SHAPE_COLORS = {
    'rectangle': 'Gazebo/Red',
    'hexagon':   'Gazebo/Blue',
    'triangle':  'Gazebo/Green',
}
VALID_SHAPES = list(SHAPE_COLORS.keys())


def _create_sdf_template(model_name, collision_geom, visual_geom, material_name, pose_z):
    return f"""<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision"><geometry>{collision_geom}</geometry></collision>
      <visual name="visual">
        <geometry>{visual_geom}</geometry>
        <material><script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>{material_name}</name>
        </script></material>
      </visual>
      <pose>0 0 {pose_z:.4f} 0 0 0</pose>
    </link>
  </model>
</sdf>"""


def _box_geom(w, d, h):
    return f"<box><size>{w:.4f} {d:.4f} {h:.4f}</size></box>"


def _polyline_geom(height, points):
    pts = "\n".join(f"<point>{x:.4f} {y:.4f}</point>" for x, y in points)
    return f"<polyline><height>{height:.4f}</height>{pts}</polyline>"


def create_obstacle_sdf(size, shape='rectangle'):
    shape = shape.strip().lower()
    if shape == 'rectangle':
        g = _box_geom(size, size, size)
        return _create_sdf_template('obstacle_rec', g, g, SHAPE_COLORS['rectangle'], size / 2)
    elif shape == 'hexagon':
        r = size / 2
        pts = [(r * math.cos(math.radians(i * 60)),
                r * math.sin(math.radians(i * 60))) for i in range(6)]
        g = _polyline_geom(size, pts)
        return _create_sdf_template('obstacle_hex', g, g, SHAPE_COLORS['hexagon'], size / 2)
    elif shape == 'triangle':
        h = size * math.sqrt(3) / 2
        pts = [(-size / 2, 0.0), (size / 2, 0.0), (0.0, h)]
        g = _polyline_geom(size, pts)
        return _create_sdf_template('obstacle_tri', g, g, SHAPE_COLORS['triangle'], size / 2)
    else:
        return create_obstacle_sdf(size, 'rectangle')


def spawn_sdf_content(model_name, sdf_content, x, y, z):
    """Write SDF to temp file and call ros2 service spawn_entity."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False)
    tmp.write(sdf_content)
    tmp.flush()
    tmp.close()
    try:
        result = subprocess.run(
            ['ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
             '-file', tmp.name,
             '-entity', model_name,
             '-x', str(x), '-y', str(y), '-z', str(z)],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    finally:
        os.unlink(tmp.name)


class SpawnRandomWorldNode(Node):
    def __init__(self):
        super().__init__('spawn_random_world')

        # Declare parameters (values injected by launch file)
        self.declare_parameter('model', os.environ.get('TURTLEBOT3_MODEL', 'waffle_pi'))
        for i in range(1, NUM_OBSTACLES + 1):
            self.declare_parameter(f'random_x_{i}', 0.0)
            self.declare_parameter(f'random_y_{i}', 0.0)
            self.declare_parameter(f'random_size_{i}', 0.2)
            self.declare_parameter(f'random_shape_{i}', 'rectangle')
        self.declare_parameter('random_robot_x', 0.0)
        self.declare_parameter('random_robot_y', 0.0)
        self.declare_parameter('random_robot_z', 0.01)
        self.declare_parameter('random_robot_yaw', 3.14)

        # Run spawn logic once after a short delay (let gzserver start)
        self.timer = self.create_timer(2.0, self._spawn_all)

    def _spawn_all(self):
        self.timer.cancel()  # run once only
        model = self.get_parameter('model').value

        # Spawn obstacles
        for i in range(1, NUM_OBSTACLES + 1):
            x     = self.get_parameter(f'random_x_{i}').value
            y     = self.get_parameter(f'random_y_{i}').value
            size  = self.get_parameter(f'random_size_{i}').value
            shape = self.get_parameter(f'random_shape_{i}').value
            if shape not in VALID_SHAPES:
                shape = 'rectangle'
            sdf = create_obstacle_sdf(size, shape)
            ok = spawn_sdf_content(f'obstacle_{i}', sdf, x, y, 0.0)
            self.get_logger().info(
                f"{'Sucess' if ok else 'Failed'} obstacle_{i} ({shape}) at ({x:.2f},{y:.2f})"
            )

        # Spawn robot
        rx  = self.get_parameter('random_robot_x').value
        ry  = self.get_parameter('random_robot_y').value
        rz  = self.get_parameter('random_robot_z').value
        ryaw = self.get_parameter('random_robot_yaw').value
        from ament_index_python.packages import get_package_share_directory
        pkg = get_package_share_directory('turtlebot3_gazebo')
        sdf_path = os.path.join(pkg, 'models', f'turtlebot3_{model}', 'model.sdf')
        result = subprocess.run(
            ['ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
             '-file', sdf_path,
             '-entity', f'turtlebot3_{model}',
             '-x', str(rx), '-y', str(ry), '-z', str(rz), '-Y', str(ryaw)],
            capture_output=True, text=True, timeout=15
        )
        self.get_logger().info(
            f"{'Sucess' if result.returncode == 0 else 'Failed'} spawned turtlebot3_{model}"
        )
        self.get_logger().info('SpawnRandomWorld complete.')


def main(args=None):
    rclpy.init(args=args)
    node = SpawnRandomWorldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

