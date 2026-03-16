import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3, Twist

class SpaceMouseRelay(Node):
    def __init__(self):
        super().__init__('spacemouse_relay')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Vector3, '/spacenav/offset', self.linear_cb, 10)
        self.create_subscription(Vector3, '/spacenav/rot_offset', self.angular_cb, 10)
        self.twist = Twist()

    def linear_cb(self, msg):
        self.twist.linear.x = msg.y * 0.8   # forward/back
        self.twist.linear.y = msg.x * 0.8   # strafe
        self.pub.publish(self.twist)

    def angular_cb(self, msg):
        self.twist.angular.z = msg.z * 2.0  # yaw
        self.pub.publish(self.twist)

def main():
    rclpy.init()
    rclpy.spin(SpaceMouseRelay())

if __name__ == '__main__':
    main()

