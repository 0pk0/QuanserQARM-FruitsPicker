import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Pose

class FruitDetector(Node):
    def __init__(self):
        super().__init__('fruit_detector')
        
        # ROS2 Publishers
        self.fruit_pub = self.create_publisher(Pose, '/fruit_position', 10)
        self.type_pub = self.create_publisher(String, '/fruit_type', 10)
        
        # Subscribe to QArm camera
        self.image_sub = self.create_subscription(Image, '/qarm/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        self.get_logger().info("Fruit Detector Node Started")

    def image_callback(self, msg):
        """Process image and detect fruit"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        fruit_data = self.detect_fruit(cv_image)

        if fruit_data:
            x, y, depth, fruit_type = fruit_data
            self.get_logger().info(f"Detected {fruit_type} at ({x}, {y}), Depth={depth}")

            # Convert to robot coordinates (assume camera calibration is done)
            Xr, Yr, Zr = self.image_to_robot_frame(x, y, depth)

            # Publish fruit position
            pose_msg = Pose()
            pose_msg.position.x = Xr
            pose_msg.position.y = Yr
            pose_msg.position.z = Zr
            self.fruit_pub.publish(pose_msg)

            # Publish fruit type
            type_msg = String()
            type_msg.data = fruit_type
            self.type_pub.publish(type_msg)

    def detect_fruit(self, image):
        """Detect fruit using color filtering"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        fruit_types = {
            "tomato": ([0, 100, 100], [10, 255, 255]),  # Red
            "banana": ([20, 100, 100], [30, 255, 255]),  # Yellow
            "strawberry": ([10, 100, 100], [20, 255, 255])  # Orange
        }

        for fruit, (lower, upper) in fruit_types.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                depth = 0.3  # Assume 30 cm depth
                return x + w // 2, y + h // 2, depth, fruit

        return None

    def image_to_robot_frame(self, x, y, depth):
        """Convert image coordinates to robot frame"""
        fx, fy = 640, 640  # Camera focal length in pixels
        cx, cy = 320, 240  # Camera center
        Xr = (x - cx) * depth / fx
        Yr = (y - cy) * depth / fy
        Zr = depth
        return Xr, Yr, Zr

def main():
    rclpy.init()
    node = FruitDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
