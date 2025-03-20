import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        
        self.box_pub = self.create_publisher(Pose, '/box_position', 10)
        self.image_sub = self.create_subscription(Image, '/qarm/camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        self.get_logger().info("Box Detector Node Started")

    def image_callback(self, msg):
        """Detect boxes for sorting"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        box_position = self.detect_boxes(cv_image)

        if box_position:
            x, y, depth = box_position
            Xr, Yr, Zr = self.image_to_robot_frame(x, y, depth)

            pose_msg = Pose()
            pose_msg.position.x = Xr
            pose_msg.position.y = Yr
            pose_msg.position.z = Zr
            self.box_pub.publish(pose_msg)

    def detect_boxes(self, image):
        """Detect box using shape & color"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([100, 50, 50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            return x + w // 2, y + h // 2, 0.2  # Assume 20 cm depth

        return None

    def image_to_robot_frame(self, x, y, depth):
        fx, fy = 640, 640
        cx, cy = 320, 240
        Xr = (x - cx) * depth / fx
        Yr = (y - cy) * depth / fy
        Zr = depth
        return Xr, Yr, Zr

def main():
    rclpy.init()
    node = BoxDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
