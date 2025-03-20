import rclpy
from rclpy.node import Node
import numpy as np
import time
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64MultiArray

class QArmCollisionPlanner(Node):
    def __init__(self):
        super().__init__('qarm_collision_planner')
        
        # ROS2 Publishers and Subscribers
        self.publisher_ = self.create_publisher(Float64MultiArray, '/qarm/joint_commands', 10)
        self.subscription = self.create_subscription(Pose, '/qarm/ee_target', self.target_callback, 10)
        
        self.get_logger().info("QArm Collision-Aware Motion Planner Node Started")

        # Define workspace obstacles (spherical regions: [x, y, z, radius])
        self.obstacles = [
            [0.15, 0.0, 0.2, 0.05],  # Example obstacle at (0.15, 0.0, 0.2) with radius 5 cm
            [0.1, 0.1, 0.15, 0.07]   # Another obstacle
        ]

        # Joint limits (in radians)
        self.joint_limits = {
            "q1": (-np.pi, np.pi),
            "q2": (-np.pi/2, np.pi/2),
            "q3": (-np.pi/2, np.pi/2),
            "q4": (-np.pi, np.pi)
        }

        # Time settings
        self.rate = 10  # Hz
        self.dt = 1.0 / self.rate  # Time step

        # Current joint positions
        self.current_joints = np.array([0.0, 0.0, 0.0, 0.0])

    def target_callback(self, msg):
        """ Receives target end-effector position, checks collision, and computes trajectory """
        x, y, z = msg.position.x, msg.position.y, msg.position.z
        self.get_logger().info(f"Received Target Position: {(x, y, z)}")

        if self.check_collision(x, y, z):
            self.get_logger().error("Target position is inside an obstacle! Aborting motion.")
            return

        target_joints = self.inverse_kinematics(x, y, z)
        if target_joints is None:
            self.get_logger().error("IK Solution Not Found! Motion Aborted.")
            return

        # Validate with Forward Kinematics
        fk_position = self.forward_kinematics(target_joints)
        error = np.linalg.norm(np.array([x, y, z]) - np.array(fk_position))

        if error > 0.01:  # Tolerance of 1 cm
            self.get_logger().error(f"FK Validation Failed! Error: {error:.3f} m")
            return

        self.execute_trajectory(target_joints)

    def check_collision(self, x, y, z):
        """ Check if the target position is inside any obstacle """
        for obs in self.obstacles:
            ox, oy, oz, radius = obs
            distance = np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2)
            if distance < radius:
                return True  # Collision detected
        return False  # No collision

    def inverse_kinematics(self, px, py, pz):
        """ Compute inverse kinematics for QArm """
        try:
            theta1 = np.arctan2(py, px)
            r = np.sqrt(px**2 + py**2)
            s = pz - 0.127  # Adjust height based on base height

            # Compute theta2 and theta3 using trigonometry
            L2, L3, beta = 0.130, 0.124, np.radians(10.8)
            D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)

            if abs(D) > 1:
                return None

            theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
            theta2 = np.arctan2(s, r) - np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
            theta4 = 0.0  # Default wrist angle

            # Apply joint limits
            theta1 = np.clip(theta1, *self.joint_limits["q1"])
            theta2 = np.clip(theta2, *self.joint_limits["q2"])
            theta3 = np.clip(theta3, *self.joint_limits["q3"])
            theta4 = np.clip(theta4, *self.joint_limits["q4"])

            return np.array([theta1, theta2, theta3, theta4])

        except Exception:
            return None

    def forward_kinematics(self, joints):
        """ Compute forward kinematics based on calculated transformation matrices """
        q1, q2, q3, q4 = joints

        px = np.cos(q1) * (0.1798 * np.cos(q2) + 0.195 * np.cos(q2 + q3 - np.radians(10.8)))
        py = np.sin(q1) * (0.1798 * np.cos(q2) + 0.195 * np.cos(q2 + q3 - np.radians(10.8)))
        pz = 0.127 + 0.1798 * np.sin(q2) - 0.195 * np.sin(q2 + q3 - np.radians(10.8))

        return [px, py, pz]

    def execute_trajectory(self, target_joints):
        """ Generates and executes a smooth trajectory """
        steps = 50  # Number of interpolation steps
        trajectory = np.linspace(self.current_joints, target_joints, steps)

        for joints in trajectory:
            self.send_joint_command(joints)
            time.sleep(self.dt)

        self.current_joints = target_joints
        self.get_logger().info(f"Motion Complete. Final Joint Positions: {target_joints}")

    def send_joint_command(self, joint_positions):
        """ Publishes joint commands to QArm """
        msg = Float64MultiArray()
        msg.data = joint_positions.tolist()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = QArmCollisionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
