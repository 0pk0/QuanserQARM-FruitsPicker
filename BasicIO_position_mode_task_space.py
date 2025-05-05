import os
import sys
import numpy as np
import time
import pygame
import cv2
from math import cos, sin, pi, atan2, sqrt, acos

sys.path.append(r"C:\Users\Yash\Documents\Quanser\0_libraries\python")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

class QArmPrecisionPicker:
    def __init__(self, hardware=False, use_camera=False):
        from pal.products.qarm import QArm
        from hal.products.qarm import QArmUtilities
        
        self.arm = QArm(hardware=int(hardware))
        self.arm_utils = QArmUtilities()
        self.use_camera = use_camera
        
        # Initialize camera with lazy loading
        self._camera = None
        self._last_frame_time = 0
        self._frame_interval = 1/15  # Target 15 FPS
        
        # Physical parameters
        self.L1 = 0.21  # Shoulder-to-elbow length (m)
        self.L2 = 0.21  # Elbow-to-wrist length (m)
        self.gripper_offset = 0.08  # End-effector offset (m)
        
        # Joint limits (radians)
        self.joint_limits = [
            (-pi, pi),     # Base
            (-pi, pi),     # Shoulder
            (-pi, pi),     # Elbow
            (-pi, pi)      # Gripper
        ]
        
        # Predefined positions
        self.positions = {
            "home": [0.4, 0.0, 0.3],
            "pick_approach": [0.5, 0.0, 0.25],  # High approach to pick
            "pick": [0.65, 0.0, 0.3],          # Actual pick position
            "place_approach": [-0.5, 0.4, 0.25], # High approach to place
            "place": [-0.5, 0.45, 0.25]         # Actual place position
        }
        
        # Current state
        self.grip_cmd = 0.0  # 0 = open, 1 = closed
        self.led_color = np.array([0, 1, 0])  # Green LED
        self.current_joints = np.array([0.0, 0.5, -0.5, 0.0])
        
        # Initialize pygame for gripper control
        pygame.init()
        self.screen = pygame.display.set_mode((300, 100))
        pygame.display.set_caption('Gripper Control')
        self.font = pygame.font.SysFont(None, 36)
        
        # Cycle time tracking
        self.cycle_start_time = 0
        self.last_cycle_time = 0

    @property
    def camera(self):
        if self._camera is None and self.use_camera:
            from pal.products.qarm import QArmRealSense
            self._camera = QArmRealSense(
                mode='RGB',  # Only use RGB unless depth is needed
                hardware=0,
                deviceID=0,
                frameWidthRGB=640,
                frameHeightRGB=480,
                readMode=0
            )
        return self._camera

    def process_camera_data(self):
        """Process and display camera data if enabled"""
        if not self.use_camera or self.camera is None:
            return
            
        current_time = time.time()
        if current_time - self._last_frame_time < self._frame_interval:
            return
            
        try:
            self._last_frame_time = current_time
            self.camera.read_RGB()
            rgb_image = self.camera.imageBufferRGB
            
            # Add status overlay
            status_text = f"Gripper: {'CLOSED' if self.grip_cmd > 0.5 else 'OPEN'}"
            cycle_time_text = f"Last Cycle: {self.last_cycle_time:.2f}s"
            cv2.putText(rgb_image, status_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(rgb_image, cycle_time_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow('QArm Camera Feed', rgb_image)
            cv2.waitKey(1)  # Minimal delay
            
        except Exception as e:
            print(f"Camera error: {str(e)}")

    def validate_coordinates(self, x, y, z):
        """Check if coordinates are within workspace limits"""
        x_lim = (-0.8, 0.8)
        y_lim = (-0.8, 0.8)
        z_lim = (0.0, 0.8)
        return (x_lim[0] <= x <= x_lim[1] and 
                y_lim[0] <= y <= y_lim[1] and 
                z_lim[0] <= z <= z_lim[1])

    def calculate_joints(self, x, y, z):
        """Hybrid IK using both analytical solution and QArmUtilities"""
        try:
            # First try using the analytical solution
            phi1 = atan2(y, x)
            r = sqrt(x**2 + y**2)
            z_eff = z - self.gripper_offset
            
            # Check reachability
            if r > (self.L1 + self.L2) or z_eff < 0:
                raise ValueError("Position beyond workspace limits")
            
            # Analytical IK solution
            D = (r**2 + z_eff**2 - self.L1**2 - self.L2**2)/(2*self.L1*self.L2)
            D = np.clip(D, -1, 1)
            phi3 = acos(D)
            phi2 = atan2(z_eff, r) - atan2(self.L2*sin(phi3), self.L1 + self.L2*cos(phi3))
            
            # Apply joint limits
            phi2 = np.clip(phi2, self.joint_limits[1][0], self.joint_limits[1][1])
            phi3 = np.clip(phi3, self.joint_limits[2][0], self.joint_limits[2][1])
            
            return np.array([phi1, phi2, phi3, 0.0])
            
        except Exception as e:
            print(f"Analytical IK failed: {e}. Falling back to QArmUtilities IK.")
            # Fall back to QArmUtilities IK
            position = np.array([x, y, z])
            gamma = 0  # Wrist angle
            _, phiCmd = self.arm_utils.qarm_inverse_kinematics(
                position, gamma, self.arm.measJointPosition[0:4])
            
            if np.any(np.isnan(phiCmd)):
                raise ValueError("Position unreachable with both IK methods")
                
            return phiCmd

    def move_to_position(self, target_joints, duration=2.0):
        """Smooth movement to target joints"""
        start = self.current_joints
        steps = int(duration / 0.1)
        
        for i in range(1, steps+1):
            t = i/steps
            t_smooth = t**2 * (3 - 2*t)  # Smoothstep interpolation
            self.current_joints = start + (target_joints - start) * t_smooth
            self.arm.read_write_std(
                phiCMD=self.current_joints,
                gprCMD=self.grip_cmd,
                baseLED=self.led_color
            )
            
            # Process camera data during movement
            if self.use_camera:
                self.process_camera_data()
                
            time.sleep(0.1)
        
        self.current_joints = target_joints.copy()

    def control_gripper(self, target_state, duration=1.0):
        """Smooth gripper control"""
        steps = int(duration / 0.1)
        start = self.grip_cmd
        
        for i in range(1, steps+1):
            t = i/steps
            self.grip_cmd = start + (target_state - start) * t
            self.arm.read_write_std(
                phiCMD=self.current_joints,
                gprCMD=self.grip_cmd,
                baseLED=self.led_color
            )
            
            # Process camera data during gripper movement
            if self.use_camera:
                self.process_camera_data()
                
            time.sleep(0.1)
        
        self.grip_cmd = target_state

    def capture_image(self, filename="qarm_capture.jpg"):
        """Capture and save an image from the camera"""
        if not self.use_camera or self.camera is None:
            print("Camera is not enabled")
            return False
            
        try:
            self.camera.read_RGB()
            rgb_image = self.camera.imageBufferRGB
            cv2.imwrite(filename, rgb_image)
            print(f"Image saved as {filename}")
            return True
        except Exception as e:
            print(f"Failed to capture image: {e}")
            return False

    def run_pick_place_cycle(self):
        """Complete pick-and-place cycle with proper approach/depart positions"""
        self.cycle_start_time = time.time()
        try:
            print("\n=== STARTING PICK-AND-PLACE CYCLE ===")
            
            # 1. Move to approach position above pick location
            print("Moving to pick approach position...")
            pick_approach_joints = self.calculate_joints(*self.positions["pick_approach"])
            self.move_to_position(pick_approach_joints)
            
            # Optional: Capture image before picking
            if self.use_camera:
                self.capture_image("before_pick.jpg")
            
            # 2. Move down to actual pick position
            print("Moving to pick position...")
            pick_joints = self.calculate_joints(*self.positions["pick"])
            self.move_to_position(pick_joints)
            
            # 3. Close gripper to grasp object
            print("Closing gripper...")
            self.control_gripper(1.0)  # Close gripper
            time.sleep(1.0)  # Ensure grip is secure
            
            # 4. Lift back to approach height
            print("Lifting object...")
            self.move_to_position(pick_approach_joints)
            
            # 5. Move to approach position above place location
            print("Moving to place approach position...")
            place_approach_joints = self.calculate_joints(*self.positions["place_approach"])
            self.move_to_position(place_approach_joints)
            
            # Optional: Capture image before placing
            if self.use_camera:
                self.capture_image("before_place.jpg")
            
            # 6. Move down to actual place position
            print("Moving to place position...")
            place_joints = self.calculate_joints(*self.positions["place"])
            self.move_to_position(place_joints)
            
            # 7. Open gripper to release object
            print("Opening gripper...")
            self.control_gripper(0.0)  # Open gripper
            time.sleep(1.0)  # Ensure object is released
            
            # 8. Lift back to approach height
            print("Lifting from place position...")
            self.move_to_position(place_approach_joints)
            
            # 9. Return to home position
            print("Returning home...")
            home_joints = self.calculate_joints(*self.positions["home"])
            self.move_to_position(home_joints)
            
            # Calculate and store cycle time
            self.last_cycle_time = time.time() - self.cycle_start_time
            print(f"=== CYCLE COMPLETED IN {self.last_cycle_time:.2f} SECONDS ===")
            return True
            
        except Exception as e:
            print(f"\nERROR DURING OPERATION: {str(e)}")
            print("Attempting recovery...")
            self.move_to_position(self.calculate_joints(*self.positions["home"]))
            return False

    def run_continuous_operation(self):
        """Run pick-and-place in continuous loop until stopped"""
        print("\n=== CONTINUOUS OPERATION MODE ===")
        print("Press 'q' in the Pygame window to stop\n")
        
        # Initialize pygame display
        pygame.display.set_caption('QArm Control - Press Q to quit')
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            # Update display
            self.screen.fill((30, 30, 30))
            status_text = self.font.render("Running cycle... Press Q to quit", True, (255, 255, 255))
            time_text = self.font.render(f"Last cycle: {self.last_cycle_time:.2f}s", True, (255, 255, 255))
            self.screen.blit(status_text, (20, 20))
            self.screen.blit(time_text, (20, 50))
            pygame.display.flip()
            clock.tick(30)
            
            # Process camera data
            if self.use_camera:
                self.process_camera_data()
            
            # Execute one complete pick-and-place cycle
            success = self.run_pick_place_cycle()
            if not success:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
        
        # Clean up
        print("\nStopping continuous operation...")
        self.move_to_position(self.calculate_joints(*self.positions["home"]))
        if self.use_camera and self._camera is not None:
            self._camera.terminate()
        cv2.destroyAllWindows()
        pygame.quit()

    def terminate(self):
        """Clean up resources"""
        print("\n=== SHUTTING DOWN ===")
        if hasattr(self, 'arm'):
            self.arm.terminate()
        if self.use_camera and self._camera is not None:
            self._camera.terminate()
        cv2.destroyAllWindows()
        pygame.quit()
        print("System shutdown complete.")

if __name__ == "__main__":
    picker = QArmPrecisionPicker(
        hardware=False,  # Set True for real hardware
        use_camera=True  # Enable camera
    )
    
    try:
        # Initialize at home position
        picker.move_to_position(picker.calculate_joints(*picker.positions["home"]))
        
        print("\n=== QARM PRECISION PICKER ===")
        print("Starting continuous pick-and-place operation...")
        print("Press 'q' in the Pygame window to stop\n")
        
        picker.run_continuous_operation()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
    finally:
        picker.terminate()
        print("Program ended cleanly")
