import os
import sys
sys.path.append(r"C:\Users\Yash\Documents\Quanser\0_libraries\python")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import time
import numpy as np
import cv2

class PygameKeyboard:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((200, 50))
        pygame.display.set_caption("QArm Control")
        self.keys = {
            'space': False, 'esc': False,
            'w': False, 'a': False, 's': False, 'd': False,
            'i': False, 'j': False, 'k': False, 'l': False,
            'comma': False, 'period': False,
            'tab': False
        }

    def read(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        pressed = pygame.key.get_pressed()
        self.keys['space'] = pressed[pygame.K_SPACE]
        self.keys['esc'] = pressed[pygame.K_ESCAPE]
        self.keys['w'] = pressed[pygame.K_w]
        self.keys['a'] = pressed[pygame.K_a]
        self.keys['s'] = pressed[pygame.K_s]
        self.keys['d'] = pressed[pygame.K_d]
        self.keys['i'] = pressed[pygame.K_i]
        self.keys['j'] = pressed[pygame.K_j]
        self.keys['k'] = pressed[pygame.K_k]
        self.keys['l'] = pressed[pygame.K_l]
        self.keys['comma'] = pressed[pygame.K_COMMA]
        self.keys['period'] = pressed[pygame.K_PERIOD]
        self.keys['tab'] = pressed[pygame.K_TAB]

    def terminate(self):
        pygame.quit()

class QArmController:
    def __init__(self, hardware=True, use_camera=False, use_conveyor=False):
        self.hardware = hardware
        self.use_camera = use_camera
        self.use_conveyor = use_conveyor
        self.keyboard = PygameKeyboard()
        
        # Initialize components with lazy loading
        self._arm = None
        self._camera = None
        self._conveyor = None
        self._qlabs = None
        
        # Control parameters
        self.joint_step = 0.05
        self.grip_step = 0.1
        self.led_color = np.array([0, 1, 0], dtype=np.float64)
        self.running = True
        self.phi_cmd = np.zeros(4, dtype=np.float64)
        self.grip_cmd = 0.0
        self.conveyor_speed = 0.5
        
        # Camera parameters from working example
        self.sampleRate = 30.0
        self.sampleTime = 1/self.sampleRate
        self.imageWidth = 640
        self.imageHeight = 480
        self.max_distance = 1.0  # meters for depth scaling
        
        # Camera state
        self._show_depth = False
        self._last_frame_time = 0
        self._colormap = cv2.COLORMAP_JET

    @property
    def arm(self):
        if self._arm is None:
            from pal.products.qarm import QArm
            self._arm = QArm(hardware=int(self.hardware))
        return self._arm

    @property
    def camera(self):
        if self._camera is None and self.use_camera:
            try:
                from pal.products.qarm import QArmRealSense
                self._camera = QArmRealSense(
                    mode='RGB&DEPTH',
                    hardware=int(self.hardware),
                    deviceID=0,
                    frameWidthRGB=self.imageWidth,
                    frameHeightRGB=self.imageHeight,
                    frameWidthDepth=self.imageWidth,
                    frameHeightDepth=self.imageHeight,
                    readMode=0
                )
                # Test camera
                self._camera.read_RGB()
                self._camera.read_depth()
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.use_camera = False
        return self._camera

    @property
    def qlabs(self):
        if self._qlabs is None and self.use_conveyor:
            from qvl.qlabs import QuanserInteractiveLabs
            self._qlabs = QuanserInteractiveLabs()
            if not self._qlabs.open(host="localhost", port=18000, timeout=5000):
                print("Failed to connect to QLabs")
                self.use_conveyor = False
        return self._qlabs

    @property
    def conveyor(self):
        if self._conveyor is None and self.use_conveyor and self.qlabs:
            from qvl.conveyor_straight import QLabsConveyorStraight
            self._conveyor = QLabsConveyorStraight(self.qlabs)
            if not self._conveyor.spawn_id(
                actorNumber=0,
                location=[1.5, 0, 0.1],
                rotation=[0, 0, 0],
                scale=[1.0, 1.0, 1.0],
                configuration=0,
                waitForConfirmation=True
            ):
                print("Failed to spawn conveyor")
                self.use_conveyor = False
        return self._conveyor

    def update_keyboard_controls(self):
        self.keyboard.read()
        
        if self.keyboard.keys['esc']:
            self.running = False
            return False
        
        if self.keyboard.keys['tab']:
            self._show_depth = not self._show_depth
            time.sleep(0.2)  # Debounce
        
        # Arm controls
        if self.keyboard.keys['a']: self.phi_cmd[0] += self.joint_step
        if self.keyboard.keys['d']: self.phi_cmd[0] -= self.joint_step
        if self.keyboard.keys['w']: self.phi_cmd[1] += self.joint_step
        if self.keyboard.keys['s']: self.phi_cmd[1] -= self.joint_step
        if self.keyboard.keys['i']: self.phi_cmd[2] += self.joint_step
        if self.keyboard.keys['k']: self.phi_cmd[2] -= self.joint_step
        if self.keyboard.keys['j']: self.phi_cmd[3] += self.joint_step
        if self.keyboard.keys['l']: self.phi_cmd[3] -= self.joint_step
        
        # Gripper control
        self.grip_cmd = 1.0 if self.keyboard.keys['space'] else 0.0
        
        # Conveyor control
        if self.use_conveyor:
            if self.keyboard.keys['comma']:
                self.conveyor_speed = max(-1.0, self.conveyor_speed - 0.1)
            if self.keyboard.keys['period']:
                self.conveyor_speed = min(1.0, self.conveyor_speed + 0.1)
            
        return True

    def process_camera_data(self):
        current_time = time.time()
        if current_time - self._last_frame_time < self.sampleTime:
            return
            
        try:
            self._last_frame_time = current_time
            start = time.time()
            
            if self._show_depth:
                # Depth view processing
                if self.camera.read_depth():
                    depth_image = self.camera.imageBufferDepthPX / self.max_distance
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=255),
                        self._colormap
                    )
                    
                    if self.use_conveyor:
                        cv2.putText(depth_colormap, f"Conveyor: {self.conveyor_speed:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('QArm - Depth View', depth_colormap)
            else:
                # RGB view processing
                if self.camera.read_RGB():
                    rgb_image = self.camera.imageBufferRGB
                    
                    if self.use_conveyor:
                        cv2.putText(rgb_image, f"Conveyor: {self.conveyor_speed:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (0, 255, 0), 1)
                    
                    cv2.imshow('QArm - RGB View', rgb_image)
            
            # Help text
            help_text = "TAB: Switch View | ESC: Quit"
            if self._show_depth and 'depth_colormap' in locals():
                cv2.putText(depth_colormap, help_text, 
                           (10, depth_colormap.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif not self._show_depth and 'rgb_image' in locals():
                cv2.putText(rgb_image, help_text, 
                           (10, rgb_image.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Timing control
            end = time.time()
            computationTime = end - start
            sleepTime = self.sampleTime - (computationTime % self.sampleTime)
            cv2.waitKey(max(1, int(1000*sleepTime)))
            
        except Exception as e:
            print(f"Camera error: {e}")

    def run(self):
        print("\n=== QARM CONTROL SYSTEM ===")
        print("Joint Controls:")
        print("  Base: A/D | Shoulder: W/S | Elbow: I/K | Wrist: J/L")
        print("Gripper: SPACE | Conveyor: ,/. | Camera: TAB | Quit: ESC")
        
        try:
            while self.running and (self._arm is None or self.arm.status):
                if not self.update_keyboard_controls():
                    break
                
                # Update arm
                self.arm.read_write_std(
                    phiCMD=self.phi_cmd,
                    gprCMD=self.grip_cmd,
                    baseLED=self.led_color
                )
                
                # Process camera
                if self.use_camera:
                    self.process_camera_data()
                
                # Update conveyor
                if self.use_conveyor and hasattr(self, '_last_conveyor_speed'):
                    if abs(self.conveyor_speed - self._last_conveyor_speed) > 0.05:
                        self.conveyor.set_speed(self.conveyor_speed)
                        self._last_conveyor_speed = self.conveyor_speed
                
                time.sleep(0.02)  # Main loop timing

        except KeyboardInterrupt:
            print("\nStopping by user request...")
        finally:
            self.terminate()

    def terminate(self):
        print("\nShutting down...")
        if self.use_conveyor and self._conveyor is not None:
            self._conveyor.set_speed(0.0)
            if self._qlabs is not None:
                self._qlabs.close()
        
        self.keyboard.terminate()
        if self._arm is not None:
            self._arm.terminate()
        if self._camera is not None:
            self._camera.terminate()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    controller = QArmController(
        hardware=False,  # Set to True for real hardware
        use_camera=True,
        use_conveyor=True
    )
    controller.run()
