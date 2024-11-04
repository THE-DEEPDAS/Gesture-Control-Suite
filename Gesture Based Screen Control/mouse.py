import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pygame
from PIL import Image
from screeninfo import get_monitors
import time
import math

class PreciseTapController:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize MediaPipe with higher confidence thresholds
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.95,  # Increased from 0.9
            min_tracking_confidence=0.95    # Increased from 0.9
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen settings
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        self.frame_width = 1280
        self.frame_height = 720
        
        # Pointer settings
        pygame.init()
        self.pointer_size = 40
        self.create_custom_pointer()
        
        # Movement tracking
        self.last_position = (0, 0)
        self.smoothing_factor = 0.5
        self.boundary_buffer = 10
        
        # Enhanced double tap detection
        self.tap_times = []
        self.double_tap_threshold = 0.4
        self.min_tap_interval = 0.15
        self.last_tap_time = 0
        self.tap_depth_threshold = 0.15
        self.last_index_depth = 0
        self.tap_start_depth = None
        self.tap_detected = False
        self.consecutive_detections = 0
        self.required_consecutive = 3
        
        # Initialize pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0

    def create_custom_pointer(self):
        """Create a distinctive purple pointer"""
        pointer_surface = pygame.Surface((self.pointer_size, self.pointer_size), pygame.SRCALPHA)
        pygame.draw.circle(pointer_surface, (128, 0, 128, 160), 
                         (self.pointer_size//2, self.pointer_size//2), 
                         self.pointer_size//2)
        pygame.draw.circle(pointer_surface, (160, 32, 240, 180),
                         (self.pointer_size//2, self.pointer_size//2),
                         self.pointer_size//3)
        pygame.draw.circle(pointer_surface, (255, 255, 255, 255),
                         (self.pointer_size//2, self.pointer_size//2),
                         self.pointer_size//6)
        pygame_string = pygame.image.tostring(pointer_surface, 'RGBA')
        self.pointer = Image.frombytes('RGBA', (self.pointer_size, self.pointer_size), pygame_string)

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = [p1.x - p2.x, p1.y - p2.y]
        v2 = [p3.x - p2.x, p3.y - p2.y]
        
        cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (
            math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
        )
        angle = math.degrees(math.acos(min(1, max(-1, cosine))))
        return angle

    def is_within_bounds(self, x, y):
        """Check if point is within screen boundaries"""
        return (self.boundary_buffer <= x <= self.frame_width - self.boundary_buffer and
                self.boundary_buffer <= y <= self.frame_height - self.boundary_buffer)

    def is_index_finger_only(self, hand_landmarks):
        """Enhanced check if only index finger is extended using angles"""
        # Get relevant landmarks
        wrist = hand_landmarks.landmark[0]
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_mcp = hand_landmarks.landmark[9]
        ring_tip = hand_landmarks.landmark[16]
        ring_mcp = hand_landmarks.landmark[13]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_mcp = hand_landmarks.landmark[17]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Calculate angles for each finger
        index_angle = self.calculate_angle(index_tip, index_mcp, wrist)
        middle_angle = self.calculate_angle(middle_tip, middle_mcp, wrist)
        ring_angle = self.calculate_angle(ring_tip, ring_mcp, wrist)
        pinky_angle = self.calculate_angle(pinky_tip, pinky_mcp, wrist)
        thumb_angle = self.calculate_angle(thumb_tip, thumb_mcp, wrist)
        
        # Check if index is extended and others are closed
        index_extended = (
            index_angle > 150 and  # Index should be straight
            index_tip.y < index_pip.y < index_mcp.y  # Index should point upward
        )
        
        others_closed = (
            middle_angle < 130 and  # Other fingers should be bent
            ring_angle < 130 and
            pinky_angle < 130 and
            thumb_angle < 120  # Thumb should be tucked
        )
        
        # Additional depth check to ensure consistent gesture
        depth_valid = (
            abs(index_tip.z - index_mcp.z) < 0.1 and  # Index should be relatively flat
            all(abs(lmk.z - wrist.z) < 0.15 for lmk in [
                middle_tip, ring_tip, pinky_tip, thumb_tip
            ])  # Other fingers should be in similar depth plane
        )
        
        return index_extended and others_closed and depth_valid

    def detect_forward_tap(self, hand_landmarks):
        """Enhanced forward tap detection with acceleration consideration"""
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        current_depth = index_tip.z
        
        if self.tap_start_depth is None:
            self.tap_start_depth = current_depth
            return False
        
        # Calculate depth change and velocity
        depth_change = current_depth - self.tap_start_depth
        depth_acceleration = depth_change - (current_depth - self.last_index_depth)
        
        # Update tracking variables
        self.last_index_depth = current_depth
        self.tap_start_depth = current_depth
        
        # Detect forward motion with acceleration threshold
        if depth_change > self.tap_depth_threshold and abs(depth_acceleration) > 0.05:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0
            
        return self.consecutive_detections >= self.required_consecutive

    def handle_double_tap(self, hand_landmarks):
        """Process double tap with enhanced requirements"""
        current_time = time.time()
        
        if not self.is_index_finger_only(hand_landmarks):
            self.tap_times = []
            return
        
        tap_detected = self.detect_forward_tap(hand_landmarks)
        
        if tap_detected and current_time - self.last_tap_time > self.min_tap_interval:
            self.tap_times.append(current_time)
            self.last_tap_time = current_time
            
            self.tap_times = [t for t in self.tap_times 
                            if current_time - t <= self.double_tap_threshold]
            
            if len(self.tap_times) >= 2:
                if self.tap_times[-1] - self.tap_times[-2] <= self.double_tap_threshold:
                    pyautogui.click()
                    self.tap_times = []

    def smooth_position(self, new_pos):
        """Apply smoothing to cursor movement"""
        if self.last_position == (0, 0):
            return new_pos
        
        x = int(self.last_position[0] * (1 - self.smoothing_factor) + 
                new_pos[0] * self.smoothing_factor)
        y = int(self.last_position[1] * (1 - self.smoothing_factor) + 
                new_pos[1] * self.smoothing_factor)
        
        x = max(0, min(x, self.screen_width))
        y = max(0, min(y, self.screen_height))
        
        self.last_position = (x, y)
        return (x, y)

    def process_frame(self, frame):
        """Process each frame for hand detection and gesture recognition"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Draw boundary
        cv2.rectangle(frame, 
                     (self.boundary_buffer, self.boundary_buffer),
                     (self.frame_width - self.boundary_buffer, 
                      self.frame_height - self.boundary_buffer),
                     (128, 0, 128), 2)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            index_tip = hand_landmarks.landmark[8]
            tip_x = int(index_tip.x * self.frame_width)
            tip_y = int(index_tip.y * self.frame_height)
            
            if self.is_within_bounds(tip_x, tip_y):
                if self.is_index_finger_only(hand_landmarks):
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(160, 32, 240), thickness=2)
                    )
                    
                    screen_x = int(np.interp(tip_x, 
                                           [self.boundary_buffer, self.frame_width - self.boundary_buffer],
                                           [0, self.screen_width]))
                    screen_y = int(np.interp(tip_y, 
                                           [self.boundary_buffer, self.frame_height - self.boundary_buffer],
                                           [0, self.screen_height]))
                    
                    screen_x, screen_y = self.smooth_position((screen_x, screen_y))
                    pyautogui.moveTo(screen_x, screen_y, duration=0)
                    self.handle_double_tap(hand_landmarks)
                    
                    cv2.circle(frame, (tip_x, tip_y), 5, (128, 0, 128), -1)
        
        return frame

    def run(self):
        """Main loop"""
        try:
            print("Precise Tap Control Started - Press 'Q' to quit")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                cv2.imshow('Precise Tap Control', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    controller = PreciseTapController()
    controller.run()