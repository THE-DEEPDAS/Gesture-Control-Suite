import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
from screeninfo import get_monitors
import pygame
from PIL import Image
import time

class EnhancedGestureController:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen settings
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        
        # Custom pointer settings
        pygame.init()
        self.pointer_size = 32
        self.create_custom_pointer()
        
        # Gesture tracking variables
        self.prev_hand_landmarks = None
        self.smoothing_factor = 0.5
        self.last_position = (0, 0)
        self.pinch_threshold = 0.06
        self.click_cooldown = 0.5
        self.last_click_time = 0
        self.pinch_start_dist = None
        self.zoom_cooldown = 0.3
        self.last_zoom_time = 0
        
        # Initialize pyautogui settings
        pyautogui.FAILSAFE = False
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.PAUSE = 0

    def create_custom_pointer(self):
        """Create a custom pointer image"""
        pointer_surface = pygame.Surface((self.pointer_size, self.pointer_size), pygame.SRCALPHA)
        
        # Draw outer circle
        pygame.draw.circle(pointer_surface, (0, 153, 255, 200), 
                         (self.pointer_size//2, self.pointer_size//2), 
                         self.pointer_size//2)
        
        # Draw inner circle
        pygame.draw.circle(pointer_surface, (255, 255, 255, 255),
                         (self.pointer_size//2, self.pointer_size//2),
                         self.pointer_size//4)
        
        # Convert to PIL image and save
        pygame_string = pygame.image.tostring(pointer_surface, 'RGBA')
        self.pointer = Image.frombytes('RGBA', (self.pointer_size, self.pointer_size), pygame_string)

    def smooth_position(self, new_pos):
        """Apply smoothing to cursor movement"""
        if self.last_position == (0, 0):
            return new_pos
        
        x = int(self.last_position[0] * (1 - self.smoothing_factor) + 
                new_pos[0] * self.smoothing_factor)
        y = int(self.last_position[1] * (1 - self.smoothing_factor) + 
                new_pos[1] * self.smoothing_factor)
        
        self.last_position = (x, y)
        return (x, y)

    def get_finger_state(self, hand_landmarks):
        """Get the state of index finger and thumb"""
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        
        # Check if index finger is extended
        index_extended = index_tip.y < index_pip.y
        
        # Calculate pinch distance
        pinch_distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        return index_extended, pinch_distance

    def handle_gestures(self, hand_landmarks):
        """Process detected hand gestures"""
        index_tip = hand_landmarks.landmark[8]
        current_time = time.time()
        
        # Convert hand position to screen coordinates
        screen_x = int(np.interp(index_tip.x, [0.0, 1.0], [0, self.screen_width]))
        screen_y = int(np.interp(index_tip.y, [0.0, 1.0], [0, self.screen_height]))
        
        # Apply smoothing
        screen_x, screen_y = self.smooth_position((screen_x, screen_y))
        
        # Move cursor
        pyautogui.moveTo(screen_x, screen_y, duration=0)
        
        # Get finger state
        index_extended, pinch_distance = self.get_finger_state(hand_landmarks)
        
        # Handle tap gesture (extended index finger)
        if index_extended and current_time - self.last_click_time > self.click_cooldown:
            if pinch_distance > self.pinch_threshold:  # Ensure not pinching
                pyautogui.click()
                self.last_click_time = current_time
        
        # Handle zoom gesture (pinch)
        if self.pinch_start_dist is None:
            self.pinch_start_dist = pinch_distance
        elif current_time - self.last_zoom_time > self.zoom_cooldown:
            zoom_factor = self.pinch_start_dist / pinch_distance
            
            if abs(zoom_factor - 1.0) > 0.2:  # Threshold for zoom activation
                if zoom_factor > 1.0:
                    pyautogui.hotkey('ctrl', '+')
                else:
                    pyautogui.hotkey('ctrl', '-')
                self.last_zoom_time = current_time
                self.pinch_start_dist = pinch_distance

    def process_frame(self, frame):
        """Process each frame for hand detection and gesture recognition"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks for visual feedback
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Process gestures
            self.handle_gestures(hand_landmarks)
            
        return frame

    def run(self):
        """Main loop for the gesture controller"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Enhanced Gesture Control (Press Q to quit)', processed_frame)
                
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
    # Set up the controller and run
    controller = EnhancedGestureController()
    controller.run()