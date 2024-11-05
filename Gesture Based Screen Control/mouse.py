import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from math import hypot

class GestureMouseController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Initialize state variables
        self.prev_x = self.prev_y = None
        self.last_click_time = 0
        self.click_cooldown = 0.3  # seconds
        
        # Smoothing parameters
        self.smooth_x = self.smooth_y = 0
        self.smoothing = 0.5  # Reduced smoothing for more responsive movement
        
        # Frame processing
        self.frame_reduction = 50  # Reduced frame boundary for better range of motion
        
        # Scroll settings
        self.scroll_speed = 10  # Adjusted scroll sensitivity
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.05  # seconds
        self.prev_scroll_y = None

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_states(self, hand_landmarks):
        """Determine which fingers are raised"""
        fingers = []
        
        # Thumb
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers.append(True)
        else:
            fingers.append(False)
            
        # Other fingers
        tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                fingers.append(True)
            else:
                fingers.append(False)
                
        return fingers

    def check_pinch(self, hand_landmarks):
        """Check if thumb and index finger are pinched"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < 0.03

    def handle_mouse_movement(self, hand_landmarks, frame_shape):
        """Handle mouse pointer movement with natural mapping"""
        index_tip = hand_landmarks.landmark[8]
        
        # Convert coordinates to pixel values
        x = int(index_tip.x * frame_shape[1])
        y = int(index_tip.y * frame_shape[0])
        
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            self.smooth_x, self.smooth_y = x, y
            return
        
        # Calculate target position (natural direction)
        target_x = np.interp(x, (self.frame_reduction, frame_shape[1] - self.frame_reduction), 
                            (0, self.screen_width))
        target_y = np.interp(y, (self.frame_reduction, frame_shape[0] - self.frame_reduction), 
                            (0, self.screen_height))
        
        # Apply smoothing
        self.smooth_x = int(self.smooth_x * self.smoothing + target_x * (1 - self.smoothing))
        self.smooth_y = int(self.smooth_y * self.smoothing + target_y * (1 - self.smoothing))
        
        # Move mouse
        pyautogui.moveTo(self.smooth_x, self.smooth_y)
        
        self.prev_x, self.prev_y = x, y

    def handle_scrolling(self, hand_landmarks, frame_shape):
        """Handle vertical scrolling based on hand movement"""
        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
            
        middle_tip = hand_landmarks.landmark[12]
        current_y = middle_tip.y * frame_shape[0]
        
        if self.prev_scroll_y is not None:
            # Calculate vertical movement
            dy = current_y - self.prev_scroll_y
            
            # Apply scrolling with direction threshold
            if abs(dy) > 0.005:  # Reduced threshold for more sensitive scrolling
                scroll_amount = int(dy * self.scroll_speed)
                pyautogui.scroll(scroll_amount)  # Positive dy = scroll down, negative = scroll up
                self.last_scroll_time = current_time
        
        self.prev_scroll_y = current_y

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                fingers = self.get_finger_states(hand_landmarks)
                
                # Fist gesture (no fingers raised)
                if not any(fingers[1:]):
                    self.prev_x = self.prev_y = None
                    self.prev_scroll_y = None
                    continue
                
                # Hovering and Clicking Mode (only index finger)
                if fingers[1] and not fingers[2]:
                    self.prev_scroll_y = None  # Reset scroll state
                    self.handle_mouse_movement(hand_landmarks, frame.shape)
                    
                    # Check for pinch gesture
                    if self.check_pinch(hand_landmarks):
                        current_time = time.time()
                        if current_time - self.last_click_time > self.click_cooldown:
                            pyautogui.click()
                            self.last_click_time = current_time
                
                # Scrolling Mode (index and middle fingers)
                elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                    self.prev_x = self.prev_y = None  # Reset mouse movement state
                    index_tip = hand_landmarks.landmark[8]
                    middle_tip = hand_landmarks.landmark[12]
                    if self.calculate_distance(index_tip, middle_tip) < 0.04:
                        self.handle_scrolling(hand_landmarks, frame.shape)
            
            cv2.imshow('Gesture Mouse Control', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Reduce mouse lag
    pyautogui.MINIMUM_DURATION = 0
    pyautogui.MINIMUM_SLEEP = 0
    pyautogui.PAUSE = 0
    
    controller = GestureMouseController()
    controller.run()