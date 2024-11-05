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
        self.prev_index_y = None  # Initialize prev_index_y here
        self.last_click_time = 0
        self.click_cooldown = 0.3  # seconds
        
        # Smoothing parameters
        self.smooth_x = self.smooth_y = 0
        self.smoothing = 0.5
        
        # Frame processing
        self.frame_reduction = 50
        
        # Enhanced scroll settings
        self.scroll_speed = 15
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.05
        self.prev_scroll_y = None
        self.scroll_buffer = []
        self.max_buffer_size = 5
        
        # Improved pinch detection
        self.pinch_threshold = 0.03
        self.thumb_active_threshold = 0.3
        self.last_pinch_state = False
        self.pinch_debounce_time = 0.2
        self.last_pinch_time = 0

    def calculate_distance(self, p1, p2):
        return hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_states(self, hand_landmarks):
        """Enhanced finger state detection with improved thumb tracking"""
        fingers = []
        
        # Improved thumb detection using both x and y coordinates
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Calculate thumb angle and position
        thumb_angle = np.arctan2(thumb_tip.y - thumb_ip.y, thumb_tip.x - thumb_ip.x)
        thumb_extended = (thumb_tip.x - thumb_mcp.x) > self.thumb_active_threshold
        
        fingers.append(thumb_extended)
            
        # Other fingers
        tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                fingers.append(True)
            else:
                fingers.append(False)
                
        return fingers

    def check_pinch(self, hand_landmarks):
        """Enhanced pinch detection with thumb position verification"""
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate primary pinch distance
        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        
        # Additional checks for thumb position
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # Verify thumb is in a pinching position
        thumb_forward = thumb_tip.z < thumb_ip.z
        thumb_above_base = thumb_tip.y < thumb_mcp.y
        
        current_time = time.time()
        if (pinch_distance < self.pinch_threshold and 
            thumb_forward and 
            thumb_above_base and 
            current_time - self.last_pinch_time > self.pinch_debounce_time):
            
            if not self.last_pinch_state:
                self.last_pinch_state = True
                self.last_pinch_time = current_time
                return True
        else:
            self.last_pinch_state = False
            
        return False

    def handle_mouse_movement(self, hand_landmarks, frame_shape):
        """Existing mouse movement handler - unchanged as requested"""
        index_tip = hand_landmarks.landmark[8]
        
        x = int(index_tip.x * frame_shape[1])
        y = int(index_tip.y * frame_shape[0])
        
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            self.smooth_x, self.smooth_y = x, y
            return
        
        target_x = np.interp(x, (self.frame_reduction, frame_shape[1] - self.frame_reduction), 
                            (0, self.screen_width))
        target_y = np.interp(y, (self.frame_reduction, frame_shape[0] - self.frame_reduction), 
                            (0, self.screen_height))
        
        self.smooth_x = int(self.smooth_x * self.smoothing + target_x * (1 - self.smoothing))
        self.smooth_y = int(self.smooth_y * self.smoothing + target_y * (1 - self.smoothing))
        
        pyautogui.moveTo(self.smooth_x, self.smooth_y)
        
        self.prev_x, self.prev_y = x, y

    def handle_scrolling(self, hand_landmarks, frame_shape):
        """Enhanced scrolling with pressure/intensity-based sensitivity"""
        current_time = time.time()
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
            
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Calculate finger separation as a proxy for "pressure"
        finger_distance = self.calculate_distance(index_tip, middle_tip)
        current_y = (index_tip.y + middle_tip.y) / 2 * frame_shape[0]
        
        # Add to rolling buffer for smooth movement detection
        self.scroll_buffer.append(current_y)
        if len(self.scroll_buffer) > self.max_buffer_size:
            self.scroll_buffer.pop(0)
        
        if len(self.scroll_buffer) >= 2:
            # Calculate moving average for smoother scrolling
            avg_movement = (self.scroll_buffer[-1] - self.scroll_buffer[0]) / len(self.scroll_buffer)
            
            # Adjust scroll speed based on finger separation (pressure proxy)
            pressure_factor = 1.0 + (0.05 - finger_distance) * 20  # Increases speed when fingers are closer
            scroll_amount = int(avg_movement * self.scroll_speed * pressure_factor)
            
            # Apply threshold to prevent accidental scrolling
            if abs(scroll_amount) > 1:
                pyautogui.scroll(scroll_amount)
                self.last_scroll_time = current_time

    def run(self):
        cap = cv2.VideoCapture(0)

        last_click_time = 0
        click_cooldown = 0.5  # Cooldown to avoid multiple clicks
        tap_threshold_y = 0.05  # Threshold for downward movement
        stable_threshold = 0.02  # Threshold for stabilization after tap

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                fingers = self.get_finger_states(hand_landmarks)
                
                # Refined gesture detection
                if not any(fingers[1:]):  # Fist gesture
                    self.prev_index_y = None
                    continue
                
                # Hovering and Clicking Mode
                if fingers[1] and not fingers[2]:  # Only index finger is up
                    index_tip = hand_landmarks.landmark[8]
                    current_time = time.time()

                    # Check for tap gesture
                    if self.prev_index_y is not None:
                        # Check if index finger has moved downward significantly
                        if index_tip.y > self.prev_index_y + tap_threshold_y:
                            # Wait for stabilization after downward movement
                            stable_position = index_tip.y
                            stable_time = time.time()

                            # Check if index finger stabilizes at a new position
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                frame = cv2.flip(frame, 1)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = self.hands.process(frame_rgb)
                                
                                if results.multi_hand_landmarks:
                                    hand_landmarks = results.multi_hand_landmarks[0]
                                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                                    
                                    index_tip = hand_landmarks.landmark[8]
                                    
                                    # Check if index finger remains stable within the threshold
                                    if abs(index_tip.y - stable_position) < stable_threshold:
                                        # Perform click at the original position
                                        if (current_time - last_click_time) > click_cooldown:
                                            pyautogui.click()  # Click action
                                            last_click_time = current_time
                                        break
                                cv2.imshow('Gesture Mouse Control', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            continue

                    self.prev_index_y = index_tip.y  # Update previous index y-coordinate
                    
                    self.handle_mouse_movement(hand_landmarks, frame.shape)

                # Scrolling Mode
                elif fingers[1] and fingers[2]:  # Index and middle fingers are up
                    self.handle_scrolling(hand_landmarks, frame.shape)
                    
            cv2.imshow('Gesture Mouse Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureMouseController()
    controller.run()