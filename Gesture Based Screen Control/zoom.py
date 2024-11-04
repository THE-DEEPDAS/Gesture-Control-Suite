import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
from screeninfo import get_monitors

class ZoomController:
    def __init__(self):
        # Initialize camera to take the input.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize MediaPipe for hands detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8, # We just put this as we felt it is enough you can tweak that
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Zoom control variables(added cooldown to prevent too rapid changes)
        self.pinch_start_dist = None
        self.zoom_cooldown = 0.3
        self.last_zoom_time = 0
        self.pinch_threshold = 0.08
        
        # Initialize pyautogui
        pyautogui.FAILSAFE = False

    def calculate_pinch_distance(self, hand_landmarks): # Self explainatory
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        return math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )

    def handle_zoom_gesture(self, hand_landmarks):
        current_time = time.time()
        pinch_distance = self.calculate_pinch_distance(hand_landmarks)
        
        # Initialize pinch start distance if not set
        if self.pinch_start_dist is None:
            self.pinch_start_dist = pinch_distance
            return
        
        # Check if enough time has passed since last zoom action
        if current_time - self.last_zoom_time > self.zoom_cooldown:
            zoom_factor = self.pinch_start_dist / pinch_distance
            
            # Apply zoom if change is significant(we just use keyboard combination for doing that)
            if abs(zoom_factor - 1.0) > 0.2:
                if zoom_factor > 1.0:
                    pyautogui.hotkey('ctrl', '-')
                else:
                    pyautogui.hotkey('ctrl', '+')
                    
                self.last_zoom_time = current_time
                self.pinch_start_dist = pinch_distance

    def process_frame(self, frame):
        """Process each frame for hand detection and gesture recognition"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks with blue color scheme
            self.mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Process zoom gesture
            self.handle_zoom_gesture(hand_landmarks)
        else:
            self.pinch_start_dist = None
            
        return frame

    def run(self):
        """Main loop for the zoom controller"""
        try:
            print("Zoom Control Started!!")
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                
                # Add text overlay for instructions
                cv2.putText(processed_frame, "Zoom Control Mode - Pinch to zoom in/out",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Zoom Gesture Control', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cleanup()

    def cleanup(self): # Standard stuff
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = ZoomController()
    controller.run()