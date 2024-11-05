import cv2
import numpy as np
import pygame
import time
from datetime import datetime, timedelta
import mediapipe as mp

class PomodoroTimer:
    def __init__(self):
        # Initialize times for different modes (in seconds che aa)
        self.WORK_TIME = 25
        self.SHORT_BREAK = 5
        self.LONG_BREAK = 15
        
        # Initialize states
        self.current_mode = "WORK"  # WORK, SHORT_BREAK, LONG_BREAK
        self.time_remaining = self.WORK_TIME * 60
        self.is_running = False
        self.is_muted = False
        self.focus_mode = False
        
        # Initialize pygame for sound
        pygame.mixer.init()
        self.background_sound = pygame.mixer.Sound("./bg_sound.wav") 
        self.background_sound.set_volume(0.3)
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
    def detect_gesture(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Get landmark positions
            landmarks = []
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
            
            # Detect gestures
            self.process_gestures(landmarks)
        
        return frame
    
    def process_gestures(self, landmarks):
        # Thumbs up detection (start)
        if self.detect_thumbs_up(landmarks):
            self.start_timer()
            
        # Thumbs down detection (stop)
        elif self.detect_thumbs_down(landmarks):
            self.stop_timer()
            
        # Pinch out detection (add time)
        elif self.detect_pinch_out(landmarks):
            self.add_time()
            
        # Pinch in detection (decrease time)
        elif self.detect_pinch_in(landmarks):
            self.decrease_time()
            
        # Finger on lips detection (mute/unmute)
        elif self.detect_finger_on_lips(landmarks):
            self.toggle_mute()
            
        # Focus mode gesture detection
        elif self.detect_focus_gesture(landmarks):
            self.toggle_focus_mode()
    
    def detect_thumbs_up(self, landmarks):
        # Implement thumbs up detection logic
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        return thumb_tip[1] < thumb_base[1]
    
    def detect_thumbs_down(self, landmarks):
        # Implement thumbs down detection logic
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        return thumb_tip[1] > thumb_base[1]
    
    def detect_pinch_out(self, landmarks):
        # Implement pinch out detection logic
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        return distance > 100  # Adjust threshold as needed
    
    def detect_pinch_in(self, landmarks):
        # Implement pinch in detection logic
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        return distance < 30  # Adjust threshold as needed
    
    def detect_finger_on_lips(self, landmarks):
        # Implement finger on lips detection logic
        index_tip = landmarks[8]
        # Define approximate lip region (adjust as needed)
        return 0.4 < index_tip[1] / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) < 0.6
    
    def detect_focus_gesture(self, landmarks):
        # Implement focus mode gesture detection (pinch with middle and thumb)
        thumb_tip = landmarks[4]
        middle_tip = landmarks[12]
        distance = np.sqrt((thumb_tip[0] - middle_tip[0])**2 + (thumb_tip[1] - middle_tip[1])**2)
        return distance < 30  # Adjust threshold as needed
    
    def start_timer(self):
        self.is_running = True
        if not self.is_muted:
            self.background_sound.play(-1)  # Loop indefinitely
    
    def stop_timer(self):
        self.is_running = False
        self.background_sound.stop()
    
    def add_time(self):
        self.time_remaining += 5 * 60  # Add 5 minutes
    
    def decrease_time(self):
        self.time_remaining = max(60, self.time_remaining - 5 * 60)  # Subtract 5 minutes, minimum 1 minute
    
    def toggle_mute(self):
        self.is_muted = not self.is_muted
        if self.is_muted:
            self.background_sound.stop()
        elif self.is_running:
            self.background_sound.play(-1)
    
    def toggle_focus_mode(self):
        self.focus_mode = not self.focus_mode
    
    def update_timer(self):
        if self.is_running:
            self.time_remaining -= 1
            if self.time_remaining <= 0:
                self.switch_mode()
    
    def switch_mode(self):
        if self.current_mode == "WORK":
            self.current_mode = "SHORT_BREAK"
            self.time_remaining = self.SHORT_BREAK * 60
        elif self.current_mode == "SHORT_BREAK":
            self.current_mode = "WORK"
            self.time_remaining = self.WORK_TIME * 60
    
    def format_time(self):
        minutes = self.time_remaining // 60
        seconds = self.time_remaining % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            frame = self.detect_gesture(frame)
            
            # Update timer
            if self.is_running:
                self.update_timer()
            
            # Draw timer and status
            cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {self.format_time()}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {'Running' if self.is_running else 'Stopped'}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Focus Mode: {'On' if self.focus_mode else 'Off'}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Pomodoro Timer", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    timer = PomodoroTimer()
    timer.run()