# This is the file with all the submodules combined just test-extractor is not included as it is
# work under progress, note: you can access the code of each feature separately also using the submodules
import cv2
import numpy as np
import time
import os
from datetime import datetime
import mediapipe as mp
import threading
from mss import mss
import sys
from enum import Enum

class CaptureMode(Enum):
    FACE_CAPTURE = 1
    HAND_GESTURE_PHOTO = 2
    SCREEN_RECORDING = 3

class IntegratedCapture:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create output directories
        self.photo_dir = 'captured_pics'
        self.gesture_dir = 'CapturedImages'
        self.video_dir = 'CapturedVideos'
        for directory in [self.photo_dir, self.gesture_dir, self.video_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None

    def capture_face_photo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Preparing to take a photo in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"Timer: {i} seconds")
            time.sleep(1)

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            cap.release()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_region = frame[y:y + h, x:x + w]
            face_region = cv2.convertScaleAbs(face_region, alpha=1.5, beta=30)
            frame[y:y + h, x:x + w] = face_region

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.photo_dir, f'captured_photo_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        print(f"Photo captured and saved as: {filename}")
        cap.release()

    def detect_finger_count(self, landmarks):
        fingers = [landmarks[i].y < landmarks[i - 2].y for i in [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]]
        return fingers.count(True)

    def capture_gesture_photo(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                total_fingers = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    total_fingers += self.detect_finger_count(hand_landmarks.landmark)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if total_fingers > 0:
                    timer_seconds = min(total_fingers, 10)
                    print(f"Waiting {timer_seconds} seconds before taking photo...")
                    time.sleep(timer_seconds)
                    
                    screenshot_name = os.path.join(self.gesture_dir, f"gesture_photo_{int(time.time())}.jpg")
                    cv2.imwrite(screenshot_name, frame)
                    print(f"Photo taken and saved as {screenshot_name}")
                    break

            cv2.imshow("Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def record_screen(self, file_name):
        sct = mss()
        screen_size = (1920, 1080)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(file_name, fourcc, 20.0, screen_size)

        while self.is_recording:
            screenshot = sct.grab(sct.monitors[1])
            img = np.array(screenshot)
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.write(frame)

        out.release()

    def detect_thumbs_up(self, landmarks):
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        return thumb_tip.y < index_tip.y and abs(thumb_tip.x - index_tip.x) < 0.05

    def detect_hand_closing(self, landmarks):
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        fingers_closed_count = 0
        fingertip_landmarks = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        for fingertip in fingertip_landmarks:
            if landmarks[fingertip].y > wrist.y:
                fingers_closed_count += 1
        return fingers_closed_count >= 3

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            file_name = os.path.join(self.video_dir, f"recording_{int(time.time())}.avi")
            print(f"Recording started: {file_name}")
            self.recording_thread = threading.Thread(target=self.record_screen, args=(file_name,))
            self.recording_thread.start()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread:
                self.recording_thread.join()
            print("Recording stopped and saved.")

    def screen_recording_mode(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.detect_thumbs_up(hand_landmarks.landmark) and not self.is_recording:
                        self.start_recording()
                    elif self.detect_hand_closing(hand_landmarks.landmark) and self.is_recording:
                        self.stop_recording()
                        return

            cv2.imshow("Gesture Detection (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.is_recording:
                    self.stop_recording()
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    capture = IntegratedCapture()
    
    while True:
        print("\nChoose capture mode:")
        print("1. Gesture Based Photo")
        print("2. Gesture based Screenshot")
        print("3. Text-Extractor")
        print("4. Screen Recording")
        print("5. Exit")
        
        
        try:
            choice = int(input("Enter your choice (1-4): "))
            if choice == 1:
                capture.capture_face_photo()
            elif choice == 2:
                capture.capture_gesture_photo()
            elif choice == 4:
                capture.screen_recording_mode()
            elif choice == 5:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()