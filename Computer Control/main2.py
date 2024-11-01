# main focus here is to integrate the feature of text-extracction
import cv2
import numpy as np
import time
import os
from datetime import datetime
import mediapipe as mp
import threading
from mss import mss
import pytesseract
from enum import Enum

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

class CaptureMode(Enum):
    FACE_CAPTURE = 1
    HAND_GESTURE_PHOTO = 2
    TEXT_EXTRACTOR = 3
    SCREEN_RECORDING = 4

class IntegratedCapture:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Directories for saving outputs
        self.photo_dir = 'captured_pics'
        self.gesture_dir = 'CapturedImages'
        self.video_dir = 'CapturedVideos'
        self.text_extraction_dir = 'ExtractedTextImages'
        
        # Create directories if they do not exist
        for directory in [self.photo_dir, self.gesture_dir, self.video_dir, self.text_extraction_dir]:
            os.makedirs(directory, exist_ok=True)

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

    def capture_gesture_photo(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Waiting for gesture to capture photo (Press 'q' to quit)...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if self.detect_hand_open(hand_landmarks.landmark):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.join(self.gesture_dir, f'gesture_photo_{timestamp}.jpg')
                        cv2.imwrite(filename, frame)
                        print(f"Gesture photo captured and saved as: {filename}")
                        cap.release()
                        return

            cv2.imshow("Gesture Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def text_extractor(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Press 'q' to stop text extraction.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            # Convert to grayscale and apply preprocessing for better OCR accuracy
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY_INV)
            text = pytesseract.image_to_string(thresh, lang='eng')

            # Get bounding boxes for detected text
            boxes = pytesseract.image_to_boxes(thresh, lang='eng')
            h, w, _ = frame.shape

            for box in boxes.splitlines():
                b = box.split()
                x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                cv2.rectangle(frame, (x, h - y), (x2, h - y2), (0, 255, 0), 2)
            
            # Display detected text on the frame
            y0, dy = 30, 30
            for i, line in enumerate(text.splitlines()):
                y = y0 + i * dy
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Text Extraction", frame)

            # Save frame with highlighted text
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.text_extraction_dir, f'text_extraction_{timestamp}.jpg')
            cv2.imwrite(filename, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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
            choice = int(input("Enter your choice (1-5): "))
            if choice == 1:
                capture.capture_face_photo()
            elif choice == 2:
                capture.capture_gesture_photo()
            elif choice == 3:
                capture.text_extractor()
            elif choice == 4:
                capture.screen_recording_mode()
            elif choice == 5:
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
