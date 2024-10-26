import cv2
import numpy as np
import time
import os
import mediapipe as mp

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def detect_thumbs_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return thumb_tip.y < index_mcp.y

def detect_fist(landmarks):
    return all(landmarks[mp_hands.HandLandmark.THUMB_TIP].y > landmarks[mp_hands.HandLandmark.THUMB_MCP].y for _ in landmarks)

def record_screen(output_folder="CapturedVideos"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    screen_size = (1920, 1080)  # Change to your screen resolution
    file_name = os.path.join(output_folder, f"recording_{int(time.time())}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(file_name, fourcc, 20.0, screen_size)

    print("Recording started. Show fist to stop recording.")
    
    while True:
        # Capture the screen using OpenCV
        img = cv2.VideoCapture(0)  # This will capture from your default camera
        ret, frame = img.read()
        
        if not ret:
            print("Failed to capture frame.")
            break

        out.write(frame)
        cv2.imshow("Recording the video here ...", frame)

        # Check for 'q' to stop early if needed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print(f"Recording saved as {file_name}")

def handle_recording(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if detect_thumbs_up(hand_landmarks.landmark):
                print("Thumbs up detected - Starting recording.")
                record_screen()
                return True  # Indicate recording started
            elif detect_fist(hand_landmarks.landmark):
                print("Fist detected - Stopping recording.")
                return False  # Indicate recording stopped
    return True  # Continue recording if neither gesture is detected


cap = cv2.VideoCapture(0)  # Start video capture from the webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    if not handle_recording(frame):
        break

    cv2.imshow("Gestures to be inputted here.", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()