import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mss import mss
import threading

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize recording state
is_recording = False
recording_thread = None
output_folder = "CapturedVideos"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Screen recording function (runs in a separate thread)
def record_screen(file_name):
    global is_recording
    sct = mss()
    screen_size = (1920, 1080)  # Adjust to your screen resolution
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(file_name, fourcc, 20.0, screen_size)

    while is_recording:
        screenshot = sct.grab(sct.monitors[1])  # Capture the primary monitor
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out.write(frame)

    out.release()

# Gesture detection functions
def detect_thumbs_up(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < index_tip.y and abs(thumb_tip.x - index_tip.x) < 0.05  # Ensure thumb is above index finger with slight x-axis tolerance

def detect_hand_closing(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    fingers_closed_count = 0

    # Check if all fingers are pointing down
    fingertip_landmarks = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    for fingertip in fingertip_landmarks:
        if landmarks[fingertip].y > wrist.y:  # Check if any fingertip is below the wrist
            fingers_closed_count += 1

    # Consider it a "closing" gesture if at least 3 fingers are below the wrist
    return fingers_closed_count >= 3

# Start recording in a separate thread
def start_recording():
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        file_name = os.path.join(output_folder, f"recording_{int(time.time())}.avi")
        print(f"Recording started: {file_name}")
        recording_thread = threading.Thread(target=record_screen, args=(file_name,))
        recording_thread.start()

# Stop recording
def stop_recording():
    global is_recording, recording_thread
    if is_recording:
        is_recording = False
        if recording_thread:
            recording_thread.join()
        print("Recording stopped and saved.")

# Main function to handle recording based on gestures
def handle_recording(frame):
    global is_recording
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Detect gestures and toggle recording
            if detect_thumbs_up(hand_landmarks.landmark) and not is_recording:
                start_recording()
            elif detect_hand_closing(hand_landmarks.landmark) and is_recording:
                stop_recording()
                return True  # Indicate that the recording has stopped

# Start video capture for gesture detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    if handle_recording(frame):  # Check for gestures and update recording status
        break  # Exit loop if recording has been stopped

    # Show the webcam feed with gesture detection and recording status
    cv2.imshow("Gesture Detection (Press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Stop recording if 'q' is pressed
        if is_recording:
            stop_recording()
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
stop_recording()
