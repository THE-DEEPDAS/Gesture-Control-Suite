import cv2
import os
import time
import mediapipe as mp

output_folder = "CapturedImages"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def detect_finger_count(landmarks):
    fingers = [landmarks[i].y < landmarks[i - 2].y for i in 
               [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP]]
    return fingers.count(True)

def capture_screenshot_with_delay(frame, delay_seconds):
    timer_start = time.time()
    while time.time() - timer_start < delay_seconds:
        cv2.putText(frame, f"Taking photo in {int(delay_seconds - (time.time() - timer_start))}...", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.imshow("Screenshot Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    screenshot_name = os.path.join(output_folder, f"screenshot_{int(time.time())}.jpg")
    cv2.imwrite(screenshot_name, frame)
    print(f"Screenshot taken and saved as {screenshot_name}")

def take_screenshot(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_up = detect_finger_count(hand_landmarks.landmark)
            if fingers_up > 0:
                print(f"Setting timer to {fingers_up} seconds")
                capture_screenshot_with_delay(frame, fingers_up)
                return True
    return False
