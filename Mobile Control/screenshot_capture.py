import cv2
import os
import time
import mediapipe as mp
from collections import Counter

output_folder = "CapturedImages"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to detect raised fingers
def detect_finger_count(landmarks):
    fingers = [landmarks[i].y < landmarks[i - 2].y for i in 
               [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP]]
    return fingers.count(True)

# Capture screenshot with a delay and save it to the output folder
def capture_screenshot(frame):
    time.sleep(0.2)  # Small buffer time for smoother screenshot capture
    screenshot_name = os.path.join(output_folder, f"screenshot_{int(time.time())}.jpg")
    cv2.imwrite(screenshot_name, frame)
    print(f"Screenshot taken and saved as {screenshot_name}")

# Detect the most common finger count over a short sampling period
def get_most_common_finger_count(cap, duration=2):
    end_time = time.time() + duration
    finger_counts = []

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Process the frame for hand landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            total_fingers = 0
            for hand_landmarks in results.multi_hand_landmarks:
                total_fingers += detect_finger_count(hand_landmarks.landmark)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_counts.append(total_fingers)
        
        # Show the live feed
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Determine the most common finger count
    if finger_counts:
        most_common_count = Counter(finger_counts).most_common(1)[0][0]
        print(f"Most common finger count detected: {most_common_count}")
        return most_common_count
    return 0  # Default if no hand detected

# Main program execution
cap = cv2.VideoCapture(0)

while True:
    # Detect most common finger count
    common_fingers = get_most_common_finger_count(cap)
    
    if common_fingers > 0:
        # Cap the timer to 10 seconds maximum and capture screenshot
        timer_seconds = min(common_fingers, 10)
        print(f"Waiting {timer_seconds} seconds before taking screenshot...")
        time.sleep(timer_seconds)
        ret, frame = cap.read()
        if ret:
            capture_screenshot(frame)
        break  # Exit after taking one screenshot

# Cleanup
cap.release()
cv2.destroyAllWindows()
