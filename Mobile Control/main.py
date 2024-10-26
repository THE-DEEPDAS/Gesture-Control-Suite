import cv2
import mediapipe as mp
import time
import os

output_folder = "CapturedImages"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Timer variables for photo countdown
timer_start = 0
is_counting_down = False
countdown_duration = 3  # 3 seconds countdown
photo_cooldown = 1  # Extra delay to hide countdown text before taking the photo

# Gesture Detection Functions
def detect_snap(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    for finger_tip in [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                       mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                       mp_hands.HandLandmark.RING_FINGER_TIP, 
                       mp_hands.HandLandmark.PINKY_TIP]:
        distance = ((thumb_tip.x - landmarks[finger_tip].x) ** 2 +
                    (thumb_tip.y - landmarks[finger_tip].y) ** 2) ** 0.5
        if distance < 0.05:  # Snap detected if thumb is close to any fingertip
            return True
    return False

def detect_camera_hold(landmarks):
    # This function assumes that both hands are visible and positioned in a "box" shape
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    if abs(thumb_tip.x - pinky_tip.x) > 0.15 and abs(thumb_tip.y - pinky_tip.y) < 0.1:
        return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert the frame for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Disable gesture detection during countdown or photo cooldown
    if results.multi_hand_landmarks and not is_counting_down:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gestures
            if detect_snap(hand_landmarks.landmark):
                screenshot_name = os.path.join(output_folder, f"screenshot_{int(time.time())}.jpg")
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot taken and saved as {screenshot_name}")
            
            elif detect_camera_hold(hand_landmarks.landmark) and not is_counting_down:
                timer_start = time.time()
                is_counting_down = True

    # Countdown timer display for photo/selfie
    if is_counting_down:
        elapsed_time = time.time() - timer_start
        countdown_remaining = countdown_duration - int(elapsed_time)

        # Show countdown timer on the frame
        if countdown_remaining > 0:
            cv2.putText(frame, f"Taking photo in {countdown_remaining}...", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        elif elapsed_time < countdown_duration + photo_cooldown:
            # Adding extra delay to hide countdown text before taking the photo
            cv2.putText(frame, "Smile!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        else:
            # Capture photo after the countdown and cooldown are complete
            photo_name = os.path.join(output_folder, f"photo_{int(time.time())}.jpg")
            cv2.imwrite(photo_name, frame)
            print(f"Photo taken and saved as {photo_name}")
            is_counting_down = False  # Reset countdown

    # Show the video feed
    cv2.imshow("Gesture Camera Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()