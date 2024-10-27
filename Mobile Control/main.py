# More things that are to be implemented
# Screenshot Annotation Mode
# Screen Recording with Gesture Control
# Save Screenshot in Multiple Formats
# Smart Cropping
# Automatic Upload to Cloud Storage
# Gesture-Driven Timer Settings for Delayed Screenshots
# Image Filtering and Enhancement
# Automated Text Extraction and Sharing (OCR)

import cv2
import mediapipe as mp
import os
from screenshot_capture import take_screenshot
from photo_timer import PhotoTimer
from screen_record import handle_recording
from text_extractor import TextExtractor

# Define output folder for captured images and videos
output_folder = "CapturedImages"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Initialize the PhotoTimer class to handle photo capturing
photo_timer = PhotoTimer(output_folder=output_folder)
is_recording = False  # Flag to track recording state

# Initialize TextExtractor for live text extraction
extractor = TextExtractor()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame for a mirror-like effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Gesture-based screenshot capture and video recording
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Capture screenshot based on detected hand gestures
            if take_screenshot(frame):
                continue  # Screenshot was taken; skip to next frame

            # Handle video recording based on hand gestures
            is_recording = handle_recording(frame)  # Update recording status based on gestures

    # Update the PhotoTimer to manage photo countdown and capture
    if not photo_timer.update_frame(frame):
        print("Photo captured.")

    # Extract text from the current frame
    extractor.extract_text_from_frame(frame)

    # Display the frame with all overlays
    cv2.imshow("Gesture Camera Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q'

# Save extracted texts when exiting
extractor.save_extracted_text()

# Release resources and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
