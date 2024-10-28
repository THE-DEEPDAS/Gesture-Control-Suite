import cv2
import numpy as np
import time
import os
from datetime import datetime

# Create a directory for captured pictures if it doesn't exist
output_dir = 'captured_pics'
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Timer countdown
print("Preparing to take a photo in 3 seconds...")
for i in range(3, 0, -1):
    print(f"Timer: {i} seconds")
    time.sleep(1)

# Capture a single frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    cap.release()
    exit()

# Convert to grayscale for face detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Enhance lighting around detected faces
for (x, y, w, h) in faces:
    # Draw rectangle around the face (optional)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Enhance lighting in the face region
    face_region = frame[y:y + h, x:x + w]
    face_region = cv2.convertScaleAbs(face_region, alpha=1.5, beta=30)

    # Replace the face region in the original frame
    frame[y:y + h, x:x + w] = face_region

# Generate a unique filename using the current timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = os.path.join(output_dir, f'captured_photo_{timestamp}.jpg')
cv2.imwrite(filename, frame)

# Print acknowledgment message
print(f"Photo captured and saved as: {filename}")

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
