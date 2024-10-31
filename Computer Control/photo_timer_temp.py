import os
from google.cloud import storage
import cv2
import numpy as np
from datetime import datetime

# Setup: specify the local directories and your Google Cloud bucket name
CLOUD_UPLOADS_FOLDER = 'CloudUploads'
CAPTURED_PICS_FOLDER = 'captured_pics'
BUCKET_NAME = 'my-cloud-uploads-bucket'  # Replace with your actual bucket name

# Initialize the Google Cloud Storage client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"  # Replace with your JSON key filename
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Ensure the directories exist
os.makedirs(CLOUD_UPLOADS_FOLDER, exist_ok=True)
os.makedirs(CAPTURED_PICS_FOLDER, exist_ok=True)

# Function to upload file to Google Cloud Storage
def upload_to_bucket(blob_name, file_path):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    print(f'Uploaded {file_path} to {blob_name} in Google Cloud Storage.')

# Capture and save a photo to both local folders
def capture_and_save_photo():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        cap.release()
        return

    # Create a unique filename based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'photo_{timestamp}.jpg'
    
    # Paths for both folders
    cloud_uploads_path = os.path.join(CLOUD_UPLOADS_FOLDER, filename)
    captured_pics_path = os.path.join(CAPTURED_PICS_FOLDER, filename)

    # Save the photo in both directories
    cv2.imwrite(cloud_uploads_path, frame)
    cv2.imwrite(captured_pics_path, frame)
    print(f'Photo saved as {cloud_uploads_path} and {captured_pics_path}')
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()
    
    # Upload the photo to the Google Cloud Storage bucket
    upload_to_bucket(f'photos/{filename}', cloud_uploads_path)

# Run the capture and save function
capture_and_save_photo()
