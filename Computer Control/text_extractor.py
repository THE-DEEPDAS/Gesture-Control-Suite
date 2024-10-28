import pytesseract
from PIL import Image
import cv2
import time

# Specify the path to the Tesseract executable file
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Parameters
output_file = "extracted_text.txt"
cooldown_seconds = 5  # Cooldown time between extractions
last_extraction_time = 0  # Track the last extraction time

def extract_text_from_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for better accuracy
    text = pytesseract.image_to_string(gray_frame, lang="eng")  # Extract text in English
    return text.strip()

def save_text_to_file(text):
    with open(output_file, "a") as file:
        file.write(text + "\n")  # Append text to file with a newline

# Open the video capture (webcam)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Cooldown mechanism
    current_time = time.time()
    if current_time - last_extraction_time >= cooldown_seconds:
        extracted_text = extract_text_from_frame(frame)
        
        # Check if text is non-empty before saving
        if extracted_text:
            print("Extracted Text:", extracted_text)  # Print for reference
            save_text_to_file(extracted_text)
            last_extraction_time = current_time  # Update the last extraction time

    # Display the live feed
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
