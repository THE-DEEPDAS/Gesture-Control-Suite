import easyocr
import cv2
import time
#make use of gpu, ask chatgpt how to do it

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language(s) to be used (English)

# Parameters
output_file = "extracted_text.txt"
cooldown_seconds = 5  # Cooldown time between extractions
last_extraction_time = 0  # Track the last extraction time

def extract_text_from_frame(frame):
    # Use EasyOCR to extract text from the frame
    results = reader.readtext(frame)
    text = " ".join([result[1] for result in results])  # Combine extracted texts
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
