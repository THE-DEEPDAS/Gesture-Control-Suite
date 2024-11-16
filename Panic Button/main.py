import time
import pyautogui
import cv2
import numpy as np
from collections import Counter
import deepface_VGGFace as d1
import deepface_Facenet as d2
import deepface_OpenFace as d3

# Initialize variables
panic_duration = 3  # Seconds to detect sustained panic
camera_state = "on"  # Tracks current camera state
panic_emotions = ["fear", "angry", "surprise"]  # Emotions considered as "panic"

def capture_meet_window():
    """
    Captures the Google Meet window where the user's video is shown.
    Returns the region of the screenshot containing the user's video.
    """
    try:
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        # Convert the screenshot to a numpy array (OpenCV format)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # TODO: You'll need to adjust these coordinates based on your Meet layout
        # These are example coordinates - adjust them to match your Meet window
        # Format: (x, y, width, height)
        video_region = (100, 100, 400, 400)  # Example coordinates
        
        # Crop the frame to the video region
        frame = frame[video_region[1]:video_region[1]+video_region[3], video_region[0]:video_region[0]+video_region[2]]
        
        return frame
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None

def analyze_emotions(frame):
    predictions = []

    try:
        # Analyze emotion using the specific model
        result = d1.analyze(frame, actions=['emotion'], enforce_detection=False)
        predictions.append(result[0]['dominant_emotion'])
    except Exception as e:
        print(f"Model VGG-Face failed: {e}")

    try:
        # Analyze emotion using the specific model
        result = d2.analyze(frame, actions=['emotion'], enforce_detection=False)
        predictions.append(result[0]['dominant_emotion'])
    except Exception as e:
        print(f"Model Facenet failed: {e}")

    try:
        # Analyze emotion using the specific model
        result = d3.analyze(frame, actions=['emotion'], enforce_detection=False)
        predictions.append(result[0]['dominant_emotion'])
    except Exception as e:
        print(f"Model OpenFace failed: {e}")

    # Combine predictions using majority voting
    if predictions:
        final_emotion = Counter(predictions).most_common(1)[0][0]
    else:
        final_emotion = "Unknown"

    return final_emotion

def toggle_camera(state):
    """
    Toggles the camera on Google Meet.
    :param state: "off" to close the camera, "on" to open the camera.
    """
    global camera_state
    if state != camera_state:
        pyautogui.hotkey('ctrl', 'e')  # Simulates Ctrl + E hotkey
        camera_state = state
        print(f"Camera toggled {state}.")
        time.sleep(1)  # Wait for the camera state to change

def detect_panic():
    """
    Continuously detects emotions from Meet window screenshots and toggles the camera if panic is detected.
    """
    start_time = time.time()
    while True:
        frame = capture_meet_window()
        if frame is None:
            print("Failed to capture Meet window.")
            time.sleep(1)
            continue

        try:
            # Analyze the frame for emotion
            # result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analyze_emotions(frame)
            print(f"Detected emotion: {emotion}")

            if emotion in panic_emotions:
                if time.time() - start_time >= panic_duration:
                    print("Panic detected for 3 seconds. Turning off camera.")
                    toggle_camera("off")
                    return
            else:
                start_time = time.time()

            # Display the frame for debugging
            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during emotion detection: {e}")
            start_time = time.time()
        
        time.sleep(0.5)  # Add a small delay to reduce CPU usage

def monitor_reopen():
    """
    Continuously monitors for non-panic emotions to reopen the camera.
    """
    while True:
        frame = capture_meet_window()
        if frame is None:
            time.sleep(1)
            continue

        try:
            # result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # emotion = result[0]["dominant_emotion"]
            emotion = analyze_emotions(frame)
            print(f"Recheck emotion: {emotion}")

            if emotion not in panic_emotions:
                print("No panic detected. Reopening camera.")
                toggle_camera("on")
                return

            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error during emotion detection: {e}")
        
        time.sleep(0.5)

def setup():
    """
    Initial setup and instructions for the user.
    """
    print("Setup Instructions:")
    print("1. Open Google Meet and ensure your video is visible")
    print("2. Position the Meet window so your video is clearly visible")
    print("3. Press Enter when ready to start monitoring")
    input("Press Enter to continue...")
    
    # Give user time to switch back to Meet window
    print("Starting in 3 seconds...")
    time.sleep(3)

# Main script
try:
    setup()  # Run initial setup

    while True:
        if camera_state == "on":
            detect_panic()
        time.sleep(1)
        if camera_state == "off":
            monitor_reopen()

finally:
    cv2.destroyAllWindows()