
# Gesture-Based Computer Control System

This project leverages hand gestures to control various computer functionalities, including screen recording, screenshot capturing, photo taking, and text extraction from the webcam feed. The system uses gesture recognition to simplify common actions, providing an innovative way to interact with a computer.

---

## Demo Video 🎬

Explore a detailed **Demo Video** showcasing the key features of the Gesture-Based Computer Control System. Watch it in action, demonstrating gesture-controlled screen recording, automated screenshot capturing, and more! Click below:

[View Sample Video on Google Drive](https://drive.google.com/file/d/1sWiR8qSKhti5zCXLqUtJMzrJANscraKG/view)

Alternatively, you can download and play the video offline using the link above.

<video width="600" controls autoplay loop>
  <source src="https://drive.google.com/file/d/1sWiR8qSKhti5zCXLqUtJMzrJANscraKG/view" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Note*: The demo video focuses on core gesture-based functionalities. Due to technical limitations, the screen recording feature isn't shown in action as it interferes with video recording. The text-extraction feature, while implemented, has been left out due to current accuracy limitations, although code is provided for further experimentation.

---

## Project Features

### 1. Screen Recording
Control your screen recording process using simple hand gestures. Start, pause, and stop recording without the need for keyboard or mouse input.

---

### 2. Screenshot Capture
Take screenshots instantly by performing a specific hand gesture. The screenshot is saved to your device and can be accessed anytime.

---

### 3. Photo Capture
Use gestures to capture photos directly from the webcam.
Also specially the for increasing the quality of the image i have tried to brighten the image,just to add some post-processing 

---

### 4. Text Extraction (OCR)
Extract text from webcam feeds. Due to the limitations of the libraries (`pytesseract` and `easyocr`), OCR accuracy may vary depending on the text quality and lighting conditions.

## Future Enhancements

1. **Screenshot Annotation Mode**
2. **Save Screenshot in Multiple Formats**
3. **Smart Cropping**
4. **Improve the accuracy in my files for Automated Text Extraction and Sharing (OCR)**
5. **Automatic Upload to Cloud Storage using google-api-python-client(almost completed)**

## Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python main.py
   ```

## Contributing

Contributions are welcome! Please fork this repository and create a pull request with your feature.

---

## License

This project is licensed to me so please don't copy, if you contribute then you will have the access to use it so help to make it better.

---

