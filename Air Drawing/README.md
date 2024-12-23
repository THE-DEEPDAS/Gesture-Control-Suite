# Air Drawing

Air Drawing is an innovative and interactive tool that allows users to create drawings in the air using hand gestures captured through a webcam. This project leverages the power of computer vision and gesture recognition to provide a touch-free and creative drawing experience.

---

## Demo Video

Check out a **Demo Video** highlighting the Air Drawing's main features, including drawing in real world scenarios using gestures, saving it into a file, and more. Click below to see it in action:

[View Sample Video](./air%20draw%20demo.mp4)

---

## Key Features

1. **Touch-Free Drawing**:

   - Draw directly in the air by moving your index finger.
   - The canvas is updated in real-time with your hand gestures.

2. **Gesture Controls**:

   - **Draw**: Raise your index finger to start drawing.
   - **Pause**: Show a V-shape gesture (thumb and index finger spread apart).
   - **Change Color**: Use a peace sign (index and middle fingers up) to cycle through available colors.
   - **Clear Canvas**: Open your palm to erase the entire canvas.

3. **Color Options**:

   - Switch between multiple colors, including blue, green, red, yellow, and purple.

4. **Save Drawings**:

   - Press `S` to save your artwork. Saved images are stored in the `saved_drawings` folder with a timestamped filename.

5. **User-Friendly Overlay**:
   - Displays instructions and the current drawing color on the screen.

---

## Installation

1. **Prerequisites**:

   - Python 3.7+
   - Webcam (for capturing hand gestures)

2. **Required Libraries**:

   - `opencv-python`
   - `numpy`
   - `mediapipe`

   Install dependencies using:

   ```bash
   pip install opencv-python numpy mediapipe
   ```

3. **Run the Program**:
   ```bash
   python air_drawing.py
   ```

---

## How It Works

1. **Hand Detection**:

   - Uses MediaPipe Hands for precise tracking of hand landmarks.

2. **Gesture Recognition**:

   - Detects finger states (up/down) and angles between thumb and index finger to identify gestures.

3. **Drawing Mechanics**:

   - Tracks the position of the index finger and draws lines on a virtual canvas.
   - Combines the canvas with the webcam feed for a seamless user experience.

4. **Saving Artwork**:
   - Merges the drawing with the webcam frame and saves it as a `.png` image.

---

## Use Cases

1. **Creative Expression**:

   - A fun tool for artists and hobbyists to sketch and experiment with digital art.

2. **Educational Applications**:

   - Engage students in interactive drawing activities without requiring traditional tools.

3. **Touch-Free Interaction**:

   - Ideal for environments where touch-free technology is preferred, such as healthcare or public settings.

4. **Rehabilitation Therapy**:
   - Assist patients in motor skill recovery through interactive and enjoyable hand movements.

---

## Controls and Instructions

- **Draw**: Move your index finger while keeping it raised.
- **Pause**: Use the V-shape gesture (thumb and index finger spread apart).
- **Change Color**: Use the peace sign (index and middle fingers up).
- **Clear Canvas**: Open your palm to reset the canvas.
- **Save Drawing**: Press `S` to save your artwork.
- **Quit**: Press `Q` to exit the program.

---

## Limitations

1. Requires good lighting for accurate hand tracking.
2. May struggle with overlapping hands or rapid movements.

---

## Future Enhancements

1. Add multi-hand support for collaborative drawing.
2. Enable shape recognition for drawing geometric figures.
3. Incorporate voice commands for enhanced interactivity.

---

## Contribution

Contributions are welcome! Feel free to fork the repository and submit a pull request with your improvements.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking.
- OpenCV for image processing support.
