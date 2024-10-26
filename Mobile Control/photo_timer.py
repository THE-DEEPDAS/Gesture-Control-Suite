import cv2
import os
import time

class PhotoTimer:
    def __init__(self, output_folder="CapturedImages", countdown_duration=3, photo_cooldown=1):
        self.output_folder = output_folder
        self.countdown_duration = countdown_duration
        self.photo_cooldown = photo_cooldown
        self.timer_start = 0
        self.is_counting_down = False
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def start_countdown(self):
        """Starts the countdown for taking a photo."""
        self.timer_start = time.time()
        self.is_counting_down = True

    def update_frame(self, frame):
        """Updates the frame with countdown or smile text, and captures photo when time is up."""
        if self.is_counting_down:
            elapsed_time = time.time() - self.timer_start
            countdown_remaining = self.countdown_duration - int(elapsed_time)

            # Display countdown or smile on frame
            if countdown_remaining > 0:
                self._display_countdown(frame, countdown_remaining)
                return True  # Countdown is still active
            elif elapsed_time < self.countdown_duration + self.photo_cooldown:
                self._display_smile(frame)
                return True
            else:
                self.capture_photo(frame)
                self.is_counting_down = False  # Reset countdown
                return False  # Countdown finished and photo captured
        return False

    def capture_photo(self, frame):
        """Saves the current frame as a photo."""
        photo_name = os.path.join(self.output_folder, f"photo_{int(time.time())}.jpg")
        cv2.imwrite(photo_name, frame)
        print(f"Photo taken and saved as {photo_name}")

    def _display_countdown(self, frame, countdown_remaining):
        """Displays countdown text on the frame."""
        cv2.putText(frame, f"Taking photo in {countdown_remaining}...", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    def _display_smile(self, frame):
        """Displays 'Smile!' text on the frame."""
        cv2.putText(frame, "Smile!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
