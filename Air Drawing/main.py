import cv2
import numpy as np
import mediapipe as mp
import time
import os
from datetime import datetime
from collections import deque

class AirDraw:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.canvas = None
        
        # basic mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # colour and drawing setup
        self.colors = {
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255)
        }
        self.current_color = 'blue'
        self.brush_thickness = 5 
        self.prev_point = None
        
        # for tracking the gestures
        self.prev_gesture = None
        self.gesture_buffer = deque(maxlen=3)
        self.last_color_change = time.time()
        self.color_change_cooldown = 1.5
        
        # canvas is initialised for drawing
        _, frame = self.cap.read()
        self.canvas = np.zeros_like(frame)
        
        # for saving the file
        self.output_dir = "saved_drawings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_finger_angles(self, hand_landmarks):
        # calculating angles between fingers so we can detect the gesture
        def get_angle(p1, p2, p3):
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            angle = np.degrees(np.arctan2(np.cross(v1, v2), np.dot(v1, v2)))
            return abs(angle)

        landmarks = hand_landmarks.landmark
        
        thumb_index_angle = get_angle(
            landmarks[4],  # thumb tip
            landmarks[5],  # index MCP
            landmarks[8]   # index tip
        )
        
        return thumb_index_angle

    def get_finger_states(self, hand_landmarks):
        finger_states = []
        landmarks = hand_landmarks.landmark
        
        # checking thumb
        thumb_tip_x = landmarks[4].x
        thumb_base_x = landmarks[2].x
        finger_states.append(thumb_tip_x < thumb_base_x)
        
        # checking other fingers using pip (middle knuckle) as reference
        # 8: index tip, 6: index pip checks if index finger is up
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            finger_states.append(landmarks[tip].y < landmarks[pip].y)
        
        return finger_states

    def detect_gesture(self, hand_landmarks, finger_states):
        # V shape recognition
        if not finger_states:
            return 'none'
        
        thumb_index_angle = self.get_finger_angles(hand_landmarks)
        
        # drawing gesture: only index finger up
        if finger_states == [False, True, False, False, False]:
            return 'draw'
        
        # for pause
        if (finger_states[0] and finger_states[1] and not any(finger_states[2:]) 
            and 35 < thumb_index_angle < 65):  # Angle range for V shape
            return 'pause'
        
        # peace sign detection
        if finger_states == [False, True, True, False, False]:
            return 'color_change'
        
        # clearing the canvas
        if all(finger_states):
            return 'clear'
        
        return 'none'

    def get_index_finger_tip(self, hand_landmarks, frame_shape):
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * frame_shape[1])
        y = int(index_tip.y * frame_shape[0])
        return (x, y)

    def save_drawing(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawing_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        result = cv2.addWeighted(frame, 1, self.canvas, 1, 0)
        cv2.imwrite(filepath, result)
        return filepath

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            current_time = time.time()
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                finger_states = self.get_finger_states(hand_landmarks)
                current_gesture = self.detect_gesture(hand_landmarks, finger_states)
                
                index_tip = self.get_index_finger_tip(hand_landmarks, frame.shape)
                
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if current_gesture == 'draw':
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, index_tip, 
                               self.colors[self.current_color], self.brush_thickness)
                    self.prev_point = index_tip
                    
                elif current_gesture == 'pause':
                    self.prev_point = None
                    
                elif current_gesture == 'color_change':
                    if current_time - self.last_color_change >= self.color_change_cooldown:
                        colors_list = list(self.colors.keys())
                        current_idx = colors_list.index(self.current_color)
                        self.current_color = colors_list[(current_idx + 1) % len(colors_list)]
                        self.last_color_change = current_time
                    self.prev_point = None
                    
                elif current_gesture == 'clear':
                    self.canvas = np.zeros_like(frame)
                    self.prev_point = None
                    
                else:
                    self.prev_point = None
                
                if current_gesture == 'draw':
                    cv2.circle(frame, index_tip, 5, self.colors[self.current_color], -1)
            else:
                self.prev_point = None
            
            result = cv2.addWeighted(frame, 1, self.canvas, 0.5, 0)
            
                                                                #left, top thi ketla pixels chodvana
            cv2.putText(result, f"Color: {self.current_color}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Index finger: Draw", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Thumb+Index (V): Pause", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Peace sign: Change color", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Open palm: Clear canvas", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Press 's' to save", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("AirDraw", result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                saved_path = self.save_drawing(frame)
                print(f"Drawing saved to: {saved_path}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    air_draw = AirDraw()
    air_draw.run()