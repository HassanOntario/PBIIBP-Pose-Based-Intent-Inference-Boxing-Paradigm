import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

# Keypoint indices for reference
KEYPOINT_NAMES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# Define EDGES constants
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


class MoveNetDetector:
    """MoveNet pose detection wrapper for easy access to pose data."""
    
    def __init__(self, model_path='4.tflite'):
        """Initialize the MoveNet interpreter.
        
        Args:
            model_path: Path to the TFLite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.keypoints = None
        self.frame_shape = None
    
    def detect(self, frame):
        """Run pose detection on a frame.
        
        Args:
            frame: BGR image from OpenCV (numpy array).
            
        Returns:
            numpy array of shape (17, 3) with [y, x, confidence] for each keypoint.
        """
        self.frame_shape = frame.shape
        
        # Preprocess image
        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)
        input_image = tf.cast(img, dtype=tf.uint8)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Store and return keypoints (17, 3)
        self.keypoints = np.squeeze(keypoints_with_scores)
        return self.keypoints
    
    def get_keypoint(self, index, pixel_coords=False):
        """Get a specific keypoint by index.
        
        Args:
            index: Keypoint index (0-16).
            pixel_coords: If True, return pixel coordinates instead of normalized.
            
        Returns:
            tuple of (y, x, confidence) or (pixel_y, pixel_x, confidence).
        """
        if self.keypoints is None:
            raise ValueError("No keypoints available. Call detect() first.")
        
        y, x, conf = self.keypoints[index]
        
        if pixel_coords and self.frame_shape is not None:
            h, w, _ = self.frame_shape
            return (int(y * h), int(x * w), conf)
        
        return (y, x, conf)
    
    def get_keypoint_by_name(self, name, pixel_coords=False):
        """Get a keypoint by name.
        
        Args:
            name: Keypoint name (e.g., 'left_wrist', 'nose').
            pixel_coords: If True, return pixel coordinates.
            
        Returns:
            tuple of (y, x, confidence).
        """
        name_to_index = {v: k for k, v in KEYPOINT_NAMES.items()}
        if name not in name_to_index:
            raise ValueError(f"Unknown keypoint name: {name}")
        return self.get_keypoint(name_to_index[name], pixel_coords)
    
    def get_all_keypoints(self, pixel_coords=False):
        """Get all keypoints as a dictionary.
        
        Args:
            pixel_coords: If True, return pixel coordinates.
            
        Returns:
            dict mapping keypoint names to (y, x, confidence).
        """
        if self.keypoints is None:
            raise ValueError("No keypoints available. Call detect() first.")
        
        result = {}
        for idx, name in KEYPOINT_NAMES.items():
            result[name] = self.get_keypoint(idx, pixel_coords)
        return result
    
    def draw_pose(self, frame, confidence_threshold=0.4):
        """Draw keypoints and connections on the frame.
        
        Args:
            frame: BGR image to draw on (modified in place).
            confidence_threshold: Minimum confidence to draw.
        """
        if self.keypoints is None:
            return
        
        draw_connections(frame, self.keypoints, EDGES, confidence_threshold)
        draw_keypoints(frame, self.keypoints, confidence_threshold)


def draw_keypoints(frame, keypoints, confidence_threshold):
    """Draw keypoints on frame."""
    y, x, c = frame.shape
    shaped = np.multiply(keypoints, [y, x, 1])
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    """Draw skeleton connections on frame."""
    y, x, c = frame.shape
    shaped = np.multiply(keypoints, [y, x, 1])
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


def play_beep():
    """Play an auditory beep sound on macOS."""
    os.system('afplay /System/Library/Sounds/Ping.aiff &')


def run_demo():
    """Run the pose detection demo with webcam."""
    detector = MoveNetDetector(model_path='4.tflite')
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        keypoints = detector.detect(frame)
        
        # Access specific keypoints
        left_wrist = detector.get_keypoint_by_name('left_wrist')
        right_wrist = detector.get_keypoint_by_name('right_wrist')
        
        confidence_threshold = 0.4
        left_hand_visible = left_wrist[2] > confidence_threshold
        right_hand_visible = right_wrist[2] > confidence_threshold
        
        # Detect if hand(s) disappeared
        if not left_hand_visible or not right_hand_visible:
            play_beep()
            print(f"Hand disappeared! Left: {left_hand_visible}, Right: {right_hand_visible}")
        
        # Draw pose on frame
        detector.draw_pose(frame, confidence_threshold)
        
        cv2.imshow('MoveNet Lightning', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Only run demo if executed directly
if __name__ == '__main__':
    run_demo()