import cv2
import dlib
import numpy as np
import time
from ultralytics import YOLO

class HeadPoseEstimator:
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        self.detector = YOLO(yolo_model_path)
        self.predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        self.model_points = self._get_3d_model_points()
        
        # Camera parameters setup
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros((4, 1), dtype=np.float64)

        # Thresholds for head pose angles
        self.PITCH_THRESHOLD = 13.0  # Up/down
        self.YAW_THRESHOLD = 40.0    # Left/right
        self.ROLL_THRESHOLD = 18.0   # Tilt

        # Timing for persistent warning
        self.pose_start_time = None
        self.POSE_WARNING_THRESHOLD = 2.0  # seconds
        self.current_warnings = []

        # Smoothing parameters for head pose angles
        self.prev_angles = None
        self.smoothing_factor = 0.7  # Adjust between 0 (no smoothing) and 1 (max smoothing)

    def _get_3d_model_points(self):
        return np.array([
            (0.0, 0.0, 0.0),         # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)

    def _convert_yolo_to_dlib_rect(self, yolo_box):
        x1, y1, x2, y2 = map(int, yolo_box)
        return dlib.rectangle(x1, y1, x2, y2)

    def _get_2d_image_points(self, landmarks):
        return np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

    def _calculate_euler_angles(self, rotation_matrix):
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
            z = 0

        return np.degrees([x, y, z])

    def _draw_axes(self, frame, origin, rvec, tvec, length=100):
        axis_points = np.float32([
            [0, 0, 0], 
            [length, 0, 0], 
            [0, length, 0], 
            [0, 0, length]
        ]).reshape(-1, 3)
        
        img_pts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, 
            self.camera_matrix, self.distortion_coeffs
        )
        
        origin = tuple(map(int, origin))
        img_pts = img_pts.reshape(-1, 2)
        
        # Draw the three axes (X: Red, Y: Green, Z: Blue)
        cv2.line(frame, origin, tuple(img_pts[1].astype(int)), (0, 0, 255), 3)
        cv2.line(frame, origin, tuple(img_pts[2].astype(int)), (0, 255, 0), 3)
        cv2.line(frame, origin, tuple(img_pts[3].astype(int)), (255, 0, 0), 3)

    def _check_head_position(self, angles):
        warnings = []
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        # Check angle thresholds and generate warnings
        if abs(pitch) > self.PITCH_THRESHOLD:
            warnings.append(f"Head too far {'Up' if pitch > 0 else 'Down'}!")
        if abs(yaw) > self.YAW_THRESHOLD:
            warnings.append(f"Head too far {'Right' if yaw > 0 else 'Left'}!")
        if abs(roll) > self.ROLL_THRESHOLD:
            warnings.append(f"Head tilted {'Right' if roll > 0 else 'Left'}!")

        # Update warning timer based on head position
        if warnings:
            if self.pose_start_time is None:
                self.pose_start_time = time.time()
        else:
            # Reset timer when head returns to normal position
            self.pose_start_time = None
            self.current_warnings = []

        return warnings

    def estimate_head_pose(self, frame):
        # Convert the frame once to grayscale for landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)

        # Check if any face is detected
        if not results or not results[0].boxes:
            cv2.putText(frame, "Driver Out of Frame!", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.pose_start_time = None  # Reset timer when driver is out of frame
            self.prev_angles = None     # Reset smoothing state
            return frame

        for result in results:
            for box in result.boxes:
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])
                landmarks = self.predictor(gray, dlib_rect)
                
                image_points = self._get_2d_image_points(landmarks)
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.distortion_coeffs
                )

                rot_mat, _ = cv2.Rodrigues(rvec)
                angles = self._calculate_euler_angles(rot_mat)

                # Smooth the angles to reduce jitter
                if self.prev_angles is None:
                    smoothed_angles = angles
                else:
                    smoothed_angles = self.smoothing_factor * angles + (1 - self.smoothing_factor) * self.prev_angles
                self.prev_angles = smoothed_angles

                warnings = self._check_head_position(smoothed_angles)
                elapsed_time = time.time() - self.pose_start_time if self.pose_start_time else 0
                
                # Reset timer if head position is normal
                if not warnings:
                    self.pose_start_time = None
                    self.current_warnings = []
                    
                # Only show warnings after threshold
                if warnings and elapsed_time >= self.POSE_WARNING_THRESHOLD:
                    self.current_warnings = warnings
                    color = (0, 0, 255)  # Red for warning
                    y_pos = 90
                    pose_text = f"Pitch: {smoothed_angles[0]:.1f}°, Yaw: {smoothed_angles[1]:.1f}°, Roll: {smoothed_angles[2]:.1f}°"
                    cv2.putText(frame, pose_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    for i, warn in enumerate(warnings):
                        cv2.putText(frame, f"{warn} ({elapsed_time:.1f}s)", 
                                    (10, y_pos + 30 * (i + 1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    color = (0, 255, 0)  # Green for normal

                # Draw bounding box around the face
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw axes using the nose tip as the origin
                self._draw_axes(frame, image_points[0], rvec, tvec)

        return frame

class ComprehensiveFacialAnalysis(HeadPoseEstimator):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.landmark_colors = {
            'jaw': (255, 0, 0), 
            'eyebrows': (0, 255, 0),
            'nose': (255, 255, 0), 
            'eyes': (0, 0, 255),
            'mouth': (0, 255, 255)
        }

    def _draw_landmarks(self, frame, landmarks):
        # Draw jawline points
        for i in range(0, 17):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 
                       2, self.landmark_colors['jaw'], -1)
        # Draw eye landmarks
        for region in [range(36, 42), range(42, 48)]:
            for i in region:
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 
                           2, self.landmark_colors['eyes'], -1)

    def estimate_head_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)

        if not results or not results[0].boxes:
            cv2.putText(frame, "Driver Out of Frame!", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.pose_start_time = None
            self.prev_angles = None
            return frame

        for result in results:
            for box in result.boxes:
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])
                landmarks = self.predictor(gray, dlib_rect)
                
                self._draw_landmarks(frame, landmarks)
                image_points = self._get_2d_image_points(landmarks)
                
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.distortion_coeffs
                )

                rot_mat, _ = cv2.Rodrigues(rvec)
                angles = self._calculate_euler_angles(rot_mat)

                if self.prev_angles is None:
                    smoothed_angles = angles
                else:
                    smoothed_angles = self.smoothing_factor * angles + (1 - self.smoothing_factor) * self.prev_angles
                self.prev_angles = smoothed_angles

                warnings = self._check_head_position(smoothed_angles)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if warnings else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                pose_text = f"Pitch: {smoothed_angles[0]:.1f}°, Yaw: {smoothed_angles[1]:.1f}°, Roll: {smoothed_angles[2]:.1f}°"
                cv2.putText(frame, pose_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

# Usage example
if __name__ == "__main__":
    estimator = HeadPoseEstimator('best.pt')
    # Alternatively, for more comprehensive facial analysis:
    # estimator = ComprehensiveFacialAnalysis('yolov8n-face.pt')

    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = estimator.estimate_head_pose(frame)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Head Pose Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
