import cv2
import dlib
import numpy as np
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

        # Thresholds
        self.PITCH_THRESHOLD = 20.0  # Up/down
        self.YAW_THRESHOLD = 20.0    # Left/right
        self.ROLL_THRESHOLD = 20.0   # Tilt

    def _get_3d_model_points(self):
        return np.array([
            (0.0, 0.0, 0.0),         (0.0, -330.0, -65.0),     
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),   
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)  
        ], dtype=np.float64)

    def _convert_yolo_to_dlib_rect(self, yolo_box):
        x1, y1, x2, y2 = map(int, yolo_box)
        return dlib.rectangle(x1, y1, x2, y2)

    def _get_2d_image_points(self, landmarks):
        return np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye
            (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth
        ], dtype=np.float64)

    def _calculate_euler_angles(self, rotation_matrix):
        sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
            y = np.arctan2(-rotation_matrix[2,0], sy)
            z = 0

        return np.degrees([x, y, z])

    def _draw_axes(self, frame, origin, rvec, tvec, length=100):
        axis_points = np.float32([
            [0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]
        ]).reshape(-1, 3)
        
        img_pts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, 
            self.camera_matrix, self.distortion_coeffs
        )
        
        origin = tuple(map(int, origin))
        img_pts = img_pts.reshape(-1, 2)
        
        cv2.line(frame, origin, tuple(img_pts[1].astype(int)), (0,0,255), 3)
        cv2.line(frame, origin, tuple(img_pts[2].astype(int)), (0,255,0), 3)
        cv2.line(frame, origin, tuple(img_pts[3].astype(int)), (255,0,0), 3)

    def _check_head_position(self, angles):
        warnings = []
        pitch, yaw, roll = angles[0], angles[1], angles[2]

        if abs(pitch) > self.PITCH_THRESHOLD:
            warnings.append(f"Head too far {'Up' if pitch < 0 else 'Down'}!")
        if abs(yaw) > self.YAW_THRESHOLD:
            warnings.append(f"Head too far {'Right' if yaw > 0 else 'Left'}!")
        if abs(roll) > self.ROLL_THRESHOLD:
            warnings.append(f"Head tilted {'Right' if roll > 0 else 'Left'}!")

        return warnings

    def estimate_head_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)

        if not results[0].boxes:
            cv2.putText(frame, "Driver Out of Frame!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return frame

        for result in results:
            for box in result.boxes:
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)
                
                image_points = self._get_2d_image_points(landmarks)
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.distortion_coeffs
                )

                rot_mat, _ = cv2.Rodrigues(rvec)
                angles = self._calculate_euler_angles(rot_mat)
                warnings = self._check_head_position(angles)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0,0,255) if warnings else (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                self._draw_axes(frame, image_points[0], rvec, tvec)
                
                y_pos = 90
                pose_text = f"Pitch: {angles[0]:.1f}°, Yaw: {angles[1]:.1f}°, Roll: {angles[2]:.1f}°"
                cv2.putText(frame, pose_text, (10, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                for i, warn in enumerate(warnings):
                    cv2.putText(frame, warn, (10, y_pos + 30*(i+1)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        return frame

class ComprehensiveFacialAnalysis(HeadPoseEstimator):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None):
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.landmark_colors = {
            'jaw': (255,0,0), 'eyebrows': (0,255,0),
            'nose': (255,255,0), 'eyes': (0,0,255),
            'mouth': (0,255,255)
        }

    def _draw_landmarks(self, frame, landmarks):
        # Jawline
        for i in range(0, 17):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 
                      2, self.landmark_colors['jaw'], -1)
        
        # Eyes
        for region in [range(36,42), range(42,48)]:
            for i in region:
                cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 
                          2, self.landmark_colors['eyes'], -1)

    def estimate_head_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector(rgb_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)
                
                self._draw_landmarks(frame, landmarks)
                image_points = self._get_2d_image_points(landmarks)
                
                success, rvec, tvec = cv2.solvePnP(
                    self.model_points, image_points,
                    self.camera_matrix, self.distortion_coeffs
                )

                rot_mat, _ = cv2.Rodrigues(rvec)
                angles = self._calculate_euler_angles(rot_mat)
                warnings = self._check_head_position(angles)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0,0,255) if warnings else (0,255,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                pose_text = f"Pitch: {angles[0]:.1f}°, Yaw: {angles[1]:.1f}°, Roll: {angles[2]:.1f}°"
                cv2.putText(frame, pose_text, (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

# Usage example
if __name__ == "__main__":
    estimator = HeadPoseEstimator('yolov8n-face.pt')
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = estimator.estimate_head_pose(frame)
        cv2.imshow('Head Pose Estimation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()