import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from ultralytics import YOLO

class Pupil:
    def __init__(self, eye_frame):
        self.eye_frame = eye_frame
        self.pupil_position = None

    def detect_pupil(self):
        # Preprocess the eye frame
        gray_eye = cv2.cvtColor(self.eye_frame, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.bilateralFilter(gray_eye, 10, 15, 15)  # Bilateral filtering
        gray_eye = cv2.erode(gray_eye, None, iterations=2)    # Erosion

        # Binarization
        _, binary_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if contours:
            # Get the largest contour (presumed to be the iris)
            iris_contour = contours[0]
            moments = cv2.moments(iris_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                self.pupil_position = (cx, cy)

        return self.pupil_position

class Eye:
    def __init__(self, landmarks, eye_points):
        self.landmarks = landmarks
        self.eye_points = eye_points
        self.blinking_ratio = None
        self.pupil = None

    def isolate_eye(self, frame):
        # Extract eye region from the frame using landmarks
        eye_region = np.array([(self.landmarks.part(point).x, self.landmarks.part(point).y) for point in self.eye_points])
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [eye_region], 255)
        eye_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Crop the eye region
        min_x = min(eye_region[:, 0])
        max_x = max(eye_region[:, 0])
        min_y = min(eye_region[:, 1])
        max_y = max(eye_region[:, 1])
        eye_frame = eye_frame[min_y:max_y, min_x:max_x]

        return eye_frame

    def calculate_blinking_ratio(self):
        # Calculate blinking ratio based on eye landmarks
        horizontal_dist = dist.euclidean(
            (self.landmarks.part(self.eye_points[0]).x, self.landmarks.part(self.eye_points[0]).y),
            (self.landmarks.part(self.eye_points[3]).x, self.landmarks.part(self.eye_points[3]).y)
        )
        vertical_dist = dist.euclidean(
            (self.landmarks.part(self.eye_points[1]).x, self.landmarks.part(self.eye_points[1]).y),
            (self.landmarks.part(self.eye_points[2]).x, self.landmarks.part(self.eye_points[2]).y)
        )
        self.blinking_ratio = horizontal_dist / vertical_dist
        return self.blinking_ratio

    def detect_pupil(self, eye_frame):
        self.pupil = Pupil(eye_frame)
        return self.pupil.detect_pupil()

class Calibration:
    def __init__(self):
        self.threshold = None

    def calibrate(self, eye_frames):
        # Analyze multiple frames to determine the best binarization threshold
        thresholds = []
        for frame in eye_frames:
            gray_eye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary_eye = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            thresholds.append(cv2.mean(binary_eye)[0])

        self.threshold = np.mean(thresholds)
        return self.threshold

class EyeGazeEstimator:
    def __init__(self, yolo_model_path, shape_predictor_path):
        self.face_detector = YOLO(yolo_model_path)
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)
        self.calibration = Calibration()
        self.left_eye_points = [36, 37, 38, 39, 40, 41]
        self.right_eye_points = [42, 43, 44, 45, 46, 47]

    def estimate_gaze(self, frame):
        # Detect faces using YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector(rgb_frame, verbose=False)

        for result in results:
            for box in result.boxes:
                # Convert box to dlib rectangle format
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dlib_rect = dlib.rectangle(x1, y1, x2, y2)

                # Get facial landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.landmark_predictor(gray, dlib_rect)

                # Initialize Eye objects for left and right eyes
                left_eye = Eye(landmarks, self.left_eye_points)
                right_eye = Eye(landmarks, self.right_eye_points)

                # Isolate eye regions
                left_eye_frame = left_eye.isolate_eye(frame)
                right_eye_frame = right_eye.isolate_eye(frame)

                # Detect pupils
                left_pupil = left_eye.detect_pupil(left_eye_frame)
                right_pupil = right_eye.detect_pupil(right_eye_frame)

                # Calculate blinking ratio
                left_eye.calculate_blinking_ratio()
                right_eye.calculate_blinking_ratio()

                # Determine gaze direction
                if left_pupil and right_pupil:
                    gaze_direction = self._calculate_gaze_direction(left_pupil, right_pupil, left_eye_frame, right_eye_frame)
                    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def _calculate_gaze_direction(self, left_pupil, right_pupil, left_eye_frame, right_eye_frame):
        # Calculate gaze direction based on pupil positions
        left_eye_center = (left_eye_frame.shape[1] // 2, left_eye_frame.shape[0] // 2)
        right_eye_center = (right_eye_frame.shape[1] // 2, right_eye_frame.shape[0] // 2)

        left_gaze = "Left" if left_pupil[0] < left_eye_center[0] else "Right"
        right_gaze = "Left" if right_pupil[0] < right_eye_center[0] else "Right"

        if left_gaze == right_gaze:
            return left_gaze
        else:
            return "Center"

# Example usage
# if __name__ == "__main__":
#     yolo_model_path = r"./best.pt"
#     shape_predictor_path = r"./shape_predictor_68_face_landmarks.dat"
#     gaze_estimator = EyeGazeEstimator(yolo_model_path, shape_predictor_path)

#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = gaze_estimator.estimate_gaze(frame)
#         cv2.imshow("Eye Gaze Estimation", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

