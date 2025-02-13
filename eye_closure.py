import cv2
import dlib
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist

# Define the function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Define the indices for the left and right eyes
LEFT_EYE_START, LEFT_EYE_END = 36, 42
RIGHT_EYE_START, RIGHT_EYE_END = 42, 48


from head_pose import ComprehensiveFacialAnalysis
class EyeClosureDetector(ComprehensiveFacialAnalysis):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None, ear_threshold=0.25):
        """
        Initialize Eye Closure Detector.

        Args:
            yolo_model_path: Path to trained YOLOv8n model weights.
            camera_matrix: Optional camera calibration matrix.
            distortion_coeffs: Optional lens distortion coefficients.
            ear_threshold: Threshold for eye aspect ratio (EAR) to detect eye closure.
        """
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.ear_threshold = ear_threshold
        self.face_detector = YOLO(yolo_model_path)

    def _get_eye_landmarks(self, landmarks, eye_start, eye_end):
        """
        Extract the (x, y)-coordinates for the given eye region.

        Args:
            landmarks: dlib facial landmarks.
            eye_start: Starting index of the eye region.
            eye_end: Ending index of the eye region.

        Returns:
            eye_points: Array of (x, y)-coordinates for the eye region.
        """
        eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(eye_start, eye_end)])
        return eye_points

    def detect_eye_closure(self, frame):
        """
        Detect eye closure in the frame.

        Args:
            frame: Input frame.

        Returns:
            frame: Frame with eye closure detection visualization.
            is_eyes_closed: Boolean indicating if eyes are closed.
        """
        # Convert to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using YOLO
        results = self.face_detector(rgb_frame, verbose=False)

        is_eyes_closed = False

        for result in results:
            for box in result.boxes:
                # Convert box to dlib rectangle format
                dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])

                # Get facial landmarks
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = self.predictor(gray, dlib_rect)

                # Extract left and right eye landmarks
                left_eye = self._get_eye_landmarks(landmarks, LEFT_EYE_START, LEFT_EYE_END)
                right_eye = self._get_eye_landmarks(landmarks, RIGHT_EYE_START, RIGHT_EYE_END)

                # Calculate EAR for both eyes
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                # Average the EAR for both eyes
                ear = (left_ear + right_ear) / 2.0

                # Check if EAR is below the threshold
                if ear < self.ear_threshold:
                    is_eyes_closed = True
                    cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw landmarks on the frame
                self._draw_landmarks(frame, landmarks)

        return frame, is_eyes_closed

# Initialize the detector
# detector = EyeClosureDetector(yolo_model_path=r'.\best.pt')

# # Capture video from webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect eye closure
#     frame = detector.detect_eye_closure(frame)

#     # Display the frame
#     cv2.imshow('Eye Closure Detection', frame[0])

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()
