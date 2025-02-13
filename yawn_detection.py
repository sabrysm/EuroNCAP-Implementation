import cv2
import numpy as np
from ultralytics import YOLO
import dlib
from head_pose import ComprehensiveFacialAnalysis

class YawnDetector(ComprehensiveFacialAnalysis):
    def __init__(self, yolo_model_path, camera_matrix=None, distortion_coeffs=None, yawn_threshold=0.3):
        """
        Initialize Yawn Detector
        
        Args:
            yolo_model_path: Path to trained YOLOv8n model weights
            camera_matrix: Optional camera calibration matrix
            distortion_coeffs: Optional lens distortion coefficients
            yawn_threshold: Threshold for yawn detection (default: 0.5)
        """
        super().__init__(yolo_model_path, camera_matrix, distortion_coeffs)
        self.yawn_threshold = yawn_threshold
        self.face_detector = YOLO(yolo_model_path)

    def _calculate_mean_lip_positions(self, landmarks):
        """
        Calculate the mean positions of the upper and lower lips
        
        Args:
            landmarks: dlib facial landmarks
        
        Returns:
            upper_mean: Mean position of the upper lip (x, y)
            lower_mean: Mean position of the lower lip (x, y)
        """
        # Upper lip landmarks (points 50, 51, 52, 61, 62, 63)
        upper_lip_points = [50, 51, 52, 61, 62, 63]
        upper_lip_x = [landmarks.part(i).x for i in upper_lip_points]
        upper_lip_y = [landmarks.part(i).y for i in upper_lip_points]
        
        # Lower lip landmarks (points 56, 57, 58, 65, 66, 67)
        lower_lip_points = [56, 57, 58, 65, 66, 67]
        lower_lip_x = [landmarks.part(i).x for i in lower_lip_points]
        lower_lip_y = [landmarks.part(i).y for i in lower_lip_points]
        
        # Calculate mean positions
        upper_mean = (np.mean(upper_lip_x), np.mean(upper_lip_y))
        lower_mean = (np.mean(lower_lip_x), np.mean(lower_lip_y))
        
        return upper_mean, lower_mean

    def _calculate_vertical_distance(self, upper_mean, lower_mean):
        """
        Calculate the vertical distance between the upper and lower lip
        
        Args:
            upper_mean: Mean position of the upper lip (x, y)
            lower_mean: Mean position of the lower lip (x, y)
        
        Returns:
            distance: Vertical distance between the upper and lower lip
        """
        return abs(upper_mean[1] - lower_mean[1])/100

    def detect_yawn(self, frame):
        """
        Detect yawn in the frame
        
        Args:
            frame: Input frame
        
        Returns:
            frame: Frame with yawn detection visualization
            is_yawn: Boolean indicating if a yawn is detected
        """
        # Convert to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLO
        results = self.face_detector(rgb_frame, verbose=False)
        
        is_yawn = False
        
        for result in results:
            # Process each detected face
            for box in result.boxes:
                try:
                    # Convert box to dlib rectangle format with frame shape
                    dlib_rect = self._convert_yolo_to_dlib_rect(box.xyxy[0])
                    
                    # Get facial landmarks
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    landmarks = self.predictor(gray, dlib_rect)
                    
                    # Calculate mean positions of upper and lower lips
                    upper_mean, lower_mean = self._calculate_mean_lip_positions(landmarks)
                    
                    # Calculate vertical distance
                    distance = self._calculate_vertical_distance(upper_mean, lower_mean)
                    
                    # Debug print
                    print(f"Yawn distance: {distance}")
                    
                    # Check if distance exceeds yawn threshold (lowered threshold)
                    if distance > self.yawn_threshold:
                        is_yawn = True
                        cv2.putText(frame, "Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Not Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw landmarks and bounding box
                    self._draw_landmarks(frame, landmarks)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error in yawn detection: {str(e)}")
                    continue
        
        return frame, is_yawn

# Initialize YawnDetector
# yawn_detector = YawnDetector(yolo_model_path=r'.\best.pt', yawn_threshold=0.3)

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Detect yawn
#     frame, is_yawn = yawn_detector.detect_yawn(frame)
    
#     # Display the frame
#     cv2.imshow('Yawn Detection', frame)
    
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()




