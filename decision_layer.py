import threading
from queue import Queue
import cv2
from eye_closure import EyeClosureDetector
from head_pose import HeadPoseEstimator
from yawn_detection import YawnDetector
from eye_gaze import EyeGazeEstimator

class DecisionLayer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.eye_closure_detector = EyeClosureDetector(model_path)
        self.head_pose_estimator = HeadPoseEstimator(model_path)
        self.yawn_detector = YawnDetector(model_path)
        self.gaze_estimator = EyeGazeEstimator(model_path, './shape_predictor_68_face_landmarks.dat')
        
        self.results_queue = Queue()
        
    def _run_eye_closure(self, frame):
        try:
            result, is_closed = self.eye_closure_detector.detect_eye_closure(frame.copy())
            self.results_queue.put(('eye_closure', result, is_closed))
        except Exception as e:
            print(f"Eye closure error: {e}")

    def _run_head_pose(self, frame):
        try:
            result = self.head_pose_estimator.estimate_head_pose(frame.copy())
            self.results_queue.put(('head_pose', result))
        except Exception as e:
            print(f"Head pose error: {e}")

    def _run_yawn(self, frame):
        try:
            result, is_yawning = self.yawn_detector.detect_yawn(frame.copy())
            self.results_queue.put(('yawn', result, is_yawning))
        except Exception as e:
            print(f"Yawn detection error: {e}")

    def _run_gaze(self, frame):
        try:
            result = self.gaze_estimator.estimate_gaze(frame.copy())
            self.results_queue.put(('gaze', result))
        except Exception as e:
            print(f"Gaze estimation error: {e}")

    def process_frame(self, frame):
        # Clear the queue
        while not self.results_queue.empty():
            self.results_queue.get()

        # Start all detections in parallel
        threads = [
            threading.Thread(target=self._run_eye_closure, args=(frame,)),
            threading.Thread(target=self._run_head_pose, args=(frame,)),
            threading.Thread(target=self._run_yawn, args=(frame,)),
            threading.Thread(target=self._run_gaze, args=(frame,))
        ]

        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Initialize detection results
        detection_results = {
            'eye_closure': False,
            'yawning': False,
            'gaze_direction': None,
            'head_pose': None
        }
        
        final_frame = frame.copy()
        
        # Process results from the queue
        while not self.results_queue.empty():
            result = self.results_queue.get()
            detection_type = result[0]
            
            if detection_type == 'eye_closure':
                final_frame = result[1]  # Update frame with eye closure visualization
                detection_results['eye_closure'] = result[2]  # Update eye closure state
            
            elif detection_type == 'yawn':
                if not detection_results['eye_closure']:  # Only update if not already modified
                    final_frame = result[1]  # Update frame with yawn visualization
                detection_results['yawning'] = result[2]  # Update yawn state
            
            elif detection_type == 'gaze':
                final_frame = result[1]  # Update frame with gaze visualization
                
            elif detection_type == 'head_pose':
                if not detection_results['eye_closure'] and not detection_results['yawning']:
                    final_frame = result[1]  # Update frame with head pose visualization

        # Add overall status text with proper positioning
        y_position = 30
        
        # Check if driver is in frame
        if len(self.head_pose_estimator.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[0].boxes) == 0:
            cv2.putText(final_frame, "Driver Out of Frame!", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return final_frame, {'driver_in_frame': False}

        # Continue with normal detection if driver is in frame
        detection_results['driver_in_frame'] = True
        
        # Eye closure status
        if detection_results['eye_closure']:
            cv2.putText(final_frame, "Eyes Closed", (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(final_frame, "Eyes Open", (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Yawn status
        y_position += 30
        if detection_results['yawning']:
            cv2.putText(final_frame, "Yawning", (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(final_frame, "Not Yawning", (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gaze direction
        y_position += 30
        if 'gaze_direction' in detection_results and detection_results['gaze_direction']:
            cv2.putText(final_frame, f"Gaze: {detection_results['gaze_direction']}", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Head pose
        y_position += 30
        if 'head_pose' in detection_results and detection_results['head_pose']:
            cv2.putText(final_frame, f"Head: {detection_results['head_pose']}", 
                       (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return final_frame, detection_results
