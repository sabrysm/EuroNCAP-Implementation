import cv2
import mediapipe as mp
import numpy as np
import time
from config import Config
from DecisionLayer import DecisionLayer, DriverState

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(**Config.FACE_MESH_CONFIG)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(**Config.DRAWING['SPEC'])

# Landmark indices for EAR, MAR, and head pose
LEFT_EYE = Config.LANDMARKS['LEFT_EYE']
RIGHT_EYE = Config.LANDMARKS['RIGHT_EYE']
MOUTH = Config.LANDMARKS['MOUTH']

# Thresholds
EYE_CLOSED_THRESH = Config.EYE['CLOSED_THRESH']
EYE_OPEN_THRESH = Config.EYE['OPEN_THRESH']
MAR_THRESH = Config.MOUTH['MAR_THRESH']
PERCLOS_THRESH = Config.DROWSINESS['PERCLOS_THRESH']  # Update to use Config value
YAWN_FRAMES_THRESH = Config.MOUTH['YAWN_FRAMES_THRESH']  # Use Config for yawn frames threshold

# Smoothing parameters
SMOOTHING_FRAMES = Config.EYE['SMOOTHING_FRAMES']
ear_history = []       # History of EAR values for smoothing
mar_history = []       # History of MAR values for smoothing

# Drowsiness detection parameters
yawn_counter = 0       # Counter for consecutive frames with yawn
EYE_CLOSED_FRAMES = Config.DROWSINESS['EYE_CLOSED_FRAMES']  # Counter for consecutive frames with closed eyes
PERCLOS_WINDOW = Config.DROWSINESS['PERCLOS_WINDOW']  # Update to use Config value
perclos_history = []   # History of eye states for Perclos

# Initialize video capture
cap = cv2.VideoCapture(Config.CAMERA['DEVICE_ID'])

# Add at the start after other initializations
decision_layer = DecisionLayer()

# Add after camera initialization and before the main loop
fps = 0
prev_frame_time = time.time()

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points, landmarks):
    # Vertical distances
    v1 = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    v2 = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    
    # Horizontal distance
    h = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth_points, landmarks):
    # Vertical distances
    v1 = np.linalg.norm(landmarks[mouth_points[1]] - landmarks[mouth_points[7]])
    v2 = np.linalg.norm(landmarks[mouth_points[3]] - landmarks[mouth_points[5]])
    
    # Horizontal distance
    h = np.linalg.norm(landmarks[mouth_points[0]] - landmarks[mouth_points[4]])
    
    mar = (v1 + v2) / (2.0 * h)
    return mar

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    start = time.time()

    # Flip the image horizontally and convert to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image with MediaPipe Face Mesh
    results = face_mesh.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to numpy array
            landmarks = np.array([(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark], dtype=np.float32)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            # Smooth EAR using a moving average
            ear_history.append(avg_ear)
            if len(ear_history) > SMOOTHING_FRAMES:
                ear_history.pop(0)
            smoothed_ear = np.mean(ear_history)

            # Determine eye state with hysteresis
            if smoothed_ear < EYE_CLOSED_THRESH:
                eye_state = "Closed"
                EYE_CLOSED_FRAMES += 1
            elif smoothed_ear > EYE_OPEN_THRESH:
                eye_state = "Open"
                EYE_CLOSED_FRAMES = 0
            else:
                eye_state = "Partial"  # Optional: Add partial state for hysteresis

            # Calculate Perclos (Percentage of Eye Closure)
            perclos_history.append(1 if eye_state == "Closed" else 0)
            if len(perclos_history) > PERCLOS_WINDOW:
                perclos_history.pop(0)
            perclos = np.mean(perclos_history)

            # Detect drowsiness based on Perclos
            if perclos > PERCLOS_THRESH:
                drowsiness_status = "Drowsy"
            else:
                drowsiness_status = "Alert"

            # Calculate MAR for yawning detection
            mar = mouth_aspect_ratio(MOUTH, landmarks)
            mar_history.append(mar)
            if len(mar_history) > SMOOTHING_FRAMES:
                mar_history.pop(0)
            smoothed_mar = np.mean(mar_history)

            # Detect yawning with temporal filtering
            yawning_status = "Normal"
            if smoothed_mar > MAR_THRESH:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAMES_THRESH:
                    yawning_status = "Yawning"
                    cv2.putText(image, f"MAR: {smoothed_mar:.2f}", (20, 250),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                yawn_counter = 0
                yawning_status = "Normal"

            # Display eye state, Perclos, and yawning status
            cv2.putText(image, f"Eye State: {eye_state}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(image, f"EAR: {smoothed_ear:.2f}", (400, 250), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(image, f"Perclos: {perclos:.2f}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Drowsiness: {drowsiness_status}", (20, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Yawning: {yawning_status}", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw eye and mouth contours for visualization
            for eye in [LEFT_EYE, RIGHT_EYE]:
                eye_points = landmarks[eye].astype(int)
                cv2.polylines(image, [eye_points], True, (0, 255, 0), 1)
            mouth_points = landmarks[MOUTH].astype(int)
            cv2.polylines(image, [mouth_points], True, (0, 255, 0), 1)

            # Head pose estimation
            face_3d = []
            face_2d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in Config.HEAD_3D['LANDMARKS']:
                    if idx == Config.HEAD_3D['NOSE_INDEX']:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * Config.HEAD_3D['DEPTH_SCALE'])

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

            # Convert to NumPy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera matrix
            focal_length = Config.CAMERA['FOCAL_LENGTH_SCALE'] * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

            # Distortion matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP to get rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, _ = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Extract pitch, yaw, and roll
            x = angles[0] * 360  # Pitch (nodding up/down)
            y = angles[1] * 360  # Yaw (turning left/right)
            z = angles[2] * 360  # Roll (tilting sideways)

            # Determine head pose direction
            if y < -10:
                head_pose = "Looking Left"
            elif y > 10:
                head_pose = "Looking Right"
            elif x < -10:
                head_pose = "Looking Down"
            elif x > 10:
                head_pose = "Looking Up"
            else:
                head_pose = "Forward"

            # After calculating all metrics, before visualization
            frame_data = {
                'timestamp': time.time(),
                'eye_state': eye_state,
                'perclos': perclos,
                'head_pose': head_pose,
                'phone_detected': False,  # Add phone detection logic if needed
                'head_rotation_speed': abs(y - last_yaw) if 'last_yaw' in locals() else 0,
                'eye_rotation_speed': smoothed_ear - last_ear if 'last_ear' in locals() else 0
            }
            
            # Store current values for next frame
            last_yaw = y
            last_ear = smoothed_ear
            
            # Get driver state assessment
            assessment = decision_layer.update(frame_data)
            
            # Display driver state information with color coding
            state_color = {
                DriverState.ALERT: (0, 255, 0),        # Green
                DriverState.DROWSY: (0, 165, 255),     # Orange
                DriverState.MICROSLEEP: (0, 0, 255),   # Red
                DriverState.SLEEP: (0, 0, 255),        # Red
                DriverState.PHONE_USE: (255, 0, 0),    # Blue
                DriverState.LONG_DISTRACTION: (0, 165, 255)  # Orange
            }.get(assessment['state'], (0, 255, 0))
            
            # Display driver state assessment
            cv2.putText(image, f"Driver State: {assessment['state'].value}", (20, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
            cv2.putText(image, f"Risk Level: {assessment['risk']:.2f}", (20, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
            cv2.putText(image, f"Gaze Strategy: {assessment['gaze_strategy'].value}", (20, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Move FPS display to avoid overlap
            cv2.putText(image, f"FPS: {int(fps)}", (400, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display head pose direction
            cv2.putText(image, f"Head Pose: {head_pose}", (20, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Pitch: {np.round(x, 2)}", (400, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Yaw: {np.round(y, 2)}", (400, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, f"Roll: {np.round(z, 2)}", (400, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw nose direction
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

    # Calculate and display FPS
    end = time.time()
    fps = 1 / (end - start)

    # Calculate FPS - update this section
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    # Display FPS with smoothing
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(image, fps_text, (400, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow('Driver Monitoring System', image)

    # Exit on 'q' key press
    if cv2.waitKey(Config.CAMERA['FRAME_WAIT']) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()