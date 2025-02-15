import cv2

class Config:
    # MediaPipe Face Mesh configs
    FACE_MESH_CONFIG = {
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'max_num_faces': 1,
        'refine_landmarks': True
    }

    # Landmark indices
    LANDMARKS = {
        'LEFT_EYE': [362, 385, 387, 263, 373, 380],
        'RIGHT_EYE': [33, 160, 158, 133, 153, 144],
        'MOUTH': [61, 291, 39, 181, 0, 17, 269, 405]
    }

    # Eye tracking thresholds
    EYE = {
        'CLOSED_THRESH': 0.2,
        'OPEN_THRESH': 0.25,
        'SMOOTHING_FRAMES': 5
    }

    # Mouth and yawning thresholds
    MOUTH = {
        'MAR_THRESH': 0.59,
        'YAWN_FRAMES_THRESH': 15,
        'MAR_DISPLAY_THRESH': 0.64  # threshold to display MAR value
    }

    # Drowsiness detection parameters
    DROWSINESS = {
        'PERCLOS_THRESH': 0.2,
        'PERCLOS_WINDOW': 30,
        'PERCLOS_SEVERE': 0.35,
        'EYE_CLOSED_FRAMES': 0  # initial value for closed frames counter
    }

    # Head pose thresholds
    HEAD_POSE = {
        'PITCH_THRESH': 10.0,  # degrees
        'YAW_THRESH': 10.0,    # degrees
        'POSE_TIME_THRESH': 3.0  # seconds
    }

    # Decision layer parameters
    DECISION = {
        'HISTORY_WINDOW': 30,
        'DISTRACTION_TIME': 2.0,
        'RISK_WEIGHTS': {
            'PERCLOS': 0.5,
            'YAWN': 0.3,
            'HEAD_POSE': 0.2
        }
    }

    # Visualization settings
    VISUALIZATION = {
        'FONT': {
            'FACE': cv2.FONT_HERSHEY_SIMPLEX,
            'SCALE': 1,
            'THICKNESS': 2
        },
        'COLORS': {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'BLUE': (255, 0, 0),
            'YELLOW': (0, 255, 255)
        }
    }

    # Camera and image processing
    CAMERA = {
        'DEVICE_ID': 0,
        'FRAME_WAIT': 5,  # milliseconds to wait between frames
        'FOCAL_LENGTH_SCALE': 1.0  # scale factor for focal length
    }

    # Drawing specifications
    DRAWING = {
        'SPEC': {
            'thickness': 1,
            'circle_radius': 1
        },
        'NOSE_LINE': {
            'thickness': 3,
            'scale': 10  # scale factor for nose direction line
        }
    }

    # 3D Head pose estimation
    HEAD_3D = {
        'LANDMARKS': [33, 263, 1, 61, 291, 199],  # landmarks used for head pose
        'NOSE_INDEX': 1,  # index of nose landmark
        'DEPTH_SCALE': 3000  # scale factor for Z coordinate
    }