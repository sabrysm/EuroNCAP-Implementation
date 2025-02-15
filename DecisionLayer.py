import numpy as np
from enum import Enum
from config import Config
from collections import deque
from time import time

class DriverState(Enum):
    ALERT = "Alert"
    BRIEF_DISTRACTION = "Brief Distraction"
    LONG_DISTRACTION = "Long Distraction"
    VATS_WARNING = "VATS Warning"
    PHONE_USE = "Phone Use"
    DROWSY = "Drowsy"
    MICROSLEEP = "Microsleep"
    SLEEP = "Sleep"
    EMERGENCY = "Emergency"

class GazeStrategy(Enum):
    OWL = "Owl"
    LIZARD = "Lizard"
    MIXED = "Mixed"

class DecisionLayer:
    def __init__(self):
        self.config = Config.DECISION
        self.vats_buffer = deque(maxlen=30)  # 30s rolling window
        self.gaze_history = deque(maxlen=100)  # For strategy detection
        self.microsleep_threshold = 0.5  # 500ms
        self.sleep_threshold = 3.0  # 3s
        self.vats_threshold = 10.0  # 10s in 30s window
        
        # Initialize state trackers
        self.last_forward_time = time()
        self.current_distraction_start = None
        self.current_gaze_strategy = GazeStrategy.MIXED
        self.escalation_level = 0
        
    def update(self, frame_data):
        """Enhanced update with Euro NCAP compliance"""
        self._update_gaze_strategy(frame_data)
        self._update_vats_buffer(frame_data)
        
        state, risk = self._evaluate_state(frame_data)
        response = self._get_response(state, risk)
        
        return {
            'state': state,
            'risk': risk,
            'response': response,
            'gaze_strategy': self.current_gaze_strategy,
            'vats_score': self._get_vats_score()
        }
    
    def _update_gaze_strategy(self, frame_data):
        """Detect owl/lizard gaze strategy"""
        if frame_data['head_pose'] != "Forward":
            head_movement = frame_data['head_rotation_speed']
            eye_movement = frame_data['eye_rotation_speed']
            ratio = head_movement / (eye_movement + 1e-6)
            self.gaze_history.append(ratio)
            
            # Update strategy based on recent behavior
            mean_ratio = np.mean(list(self.gaze_history))
            self.current_gaze_strategy = (
                GazeStrategy.OWL if mean_ratio > 0.7
                else GazeStrategy.LIZARD if mean_ratio < 0.3
                else GazeStrategy.MIXED
            )
    
    def _update_vats_buffer(self, frame_data):
        """Manage Visual Attention Time Sharing buffer"""
        current_time = frame_data['timestamp']
        if frame_data['head_pose'] != "Forward":
            self.vats_buffer.append((current_time, frame_data['head_pose']))
            
        # Clean old entries
        while self.vats_buffer and (current_time - self.vats_buffer[0][0]) > 30:
            self.vats_buffer.popleft()
    
    def _evaluate_state(self, frame_data):
        """Determine driver state based on Euro NCAP criteria"""
        current_time = frame_data['timestamp']
        off_road_time = current_time - self.last_forward_time
        vats_score = self._get_vats_score()
        
        # Update last_forward_time when looking forward
        if frame_data['head_pose'] == "Forward":
            self.last_forward_time = current_time
            off_road_time = 0  # Reset off-road time
        
        # Check for sleep states first with stricter conditions
        if off_road_time >= self.sleep_threshold and frame_data['eye_state'] == "Closed":
            return DriverState.SLEEP, 1.0
        elif off_road_time >= self.microsleep_threshold and frame_data['eye_state'] == "Closed":
            return DriverState.MICROSLEEP, 0.8
            
        # Check drowsiness with combined metrics
        if frame_data['perclos'] > Config.DROWSINESS['PERCLOS_SEVERE']:
            return DriverState.DROWSY, 0.5
            
        # Check distraction states
        if frame_data['phone_detected']:
            return DriverState.PHONE_USE, 0.9
        elif off_road_time >= 3.0:
            return DriverState.LONG_DISTRACTION, 0.7
        elif vats_score >= self.vats_threshold:
            return DriverState.VATS_WARNING, 0.6
        elif off_road_time > 0.5:  # Brief distraction threshold
            return DriverState.BRIEF_DISTRACTION, 0.3
            
        return DriverState.ALERT, 0.0
    
    def _get_vats_score(self):
        """Calculate VATS score for 30s window"""
        if not self.vats_buffer:
            return 0.0
        return sum(1 for _ in self.vats_buffer)  # Simplified for example
    
    def _get_response(self, state, risk):
        """Generate graduated response based on state and risk"""
        if state in [DriverState.SLEEP, DriverState.EMERGENCY]:
            return {
                'warning_level': 'CRITICAL',
                'adas_sensitivity': 'HIGH',
                'intervention_required': True
            }
        elif state in [DriverState.MICROSLEEP, DriverState.PHONE_USE]:
            return {
                'warning_level': 'HIGH',
                'adas_sensitivity': 'INCREASED',
                'intervention_required': False
            }
