"""
MediaPipe Pose Integration

Body tracking for two-handed signs and body posture:
- 33-point pose landmarks
- Two-handed coordination
- Body position context
- Arm and shoulder tracking
"""

import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import mediapipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.info("MediaPipe not installed. Pose features will use fallback.")


# Pose landmark indices
POSE_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}


class PoseDetector:
    """
    Pose detection using MediaPipe.

    Provides 33-point body tracking for:
    - Two-handed coordination
    - Body position context
    - Arm and shoulder tracking
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize pose detector.

        Args:
            static_image_mode: If True, treats each image independently
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: If True, applies landmark smoothing
            enable_segmentation: If True, enables person segmentation
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        if not MEDIAPIPE_AVAILABLE:
            self.available = False
            return

        self.available = True

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.results = None
        self.landmarks = None

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect pose in image.

        Args:
            img: Input image (BGR format)

        Returns:
            Image with pose landmarks drawn
        """
        if not self.available:
            return img

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        # Process
        self.results = self.pose.process(img_rgb)

        # Draw landmarks
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return img

    def get_landmarks(
        self,
        img_shape: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get pose landmarks as list.

        Args:
            img_shape: Image shape (height, width) for coordinate conversion

        Returns:
            List of landmark dictionaries
        """
        if not self.available or not self.results or not self.results.pose_landmarks:
            return []

        landmarks = []
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            # Convert to pixel coordinates if shape provided
            if img_shape:
                h, w = img_shape
                x = landmark.x * w
                y = landmark.y * h
            else:
                x = landmark.x
                y = landmark.y

            # Get landmark name
            name = None
            for lm_name, lm_idx in POSE_LANDMARKS.items():
                if lm_idx == idx:
                    name = lm_name
                    break

            landmarks.append({
                'id': idx,
                'name': name or f'point_{idx}',
                'x': x,
                'y': y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        return landmarks

    def get_hand_positions(self) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Get left and right hand positions.

        Returns:
            Dictionary with 'left' and 'right' hand positions
        """
        landmarks = self.get_landmarks()

        left_hand = None
        right_hand = None

        for lm in landmarks:
            if lm['name'] == 'left_wrist':
                left_hand = {'x': lm['x'], 'y': lm['y'], 'visibility': lm['visibility']}
            elif lm['name'] == 'right_wrist':
                right_hand = {'x': lm['x'], 'y': lm['y'], 'visibility': lm['visibility']}

        return {
            'left': left_hand,
            'right': right_hand
        }

    def get_body_position(self) -> Dict[str, Any]:
        """
        Get body position and orientation.

        Returns:
            Dictionary with body position information
        """
        landmarks = self.get_landmarks()

        # Find key body points
        positions = {}
        for lm in landmarks:
            if lm['name'] in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'nose']:
                positions[lm['name']] = {'x': lm['x'], 'y': lm['y'], 'visibility': lm['visibility']}

        if not all(k in positions for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
            return {}

        # Calculate body center
        center_x = (positions['left_shoulder']['x'] + positions['right_shoulder']['x'] +
                    positions['left_hip']['x'] + positions['right_hip']['x']) / 4
        center_y = (positions['left_shoulder']['y'] + positions['right_shoulder']['y'] +
                    positions['left_hip']['y'] + positions['right_hip']['y']) / 4

        # Calculate shoulder width
        shoulder_width = math.hypot(
            positions['right_shoulder']['x'] - positions['left_shoulder']['x'],
            positions['right_shoulder']['y'] - positions['left_shoulder']['y']
        )

        # Calculate body height
        body_height = math.hypot(
            center_x - positions['left_hip']['x'],
            center_y - positions['left_hip']['y']
        ) * 2

        return {
            'center': {'x': center_x, 'y': center_y},
            'shoulder_width': shoulder_width,
            'body_height': body_height,
            'positions': positions,
            'is_visible': all(p.get('visibility', 0) > 0.5 for p in positions.values())
        }

    def get_two_handed_info(self) -> Dict[str, Any]:
        """
        Get information about both hands.

        Returns:
            Dictionary with two-handed coordination info
        """
        hands = self.get_hand_positions()
        body = self.get_body_position()

        result = {
            'left_hand': hands.get('left'),
            'right_hand': hands.get('right'),
            'both_visible': hands.get('left') is not None and hands.get('right') is not None
        }

        # Calculate hand distance if both visible
        if result['both_visible']:
            left = hands['left']
            right = hands['right']
            distance = math.hypot(right['x'] - left['x'], right['y'] - left['y'])
            result['hand_distance'] = distance

            # Determine hand relationship
            if left['x'] < right['x']:
                result['hands_positioned'] = 'normal'
            else:
                result['hands_positioned'] = 'crossed'

        # Add body context
        if body:
            result['body_position'] = body.get('center')
            result['body_visible'] = body.get('is_visible', False)

        return result

    def __del__(self):
        """Cleanup mediapipe resources."""
        if hasattr(self, 'pose') and self.available:
            self.pose.close()


class TwoHandedPoseAnalyzer:
    """
    Analyzes two-handed gestures and coordination.
    """

    def __init__(self, **detector_kwargs):
        self.detector = PoseDetector(**detector_kwargs)
        self.left_hand_history = []
        self.right_hand_history = []
        self.coordination_history = []

    def analyze_frame(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a frame for two-handed information.

        Args:
            img: Input image

        Returns:
            Analysis results
        """
        if not self.detector.available:
            return {'error': 'MediaPipe not available'}

        # Detect pose
        self.detector.detect(img)

        # Get two-handed info
        info = self.detector.get_two_handed_info()

        # Track history
        if info.get('left_hand'):
            self.left_hand_history.append(info['left_hand'])
        if info.get('right_hand'):
            self.right_hand_history.append(info['right_hand'])

        # Keep history manageable
        max_history = 100
        self.left_hand_history = self.left_hand_history[-max_history:]
        self.right_hand_history = self.right_hand_history[-max_history:]

        # Calculate coordination metrics
        coordination = self._calculate_coordination()
        info['coordination'] = coordination

        return info

    def _calculate_coordination(self) -> Dict[str, Any]:
        """Calculate coordination metrics between hands."""
        if len(self.left_hand_history) < 2 or len(self.right_hand_history) < 2:
            return {}

        # Get recent positions
        left_recent = self.left_hand_history[-10:]
        right_recent = self.right_hand_history[-10:]

        # Calculate synchronization (how similar are the movements)
        left_velocities = []
        right_velocities = []

        for i in range(1, len(left_recent)):
            left_vx = left_recent[i]['x'] - left_recent[i-1]['x']
            left_vy = left_recent[i]['y'] - left_recent[i-1]['y']
            left_velocities.append((left_vx, left_vy))

            right_vx = right_recent[i]['x'] - right_recent[i-1]['x']
            right_vy = right_recent[i]['y'] - right_recent[i-1]['y']
            right_velocities.append((right_vx, right_vy))

        # Calculate velocity correlation
        if left_velocities and right_velocities:
            left_arr = np.array(left_velocities)
            right_arr = np.array(right_velocities)

            # Normalize
            left_norm = np.linalg.norm(left_arr, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right_arr, axis=1, keepdims=True)

            left_norm[left_norm == 0] = 1
            right_norm[right_norm == 0] = 1

            left_normalized = left_arr / left_norm
            right_normalized = right_arr / right_norm

            # Dot product for similarity
            similarity = np.mean(np.sum(left_normalized * right_normalized, axis=1))

            return {
                'synchronization': float(similarity),
                'is_synchronized': similarity > 0.7
            }

        return {}
