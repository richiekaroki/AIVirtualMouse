"""
MediaPipe Face Mesh Integration

Facial expression tracking for non-manual markers:
- 468-point face mesh
- Eye tracking
- Mouth expressions
- Eyebrow movements
- Head orientation
"""

import math
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
    logger.info("MediaPipe not installed. Face features will use fallback.")


# Key face mesh indices
FACE_MESH_INDICES = {
    # Eyes
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],

    # Eyebrows
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],

    # Mouth
    'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267],
    'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312],

    # Nose
    'nose': [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 2],

    # Face outline
    'face_outline': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
}


class FaceDetector:
    """
    Face detection and mesh analysis using MediaPipe.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize face detector.

        Args:
            static_image_mode: If True, treats each image independently
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: If True, refines landmark positions
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        if not MEDIAPIPE_AVAILABLE:
            self.available = False
            return

        self.available = True

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.results = None

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Detect face mesh in image.

        Args:
            img: Input image (BGR format)

        Returns:
            Image with face mesh drawn
        """
        if not self.available:
            return img

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False

        # Process
        self.results = self.face_mesh.process(img_rgb)

        # Draw mesh
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        return img

    def get_landmarks(
        self,
        face_idx: int = 0,
        img_shape: Optional[Tuple[int, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get face mesh landmarks.

        Args:
            face_idx: Face index (for multiple faces)
            img_shape: Image shape for coordinate conversion

        Returns:
            List of landmark dictionaries
        """
        if not self.available or not self.results or not self.results.multi_face_landmarks:
            return []

        if face_idx >= len(self.results.multi_face_landmarks):
            return []

        face = self.results.multi_face_landmarks[face_idx]
        landmarks = []

        for idx, landmark in enumerate(face.landmark):
            if img_shape:
                h, w = img_shape
                x = landmark.x * w
                y = landmark.y * h
            else:
                x = landmark.x
                y = landmark.y

            landmarks.append({
                'id': idx,
                'x': x,
                'y': y,
                'z': landmark.z
            })

        return landmarks

    def get_eye_aspect_ratio(self, eye_indices: List[int]) -> float:
        """
        Calculate eye aspect ratio (EAR).

        Args:
            eye_indices: Indices of eye landmarks

        Returns:
            Eye aspect ratio (higher = more open)
        """
        landmarks = self.get_landmarks()

        if not landmarks or len(landmarks) < max(eye_indices) + 1:
            return 0.0

        # Get eye points
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                eye_points.append((landmarks[idx]['x'], landmarks[idx]['y']))

        if len(eye_points) < 6:
            return 0.0

        # Calculate EAR
        # Vertical distances
        v1 = math.hypot(eye_points[1][0] - eye_points[5][0], eye_points[1][1] - eye_points[5][1])
        v2 = math.hypot(eye_points[2][0] - eye_points[4][0], eye_points[2][1] - eye_points[4][1])

        # Horizontal distance
        h = math.hypot(eye_points[0][0] - eye_points[3][0], eye_points[0][1] - eye_points[3][1])

        if h == 0:
            return 0.0

        ear = (v1 + v2) / (2.0 * h)
        return ear

    def get_mouth_aspect_ratio(self) -> float:
        """
        Calculate mouth aspect ratio.

        Returns:
            Mouth aspect ratio (higher = more open)
        """
        landmarks = self.get_landmarks()

        if not landmarks:
            return 0.0

        mouth_outer = FACE_MESH_INDICES['mouth_outer']

        # Get mouth points
        mouth_points = []
        for idx in mouth_outer:
            if idx < len(landmarks):
                mouth_points.append((landmarks[idx]['x'], landmarks[idx]['y']))

        if len(mouth_points) < 7:
            return 0.0

        # Vertical distance (top to bottom)
        top = mouth_points[2]  # Upper lip
        bottom = mouth_points[6]  # Lower lip
        vertical = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

        # Horizontal distance (left to right)
        left = mouth_points[0]
        right = mouth_points[8]
        horizontal = math.hypot(left[0] - right[0], left[1] - right[1])

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    def get_eyebrow_height(self) -> Dict[str, float]:
        """
        Calculate eyebrow heights relative to eyes.

        Returns:
            Dictionary with left and right eyebrow heights
        """
        landmarks = self.get_landmarks()

        if not landmarks:
            return {'left': 0.0, 'right': 0.0}

        # Get eye centers
        left_eye = FACE_MESH_INDICES['left_eye']
        right_eye = FACE_MESH_INDICES['right_eye']

        left_eye_center = np.mean([landmarks[i]['y'] for i in left_eye if i < len(landmarks)])
        right_eye_center = np.mean([landmarks[i]['y'] for i in right_eye if i < len(landmarks)])

        # Get eyebrow centers
        left_brow = FACE_MESH_INDICES['left_eyebrow']
        right_brow = FACE_MESH_INDICES['right_eyebrow']

        left_brow_center = np.mean([landmarks[i]['y'] for i in left_brow if i < len(landmarks)])
        right_brow_center = np.mean([landmarks[i]['y'] for i in right_brow if i < len(landmarks)])

        # Calculate height (lower y = higher position)
        left_height = left_eye_center - left_brow_center
        right_height = right_eye_center - right_brow_center

        return {
            'left': float(left_height),
            'right': float(right_height)
        }

    def get_head_orientation(self) -> Dict[str, float]:
        """
        Estimate head orientation (yaw, pitch, roll).

        Returns:
            Dictionary with rotation angles in degrees
        """
        landmarks = self.get_landmarks()

        if not landmarks or len(landmarks) < 334:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}

        # Key points for head orientation
        nose_tip = landmarks[1]
        left_ear = landmarks[234]  # Approximate
        right_ear = landmarks[454]  # Approximate
        forehead = landmarks[10]
        chin = landmarks[152]

        # Calculate yaw (left-right rotation)
        ear_distance = right_ear['x'] - left_ear['x']
        nose_offset = nose_tip['x'] - (left_ear['x'] + right_ear['x']) / 2
        yaw = math.degrees(math.atan2(nose_offset, ear_distance / 2))

        # Calculate pitch (up-down rotation)
        face_height = chin['y'] - forehead['y']
        nose_offset_y = nose_tip['y'] - (forehead['y'] + chin['y']) / 2
        pitch = math.degrees(math.atan2(nose_offset_y, face_height / 2))

        # Calculate roll (tilt)
        roll = math.degrees(math.atan2(
            right_ear['y'] - left_ear['y'],
            right_ear['x'] - left_ear['x']
        ))

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }

    def __del__(self):
        """Cleanup mediapipe resources."""
        if hasattr(self, 'face_mesh') and self.available:
            self.face_mesh.close()


class FacialExpressionAnalyzer:
    """
    Analyzes facial expressions for non-manual markers.
    """

    # Expression thresholds
    EYE_OPEN_THRESHOLD = 0.25
    MOUTH_OPEN_THRESHOLD = 0.35
    EYEBROW_RAISE_THRESHOLD = 0.02

    def __init__(self, **detector_kwargs):
        self.detector = FaceDetector(**detector_kwargs)
        self.expression_history = []

    def analyze_frame(self, img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial expression in frame.

        Args:
            img: Input image

        Returns:
            Expression analysis results
        """
        if not self.detector.available:
            return {'error': 'MediaPipe not available'}

        # Detect face
        self.detector.detect(img)

        # Calculate features
        left_ear = self.detector.get_eye_aspect_ratio(FACE_MESH_INDICES['left_eye'])
        right_ear = self.detector.get_eye_aspect_ratio(FACE_MESH_INDICES['right_eye'])
        mar = self.detector.get_mouth_aspect_ratio()
        eyebrows = self.detector.get_eyebrow_height()
        head = self.detector.get_head_orientation()

        # Determine expressions
        expressions = {
            'eyes_open': left_ear > self.EYE_OPEN_THRESHOLD and right_ear > self.EYE_OPEN_THRESHOLD,
            'mouth_open': mar > self.MOUTH_OPEN_THRESHOLD,
            'eyebrows_raised': (eyebrows['left'] > self.EYEBROW_RAISE_THRESHOLD and
                               eyebrows['right'] > self.EYEBROW_RAISE_THRESHOLD),
            'left_ear': left_ear,
            'right_ear': right_ear,
            'mouth_ratio': mar,
            'eyebrow_height': eyebrows,
            'head_orientation': head
        }

        # Classify expression
        expression = self._classify_expression(expressions)
        expressions['classified'] = expression

        # Track history
        self.expression_history.append(expression)
        if len(self.expression_history) > 30:
            self.expression_history.pop(0)

        return expressions

    def _classify_expression(self, features: Dict[str, Any]) -> str:
        """Classify expression from features."""
        if not features.get('eyes_open'):
            return 'blinking'
        if features.get('mouth_open'):
            return 'surprised'
        if features.get('eyebrows_raised'):
            return 'questioning'

        # Check head orientation
        head = features.get('head_orientation', {})
        if abs(head.get('yaw', 0)) > 20:
            return 'looking_side'
        if abs(head.get('pitch', 0)) > 20:
            return 'looking_up' if head['pitch'] < 0 else 'looking_down'

        return 'neutral'

    def get_non_manual_markers(self) -> Dict[str, Any]:
        """
        Get non-manual markers for sign language.

        Returns:
            Dictionary with non-manual marker states
        """
        if not self.expression_history:
            return {}

        current = self.expression_history[-1]

        return {
            'facial_expression': current,
            'is_questioning': current == 'questioning',
            'is_negating': current in ['shaking_head', 'frowning'],
            'is_emphasizing': current in ['surprised', 'wide_eyes']
        }
