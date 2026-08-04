"""
Motion Descriptor - Structured Motion Representation

Converts raw hand landmarks into structured, reusable motion descriptors.
Core abstraction layer enabling the same motion data to drive multiple outputs.
"""

import time
import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Primitive classification constants — covers all 32 finger combos
# Format: (thumb, index, middle, ring, pinky)
PRIMITIVE_MAP: Dict[Tuple[int, ...], str] = {
    # 0 fingers
    (0, 0, 0, 0, 0): "FIST",
    # 1 finger
    (0, 1, 0, 0, 0): "POINT",
    (0, 0, 1, 0, 0): "MIDDLE_POINT",
    (0, 0, 0, 1, 0): "RING_ONLY",
    (0, 0, 0, 0, 1): "PINKY",
    (1, 0, 0, 0, 0): "THUMBS_UP",
    # 2 fingers
    (0, 1, 1, 0, 0): "PEACE_V",
    (0, 1, 0, 1, 0): "ROCK",
    (0, 1, 0, 0, 1): "SHAKA",
    (0, 0, 1, 1, 0): "TWO_MID_RING",
    (0, 0, 1, 0, 1): "TWO_MID_PINKY",
    (0, 0, 0, 1, 1): "TWO_RING_PINKY",
    (1, 1, 0, 0, 0): "GUN",
    (1, 0, 1, 0, 0): "THUMB_MIDDLE",
    (1, 0, 0, 1, 0): "THUMB_RING",
    (1, 0, 0, 0, 1): "CALL_ME",
    # 3 fingers
    (0, 1, 1, 1, 0): "THREE",
    (0, 1, 1, 0, 1): "THREE_SPREAD",
    (0, 1, 0, 1, 1): "W3",
    (0, 0, 1, 1, 1): "THREE_MID",
    (1, 1, 1, 0, 0): "THUMB_THREE",
    (1, 1, 0, 1, 0): "THUMB_ROCK",
    (1, 0, 1, 1, 0): "FORK",
    (1, 0, 1, 0, 1): "SPIDER",
    (1, 0, 0, 1, 1): "YAW",
    (1, 1, 0, 0, 1): "L_SHAPE",
    # 4 fingers
    (0, 1, 1, 1, 1): "FOUR",
    (1, 1, 1, 1, 0): "FOUR_NO_PINKY",
    (1, 1, 1, 0, 1): "FOUR_NO_RING",
    (1, 1, 0, 1, 1): "FOUR_NO_MIDDLE",
    (1, 0, 1, 1, 1): "FOUR_NO_INDEX",
    # 5 fingers
    (1, 1, 1, 1, 1): "OPEN_HAND",
}

# Motion detection thresholds
MOTION_WINDOW = 15  # frames to analyze for motion patterns
CIRCLE_MIN_FRAMES = 10
CIRCLE_MIN_RADIUS = 30  # pixels
SWIPE_MIN_DISTANCE = 100  # pixels
SWIPE_MIN_FRAMES = 4
WAVE_MIN_DIR_CHANGES = 3
WAVE_MIN_FRAMES = 8

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
MIDDLE_BASE = 9
INDEX_MCP = 5


class MotionDescriptor:
    """
    Converts hand landmarks into structured motion descriptors.

    Attributes:
        motion_history: List of recent motion descriptors
        primitives_seen: Set of all primitives encountered
        recording_start_time: When current recording session started
        max_history: Maximum frames to keep in memory (0 = unlimited)
    """

    __slots__ = ('motion_history', 'primitives_seen', 'recording_start_time', 'max_history')

    def __init__(self, max_history: int = 1000) -> None:
        self.motion_history: List[Dict[str, Any]] = []
        self.primitives_seen: Set[str] = set()
        self.recording_start_time: Optional[float] = None
        self.max_history: int = max_history or 0

    def create_descriptor(
        self,
        lmList: List[List[float]],
        fingers: List[int],
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a structured representation of hand motion state.

        Args:
            lmList: List of 21 landmarks as [id, x, y] or [x, y, z]
            fingers: List of 5 binary values (0/1) for each finger
            frame_shape: Optional (height, width) for coordinate normalization

        Returns:
            Dictionary containing structured motion data, or None if invalid input
        """
        if not lmList or len(lmList) < 21:
            return None
        if not fingers or len(fingers) != 5:
            return None

        timestamp = time.time()
        if self.recording_start_time is None:
            self.recording_start_time = timestamp
        relative_time = timestamp - self.recording_start_time

        # Extract key landmarks once for reuse
        wrist_x, wrist_y = lmList[WRIST][1], lmList[WRIST][2]
        index_mcp_x = lmList[INDEX_MCP][1]

        descriptor: Dict[str, Any] = {
            'timestamp': timestamp,
            'relative_time': relative_time,
            'frame_num': len(self.motion_history),
            'hand': self._detect_hand_from_positions(wrist_x, index_mcp_x),
            'fingers_extended': fingers,
            'finger_count': sum(fingers),
            'handshape_code': self._encode_handshape(fingers),
            'landmarks': {
                'wrist': {'x': wrist_x, 'y': wrist_y},
                'thumb_tip': {'x': lmList[THUMB_TIP][1], 'y': lmList[THUMB_TIP][2]},
                'index_tip': {'x': lmList[INDEX_TIP][1], 'y': lmList[INDEX_TIP][2]},
                'middle_tip': {'x': lmList[MIDDLE_TIP][1], 'y': lmList[MIDDLE_TIP][2]},
                'ring_tip': {'x': lmList[RING_TIP][1], 'y': lmList[RING_TIP][2]},
                'pinky_tip': {'x': lmList[PINKY_TIP][1], 'y': lmList[PINKY_TIP][2]},
            },
            'features': {
                'pinch_distance': self._calculate_pinch(lmList),
                'hand_openness': self._calculate_openness(fingers),
                'hand_span': self._calculate_span(lmList),
                'palm_center': self._calculate_palm_center(lmList),
            },
            'primitive': self._classify_primitive(fingers, lmList),
            'confidence': self._calculate_confidence(fingers, lmList),
            'velocity': self._calculate_velocity(lmList) if len(self.motion_history) > 0 else None,
        }

        if frame_shape:
            descriptor['normalized'] = self._normalize_coordinates(descriptor, frame_shape)

        self.primitives_seen.add(descriptor['primitive'])
        self.motion_history.append(descriptor)

        if self.max_history > 0 and len(self.motion_history) > self.max_history:
            excess = len(self.motion_history) - self.max_history
            self.motion_history = self.motion_history[excess:]

        return descriptor

    def _detect_hand_from_positions(self, wrist_x: float, index_mcp_x: float) -> str:
        """
        Detect whether the hand is left or right based on landmark positions.

        Args:
            wrist_x: X coordinate of wrist landmark
            index_mcp_x: X coordinate of index finger MCP joint

        Returns:
            'right' if index MCP is to the right of wrist, 'left' otherwise
        """
        return "right" if index_mcp_x > wrist_x else "left"

    def _encode_handshape(self, fingers: List[int]) -> str:
        """
        Encode finger configuration as compact binary string.

        Args:
            fingers: List of 5 binary values [thumb, index, middle, ring, pinky]

        Returns:
            String like '11000' representing finger states
        """
        return ''.join(str(f) for f in fingers)

    def _calculate_pinch(self, lmList: List[List[float]]) -> float:
        """
        Calculate Euclidean distance between thumb and index finger tips.

        Args:
            lmList: List of hand landmarks

        Returns:
            Distance in pixels, or 0.0 if landmarks are invalid
        """
        if len(lmList) < 9:
            return 0.0
        x1, y1 = lmList[THUMB_TIP][1], lmList[THUMB_TIP][2]
        x2, y2 = lmList[INDEX_TIP][1], lmList[INDEX_TIP][2]
        return math.hypot(x2 - x1, y2 - y1)

    def _calculate_openness(self, fingers: List[int]) -> float:
        """
        Calculate hand openness as ratio of extended fingers.

        Args:
            fingers: List of 5 binary values

        Returns:
            Float between 0.0 (fist) and 1.0 (fully open)
        """
        return sum(fingers) / len(fingers)

    def _calculate_span(self, lmList: List[List[float]]) -> float:
        """
        Calculate distance between thumb tip and pinky tip.

        Args:
            lmList: List of hand landmarks

        Returns:
            Distance in pixels, or 0.0 if landmarks are invalid
        """
        if len(lmList) < 21:
            return 0.0
        x1, y1 = lmList[THUMB_TIP][1], lmList[THUMB_TIP][2]
        x2, y2 = lmList[PINKY_TIP][1], lmList[PINKY_TIP][2]
        return math.hypot(x2 - x1, y2 - y1)

    def _calculate_palm_center(self, lmList: List[List[float]]) -> Dict[str, float]:
        """
        Approximate palm center from wrist and middle finger base.

        Args:
            lmList: List of hand landmarks

        Returns:
            Dictionary with 'x' and 'y' coordinates
        """
        if len(lmList) < 10:
            return {'x': 0.0, 'y': 0.0}
        wrist_x, wrist_y = lmList[WRIST][1], lmList[WRIST][2]
        middle_base_x, middle_base_y = lmList[MIDDLE_BASE][1], lmList[MIDDLE_BASE][2]
        return {
            'x': (wrist_x + middle_base_x) / 2,
            'y': (wrist_y + middle_base_y) / 2
        }

    def _calculate_confidence(self, fingers: List[int], lmList: List[List[float]]) -> float:
        """
        Calculate a heuristic confidence score for the current gesture.

        Based on how clearly the finger states match and landmark stability.
        Returns a value between 0.0 and 1.0.
        """
        if not fingers or len(lmList) < 21:
            return 0.0

        extended = sum(fingers)
        if extended == 0 or extended == 5:
            base = 0.9
        elif extended in (1, 4):
            base = 0.85
        else:
            base = 0.75

        wrist = lmList[WRIST]
        middle_base = lmList[MIDDLE_BASE]
        spread = math.hypot(middle_base[1] - wrist[1], middle_base[2] - wrist[2])
        if spread < 20:
            base *= 0.7
        elif spread > 100:
            base = min(base * 1.05, 1.0)

        return round(base, 3)

    def _calculate_velocity(self, lmList: List[List[float]]) -> Optional[Dict[str, float]]:
        """
        Calculate velocity of index finger tip since last frame.

        Args:
            lmList: List of hand landmarks

        Returns:
            Dictionary with velocity components and magnitude, or None
        """
        if not self.motion_history or len(lmList) < 9:
            return None

        curr_x, curr_y = lmList[INDEX_TIP][1], lmList[INDEX_TIP][2]
        prev_landmarks = self.motion_history[-1]['landmarks']
        prev_x = prev_landmarks['index_tip']['x']
        prev_y = prev_landmarks['index_tip']['y']

        curr_time = time.time()
        prev_time = self.motion_history[-1]['timestamp']
        dt = curr_time - prev_time

        if dt <= 0:
            return None

        vx = (curr_x - prev_x) / dt
        vy = (curr_y - prev_y) / dt
        magnitude = math.hypot(vx, vy)

        return {
            'vx': vx,
            'vy': vy,
            'magnitude': magnitude,
            'direction': math.atan2(vy, vx)
        }

    def _classify_primitive(self, fingers: List[int], lmList: List[List[float]]) -> str:
        """
        Classify hand configuration into gesture primitive.
        Uses finger states for static gestures + velocity history for motion gestures.

        Args:
            fingers: List of 5 binary values
            lmList: List of hand landmarks

        Returns:
            String name of the classified primitive
        """
        # 1) Check for motion-based gestures first (they override static)
        motion_gesture = self._detect_motion_gesture(lmList)
        if motion_gesture:
            return motion_gesture

        # 2) Check for PINCH (thumb + index close)
        pinch_dist = self._calculate_pinch(lmList)
        finger_count = sum(fingers)

        if fingers[0] == 1 and fingers[1] == 1 and finger_count == 2:
            return "OK_SIGN" if pinch_dist < 40 else "GUN"

        if fingers[0] == 1 and fingers[1] == 1 and finger_count == 3 and fingers[2] == 0:
            return "SPIDER" if pinch_dist < 50 else "GUN"

        # 3) Check for THUMBS_UP with angle validation
        if fingers == [1, 0, 0, 0, 0]:
            if self._is_thumb_up(lmList):
                return "THUMBS_UP"

        # 4) Lookup table for all other combos
        fingers_tuple = tuple(fingers)
        if fingers_tuple in PRIMITIVE_MAP:
            return PRIMITIVE_MAP[fingers_tuple]

        return f"UNKNOWN_{self._encode_handshape(fingers)}"

    def _is_thumb_up(self, lmList: List[List[float]]) -> bool:
        """Check if thumb is pointing upward (not sideways)."""
        if len(lmList) < 5:
            return False
        thumb_tip_y = lmList[THUMB_TIP][2]
        thumb_ip_y = lmList[3][2]  # Thumb IP joint
        wrist_y = lmList[WRIST][2]
        # Thumb tip should be significantly above thumb IP joint
        return thumb_tip_y < thumb_ip_y - 15

    def _detect_motion_gesture(self, lmList: List[List[float]]) -> Optional[str]:
        """
        Detect motion-based gestures from velocity/position history.
        Analyzes the last N frames to detect CIRCLE, WAVE, SWIPE patterns.
        """
        if len(self.motion_history) < MOTION_WINDOW:
            return None

        recent = self.motion_history[-MOTION_WINDOW:]
        wrist_positions = [(m['landmarks']['wrist']['x'], m['landmarks']['wrist']['y']) for m in recent]
        index_positions = [(m['landmarks']['index_tip']['x'], m['landmarks']['index_tip']['y']) for m in recent]

        # Use index tip for most gestures (more expressive than wrist)
        positions = index_positions

        # Detect CIRCLE: points form a rough circle
        circle = self._detect_circle(positions)
        if circle:
            return "CIRCLE"

        # Detect WAVE: rapid left-right oscillation
        wave = self._detect_wave(wrist_positions)
        if wave:
            return "WAVE"

        # Detect SWIPE: large unidirectional movement
        swipe = self._detect_swipe(positions)
        if swipe:
            return swipe

        return None

    def _detect_circle(self, positions: List[Tuple[float, float]]) -> bool:
        """Detect if positions form a circular pattern."""
        if len(positions) < CIRCLE_MIN_FRAMES:
            return False

        # Calculate centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)

        # Calculate average radius
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in positions]
        avg_radius = sum(radii) / len(radii)

        if avg_radius < CIRCLE_MIN_RADIUS:
            return False

        # Check if points are roughly equidistant from center (low variance)
        variance = sum((r - avg_radius) ** 2 for r in radii) / len(radii)
        # Also check that points go around (angular spread)
        angles = [math.atan2(p[1] - cy, p[0] - cx) for p in positions]
        angle_range = max(angles) - min(angles)

        # Low variance + large angle spread = circle
        return variance < (avg_radius * 0.6) ** 2 and angle_range > 3.0

    def _detect_wave(self, positions: List[Tuple[float, float]]) -> bool:
        """Detect oscillating left-right motion (wave)."""
        if len(positions) < WAVE_MIN_FRAMES:
            return False

        # Count direction changes along X axis
        dir_changes = 0
        for i in range(2, len(positions)):
            dx_prev = positions[i-1][0] - positions[i-2][0]
            dx_curr = positions[i][0] - positions[i-1][0]
            if dx_prev * dx_curr < 0 and abs(dx_curr) > 3:
                dir_changes += 1

        # Check total X displacement is significant
        total_dx = abs(positions[-1][0] - positions[0][0])

        return dir_changes >= WAVE_MIN_DIR_CHANGES and total_dx > 40

    def _detect_swipe(self, positions: List[Tuple[float, float]]) -> Optional[str]:
        """Detect unidirectional swipe movement."""
        if len(positions) < SWIPE_MIN_FRAMES:
            return None

        total_dx = positions[-1][0] - positions[0][0]
        total_dy = positions[-1][1] - positions[0][1]

        # Check if movement is primarily horizontal or vertical
        if abs(total_dx) > SWIPE_MIN_DISTANCE and abs(total_dx) > abs(total_dy) * 2:
            return "SWIPE_LEFT" if total_dx < 0 else "SWIPE_RIGHT"

        return None

    def _normalize_coordinates(
        self,
        descriptor: Dict[str, Any],
        frame_shape: Tuple[int, int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize coordinates to [0, 1] range.

        Args:
            descriptor: Motion descriptor with landmarks
            frame_shape: Tuple of (height, width)

        Returns:
            Dictionary of normalized coordinates
        """
        height, width = frame_shape
        if height <= 0 or width <= 0:
            logger.warning("Invalid frame shape: %s", frame_shape)
            return {}

        normalized: Dict[str, Dict[str, float]] = {}
        for landmark_name, coords in descriptor['landmarks'].items():
            normalized[landmark_name] = {
                'x': coords['x'] / width,
                'y': coords['y'] / height
            }
        palm = descriptor['features']['palm_center']
        normalized['palm_center'] = {
            'x': palm['x'] / width,
            'y': palm['y'] / height
        }
        return normalized

    def get_motion_sequence(self, window_seconds: float = 2.0) -> List[Dict[str, Any]]:
        """
        Get recent motion history within time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            List of motion descriptors within the time window
        """
        if not self.motion_history:
            return []
        cutoff_time = time.time() - window_seconds
        return [m for m in self.motion_history if m['timestamp'] > cutoff_time]

    def get_primitive_sequence(self, window_seconds: float = 2.0) -> List[str]:
        """
        Get sequence of primitives within time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            List of primitive names
        """
        return [m['primitive'] for m in self.get_motion_sequence(window_seconds)]

    def save_sequence(
        self,
        filename: str,
        gesture_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save motion sequence to JSON file.

        Args:
            filename: Path to save the JSON file
            gesture_name: Name of the gesture
            metadata: Optional additional metadata

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.motion_history:
            logger.warning("No motion history to save")
            return False

        start_time = self.motion_history[0]['timestamp']
        end_time = self.motion_history[-1]['timestamp']
        duration = end_time - start_time
        fps = len(self.motion_history) / duration if duration > 0 else 0

        data = {
            'metadata': {
                'gesture_name': gesture_name,
                'recorded_at': datetime.now().isoformat(),
                'duration_seconds': duration,
                'total_frames': len(self.motion_history),
                'average_fps': fps,
                'primitives_used': list(self.primitives_seen),
                'custom': metadata or {}
            },
            'frames': self.motion_history
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved %d frames to %s", len(self.motion_history), filename)
            return True
        except (IOError, OSError) as e:
            logger.error("Error saving motion data: %s", e)
            return False

    def clear_history(self) -> None:
        """Clear motion history before starting new recording."""
        self.motion_history = []
        self.primitives_seen = set()
        self.recording_start_time = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about recorded motion.

        Returns:
            Dictionary containing motion statistics
        """
        if not self.motion_history:
            return {'error': 'No motion history'}

        duration = self.motion_history[-1]['timestamp'] - self.motion_history[0]['timestamp']
        fps = len(self.motion_history) / duration if duration > 0 else 0

        # Count primitives efficiently using Counter
        from collections import Counter
        primitive_counter = Counter(m['primitive'] for m in self.motion_history)
        primitive_counts = dict(primitive_counter)

        # Collect velocities using list comprehension
        velocities = [
            m['velocity']['magnitude']
            for m in self.motion_history
            if m['velocity'] is not None and 'magnitude' in m['velocity']
        ]

        velocity_stats = None
        if velocities:
            velocity_stats = {
                'mean': sum(velocities) / len(velocities),
                'max': max(velocities),
                'min': min(velocities)
            }

        return {
            'duration_seconds': duration,
            'total_frames': len(self.motion_history),
            'average_fps': fps,
            'primitive_counts': primitive_counts,
            'unique_primitives': len(primitive_counts),
            'velocity_stats': velocity_stats
        }


if __name__ == "__main__":
    print("MotionDescriptor Module")
    print("=" * 50)
    print("Converts raw hand landmarks into structured motion descriptors.")
    print()
    print("Import: from hand_motion import MotionDescriptor")
