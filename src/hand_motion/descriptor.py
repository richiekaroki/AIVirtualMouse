"""
Motion Descriptor - Structured Motion Representation

Converts raw hand landmarks into structured, reusable motion descriptors.
Core abstraction layer enabling the same motion data to drive multiple outputs.
"""

import time
import json
import math
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MotionDescriptor:
    """
    Converts hand landmarks into structured motion descriptors.

    Attributes:
        motion_history: List of recent motion descriptors
        primitives_seen: Set of all primitives encountered
        recording_start_time: When current recording session started
        max_history: Maximum frames to keep in memory (0 = unlimited)
    """

    def __init__(self, max_history: int = 1000):
        self.motion_history: List[Dict] = []
        self.primitives_seen: set = set()
        self.recording_start_time: Optional[float] = None
        self.max_history: int = max_history or 0

    def create_descriptor(self, lmList: List, fingers: List[int],
                          frame_shape: Optional[Tuple[int, int]] = None) -> Optional[Dict]:
        """Create a structured representation of hand motion state."""
        if not lmList or len(lmList) < 21:
            return None
        if not fingers or len(fingers) != 5:
            return None

        timestamp = time.time()
        if self.recording_start_time is None:
            self.recording_start_time = timestamp
        relative_time = timestamp - self.recording_start_time

        descriptor = {
            'timestamp': timestamp,
            'relative_time': relative_time,
            'frame_num': len(self.motion_history),
            'hand': self._detect_hand(lmList),
            'fingers_extended': fingers,
            'finger_count': sum(fingers),
            'handshape_code': self._encode_handshape(fingers),
            'landmarks': {
                'wrist': {'x': lmList[0][1], 'y': lmList[0][2]},
                'thumb_tip': {'x': lmList[4][1], 'y': lmList[4][2]},
                'index_tip': {'x': lmList[8][1], 'y': lmList[8][2]},
                'middle_tip': {'x': lmList[12][1], 'y': lmList[12][2]},
                'ring_tip': {'x': lmList[16][1], 'y': lmList[16][2]},
                'pinky_tip': {'x': lmList[20][1], 'y': lmList[20][2]},
            },
            'features': {
                'pinch_distance': self._calculate_pinch(lmList),
                'hand_openness': self._calculate_openness(fingers),
                'hand_span': self._calculate_span(lmList),
                'palm_center': self._calculate_palm_center(lmList),
            },
            'primitive': self._classify_primitive(fingers, lmList),
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

    def _detect_hand(self, lmList: List) -> str:
        """Detect whether the hand is left or right."""
        if len(lmList) < 6:
            return "unknown"
        wrist_x = lmList[0][1]
        index_mcp_x = lmList[5][1]
        return "right" if index_mcp_x > wrist_x else "left"

    def _encode_handshape(self, fingers: List[int]) -> str:
        """Encode finger configuration as compact string (e.g. '11000')."""
        return ''.join(str(f) for f in fingers)

    def _calculate_pinch(self, lmList: List) -> float:
        """Distance between thumb and index finger tips."""
        if len(lmList) < 9:
            return 0.0
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        return math.hypot(x2 - x1, y2 - y1)

    def _calculate_openness(self, fingers: List[int]) -> float:
        """Hand openness (0.0=fist, 1.0=fully open)."""
        return sum(fingers) / len(fingers)

    def _calculate_span(self, lmList: List) -> float:
        """Distance between thumb tip and pinky tip."""
        if len(lmList) < 21:
            return 0.0
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[20][1], lmList[20][2]
        return math.hypot(x2 - x1, y2 - y1)

    def _calculate_palm_center(self, lmList: List) -> Dict[str, float]:
        """Approximate palm center from wrist and middle finger base."""
        if len(lmList) < 10:
            return {'x': 0, 'y': 0}
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]
        middle_base_x, middle_base_y = lmList[9][1], lmList[9][2]
        return {
            'x': (wrist_x + middle_base_x) / 2,
            'y': (wrist_y + middle_base_y) / 2
        }

    def _calculate_velocity(self, lmList: List) -> Optional[Dict[str, float]]:
        """Velocity of index finger tip since last frame."""
        if len(self.motion_history) < 1 or len(lmList) < 9:
            return None

        curr_x, curr_y = lmList[8][1], lmList[8][2]
        prev_landmarks = self.motion_history[-1]['landmarks']
        prev_x = prev_landmarks['index_tip']['x']
        prev_y = prev_landmarks['index_tip']['y']

        curr_time = time.time()
        prev_time = self.motion_history[-1]['timestamp']
        dt = curr_time - prev_time

        if dt == 0:
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

    def _classify_primitive(self, fingers: List[int], lmList: List) -> str:
        """Classify motion into gesture primitives."""
        if fingers == [0, 1, 0, 0, 0]:
            return "POINT"
        elif fingers == [0, 1, 1, 0, 0]:
            return "PEACE_V"
        elif fingers == [1, 1, 1, 1, 1]:
            return "OPEN_HAND"
        elif sum(fingers) == 0:
            return "FIST"
        elif fingers == [1, 0, 0, 0, 0]:
            return "THUMBS_UP"
        elif fingers == [1, 1, 0, 0, 0]:
            pinch_dist = self._calculate_pinch(lmList)
            return "OK_SIGN" if pinch_dist < 40 else "PINCH_READY"
        elif fingers == [0, 1, 1, 1, 0]:
            return "THREE"
        elif fingers == [0, 1, 1, 1, 1]:
            return "FOUR"
        elif fingers == [0, 0, 0, 0, 1]:
            return "PINKY"
        else:
            return f"UNKNOWN_{self._encode_handshape(fingers)}"

    def _normalize_coordinates(self, descriptor: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """Normalize coordinates to [0, 1] range."""
        height, width = frame_shape
        normalized = {}
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

    def get_motion_sequence(self, window_seconds: float = 2.0) -> List[Dict]:
        """Returns recent motion history within time window."""
        if not self.motion_history:
            return []
        cutoff_time = time.time() - window_seconds
        return [m for m in self.motion_history if m['timestamp'] > cutoff_time]

    def get_primitive_sequence(self, window_seconds: float = 2.0) -> List[str]:
        """Returns sequence of primitives within time window."""
        return [m['primitive'] for m in self.get_motion_sequence(window_seconds)]

    def save_sequence(self, filename: str, gesture_name: str, metadata: Optional[Dict] = None):
        """Save motion sequence to JSON file."""
        if not self.motion_history:
            logger.warning("No motion history to save")
            return

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
        except Exception as e:
            logger.error("Error saving motion data: %s", e)

    def clear_history(self):
        """Clear motion history before starting new recording."""
        self.motion_history = []
        self.primitives_seen = set()
        self.recording_start_time = None

    def get_statistics(self) -> Dict:
        """Calculate statistics about recorded motion."""
        if not self.motion_history:
            return {'error': 'No motion history'}

        duration = self.motion_history[-1]['timestamp'] - self.motion_history[0]['timestamp']
        fps = len(self.motion_history) / duration if duration > 0 else 0

        primitives = [m['primitive'] for m in self.motion_history]
        unique_primitives = set(primitives)
        primitive_counts = {p: primitives.count(p) for p in unique_primitives}

        velocities = [m['velocity']['magnitude'] for m in self.motion_history
                      if m['velocity'] is not None]

        return {
            'duration_seconds': duration,
            'total_frames': len(self.motion_history),
            'average_fps': fps,
            'primitive_counts': primitive_counts,
            'unique_primitives': len(unique_primitives),
            'velocity_stats': {
                'mean': sum(velocities) / len(velocities) if velocities else 0,
                'max': max(velocities) if velocities else 0,
                'min': min(velocities) if velocities else 0
            } if velocities else None
        }


if __name__ == "__main__":
    print("MotionDescriptor Module")
    print("=" * 50)
    print("Converts raw hand landmarks into structured motion descriptors.")
    print()
    print("Import: from hand_motion import MotionDescriptor")
