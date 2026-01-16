"""
MotionDescriptor - Structured Motion Representation

Converts raw hand landmarks into structured, reusable motion descriptors.
This abstraction layer enables the same motion data to drive multiple outputs:
- Cursor control (current)
- JSON export for training data
- Animation systems
- Validation tools
- Real-time translation

Key Design Principle: Motion is data, not just input.

For sign language systems, this means:
- Same capture → different outputs
- Motion sequences can be logged, analyzed, replayed
- Enables building training datasets
- Supports validation with deaf community

Author: Richard Kabue Karoki
Created: January 2026
"""

import time
import json
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class MotionDescriptor:
    """
    Converts hand landmarks into structured motion descriptors.
    
    A motion descriptor captures the semantic meaning of hand position
    and configuration at a point in time. This is critical for sign
    language because:
    
    1. Signs are made of sequences, not single frames
    2. The same motion data needs multiple uses (animation, logging, etc.)
    3. Motion must be validated for semantic accuracy, not just visual quality
    
    Attributes:
        motion_history: List of recent motion descriptors
        primitives_seen: Set of all primitives encountered
        recording_start_time: When current recording session started
    """
    
    def __init__(self):
        self.motion_history: List[Dict] = []
        self.primitives_seen: set = set()
        self.recording_start_time: Optional[float] = None
        
    def create_descriptor(self, lmList: List, fingers: List[int], 
                         frame_shape: Optional[Tuple[int, int]] = None) -> Optional[Dict]:
        """
        Creates a structured representation of hand motion state.
        
        This is the core abstraction: instead of immediately acting on gestures,
        we create a structured representation that can be:
        - Logged for later analysis
        - Exported as training data
        - Used to drive animations
        - Validated for semantic correctness
        
        Args:
            lmList: List of [id, x, y] landmark positions (21 landmarks)
            fingers: List of 5 binary values [thumb, index, middle, ring, pinky]
            frame_shape: Optional (height, width) for normalization
            
        Returns:
            Dictionary containing structured motion data, or None if invalid
        """
        if not lmList or len(lmList) < 21:
            return None
        
        if not fingers or len(fingers) != 5:
            return None
        
        # Timestamp - critical for temporal sequences
        timestamp = time.time()
        if self.recording_start_time is None:
            self.recording_start_time = timestamp
        relative_time = timestamp - self.recording_start_time
        
        descriptor = {
            # Temporal information
            'timestamp': timestamp,
            'relative_time': relative_time,
            'frame_num': len(self.motion_history),
            
            # Hand identification
            'hand': 'right',  # TODO: Detect left vs right
            
            # Handshape representation (one of 4 sign language parameters)
            'fingers_extended': fingers,
            'finger_count': sum(fingers),
            'handshape_code': self._encode_handshape(fingers),
            
            # Key landmark positions (Location parameter)
            'landmarks': {
                'wrist': {'x': lmList[0][1], 'y': lmList[0][2]},
                'thumb_tip': {'x': lmList[4][1], 'y': lmList[4][2]},
                'index_tip': {'x': lmList[8][1], 'y': lmList[8][2]},
                'middle_tip': {'x': lmList[12][1], 'y': lmList[12][2]},
                'ring_tip': {'x': lmList[16][1], 'y': lmList[16][2]},
                'pinky_tip': {'x': lmList[20][1], 'y': lmList[20][2]},
            },
            
            # Derived geometric features
            'features': {
                'pinch_distance': self._calculate_pinch(lmList),
                'hand_openness': self._calculate_openness(fingers),
                'hand_span': self._calculate_span(lmList),
                'palm_center': self._calculate_palm_center(lmList),
            },
            
            # Motion primitive classification (building blocks of signs)
            'primitive': self._classify_primitive(fingers, lmList),
            
            # Velocity (Movement parameter - derived from history)
            'velocity': self._calculate_velocity(lmList) if len(self.motion_history) > 0 else None,
        }
        
        # Normalize coordinates if frame shape provided
        if frame_shape:
            descriptor['normalized'] = self._normalize_coordinates(descriptor, frame_shape)
        
        # Track primitives seen
        self.primitives_seen.add(descriptor['primitive'])
        
        # Add to history
        self.motion_history.append(descriptor)
        
        return descriptor
    
    def _encode_handshape(self, fingers: List[int]) -> str:
        """
        Encode finger configuration as compact string.
        Example: [1,1,0,0,0] -> "11000"
        
        This is a basic handshape classifier. In real sign language systems,
        handshape includes:
        - Finger extension (captured here)
        - Finger curl amount (not captured)
        - Thumb position relative to fingers (not captured)
        - 3D orientation (not captured)
        """
        return ''.join(str(f) for f in fingers)
    
    def _calculate_pinch(self, lmList: List) -> float:
        """
        Calculate distance between thumb and index finger tips.
        
        Pinch is a common gesture primitive in sign language. The distance
        indicates whether fingers are touching (closed pinch) or separated
        (open pinch), which can change meaning.
        """
        if len(lmList) < 9:
            return 0.0
        
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
        
        return math.hypot(x2 - x1, y2 - y1)
    
    def _calculate_openness(self, fingers: List[int]) -> float:
        """
        Calculate how "open" the hand is (0.0 = fist, 1.0 = fully open).
        
        This is a simplified metric. Real handshape classification needs
        to consider finger curl, not just extension.
        """
        return sum(fingers) / len(fingers)
    
    def _calculate_span(self, lmList: List) -> float:
        """
        Calculate distance between thumb tip and pinky tip.
        
        Hand span indicates how spread the fingers are, which is relevant
        for signs that require wide vs. narrow hand configurations.
        """
        if len(lmList) < 21:
            return 0.0
        
        x1, y1 = lmList[4][1], lmList[4][2]   # Thumb tip
        x2, y2 = lmList[20][1], lmList[20][2] # Pinky tip
        
        return math.hypot(x2 - x1, y2 - y1)
    
    def _calculate_palm_center(self, lmList: List) -> Dict[str, float]:
        """
        Calculate approximate palm center from wrist and middle finger base.
        
        Palm center is useful for tracking hand location (one of the 4
        core sign language parameters).
        """
        if len(lmList) < 10:
            return {'x': 0, 'y': 0}
        
        # Average of wrist and middle finger MCP (base)
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]
        middle_base_x, middle_base_y = lmList[9][1], lmList[9][2]
        
        return {
            'x': (wrist_x + middle_base_x) / 2,
            'y': (wrist_y + middle_base_y) / 2
        }
    
    def _calculate_velocity(self, lmList: List) -> Optional[Dict[str, float]]:
        """
        Calculate velocity of index finger tip since last frame.
        
        Movement is one of the 4 core sign language parameters. Velocity
        captures speed and direction, which are linguistically significant.
        
        Returns:
            Dict with vx, vy, magnitude, or None if insufficient history
        """
        if len(self.motion_history) < 1:
            return None
        
        if len(lmList) < 9:
            return None
        
        # Current index tip position
        curr_x, curr_y = lmList[8][1], lmList[8][2]
        
        # Previous index tip position
        prev_landmarks = self.motion_history[-1]['landmarks']
        prev_x = prev_landmarks['index_tip']['x']
        prev_y = prev_landmarks['index_tip']['y']
        
        # Time difference
        curr_time = time.time()
        prev_time = self.motion_history[-1]['timestamp']
        dt = curr_time - prev_time
        
        if dt == 0:
            return None
        
        # Velocity components
        vx = (curr_x - prev_x) / dt
        vy = (curr_y - prev_y) / dt
        magnitude = math.hypot(vx, vy)
        
        return {
            'vx': vx,
            'vy': vy,
            'magnitude': magnitude,
            'direction': math.atan2(vy, vx)  # Radians
        }
    
    def _classify_primitive(self, fingers: List[int], lmList: List) -> str:
        """
        Classify motion into gesture primitives.
        
        These are building blocks of signs, not complete words. Real sign
        language classification requires:
        - Temporal context (sequences)
        - Body position
        - Facial expressions
        - Movement dynamics
        
        This method only captures handshape-based primitives as a starting point.
        """
        # Classic pointing gesture
        if fingers == [0, 1, 0, 0, 0]:
            return "POINT"
        
        # Peace sign / V-shape
        elif fingers == [0, 1, 1, 0, 0]:
            return "PEACE_V"
        
        # Fully open hand
        elif fingers == [1, 1, 1, 1, 1]:
            return "OPEN_HAND"
        
        # Closed fist
        elif sum(fingers) == 0:
            return "FIST"
        
        # Thumb up
        elif fingers == [1, 0, 0, 0, 0]:
            return "THUMBS_UP"
        
        # OK sign (thumb + index extended, others down)
        elif fingers == [1, 1, 0, 0, 0]:
            # Check if thumb and index are close (forming circle)
            pinch_dist = self._calculate_pinch(lmList)
            if pinch_dist < 40:  # Threshold for "touching"
                return "OK_SIGN"
            else:
                return "PINCH_READY"
        
        # Three fingers (index, middle, ring)
        elif fingers == [0, 1, 1, 1, 0]:
            return "THREE"
        
        # Four fingers (no thumb)
        elif fingers == [0, 1, 1, 1, 1]:
            return "FOUR"
        
        # Pinky extended
        elif fingers == [0, 0, 0, 0, 1]:
            return "PINKY"
        
        # Unknown/composite handshape
        else:
            return f"UNKNOWN_{self._encode_handshape(fingers)}"
    
    def _normalize_coordinates(self, descriptor: Dict, frame_shape: Tuple[int, int]) -> Dict:
        """
        Normalize coordinates to [0, 1] range based on frame dimensions.
        
        Normalization is critical for:
        - Comparing gestures across different camera resolutions
        - Training ML models
        - Animation retargeting to different character sizes
        """
        height, width = frame_shape
        
        normalized = {}
        for landmark_name, coords in descriptor['landmarks'].items():
            normalized[landmark_name] = {
                'x': coords['x'] / width,
                'y': coords['y'] / height
            }
        
        # Normalize palm center
        palm = descriptor['features']['palm_center']
        normalized['palm_center'] = {
            'x': palm['x'] / width,
            'y': palm['y'] / height
        }
        
        return normalized
    
    def get_motion_sequence(self, window_seconds: float = 2.0) -> List[Dict]:
        """
        Returns recent motion history within time window.
        
        Critical for sign language: signs are SEQUENCES, not single frames.
        A "wave" isn't just "open hand" but "open hand + sideways motion + repetition"
        
        Args:
            window_seconds: How far back to look (default 2 seconds)
            
        Returns:
            List of motion descriptors within window
        """
        if not self.motion_history:
            return []
        
        cutoff_time = time.time() - window_seconds
        return [m for m in self.motion_history if m['timestamp'] > cutoff_time]
    
    def get_primitive_sequence(self, window_seconds: float = 2.0) -> List[str]:
        """
        Returns sequence of primitives within time window.
        
        Example: ["OPEN_HAND", "OPEN_HAND", "FIST", "FIST", "OPEN_HAND"]
        
        This sequence data is what sign language recognition needs to analyze.
        """
        sequence = self.get_motion_sequence(window_seconds)
        return [m['primitive'] for m in sequence]
    
    def save_sequence(self, filename: str, gesture_name: str, metadata: Optional[Dict] = None):
        """
        Save motion sequence to JSON file.
        
        This enables:
        - Building training datasets
        - Offline analysis
        - Sharing with researchers
        - Validation with deaf community
        - Animation retargeting
        
        Args:
            filename: Path to save JSON file
            gesture_name: Name/label of the gesture
            metadata: Optional additional info (signer name, date, etc.)
        """
        if not self.motion_history:
            print("Warning: No motion history to save")
            return
        
        # Calculate duration
        start_time = self.motion_history[0]['timestamp']
        end_time = self.motion_history[-1]['timestamp']
        duration = end_time - start_time
        
        # Calculate average FPS
        fps = len(self.motion_history) / duration if duration > 0 else 0
        
        # Compile data
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
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(self.motion_history)} frames to {filename}")
            print(f"  Duration: {duration:.2f}s, FPS: {fps:.1f}")
            print(f"  Primitives: {', '.join(self.primitives_seen)}")
        except Exception as e:
            print(f"✗ Error saving motion data: {e}")
    
    def clear_history(self):
        """Clear motion history. Call this before starting new recording."""
        self.motion_history = []
        self.primitives_seen = set()
        self.recording_start_time = None
    
    def get_statistics(self) -> Dict:
        """
        Calculate statistics about recorded motion.
        
        Useful for:
        - Quality checking recordings
        - Debugging motion capture
        - Comparing gestures
        """
        if not self.motion_history:
            return {'error': 'No motion history'}
        
        # Temporal stats
        duration = self.motion_history[-1]['timestamp'] - self.motion_history[0]['timestamp']
        fps = len(self.motion_history) / duration if duration > 0 else 0
        
        # Primitive distribution
        primitives = [m['primitive'] for m in self.motion_history]
        unique_primitives = set(primitives)
        primitive_counts = {p: primitives.count(p) for p in unique_primitives}
        
        # Velocity stats (if available)
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


# Example usage and testing
if __name__ == "__main__":
    """
    This demonstrates how MotionDescriptor decouples motion capture from output.
    
    In the real application:
    - AiVirtualMouseProject.py uses this for cursor control + recording
    - Future animation system uses this for 3D rig control
    - Analysis tools use this for visualization
    - ML training uses this for datasets
    """
    
    print("MotionDescriptor Module")
    print("=" * 50)
    print("This module converts raw hand landmarks into structured")
    print("motion descriptors for sign language systems.")
    print()
    print("Key Features:")
    print("  - Temporal sequence capture")
    print("  - Gesture primitive classification")
    print("  - JSON export for training data")
    print("  - Velocity and motion analysis")
    print("  - Reusable across multiple output systems")
    print()
    print("Import this in your main script:")
    print("  from MotionDescriptor import MotionDescriptor")