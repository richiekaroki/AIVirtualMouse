"""
Fingerspelling Detection

Recognizes individual letters (A-Z) from hand landmark configurations.
Uses handshape geometry to classify static fingerspelling poses.

Usage:
    detector = FingerspellingDetector()
    letter = detector.recognize(landmarks)
"""

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ASL fingerspelling reference shapes (21-landmark geometry)
# Each letter is defined by finger extension patterns and thumb position
ASL_ALPHABET: Dict[str, Dict[str, Any]] = {
    "A": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist with thumb beside index"},
    "B": {"fingers": [0, 1, 1, 1, 1], "thumb_across": False, "description": "Four fingers up, thumb tucked"},
    "C": {"fingers": [0, 1, 1, 1, 1], "thumb_across": False, "description": "Curved hand like holding a ball"},
    "D": {"fingers": [0, 1, 0, 0, 0], "thumb_across": False, "description": "Index up, others touch thumb"},
    "E": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist with thumb under fingers"},
    "F": {"fingers": [0, 1, 1, 1, 0], "thumb_across": False, "description": "Three fingers up, index+thumb circle"},
    "G": {"fingers": [0, 1, 0, 0, 0], "thumb_across": True, "description": "Index and thumb pointing side"},
    "H": {"fingers": [0, 1, 1, 0, 0], "thumb_across": True, "description": "Index and middle horizontal"},
    "I": {"fingers": [0, 0, 0, 0, 1], "thumb_across": False, "description": "Pinky up only"},
    "J": {"fingers": [0, 0, 0, 0, 1], "thumb_across": False, "description": "Pinky draws J shape"},
    "K": {"fingers": [0, 1, 1, 0, 0], "thumb_across": False, "description": "Peace with thumb between"},
    "L": {"fingers": [0, 1, 0, 0, 0], "thumb_across": True, "description": "L shape with index and thumb"},
    "M": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist, thumb under 3 fingers"},
    "N": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist, thumb under 2 fingers"},
    "O": {"fingers": [0, 1, 1, 1, 1], "thumb_across": False, "description": "All fingers touch thumb tip"},
    "P": {"fingers": [0, 1, 1, 0, 0], "thumb_across": True, "description": "Like K but pointing down"},
    "Q": {"fingers": [0, 1, 0, 0, 0], "thumb_across": True, "description": "Like G but pointing down"},
    "R": {"fingers": [0, 1, 1, 0, 0], "thumb_across": False, "description": "Crossed index and middle"},
    "S": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist with thumb in front"},
    "T": {"fingers": [0, 0, 0, 0, 0], "thumb_across": True, "description": "Fist with thumb between index and middle"},
    "U": {"fingers": [0, 1, 1, 0, 0], "thumb_across": False, "description": "Index and middle up together"},
    "V": {"fingers": [0, 1, 1, 0, 0], "thumb_across": False, "description": "Peace sign"},
    "W": {"fingers": [0, 1, 1, 1, 0], "thumb_across": False, "description": "Three fingers spread up"},
    "X": {"fingers": [0, 0, 0, 0, 0], "thumb_across": False, "description": "Index hooked"},
    "Y": {"fingers": [0, 0, 0, 0, 1], "thumb_across": True, "description": "Thumb and pinky out (shaka)"},
    "Z": {"fingers": [0, 1, 0, 0, 0], "thumb_across": False, "description": "Index traces Z shape"},
}

# Fingertip and MCP landmark indices
TIP_IDS = [4, 8, 12, 16, 20]
MCP_IDS = [2, 5, 9, 13, 17]
WRIST_ID = 0


class FingerspellingDetector:
    """
    Detects fingerspelling letters from hand landmarks.

    Uses a combination of:
    - Finger extension detection (tip vs MCP positions)
    - Thumb position relative to palm
    - Inter-finger distances for distinguishing similar letters
    """

    def __init__(self):
        self.letter_buffer: List[str] = []
        self.buffer_size = 5
        self.stable_letter: Optional[str] = None
        self.stable_frames = 0
        self.min_stable_frames = 3

    def recognize(self, lm_list: List[List[float]]) -> Optional[str]:
        """
        Recognize a fingerspelling letter from landmarks.

        Args:
            lm_list: 21 landmarks [[id, x, y], ...]

        Returns:
            Recognized letter (A-Z) or None
        """
        if len(lm_list) < 21:
            return None

        points = np.array([[lm[1], lm[2]] for lm in lm_list[:21]], dtype=np.float32)

        # Detect which fingers are extended
        fingers = self._detect_fingers(points)

        # Detect thumb position
        thumb_across = self._is_thumb_across(points)

        # Compute additional features for disambiguation
        features = self._compute_features(points)

        # Match against reference alphabet
        best_letter = self._match_letter(fingers, thumb_across, features)

        # Temporal smoothing: require N consecutive same-letter detections
        self.letter_buffer.append(best_letter or "")
        if len(self.letter_buffer) > self.buffer_size:
            self.letter_buffer.pop(0)

        # Check for stable letter
        if best_letter and self.letter_buffer.count(best_letter) >= self.min_stable_frames:
            if self.stable_letter == best_letter:
                return None  # Already returned this letter
            self.stable_letter = best_letter
            self.stable_frames += 1
            return best_letter

        return None

    def get_stable_letter(self) -> Optional[str]:
        """Get the current stable letter (without consuming it)."""
        return self.stable_letter

    def reset(self):
        """Reset detection state."""
        self.letter_buffer.clear()
        self.stable_letter = None
        self.stable_frames = 0

    def _detect_fingers(self, points: np.ndarray) -> List[int]:
        """Detect extended fingers using tip vs MCP positions."""
        fingers = []
        # Thumb: compare x position (left/right hand aware)
        if points[TIP_IDS[0]][0] < points[MCP_IDS[0]][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: tip y < MCP y means extended (screen coords)
        for tip, mcp in zip(TIP_IDS[1:], MCP_IDS[1:]):
            if points[tip][1] < points[mcp][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def _is_thumb_across(self, points: np.ndarray) -> bool:
        """Check if thumb is across the palm (tucked or crossed)."""
        thumb_tip = points[TIP_IDS[0]]
        index_mcp = points[MCP_IDS[1]]
        ring_mcp = points[MCP_IDS[3]]
        palm_center = (index_mcp + ring_mcp) / 2

        dist_to_palm = math.hypot(thumb_tip[0] - palm_center[0], thumb_tip[1] - palm_center[1])
        palm_width = math.hypot(index_mcp[0] - ring_mcp[0], index_mcp[1] - ring_mcp[1])

        if palm_width == 0:
            return False

        return dist_to_palm / palm_width < 0.6

    def _compute_features(self, points: np.ndarray) -> Dict[str, float]:
        """Compute additional features for letter disambiguation."""
        features = {}

        # Index-middle finger spread
        idx_tip = points[TIP_IDS[1]]
        mid_tip = points[TIP_IDS[2]]
        features["spread"] = math.hypot(idx_tip[0] - mid_tip[0], idx_tip[1] - mid_tip[1])

        # Index finger curl (distance from tip to MCP, normalized)
        idx_tip = points[TIP_IDS[1]]
        idx_mcp = points[MCP_IDS[1]]
        wrist = points[WRIST_ID]
        wrist_to_mcp = math.hypot(idx_mcp[0] - wrist[0], idx_mcp[1] - wrist[1]) + 1e-8
        features["index_curl"] = math.hypot(idx_tip[0] - idx_mcp[0], idx_tip[1] - idx_mcp[1]) / wrist_to_mcp

        # Middle-ringed finger overlap
        mid_tip = points[TIP_IDS[2]]
        ring_tip = points[TIP_IDS[3]]
        features["mid_ring_dist"] = math.hypot(mid_tip[0] - ring_tip[0], mid_tip[1] - ring_tip[1])

        return features

    def _match_letter(
        self,
        fingers: List[int],
        thumb_across: bool,
        features: Dict[str, float],
    ) -> Optional[str]:
        """Match finger configuration to best letter."""
        best_match = None
        best_score = -1

        for letter, ref in ASL_ALPHABET.items():
            score = 0
            ref_fingers = ref["fingers"]
            ref_thumb = ref["thumb_across"]

            # Finger match score
            for f, r in zip(fingers, ref_fingers):
                if f == r:
                    score += 1

            # Thumb match
            if thumb_across == ref_thumb:
                score += 1

            # Disambiguation features
            if letter in ("K", "V") and features.get("spread", 0) > 20:
                score += 0.5
            if letter == "X" and features.get("index_curl", 0) < 0.5:
                score += 0.5

            if score > best_score:
                best_score = score
                best_match = letter

        # Require at least 5/6 features to match
        if best_score >= 5:
            return best_match
        return None

    def get_alphabet_reference(self) -> Dict[str, str]:
        """Get human-readable descriptions of all letters."""
        return {k: v["description"] for k, v in ASL_ALPHABET.items()}
