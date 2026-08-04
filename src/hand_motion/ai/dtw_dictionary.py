"""
DTW Sign Language Dictionary

Dynamic Time Warping template matching for sign language recognition.
Compares incoming landmark sequences against stored templates to
recognize signs with temporal variation.

Usage:
    dictionary = DTWDictionary()
    dictionary.add_template("hello", hello_landmarks)
    result = dictionary.recognize(input_landmarks)
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DTWMatch:
    """Result of a DTW match against a template."""
    label: str
    distance: float
    confidence: float
    path_length: int


class DTWEngine:
    """
    Core Dynamic Time Warping implementation.

    Supports both standard DTW and constrained (Sakoe-Chiba band) DTW.
    """

    def __init__(self, band_width: int = 0):
        """
        Args:
            band_width: Sakoe-Chiba band width (0 = unconstrained)
        """
        self.band_width = band_width

    def distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Compute DTW distance between two sequences.

        Args:
            seq1: (T1, D) feature sequence
            seq2: (T2, D) feature sequence

        Returns:
            Normalized DTW distance
        """
        T1, T2 = len(seq1), len(seq2)
        if T1 == 0 or T2 == 0:
            return float("inf")

        # Cost matrix
        cost = np.full((T1 + 1, T2 + 1), np.inf)
        cost[0, 0] = 0.0

        for i in range(1, T1 + 1):
            j_start = max(1, i - self.band_width) if self.band_width > 0 else 1
            j_end = min(T2, i + self.band_width) if self.band_width > 0 else T2

            for j in range(j_start, j_end + 1):
                d = float(np.sum((seq1[i - 1] - seq2[j - 1]) ** 2))
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        # Backtrack to find path length
        path_length = 0
        i, j = T1, T2
        while i > 0 or j > 0:
            path_length += 1
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]]
                argmin = np.argmin(candidates)
                if argmin == 0:
                    i -= 1; j -= 1
                elif argmin == 1:
                    i -= 1
                else:
                    j -= 1

        # Normalize by path length
        return cost[T1, T2] / max(path_length, 1)

    def distance_matrix(
        self,
        templates: Dict[str, np.ndarray],
        query: np.ndarray,
    ) -> List[DTWMatch]:
        """
        Compute DTW distance from query to all templates.

        Args:
            templates: Dict of label -> landmark sequence
            query: Input landmark sequence

        Returns:
            Sorted list of DTWMatch (best first)
        """
        matches = []
        for label, template in templates.items():
            dist = self.distance(query, template)
            matches.append(DTWMatch(
                label=label,
                distance=dist,
                confidence=0.0,  # filled below
                path_length=0,
            ))

        # Convert distances to confidence scores
        if matches:
            distances = np.array([m.distance for m in matches])
            min_d, max_d = distances.min(), distances.max()
            range_d = max_d - min_d if max_d > min_d else 1.0
            for m in matches:
                m.confidence = 1.0 - (m.distance - min_d) / range_d

        matches.sort(key=lambda m: m.distance)
        return matches


class DTWDictionary:
    """
    Template-based sign language dictionary using DTW.

    Stores landmark sequences as templates and recognizes incoming
    gestures by comparing against all stored templates.
    """

    DEFAULT_TEMPLATE_DIR = "models/dtw_templates"

    def __init__(self, template_dir: Optional[str] = None, band_width: int = 10):
        """
        Args:
            template_dir: Directory to store/load templates
            band_width: Sakoe-Chiba band width for DTW
        """
        self.template_dir = template_dir or self.DEFAULT_TEMPLATE_DIR
        self.engine = DTWEngine(band_width=band_width)
        self.templates: Dict[str, List[np.ndarray]] = {}
        self._load_templates()

    def add_template(
        self,
        label: str,
        landmarks: np.ndarray,
        overwrite: bool = False,
    ):
        """
        Add a template for a sign.

        Args:
            label: Sign label (e.g. "hello")
            landmarks: (T, D) landmark sequence
            overwrite: If True, replace existing templates for this label
        """
        if overwrite or label not in self.templates:
            self.templates[label] = []
        self.templates[label].append(np.array(landmarks, dtype=np.float32))
        self._save_templates()

    def remove_template(self, label: str):
        """Remove all templates for a label."""
        self.templates.pop(label, None)
        self._save_templates()

    def list_labels(self) -> List[str]:
        """List all stored sign labels."""
        return list(self.templates.keys())

    def get_template_count(self) -> Dict[str, int]:
        """Get number of templates per label."""
        return {k: len(v) for k, v in self.templates.items()}

    def recognize(
        self,
        landmarks: np.ndarray,
        top_k: int = 3,
        threshold: float = 0.5,
    ) -> List[DTWMatch]:
        """
        Recognize a gesture by matching against all templates.

        Args:
            landmarks: (T, D) input landmark sequence
            top_k: Number of top matches to return
            threshold: Minimum confidence to include in results

        Returns:
            List of DTWMatch, best first
        """
        if not self.templates:
            return []

        # Average templates per label for faster matching
        avg_templates = {}
        for label, tmpls in self.templates.items():
            avg_templates[label] = np.mean(tmpls, axis=0)

        matches = self.engine.distance_matrix(avg_templates, np.array(landmarks))
        return [m for m in matches[:top_k] if m.confidence >= threshold]

    def recognize_realtime(
        self,
        landmarks: np.ndarray,
        min_frames: int = 10,
        threshold: float = 0.6,
    ) -> Optional[DTWMatch]:
        """
        Lightweight real-time recognition (single best match).

        Args:
            landmarks: (T, D) input, can be partial
            min_frames: Minimum frames before attempting recognition
            threshold: Minimum confidence

        Returns:
            Best DTWMatch or None
        """
        if len(landmarks) < min_frames or not self.templates:
            return None

        matches = self.recognize(landmarks, top_k=1, threshold=threshold)
        return matches[0] if matches else None

    def _save_templates(self):
        """Save templates to disk."""
        os.makedirs(self.template_dir, exist_ok=True)
        save_data = {}
        for label, tmpls in self.templates.items():
            save_data[label] = [t.tolist() for t in tmpls]

        path = os.path.join(self.template_dir, "templates.json")
        with open(path, "w") as f:
            json.dump(save_data, f)
        logger.debug("Saved %d templates to %s", len(save_data), path)

    def _load_templates(self):
        """Load templates from disk."""
        path = os.path.join(self.template_dir, "templates.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            for label, tmpls in data.items():
                self.templates[label] = [np.array(t, dtype=np.float32) for t in tmpls]
            logger.info("Loaded %d sign templates from %s", len(data), path)
        except Exception as e:
            logger.warning("Failed to load templates: %s", e)

    def export_templates(self, filepath: str):
        """Export all templates to a JSON file."""
        data = {}
        for label, tmpls in self.templates.items():
            data[label] = [t.tolist() for t in tmpls]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_templates(self, filepath: str):
        """Import templates from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        for label, tmpls in data.items():
            if label not in self.templates:
                self.templates[label] = []
            for t in tmpls:
                self.templates[label].append(np.array(t, dtype=np.float32))
        self._save_templates()
