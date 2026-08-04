"""
Data Augmentation Pipeline for Hand Landmarks

Provides geometric and temporal augmentations for training data:
- Rotation (2D around wrist)
- Scaling (uniform)
- Noise injection (Gaussian)
- Time warping (non-linear temporal distortion)
- Mirror (horizontal flip)
- Jitter (per-point perturbation)
"""

import math
import random
from typing import List, Tuple, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)


class LandmarkAugmenter:
    """
    Augmentation pipeline for 21-point hand landmark sequences.

    All transforms preserve the 21-landmark structure and operate
    on (id, x, y) format used throughout the project.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        noise_std: float = 3.0,
        jitter_std: float = 2.0,
        time_warp_strength: float = 0.2,
        mirror_probability: float = 0.5,
        image_width: int = 640,
    ):
        """
        Args:
            rotation_range: Max rotation angle in degrees
            scale_range: (min, max) scaling factor
            noise_std: Std dev of Gaussian noise on all points
            jitter_std: Std dev of per-point jitter
            time_warp_strength: Strength of time warping (0 = off)
            mirror_probability: Probability of applying mirror flip
            image_width: Width of the image frame (for mirror)
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.jitter_std = jitter_std
        self.time_warp_strength = time_warp_strength
        self.mirror_probability = mirror_probability
        self.image_width = image_width

    def augment_sequence(
        self,
        frames: List[List[List[float]]],
        probability: float = 0.8,
    ) -> List[List[List[float]]]:
        """
        Apply random augmentations to a sequence of landmark frames.

        Args:
            frames: List of frames, each frame = [[id, x, y], ...]
            probability: Probability of applying each augmentation

        Returns:
            Augmented sequence (same length)
        """
        if not frames or random.random() > probability:
            return frames

        # Decide which augmentations to apply
        do_rotate = random.random() < probability
        do_scale = random.random() < probability
        do_noise = random.random() < probability
        do_jitter = random.random() < probability
        do_mirror = random.random() < self.mirror_probability

        # Compute per-sequence random params
        angle = random.uniform(-self.rotation_range, self.rotation_range) if do_rotate else 0.0
        scale = random.uniform(*self.scale_range) if do_scale else 1.0

        augmented = []
        for frame in frames:
            new_frame = [list(lm) for lm in frame]  # deep copy

            if do_mirror:
                new_frame = self._mirror(new_frame)

            if do_rotate or do_scale:
                new_frame = self._rotate_and_scale(new_frame, angle, scale)

            if do_noise:
                new_frame = self._add_noise(new_frame)

            if do_jitter:
                new_frame = self._jitter(new_frame)

            augmented.append(new_frame)

        # Time warping operates on the full sequence
        if self.time_warp_strength > 0 and random.random() < probability:
            augmented = self._time_warp(augmented)

        return augmented

    def augment_single(
        self,
        lm_list: List[List[float]],
        probability: float = 0.8,
    ) -> List[List[float]]:
        """
        Augment a single frame of landmarks.

        Args:
            lm_list: [[id, x, y], ...] for 21 points
            probability: Probability of applying augmentation

        Returns:
            Augmented landmarks
        """
        if not lm_list or random.random() > probability:
            return lm_list

        frame = [list(lm) for lm in lm_list]

        if random.random() < self.mirror_probability:
            frame = self._mirror(frame)

        if random.random() < 0.8:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            scale = random.uniform(*self.scale_range)
            frame = self._rotate_and_scale(frame, angle, scale)

        if random.random() < 0.8:
            frame = self._add_noise(frame)

        if random.random() < 0.8:
            frame = self._jitter(frame)

        return frame

    def _mirror(self, frame: List[List[float]]) -> List[List[float]]:
        """Horizontally flip landmarks across image center."""
        return [[lm[0], self.image_width - lm[1], lm[2]] for lm in frame]

    def _rotate_and_scale(
        self,
        frame: List[List[float]],
        angle_deg: float,
        scale: float,
    ) -> List[List[float]]:
        """Rotate around wrist and scale uniformly."""
        if len(frame) < 21:
            return frame

        wrist_x, wrist_y = frame[0][1], frame[0][2]
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad) * scale
        sin_a = math.sin(angle_rad) * scale

        result = []
        for lm in frame:
            dx = lm[1] - wrist_x
            dy = lm[2] - wrist_y
            new_x = wrist_x + dx * cos_a - dy * sin_a
            new_y = wrist_y + dx * sin_a + dy * cos_a
            result.append([lm[0], new_x, new_y])
        return result

    def _add_noise(self, frame: List[List[float]]) -> List[List[float]]:
        """Add Gaussian noise to all points."""
        return [
            [lm[0], lm[1] + random.gauss(0, self.noise_std), lm[2] + random.gauss(0, self.noise_std)]
            for lm in frame
        ]

    def _jitter(self, frame: List[List[float]]) -> List[List[float]]:
        """Apply independent per-point jitter."""
        return [
            [lm[0], lm[1] + random.gauss(0, self.jitter_std), lm[2] + random.gauss(0, self.jitter_std)]
            for lm in frame
        ]

    def _time_warp(self, frames: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Non-linear time warping: randomly stretch/compress sub-segments.

        Uses a piecewise-linear time map to remap frame indices.
        """
        n = len(frames)
        if n < 4:
            return frames

        strength = self.time_warp_strength
        # Generate 3 random breakpoints
        pts = sorted([0] + sorted([random.randint(1, n - 2) for _ in range(3)]) + [n - 1])

        # Build mapping: old_idx -> new_idx (float)
        mapping = {}
        for i in range(len(pts) - 1):
            old_start, old_end = pts[i], pts[i + 1]
            new_start = int(old_start * (1 + random.uniform(-strength, strength)))
            new_start = max(0, min(n - 1, new_start))
            new_end = int(old_end * (1 + random.uniform(-strength, strength)))
            new_end = max(0, min(n - 1, new_end))
            seg_len = old_end - old_start
            if seg_len == 0:
                continue
            for j in range(seg_len):
                t = j / seg_len
                old_idx = old_start + j
                new_idx = new_start + t * (new_end - new_start)
                mapping[old_idx] = new_idx

        if not mapping:
            return frames

        # Sample frames at mapped positions
        result = []
        for i in range(n):
            if i in mapping:
                src_idx = int(round(mapping[i]))
            else:
                src_idx = i
            src_idx = max(0, min(n - 1, src_idx))
            result.append(frames[src_idx])

        return result
