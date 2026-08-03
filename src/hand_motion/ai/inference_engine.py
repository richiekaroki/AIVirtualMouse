"""
Real-time Inference Engine

Provides low-latency gesture recognition for live video streams:
- Frame buffering
- Sliding window inference
- Result smoothing
- Performance monitoring
"""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a single inference step."""
    gesture: str
    confidence: float
    timestamp: float
    latency_ms: float
    frame_number: int
    all_probs: Dict[str, float] = field(default_factory=dict)


class FrameBuffer:
    """
    Circular buffer for storing landmark sequences.
    """

    def __init__(self, max_length: int = 30, feature_dim: int = 63):
        """
        Initialize frame buffer.

        Args:
            max_length: Maximum sequence length
            feature_dim: Dimension of feature vector per frame
        """
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=max_length)
        self.frame_count = 0

    def add_frame(self, landmarks: np.ndarray) -> None:
        """
        Add a frame of landmarks to the buffer.

        Args:
            landmarks: Flat array of shape (feature_dim,)
        """
        if landmarks.shape[0] != self.feature_dim:
            logger.warning("Expected feature dim %d, got %d", self.feature_dim, landmarks.shape[0])
            return

        self.buffer.append(landmarks)
        self.frame_count += 1

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence as numpy array.

        Returns:
            Array of shape (sequence_length, feature_dim) or None if empty
        """
        if len(self.buffer) == 0:
            return None

        return np.array(list(self.buffer))

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for inference."""
        return len(self.buffer) >= self.max_length

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.frame_count = 0

    @property
    def length(self) -> int:
        return len(self.buffer)


class PerformanceMonitor:
    """
    Monitor inference performance metrics.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of samples to keep for averaging
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()

    def record_inference(self, latency_ms: float, confidence: float) -> None:
        """Record a single inference result."""
        self.latencies.append(latency_ms)
        self.confidences.append(confidence)
        self.total_frames += 1

        now = time.time()
        if self.frame_times:
            self.frame_times.append(now - self.frame_times[-1] if self.frame_times else 0)
        else:
            self.frame_times.append(0)

    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.latencies:
            return {
                'avg_latency_ms': 0,
                'avg_fps': 0,
                'avg_confidence': 0,
                'total_frames': self.total_frames
            }

        elapsed = time.time() - self.start_time
        return {
            'avg_latency_ms': np.mean(self.latencies),
            'max_latency_ms': np.max(self.latencies),
            'min_latency_ms': np.min(self.latencies),
            'avg_fps': self.total_frames / elapsed if elapsed > 0 else 0,
            'avg_confidence': np.mean(self.confidences),
            'total_frames': self.total_frames,
            'uptime_seconds': elapsed
        }


class InferenceEngine:
    """
    Real-time gesture inference engine.

    Provides low-latency gesture recognition with:
    - Frame buffering
    - Sliding window inference
    - Result smoothing
    - Performance monitoring
    """

    def __init__(
        self,
        classifier: Any,
        sequence_length: int = 30,
        feature_dim: int = 63,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize inference engine.

        Args:
            classifier: GestureClassifier instance
            sequence_length: Number of frames for inference
            feature_dim: Dimension of feature vector
            smoothing_window: Window size for result smoothing
            confidence_threshold: Minimum confidence for prediction
        """
        self.classifier = classifier
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold

        self.frame_buffer = FrameBuffer(sequence_length, feature_dim)
        self.performance = PerformanceMonitor()

        # Result smoothing
        self.recent_results = deque(maxlen=smoothing_window)

        # Callbacks
        self.on_gesture_detected: Optional[Callable] = None

        # State
        self.is_running = False
        self.frame_number = 0

    def process_frame(self, landmarks: np.ndarray) -> Optional[InferenceResult]:
        """
        Process a single frame of landmarks.

        Args:
            landmarks: Flat array of shape (feature_dim,)

        Returns:
            InferenceResult if prediction made, None otherwise
        """
        start_time = time.time()
        self.frame_number += 1

        # Add to buffer
        self.frame_buffer.add_frame(landmarks)

        # Check if ready for inference
        if not self.frame_buffer.is_ready():
            return None

        # Get sequence
        sequence = self.frame_buffer.get_sequence()
        if sequence is None:
            return None

        # Run inference
        try:
            result = self.classifier.classify(sequence)
        except Exception as e:
            logger.error("Inference error: %s", e)
            return None

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Apply confidence threshold
        if result['confidence'] < self.confidence_threshold:
            gesture = 'uncertain'
        else:
            gesture = result['gesture']

        # Create inference result
        inference_result = InferenceResult(
            gesture=gesture,
            confidence=result['confidence'],
            timestamp=time.time(),
            latency_ms=latency_ms,
            frame_number=self.frame_number,
            all_probs=result.get('all_probs', {})
        )

        # Smooth results
        self.recent_results.append(inference_result)
        smoothed = self._smooth_results()

        # Record performance
        self.performance.record_inference(latency_ms, result['confidence'])

        # Trigger callback if gesture detected
        if smoothed and self.on_gesture_detected:
            self.on_gesture_detected(smoothed)

        return smoothed

    def _smooth_results(self) -> Optional[InferenceResult]:
        """Smooth recent results using majority voting."""
        if not self.recent_results:
            return None

        # Get most recent result
        latest = self.recent_results[-1]

        # Count gesture occurrences in window
        gesture_counts = {}
        for result in self.recent_results:
            gesture = result.gesture
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        # Find majority gesture
        if gesture_counts:
            majority_gesture = max(gesture_counts, key=gesture_counts.get)
            majority_count = gesture_counts[majority_gesture]

            # Update result if majority agrees
            if majority_count >= len(self.recent_results) * 0.6:
                latest = InferenceResult(
                    gesture=majority_gesture,
                    confidence=latest.confidence,
                    timestamp=latest.timestamp,
                    latency_ms=latest.latency_ms,
                    frame_number=latest.frame_number,
                    all_probs=latest.all_probs
                )

        return latest

    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return self.performance.get_stats()

    def reset(self) -> None:
        """Reset the inference engine state."""
        self.frame_buffer.clear()
        self.recent_results.clear()
        self.frame_number = 0

    def start(self) -> None:
        """Start the inference engine."""
        self.is_running = True
        self.reset()
        logger.info("Inference engine started")

    def stop(self) -> None:
        """Stop the inference engine."""
        self.is_running = False
        logger.info("Inference engine stopped")

    def get_current_gesture(self) -> Optional[str]:
        """Get the most recent gesture prediction."""
        if self.recent_results:
            return self.recent_results[-1].gesture
        return None

    def get_confidence(self) -> float:
        """Get the confidence of the most recent prediction."""
        if self.recent_results:
            return self.recent_results[-1].confidence
        return 0.0
