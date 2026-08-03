"""Tests for pose detection module."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestPoseLandmarks:
    """Test pose landmark constants."""

    def test_landmark_count(self):
        from hand_motion.pose import POSE_LANDMARKS
        assert len(POSE_LANDMARKS) == 33

    def test_landmark_names(self):
        from hand_motion.pose import POSE_LANDMARKS
        expected_names = ['nose', 'left_eye', 'right_eye', 'left_shoulder',
                         'right_shoulder', 'left_wrist', 'right_wrist']
        for name in expected_names:
            assert name in POSE_LANDMARKS


class TestPoseDetector:
    """Test PoseDetector class."""

    def test_init_without_mediapipe(self):
        from hand_motion.pose import PoseDetector
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            detector = PoseDetector()
            assert detector.available is False

    def test_detect_without_mediapipe(self):
        from hand_motion.pose import PoseDetector
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            detector = PoseDetector()
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = detector.detect(img)
            assert result.shape == img.shape

    def test_get_landmarks_without_mediapipe(self):
        from hand_motion.pose import PoseDetector
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            detector = PoseDetector()
            landmarks = detector.get_landmarks()
            assert landmarks == []

    def test_get_hand_positions_without_mediapipe(self):
        from hand_motion.pose import PoseDetector
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            detector = PoseDetector()
            hands = detector.get_hand_positions()
            assert hands == {'left': None, 'right': None}


class TestTwoHandedPoseAnalyzer:
    """Test TwoHandedPoseAnalyzer class."""

    def test_init(self):
        from hand_motion.pose import TwoHandedPoseAnalyzer
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            analyzer = TwoHandedPoseAnalyzer()
            assert hasattr(analyzer, 'left_hand_history')
            assert hasattr(analyzer, 'right_hand_history')

    def test_analyze_frame_without_mediapipe(self):
        from hand_motion.pose import TwoHandedPoseAnalyzer
        with patch('hand_motion.pose.MEDIAPIPE_AVAILABLE', False):
            analyzer = TwoHandedPoseAnalyzer()
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(img)
            assert 'error' in result
