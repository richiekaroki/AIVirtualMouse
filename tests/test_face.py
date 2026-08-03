"""Tests for face detection module."""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestFaceMeshIndices:
    """Test face mesh index constants."""

    def test_eye_indices_exist(self):
        from hand_motion.face import FACE_MESH_INDICES
        assert 'left_eye' in FACE_MESH_INDICES
        assert 'right_eye' in FACE_MESH_INDICES

    def test_mouth_indices_exist(self):
        from hand_motion.face import FACE_MESH_INDICES
        assert 'mouth_outer' in FACE_MESH_INDICES
        assert 'mouth_inner' in FACE_MESH_INDICES

    def test_eyebrow_indices_exist(self):
        from hand_motion.face import FACE_MESH_INDICES
        assert 'left_eyebrow' in FACE_MESH_INDICES
        assert 'right_eyebrow' in FACE_MESH_INDICES


class TestFaceDetector:
    """Test FaceDetector class."""

    def test_init_without_mediapipe(self):
        from hand_motion.face import FaceDetector
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            detector = FaceDetector()
            assert detector.available is False

    def test_detect_without_mediapipe(self):
        from hand_motion.face import FaceDetector
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            detector = FaceDetector()
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = detector.detect(img)
            assert result.shape == img.shape

    def test_get_landmarks_without_mediapipe(self):
        from hand_motion.face import FaceDetector
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            detector = FaceDetector()
            landmarks = detector.get_landmarks()
            assert landmarks == []


class TestFacialExpressionAnalyzer:
    """Test FacialExpressionAnalyzer class."""

    def test_init(self):
        from hand_motion.face import FacialExpressionAnalyzer
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            analyzer = FacialExpressionAnalyzer()
            assert hasattr(analyzer, 'expression_history')
            assert hasattr(analyzer, 'detector')

    def test_analyze_frame_without_mediapipe(self):
        from hand_motion.face import FacialExpressionAnalyzer
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            analyzer = FacialExpressionAnalyzer()
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = analyzer.analyze_frame(img)
            assert 'error' in result

    def test_get_non_manual_markers_empty(self):
        from hand_motion.face import FacialExpressionAnalyzer
        with patch('hand_motion.face.MEDIAPIPE_AVAILABLE', False):
            analyzer = FacialExpressionAnalyzer()
            markers = analyzer.get_non_manual_markers()
            assert markers == {}
