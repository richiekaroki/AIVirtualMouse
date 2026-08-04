"""Tests for HandDetector module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from hand_motion.detection import HandDetector, handDetector


class TestHandDetectorInit:
    def test_default_init(self):
        detector = HandDetector()
        assert detector.mode is True
        assert detector.maxHands == 2
        assert detector.detectionCon == 0.5
        assert detector.trackCon == 0.5

    def test_custom_init(self):
        detector = HandDetector(mode=True, maxHands=1, detectionCon=0.8, trackCon=0.8)
        assert detector.mode is True
        assert detector.maxHands == 1
        assert detector.detectionCon == 0.8
        assert detector.trackCon == 0.8

    def test_tip_ids(self):
        detector = HandDetector()
        assert detector.tipIds == [4, 8, 12, 16, 20]

    def test_backward_compatibility(self):
        assert handDetector is HandDetector


class TestFindHands:
    def test_returns_image(self):
        detector = HandDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.findHands(img, draw=False)
        assert result is not None
        assert result.shape == img.shape

    def test_draw_enabled(self):
        detector = HandDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.findHands(img, draw=True)
        assert result is not None

    def test_none_image(self):
        detector = HandDetector()
        result = detector.findHands(None, draw=False)
        assert result is None

    def test_empty_image(self):
        detector = HandDetector()
        img = np.array([], dtype=np.uint8)
        result = detector.findHands(img, draw=False)
        assert result is not None


class TestFindPosition:
    def test_no_hands(self):
        detector = HandDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lmList, bbox = detector.findPosition(img)
        assert lmList == []
        assert bbox == []

    def test_with_results(self):
        detector = HandDetector()
        detector.results = [{
            "lmList": [(i, i*10, i*20) for i in range(21)],
            "bbox": (100, 100, 200, 200)
        }]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lmList, bbox = detector.findPosition(img)
        assert len(lmList) == 21
        assert len(bbox) == 4

    def test_hand_index(self):
        detector = HandDetector()
        detector.results = [
            {"lmList": [(i, i*10, i*20) for i in range(21)], "bbox": (100, 100, 200, 200)},
            {"lmList": [(i, i*5, i*10) for i in range(21)], "bbox": (50, 50, 100, 100)}
        ]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        lmList, bbox = detector.findPosition(img, handNo=1)
        assert len(lmList) == 21
        assert bbox == [50, 50, 150, 150]


class TestFingersUp:
    def test_no_hands(self):
        detector = HandDetector()
        result = detector.fingersUp()
        assert result == []

    def test_with_hand(self):
        detector = HandDetector()
        detector.results = [{"lmList": [(i, i*10, i*20) for i in range(21)]}]
        fingers = detector.fingersUp()
        assert isinstance(fingers, list)


class TestFindDistance:
    def test_no_landmarks(self):
        detector = HandDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        length, img_out, lineInfo = detector.findDistance(8, 12, img, draw=False)
        assert length == 0.0
        assert lineInfo == [0, 0, 0, 0, 0, 0]

    def test_with_landmarks(self):
        detector = HandDetector()
        detector.lmList = [[i, i*10, i*20] for i in range(21)]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        length, img_out, lineInfo = detector.findDistance(8, 12, img, draw=False)
        assert length >= 0
        assert len(lineInfo) == 6

    def test_draw_enabled(self):
        detector = HandDetector()
        detector.lmList = [[i, i*10, i*20] for i in range(21)]
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        length, img_out, lineInfo = detector.findDistance(8, 12, img, draw=True)
        assert img_out is not None
