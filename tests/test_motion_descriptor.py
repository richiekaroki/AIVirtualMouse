"""Tests for MotionDescriptor module."""

import json
import os
import time
import pytest
from collections import Counter

from hand_motion.descriptor import MotionDescriptor, PRIMITIVE_MAP


def make_landmarks(x_start: int = 100, y_start: int = 200):
    """Helper to create 21 fake hand landmarks."""
    return [[i, x_start + i * 10, y_start + i * 5] for i in range(21)]


class TestMotionDescriptorInit:
    def test_default_max_history(self):
        md = MotionDescriptor()
        assert md.max_history == 1000

    def test_custom_max_history(self):
        md = MotionDescriptor(max_history=500)
        assert md.max_history == 500

    def test_unlimited_history(self):
        md = MotionDescriptor(max_history=0)
        assert md.max_history == 0

    def test_empty_state(self):
        md = MotionDescriptor()
        assert md.motion_history == []
        assert md.primitives_seen == set()
        assert md.recording_start_time is None


class TestCreateDescriptor:
    def test_creates_descriptor(self):
        md = MotionDescriptor()
        lm = make_landmarks()
        fingers = [0, 1, 0, 0, 0]
        d = md.create_descriptor(lm, fingers, frame_shape=(480, 640))
        assert d is not None
        assert d['primitive'] == 'POINT'
        assert d['handshape_code'] == '01000'

    def test_returns_none_on_empty_landmarks(self):
        md = MotionDescriptor()
        d = md.create_descriptor([], [0, 1, 0, 0, 0])
        assert d is None

    def test_returns_none_on_insufficient_landmarks(self):
        md = MotionDescriptor()
        d = md.create_descriptor([[i, 0, 0] for i in range(10)], [0, 1, 0, 0, 0])
        assert d is None

    def test_returns_none_on_invalid_fingers(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [])
        assert d is None

    def test_returns_none_on_wrong_finger_count(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0])
        assert d is None

    def test_adds_to_history(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        assert len(md.motion_history) == 1
        assert d is md.motion_history[0]

    def test_sets_recording_start_time(self):
        md = MotionDescriptor()
        assert md.recording_start_time is None
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        assert md.recording_start_time is not None

    def test_tracks_primitives_seen(self):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        md.create_descriptor(make_landmarks(), [1, 1, 1, 1, 1])
        assert 'POINT' in md.primitives_seen
        assert 'OPEN_HAND' in md.primitives_seen

    def test_descriptor_has_required_keys(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        required = ['timestamp', 'relative_time', 'frame_num', 'hand',
                     'fingers_extended', 'finger_count', 'handshape_code',
                     'landmarks', 'features', 'primitive', 'velocity']
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_normalized_coordinates(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0],
                                 frame_shape=(480, 640))
        assert 'normalized' in d
        for name, coords in d['normalized'].items():
            assert 0 <= coords['x'] <= 1
            assert 0 <= coords['y'] <= 1


class TestPrimitiveClassification:
    @pytest.mark.parametrize("fingers,expected", [
        ([0, 1, 0, 0, 0], "POINT"),
        ([0, 1, 1, 0, 0], "PEACE_V"),
        ([1, 1, 1, 1, 1], "OPEN_HAND"),
        ([0, 0, 0, 0, 0], "FIST"),
        ([1, 0, 0, 0, 0], "THUMBS_UP"),
        ([0, 1, 1, 1, 0], "THREE"),
        ([0, 1, 1, 1, 1], "FOUR"),
        ([0, 0, 0, 0, 1], "PINKY"),
    ])
    def test_classifies_primitives(self, fingers, expected):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), fingers)
        assert d['primitive'] == expected

    def test_unknown_primitive(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [1, 0, 1, 0, 1])
        assert d['primitive'].startswith("UNKNOWN_")

    def test_primitive_map_completeness(self):
        """Test that PRIMITIVE_MAP contains expected primitives."""
        assert len(PRIMITIVE_MAP) >= 8
        assert "POINT" in PRIMITIVE_MAP.values()


class TestHandDetection:
    def test_right_hand(self):
        md = MotionDescriptor()
        lm = make_landmarks(x_start=100)
        lm[0] = [0, 100, 200]
        lm[5] = [5, 200, 250]
        d = md.create_descriptor(lm, [1, 1, 1, 1, 1])
        assert d['hand'] == 'right'

    def test_left_hand(self):
        md = MotionDescriptor()
        lm = make_landmarks(x_start=300)
        lm[0] = [0, 300, 200]
        lm[5] = [5, 200, 250]
        d = md.create_descriptor(lm, [1, 1, 1, 1, 1])
        assert d['hand'] == 'left'


class TestMaxHistoryPruning:
    def test_prunes_old_frames(self):
        md = MotionDescriptor(max_history=5)
        for _ in range(10):
            md.create_descriptor(make_landmarks(), [1, 1, 1, 1, 1])
        assert len(md.motion_history) == 5

    def test_unlimited_no_pruning(self):
        md = MotionDescriptor(max_history=0)
        for _ in range(20):
            md.create_descriptor(make_landmarks(), [1, 1, 1, 1, 1])
        assert len(md.motion_history) == 20


class TestFeatures:
    def test_pinch_distance(self):
        md = MotionDescriptor()
        lm = make_landmarks()
        d = md.create_descriptor(lm, [1, 1, 0, 0, 0])
        assert d['features']['pinch_distance'] >= 0

    def test_hand_openness_fist(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 0, 0, 0, 0])
        assert d['features']['hand_openness'] == 0.0

    def test_hand_openness_open(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [1, 1, 1, 1, 1])
        assert d['features']['hand_openness'] == 1.0

    def test_hand_span(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [1, 1, 1, 1, 1])
        assert d['features']['hand_span'] > 0

    def test_palm_center(self):
        md = MotionDescriptor()
        lm = make_landmarks()
        d = md.create_descriptor(lm, [1, 1, 1, 1, 1])
        palm = d['features']['palm_center']
        assert 'x' in palm and 'y' in palm


class TestVelocity:
    def test_no_velocity_on_first_frame(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        assert d['velocity'] is None

    def test_velocity_on_second_frame(self):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        time.sleep(0.01)
        lm2 = make_landmarks(x_start=110)
        d = md.create_descriptor(lm2, [0, 1, 0, 0, 0])
        assert d['velocity'] is not None
        assert 'magnitude' in d['velocity']
        assert 'direction' in d['velocity']


class TestClearHistory:
    def test_clears_all_state(self):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        md.clear_history()
        assert md.motion_history == []
        assert md.primitives_seen == set()
        assert md.recording_start_time is None


class TestStatistics:
    def test_empty_history(self):
        md = MotionDescriptor()
        stats = md.get_statistics()
        assert 'error' in stats

    def test_stats_with_data(self):
        md = MotionDescriptor()
        for _ in range(5):
            md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
            time.sleep(0.01)
        stats = md.get_statistics()
        assert stats['total_frames'] == 5
        assert stats['duration_seconds'] >= 0
        assert 'POINT' in stats['primitive_counts']

    def test_velocity_stats(self):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        time.sleep(0.01)
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        stats = md.get_statistics()
        assert stats['velocity_stats'] is not None
        assert 'mean' in stats['velocity_stats']
        assert 'max' in stats['velocity_stats']
        assert 'min' in stats['velocity_stats']


class TestSaveSequence:
    def test_saves_json(self, tmp_path):
        md = MotionDescriptor()
        for _ in range(3):
            md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        filepath = str(tmp_path / "test.json")
        result = md.save_sequence(filepath, "test_gesture")
        assert result is True
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data['metadata']['gesture_name'] == 'test_gesture'
        assert len(data['frames']) == 3

    def test_empty_history_no_save(self, tmp_path):
        md = MotionDescriptor()
        filepath = str(tmp_path / "empty.json")
        result = md.save_sequence(filepath, "empty")
        assert result is False
        assert not os.path.exists(filepath)

    def test_saves_with_metadata(self, tmp_path):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        filepath = str(tmp_path / "test.json")
        md.save_sequence(filepath, "test", metadata={'attempt': 1})
        with open(filepath) as f:
            data = json.load(f)
        assert data['metadata']['custom']['attempt'] == 1


class TestMotionSequence:
    def test_empty_sequence(self):
        md = MotionDescriptor()
        seq = md.get_motion_sequence(window_seconds=2.0)
        assert seq == []

    def test_primitives_sequence(self):
        md = MotionDescriptor()
        md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        seq = md.get_primitive_sequence(window_seconds=2.0)
        assert seq == ['POINT']


class TestEdgeCases:
    def test_invalid_frame_shape(self):
        md = MotionDescriptor()
        d = md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0],
                                 frame_shape=(-1, -1))
        assert d is not None
        assert 'normalized' in d
        assert d['normalized'] == {}

    def test_concurrent_access(self):
        """Test that multiple descriptors can be created quickly."""
        md = MotionDescriptor()
        for _ in range(100):
            md.create_descriptor(make_landmarks(), [0, 1, 0, 0, 0])
        assert len(md.motion_history) == 100

    def test_large_landmark_list(self):
        """Test with more landmarks than expected."""
        md = MotionDescriptor()
        lm = make_landmarks() + [[21, 100, 200] for _ in range(10)]
        d = md.create_descriptor(lm, [0, 1, 0, 0, 0])
        assert d is not None

    def test_slots_optimization(self):
        """Test that __slots__ is properly defined."""
        md = MotionDescriptor()
        assert hasattr(md, '__slots__')
