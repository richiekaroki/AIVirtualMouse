"""Tests for MotionAnalyzer module."""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from hand_motion.analyzer import MotionAnalyzer, GestureComparator


def create_test_json(tmp_path, gesture_name="test_gesture", frames=5):
    """Helper to create test JSON file."""
    data = {
        "metadata": {
            "gesture_name": gesture_name,
            "recorded_at": "2026-01-15T10:00:00",
            "duration_seconds": 2.5,
            "total_frames": frames,
            "average_fps": 30.0,
            "primitives_used": ["POINT", "OPEN_HAND"]
        },
        "frames": [
            {
                "timestamp": 1000000000 + i * 0.033,
                "relative_time": i * 0.033,
                "frame_num": i,
                "hand": "right",
                "fingers_extended": [0, 1, 0, 0, 0],
                "finger_count": 1,
                "handshape_code": "01000",
                "landmarks": {
                    "wrist": {"x": 100 + i, "y": 200 + i},
                    "thumb_tip": {"x": 110 + i, "y": 210 + i},
                    "index_tip": {"x": 120 + i, "y": 220 + i},
                    "middle_tip": {"x": 130 + i, "y": 230 + i},
                    "ring_tip": {"x": 140 + i, "y": 240 + i},
                    "pinky_tip": {"x": 150 + i, "y": 250 + i}
                },
                "features": {
                    "pinch_distance": 50.0,
                    "hand_openness": 0.2,
                    "hand_span": 100.0,
                    "palm_center": {"x": 125 + i, "y": 225 + i}
                },
                "primitive": "POINT" if i % 2 == 0 else "OPEN_HAND",
                "velocity": {
                    "vx": 10.0,
                    "vy": 5.0,
                    "magnitude": 11.18,
                    "direction": 0.46
                } if i > 0 else None
            }
            for i in range(frames)
        ]
    }
    
    filepath = tmp_path / f"{gesture_name}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return filepath, data


class TestMotionAnalyzerInit:
    def test_load_valid_file(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        assert analyzer.gesture_name == "test_gesture"
        assert len(analyzer.frames) == 5

    def test_invalid_file(self, tmp_path):
        filepath = tmp_path / "nonexistent.json"
        with pytest.raises(ValueError):
            MotionAnalyzer(str(filepath))

    def test_invalid_json(self, tmp_path):
        filepath = tmp_path / "invalid.json"
        with open(filepath, 'w') as f:
            f.write("not valid json")
        with pytest.raises(ValueError):
            MotionAnalyzer(str(filepath))

    def test_missing_metadata(self, tmp_path):
        filepath = tmp_path / "no_metadata.json"
        with open(filepath, 'w') as f:
            json.dump({"frames": []}, f)
        with pytest.raises(ValueError):
            MotionAnalyzer(str(filepath))

    def test_missing_frames(self, tmp_path):
        filepath = tmp_path / "no_frames.json"
        with open(filepath, 'w') as f:
            json.dump({"metadata": {"gesture_name": "test"}}, f)
        with pytest.raises(ValueError):
            MotionAnalyzer(str(filepath))


class TestPrintSummary:
    def test_print_summary(self, tmp_path, capsys):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        analyzer.print_summary()
        captured = capsys.readouterr()
        assert "Motion Analysis" in captured.out
        assert "test_gesture" in captured.out


class TestPlotTrajectory:
    def test_plot_trajectory(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_trajectory(save_path=str(tmp_path / "test.png"))
        
        assert (tmp_path / "test.png").exists()

    def test_plot_palm_center(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_trajectory(landmark='palm_center')
        
        assert True  # No exception raised


class TestPlotPrimitivesTimeline:
    def test_plot_primitives(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_primitives_timeline(save_path=str(tmp_path / "primitives.png"))
        
        assert (tmp_path / "primitives.png").exists()


class TestPlotVelocityProfile:
    def test_plot_velocity(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_velocity_profile(save_path=str(tmp_path / "velocity.png"))
        
        assert (tmp_path / "velocity.png").exists()


class TestPlotHandOpenness:
    def test_plot_openness(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_hand_openness(save_path=str(tmp_path / "openness.png"))
        
        assert (tmp_path / "openness.png").exists()


class TestPlotPrimitiveDistribution:
    def test_plot_distribution(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        
        with patch('matplotlib.pyplot.show'):
            analyzer.plot_primitive_distribution(save_path=str(tmp_path / "distribution.png"))
        
        assert (tmp_path / "distribution.png").exists()


class TestGenerateAllPlots:
    def test_generate_all(self, tmp_path):
        filepath, _ = create_test_json(tmp_path)
        analyzer = MotionAnalyzer(str(filepath))
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        with patch('matplotlib.pyplot.show'):
            analyzer.generate_all_plots(output_dir=str(output_dir))
        
        assert len(list(output_dir.glob("*.png"))) == 5


class TestGestureComparator:
    def test_compare_statistics(self, tmp_path, capsys):
        filepath1, _ = create_test_json(tmp_path, "gesture1")
        filepath2, _ = create_test_json(tmp_path, "gesture2")
        
        comparator = GestureComparator(str(filepath1), str(filepath2))
        comparator.compare_statistics()
        
        captured = capsys.readouterr()
        assert "Gesture Comparison" in captured.out

    def test_compare_trajectories(self, tmp_path):
        filepath1, _ = create_test_json(tmp_path, "gesture1")
        filepath2, _ = create_test_json(tmp_path, "gesture2")
        
        comparator = GestureComparator(str(filepath1), str(filepath2))
        
        with patch('matplotlib.pyplot.show'):
            comparator.compare_trajectories(save_path=str(tmp_path / "comparison.png"))
        
        assert (tmp_path / "comparison.png").exists()
