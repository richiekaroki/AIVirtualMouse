"""Tests for 3D animation export module."""
import pytest
import json
import tempfile
import os
import numpy as np


class TestAnimationExporter:
    """Test AnimationExporter class."""

    def test_init(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()
        assert hasattr(exporter, 'exporters')
        assert 'blender' in exporter.exporters
        assert 'threejs' in exporter.exporters
        assert 'bvh' in exporter.exporters

    def test_export_invalid_format(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export([], format='invalid')

    def test_export_blender(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()

        frames = [
            {
                'timestamp_ms': 0,
                'hand_landmarks': {
                    'landmarks': [{'x': 0.1, 'y': 0.2, 'z': 0.3}]
                }
            },
            {
                'timestamp_ms': 33.33,
                'hand_landmarks': {
                    'landmarks': [{'x': 0.4, 'y': 0.5, 'z': 0.6}]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test')
            result = exporter.export(frames, format='blender', output_path=output_path)
            assert result.endswith('.py')
            assert os.path.exists(result)

    def test_export_threejs(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()

        frames = [
            {
                'timestamp_ms': 0,
                'hand_landmarks': {
                    'landmarks': [{'x': 0.1, 'y': 0.2, 'z': 0.3}]
                }
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test')
            result = exporter.export(frames, format='threejs', output_path=output_path)
            assert result.endswith('.json')
            assert os.path.exists(result)

            # Verify JSON structure
            with open(result) as f:
                data = json.load(f)
            assert 'name' in data
            assert 'tracks' in data

    def test_export_bvh(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()

        frames = [
            {
                'timestamp_ms': 0,
                'hand_landmarks': {
                    'landmarks': [{'x': 0.1, 'y': 0.2, 'z': 0.3}] * 21
                }
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test')
            result = exporter.export(frames, format='bvh', output_path=output_path)
            assert result.endswith('.bvh')
            assert os.path.exists(result)

    def test_export_empty_frames(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test')
            result = exporter.export([], format='blender', output_path=output_path)
            assert os.path.exists(result)


class TestThreeJSViewer:
    """Test Three.js viewer creation."""

    def test_create_viewer(self):
        from hand_motion.animation import AnimationExporter
        exporter = AnimationExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            anim_path = os.path.join(tmpdir, 'animation.json')
            viewer_path = os.path.join(tmpdir, 'viewer.html')

            # Create dummy animation file
            with open(anim_path, 'w') as f:
                json.dump({'name': 'test', 'tracks': []}, f)

            result = exporter.create_threejs_viewer(anim_path, viewer_path)
            assert result == viewer_path
            assert os.path.exists(viewer_path)

            # Verify HTML content
            with open(viewer_path) as f:
                content = f.read()
            assert 'Three.js' in content or 'three.js' in content
