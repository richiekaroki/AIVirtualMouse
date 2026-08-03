"""Tests for database storage module."""
import pytest
import json
import tempfile
import os
from pathlib import Path


class TestMotionDatabase:
    """Test MotionDatabase class."""

    def test_init_in_memory(self):
        from hand_motion.database import MotionDatabase
        db = MotionDatabase(':memory:')
        assert db.conn is not None
        db.close()

    def test_init_file(self):
        from hand_motion.database import MotionDatabase
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            db = MotionDatabase(db_path)
            assert db.conn is not None
            db.close()
        finally:
            os.unlink(db_path)

    def test_context_manager(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            assert db.conn is not None

    def test_add_recording(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(
                name='test_recording',
                source_file='test.mp4',
                duration_ms=5000.0,
                frame_count=150
            )
            assert rec_id > 0

            recording = db.get_recording(rec_id)
            assert recording is not None
            assert recording['name'] == 'test_recording'

    def test_add_frame(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            frame_id = db.add_frame(
                recording_id=rec_id,
                frame_index=0,
                timestamp_ms=33.33,
                hand_landmarks=[{'x': 0.5, 'y': 0.5}]
            )
            assert frame_id > 0

    def test_add_frames_batch(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            frames = [
                {'frame_index': i, 'timestamp_ms': i * 33.33}
                for i in range(10)
            ]
            count = db.add_frames_batch(rec_id, frames)
            assert count == 10

    def test_add_annotation(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            ann_id = db.add_annotation(
                recording_id=rec_id,
                label='HELLO',
                frame_start=0,
                frame_end=30,
                category='sign'
            )
            assert ann_id > 0

    def test_create_dataset(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            ds_id = db.create_dataset(
                name='test_dataset',
                description='A test dataset'
            )
            assert ds_id > 0

    def test_add_recording_to_dataset(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            ds_id = db.create_dataset(name='test')
            rec_id = db.add_recording(name='test')
            db.add_recording_to_dataset(ds_id, rec_id, split='train')

            stats = db.get_statistics()
            assert stats['recordings'] == 1
            assert stats['datasets'] == 1

    def test_get_frames(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            for i in range(5):
                db.add_frame(rec_id, i, i * 33.33)

            frames = db.get_frames(rec_id)
            assert len(frames) == 5

    def test_get_frames_with_range(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            for i in range(10):
                db.add_frame(rec_id, i, i * 33.33)

            frames = db.get_frames(rec_id, frame_start=2, frame_end=7)
            assert len(frames) == 6

    def test_get_annotations(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            db.add_annotation(rec_id, 'HELLO', category='sign')
            db.add_annotation(rec_id, 'WORLD', category='sign')

            anns = db.get_annotations(recording_id=rec_id)
            assert len(anns) == 2

    def test_search_by_label(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            db.add_annotation(rec_id, 'HELLO')

            results = db.search_by_label('HELLO')
            assert len(results) == 1

    def test_get_statistics(self):
        from hand_motion.database import MotionDatabase
        with MotionDatabase(':memory:') as db:
            stats = db.get_statistics()
            assert 'recordings' in stats
            assert 'frames' in stats
            assert 'annotations' in stats
            assert 'datasets' in stats

    def test_export_recording_frames(self):
        from hand_motion.database import MotionDatabase
        import numpy as np
        with MotionDatabase(':memory:') as db:
            rec_id = db.add_recording(name='test')
            db.add_frame(rec_id, 0, 0, hand_landmarks=[{'x': 0.5}])

            data = db.export_recording_frames(rec_id)
            assert len(data) == 1
