"""Tests for gloss annotation module."""
import pytest
import json
import tempfile
import os


class TestGlossUnit:
    """Test GlossUnit dataclass."""

    def test_create_sign_unit(self):
        from hand_motion.gloss import GlossUnit, GlossCategory
        unit = GlossUnit(
            label='HELLO',
            category=GlossCategory.SIGN,
            start_ms=0,
            end_ms=500
        )
        assert unit.label == 'HELLO'
        assert unit.category == GlossCategory.SIGN
        assert unit.duration_ms == 500

    def test_to_dict(self):
        from hand_motion.gloss import GlossUnit, GlossCategory
        unit = GlossUnit(
            label='HELLO',
            category=GlossCategory.SIGN,
            start_ms=0,
            end_ms=500
        )
        d = unit.to_dict()
        assert d['label'] == 'HELLO'
        assert d['category'] == 'sign'


class TestGlossAnnotation:
    """Test GlossAnnotation class."""

    def test_create_annotation(self):
        from hand_motion.gloss import GlossAnnotation
        ann = GlossAnnotation(recording_name='test')
        assert ann.recording_name == 'test'
        assert len(ann.units) == 0

    def test_add_unit(self):
        from hand_motion.gloss import GlossAnnotation, GlossUnit, GlossCategory
        ann = GlossAnnotation()
        unit = ann.add_unit(
            label='HELLO',
            category=GlossCategory.SIGN,
            start_ms=0,
            end_ms=500
        )
        assert len(ann.units) == 1
        assert unit.label == 'HELLO'

    def test_get_transcription(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation()
        ann.add_unit('HELLO', GlossCategory.SIGN, 0, 500)
        ann.add_unit('WORLD', GlossCategory.SIGN, 500, 1000)
        assert ann.get_transcription() == 'HELLO WORLD'

    def test_get_signs(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation()
        ann.add_unit('HELLO', GlossCategory.SIGN, 0, 500)
        ann.add_unit('blink', GlossCategory.NON_MANUAL, 0, 200)
        signs = ann.get_signs()
        assert len(signs) == 1
        assert signs[0].label == 'HELLO'

    def test_json_export_import(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation(recording_name='test')
        ann.add_unit('HELLO', GlossCategory.SIGN, 0, 500)
        ann.add_unit('WORLD', GlossCategory.SIGN, 500, 1000)

        json_str = ann.to_json()
        loaded = GlossAnnotation.from_json(json_str)

        assert loaded.recording_name == 'test'
        assert len(loaded.units) == 2
        assert loaded.units[0].label == 'HELLO'

    def test_save_load_json(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation(recording_name='test')
        ann.add_unit('HELLO', GlossCategory.SIGN, 0, 500)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            ann.save(path)
            loaded = GlossAnnotation.load(path)
            assert loaded.recording_name == 'test'
            assert len(loaded.units) == 1
        finally:
            os.unlink(path)

    def test_duration(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation()
        ann.add_unit('A', GlossCategory.SIGN, 0, 500)
        ann.add_unit('B', GlossCategory.SIGN, 500, 1500)
        assert ann.duration_ms == 1500

    def test_sign_count(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation()
        ann.add_unit('HELLO', GlossCategory.SIGN, 0, 500)
        ann.add_unit('blink', GlossCategory.NON_MANUAL, 0, 200)
        ann.add_unit('WORLD', GlossCategory.SIGN, 500, 1000)
        assert ann.sign_count == 2

    def test_fps(self):
        from hand_motion.gloss import GlossAnnotation, GlossCategory
        ann = GlossAnnotation()
        ann.add_unit('A', GlossCategory.SIGN, 0, 500)
        ann.add_unit('B', GlossCategory.SIGN, 500, 1000)
        ann.add_unit('C', GlossCategory.SIGN, 1000, 1500)
        # 3 signs in 1.5 seconds = 2 signs per second
        assert ann.fps == pytest.approx(2.0, rel=0.1)


class TestGlossAnnotator:
    """Test GlossAnnotator class."""

    def test_create_annotation(self):
        from hand_motion.gloss import GlossAnnotator
        annotator = GlossAnnotator()
        ann = annotator.create_annotation(1, 'test_video')
        assert ann.recording_id == 1

    def test_add_sign(self):
        from hand_motion.gloss import GlossAnnotator
        annotator = GlossAnnotator()
        annotator.create_annotation(1)
        unit = annotator.add_sign(1, 'HELLO', 0, 500)
        assert unit is not None
        assert unit.label == 'HELLO'

    def test_add_non_manual(self):
        from hand_motion.gloss import GlossAnnotator, NonManualMarker
        annotator = GlossAnnotator()
        annotator.create_annotation(1)
        unit = annotator.add_non_manual(1, NonManualMarker.EYEBROW_RAISE, 0, 300)
        assert unit is not None

    def test_validate_annotation(self):
        from hand_motion.gloss import GlossAnnotator
        annotator = GlossAnnotator()
        annotator.create_annotation(1)
        annotator.add_sign(1, 'HELLO', 0, 500)
        warnings = annotator.validate_annotation(1)
        assert isinstance(warnings, list)

    def test_validate_overlapping(self):
        from hand_motion.gloss import GlossAnnotator
        annotator = GlossAnnotator()
        annotator.create_annotation(1)
        annotator.add_sign(1, 'HELLO', 0, 500)
        annotator.add_sign(1, 'WORLD', 250, 750)
        warnings = annotator.validate_annotation(1)
        assert len(warnings) > 0
        assert any('Overlapping' in w for w in warnings)
