"""
Gloss Annotation Tools

Linguistic annotation for sign language:
- Gloss transcription
- Syntactic marking
- Non-manual marker annotation
- Time-aligned annotations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class GlossCategory(Enum):
    """Sign language gloss categories."""
    SIGN = "sign"
    FINGERSPELLING = "fingerspelling"
    NON_MANUAL = "non_manual"
    PAUSE = "pause"
    HOLD = "hold"


class NonManualMarker(Enum):
    """Non-manual marker types."""
    EYEBROW_RAISE = "eyebrow_raise"
    EYEBROW_FURROW = "eyebrow_furrow"
    EYE_WIDEN = "eye_widen"
    EYE_SQUINT = "eye_squint"
    HEAD_NOD = "head_nod"
    HEAD_SHAKE = "head_shake"
    HEAD_TILT = "head_tilt"
    MOUTH_OPEN = "mouth_open"
    MOUTH_CLOSE = "mouth_close"
    TONGUE_OUT = "tongue_out"
    CHEEK_PUFF = "cheek_puff"
    NOSE_WRINKLE = "nose_wrinkle"


@dataclass
class GlossUnit:
    """
    A single gloss unit in an annotation.
    """
    label: str
    category: GlossCategory
    start_ms: float
    end_ms: float
    confidence: float = 1.0
    hand: str = "both"
    dominant_hand: str = "right"
    non_manuals: List[str] = field(default_factory=list)
    notes: str = ""
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['category'] = self.category.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GlossUnit':
        data['category'] = GlossCategory(data['category'])
        return cls(**data)


@dataclass
class GlossAnnotation:
    """
    Complete gloss annotation for a recording.
    """
    recording_id: Optional[int] = None
    recording_name: str = ""
    units: List[GlossUnit] = field(default_factory=list)
    language: str = "ASL"
    annotator: str = ""
    notes: str = ""

    def add_unit(
        self,
        label: str,
        category: GlossCategory,
        start_ms: float,
        end_ms: float,
        **kwargs
    ) -> GlossUnit:
        """Add a gloss unit to the annotation."""
        unit = GlossUnit(
            label=label,
            category=category,
            start_ms=start_ms,
            end_ms=end_ms,
            **kwargs
        )
        self.units.append(unit)
        self.units.sort(key=lambda u: u.start_ms)
        return unit

    def remove_unit(self, index: int):
        """Remove a gloss unit by index."""
        if 0 <= index < len(self.units):
            self.units.pop(index)

    @property
    def duration_ms(self) -> float:
        """Total duration of annotation."""
        if not self.units:
            return 0
        return max(u.end_ms for u in self.units) - min(u.start_ms for u in self.units)

    @property
    def sign_count(self) -> int:
        """Number of signs in annotation."""
        return sum(1 for u in self.units if u.category == GlossCategory.SIGN)

    @property
    def fps(self) -> float:
        """Average signs per second."""
        duration_s = self.duration_ms / 1000
        if duration_s <= 0:
            return 0
        return self.sign_count / duration_s

    def get_signs(self) -> List[GlossUnit]:
        """Get all sign units."""
        return [u for u in self.units if u.category == GlossCategory.SIGN]

    def get_non_manuals(self) -> List[GlossUnit]:
        """Get all non-manual units."""
        return [u for u in self.units if u.category == GlossCategory.NON_MANUAL]

    def get_transcription(self, separator: str = " ") -> str:
        """
        Get gloss transcription as string.

        Args:
            separator: Separator between glosses

        Returns:
            Gloss transcription string
        """
        return separator.join(u.label for u in self.units)

    def to_eaf(self) -> str:
        """
        Export as ELAN Annotation Format (simplified XML).

        Returns:
            EAF XML string
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<ANNOTATION_DOCUMENT>')
        lines.append('  <TIME_ORDER>')

        # Collect all time points
        time_points = set()
        for unit in self.units:
            time_points.add(unit.start_ms)
            time_points.add(unit.end_ms)

        time_points = sorted(time_points)
        time_slots = {}
        for i, t in enumerate(time_points):
            time_slots[t] = f"ts{i+1}"
            lines.append(f'    <TIME_SLOT TIME_SLOT_ID="ts{i+1}" TIME_VALUE="{int(t)}"/>')

        lines.append('  </TIME_ORDER>')
        lines.append('  <ANNOTATION_LIST>')

        for i, unit in enumerate(self.units):
            lines.append(f'    <ALIGNABLE_ANNOTATION ANNOTATION_ID="a{i+1}"')
            lines.append(f'      TIME_SLOT_REF1="{time_slots[unit.start_ms]}"')
            lines.append(f'      TIME_SLOT_REF2="{time_slots[unit.end_ms]}">')
            lines.append(f'      <ANNOTATION_VALUE>{unit.label}</ANNOTATION_VALUE>')
            lines.append(f'    </ALIGNABLE_ANNOTATION>')

        lines.append('  </ANNOTATION_LIST>')
        lines.append('</ANNOTATION_DOCUMENT>')

        return '\n'.join(lines)

    def to_json(self) -> str:
        """Export as JSON."""
        data = {
            'recording_id': self.recording_id,
            'recording_name': self.recording_name,
            'language': self.language,
            'annotator': self.annotator,
            'notes': self.notes,
            'units': [u.to_dict() for u in self.units]
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'GlossAnnotation':
        """Import from JSON."""
        data = json.loads(json_str)
        annotation = cls(
            recording_id=data.get('recording_id'),
            recording_name=data.get('recording_name', ''),
            language=data.get('language', 'ASL'),
            annotator=data.get('annotator', ''),
            notes=data.get('notes', '')
        )
        for unit_data in data.get('units', []):
            annotation.units.append(GlossUnit.from_dict(unit_data))
        return annotation

    def save(self, path: str):
        """Save annotation to file."""
        file_path = Path(path)
        if file_path.suffix == '.json':
            file_path.write_text(self.to_json(), encoding='utf-8')
        elif file_path.suffix == '.eaf':
            file_path.write_text(self.to_eaf(), encoding='utf-8')
        else:
            file_path.write_text(self.to_json(), encoding='utf-8')

    @classmethod
    def load(cls, path: str) -> 'GlossAnnotation':
        """Load annotation from file."""
        file_path = Path(path)
        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix == '.json':
            return cls.from_json(content)
        else:
            return cls.from_json(content)


class GlossAnnotator:
    """
    Tool for creating and editing gloss annotations.
    """

    def __init__(self):
        self.annotations: Dict[int, GlossAnnotation] = {}

    def create_annotation(
        self,
        recording_id: int,
        recording_name: str = "",
        language: str = "ASL"
    ) -> GlossAnnotation:
        """
        Create a new annotation for a recording.

        Args:
            recording_id: Recording ID
            recording_name: Recording name
            language: Sign language code

        Returns:
            New GlossAnnotation
        """
        annotation = GlossAnnotation(
            recording_id=recording_id,
            recording_name=recording_name,
            language=language
        )
        self.annotations[recording_id] = annotation
        return annotation

    def get_annotation(self, recording_id: int) -> Optional[GlossAnnotation]:
        """Get annotation for a recording."""
        return self.annotations.get(recording_id)

    def add_sign(
        self,
        recording_id: int,
        label: str,
        start_ms: float,
        end_ms: float,
        **kwargs
    ) -> Optional[GlossUnit]:
        """
        Add a sign to an annotation.

        Args:
            recording_id: Recording ID
            label: Sign gloss label
            start_ms: Start time in ms
            end_ms: End time in ms

        Returns:
            GlossUnit if added, None if annotation not found
        """
        annotation = self.annotations.get(recording_id)
        if not annotation:
            return None

        return annotation.add_unit(
            label=label,
            category=GlossCategory.SIGN,
            start_ms=start_ms,
            end_ms=end_ms,
            **kwargs
        )

    def add_non_manual(
        self,
        recording_id: int,
        marker: NonManualMarker,
        start_ms: float,
        end_ms: float,
        **kwargs
    ) -> Optional[GlossUnit]:
        """
        Add a non-manual marker to an annotation.

        Args:
            recording_id: Recording ID
            marker: Non-manual marker type
            start_ms: Start time in ms
            end_ms: End time in ms

        Returns:
            GlossUnit if added, None if annotation not found
        """
        annotation = self.annotations.get(recording_id)
        if not annotation:
            return None

        return annotation.add_unit(
            label=marker.value,
            category=GlossCategory.NON_MANUAL,
            start_ms=start_ms,
            end_ms=end_ms,
            **kwargs
        )

    def validate_annotation(self, recording_id: int) -> List[str]:
        """
        Validate an annotation for consistency.

        Args:
            recording_id: Recording ID

        Returns:
            List of validation warnings
        """
        annotation = self.annotations.get(recording_id)
        if not annotation:
            return ["Annotation not found"]

        warnings = []

        # Check for overlapping units
        for i, u1 in enumerate(annotation.units):
            for u2 in annotation.units[i+1:]:
                if u1.start_ms < u2.end_ms and u2.start_ms < u1.end_ms:
                    warnings.append(
                        f"Overlapping units: '{u1.label}' and '{u2.label}'"
                    )

        # Check for very short units
        for unit in annotation.units:
            if unit.duration_ms < 50:
                warnings.append(
                    f"Very short unit: '{u1.label}' ({unit.duration_ms:.1f}ms)"
                )

        # Check for very long units
        for unit in annotation.units:
            if unit.duration_ms > 5000:
                warnings.append(
                    f"Very long unit: '{unit.label}' ({unit.duration_ms:.1f}ms)"
                )

        return warnings

    def export_all(self, output_dir: str):
        """
        Export all annotations to a directory.

        Args:
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for rec_id, annotation in self.annotations.items():
            filename = f"annotation_{rec_id}.json"
            filepath = output_path / filename
            annotation.save(str(filepath))

    def import_from_elan(self, eaf_path: str, recording_id: int) -> Optional[GlossAnnotation]:
        """
        Import from ELAN EAF file (simplified).

        Args:
            eaf_path: Path to EAF file
            recording_id: Recording ID to associate with

        Returns:
            GlossAnnotation if successful
        """
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(eaf_path)
            root = tree.getroot()

            annotation = self.create_annotation(recording_id)

            # Parse time slots
            time_slots = {}
            for ts in root.findall('.//TIME_SLOT'):
                ts_id = ts.get('TIME_SLOT_ID')
                ts_value = int(ts.get('TIME_VALUE', 0))
                time_slots[ts_id] = ts_value

            # Parse annotations
            for align in root.findall('.//ALIGNABLE_ANNOTATION'):
                ref1 = align.get('TIME_SLOT_REF1')
                ref2 = align.get('TIME_SLOT_REF2')
                value = align.find('ANNOTATION_VALUE').text

                start_ms = time_slots.get(ref1, 0)
                end_ms = time_slots.get(ref2, 0)

                annotation.add_unit(
                    label=value,
                    category=GlossCategory.SIGN,
                    start_ms=start_ms,
                    end_ms=end_ms
                )

            return annotation

        except Exception as e:
            logger.error(f"Failed to import EAF file: {e}")
            return None
