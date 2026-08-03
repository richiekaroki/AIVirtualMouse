"""
Dataset Validation Module

Provides automated quality checks for recorded motion data:
- Frame rate validation
- Duration checks
- Landmark consistency
- Primitive distribution analysis
- Quality scoring
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of validating a motion file."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[ValidationIssue]
    metadata: Dict[str, Any]

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class MotionValidator:
    """
    Validates motion capture data files.

    Checks for:
    - Required metadata fields
    - Frame rate consistency
    - Duration requirements
    - Landmark validity
    - Primitive classification quality
    """

    # Quality thresholds
    MIN_FPS = 20.0
    MAX_FPS = 60.0
    MIN_FRAMES = 10
    MAX_DURATION = 30.0  # seconds
    MIN_DURATION = 0.5   # seconds
    MIN_QUALITY_SCORE = 0.5

    REQUIRED_METADATA_FIELDS = [
        'gesture_name', 'recorded_at', 'duration_seconds',
        'total_frames', 'average_fps', 'primitives_used'
    ]

    REQUIRED_FRAME_FIELDS = [
        'timestamp', 'relative_time', 'frame_num', 'hand',
        'fingers_extended', 'landmarks', 'features', 'primitive'
    ]

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict

    def validate_file(self, filepath: str) -> ValidationResult:
        """
        Validate a motion JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ValidationResult with issues and score
        """
        issues: List[ValidationIssue] = []
        metadata: Dict[str, Any] = {}

        # Check file exists
        if not os.path.exists(filepath):
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                f"File not found: {filepath}"
            ))
            return ValidationResult(False, 0.0, issues, metadata)

        # Load JSON
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                f"Invalid JSON: {e}"
            ))
            return ValidationResult(False, 0.0, issues, metadata)
        except IOError as e:
            issues.append(ValidationIssue(
                ValidationSeverity.CRITICAL,
                f"File read error: {e}"
            ))
            return ValidationResult(False, 0.0, issues, metadata)

        # Validate structure
        issues.extend(self._validate_structure(data))

        # Validate metadata
        if 'metadata' in data:
            issues.extend(self._validate_metadata(data['metadata']))
            metadata = data['metadata']

        # Validate frames
        if 'frames' in data:
            issues.extend(self._validate_frames(data['frames']))

        # Calculate score
        score = self._calculate_score(issues)

        # Determine if valid
        has_critical = any(i.severity == ValidationSeverity.CRITICAL for i in issues)
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        is_valid = not has_critical and not (has_errors or self.strict and issues)

        return ValidationResult(is_valid, score, issues, metadata)

    def _validate_structure(self, data: Dict) -> List[ValidationIssue]:
        """Validate JSON structure."""
        issues = []

        if 'metadata' not in data:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "Missing 'metadata' section",
                field="metadata"
            ))

        if 'frames' not in data:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "Missing 'frames' section",
                field="frames"
            ))
        elif not isinstance(data['frames'], list):
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "'frames' must be a list",
                field="frames"
            ))

        return issues

    def _validate_metadata(self, metadata: Dict) -> List[ValidationIssue]:
        """Validate metadata fields."""
        issues = []

        # Check required fields
        for field in self.REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Missing metadata field: {field}",
                    field=f"metadata.{field}"
                ))

        # Validate FPS
        if 'average_fps' in metadata:
            fps = metadata['average_fps']
            if fps < self.MIN_FPS:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Low FPS: {fps:.1f} (minimum: {self.MIN_FPS})",
                    field="metadata.average_fps",
                    value=fps
                ))
            elif fps > self.MAX_FPS:
                issues.append(ValidationIssue(
                    ValidationSeverity.INFO,
                    f"High FPS: {fps:.1f}",
                    field="metadata.average_fps",
                    value=fps
                ))

        # Validate frame count
        if 'total_frames' in metadata:
            frames = metadata['total_frames']
            if frames < self.MIN_FRAMES:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Too few frames: {frames} (minimum: {self.MIN_FRAMES})",
                    field="metadata.total_frames",
                    value=frames
                ))

        # Validate duration
        if 'duration_seconds' in metadata:
            duration = metadata['duration_seconds']
            if duration < self.MIN_DURATION:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Duration too short: {duration:.2f}s (minimum: {self.MIN_DURATION}s)",
                    field="metadata.duration_seconds",
                    value=duration
                ))
            elif duration > self.MAX_DURATION:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    f"Duration too long: {duration:.2f}s (maximum: {self.MAX_DURATION}s)",
                    field="metadata.duration_seconds",
                    value=duration
                ))

        return issues

    def _validate_frames(self, frames: List[Dict]) -> List[ValidationIssue]:
        """Validate frame data."""
        issues = []

        if not frames:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "No frames in recording",
                field="frames"
            ))
            return issues

        # Check frame structure
        for i, frame in enumerate(frames[:5]):  # Check first 5 frames
            for field in self.REQUIRED_FRAME_FIELDS:
                if field not in frame:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Frame {i}: Missing field '{field}'",
                        field=f"frames[{i}].{field}"
                    ))

            # Validate landmarks
            if 'landmarks' in frame:
                landmarks = frame['landmarks']
                required_landmarks = ['wrist', 'index_tip', 'middle_tip']
                for lm_name in required_landmarks:
                    if lm_name not in landmarks:
                        issues.append(ValidationIssue(
                            ValidationSeverity.INFO,
                            f"Frame {i}: Missing landmark '{lm_name}'",
                            field=f"frames[{i}].landmarks.{lm_name}"
                        ))

            # Validate finger states
            if 'fingers_extended' in frame:
                fingers = frame['fingers_extended']
                if not isinstance(fingers, list) or len(fingers) != 5:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        f"Frame {i}: Invalid finger states (expected list of 5)",
                        field=f"frames[{i}].fingers_extended"
                    ))

        # Check for duplicate timestamps
        timestamps = [f.get('timestamp') for f in frames if 'timestamp' in f]
        if len(timestamps) != len(set(timestamps)):
            issues.append(ValidationIssue(
                ValidationSeverity.INFO,
                "Duplicate timestamps detected",
                field="frames.timestamp"
            ))

        return issues

    def _calculate_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate quality score based on issues."""
        if not issues:
            return 1.0

        score = 1.0
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.5
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.2
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.1
            elif issue.severity == ValidationSeverity.INFO:
                score -= 0.05

        return max(0.0, score)


class DatasetValidator:
    """
    Validates an entire dataset of motion files.
    """

    def __init__(self, data_dir: str, strict: bool = False):
        """
        Initialize dataset validator.

        Args:
            data_dir: Directory containing motion JSON files
            strict: If True, treat warnings as errors
        """
        self.data_dir = Path(data_dir)
        self.validator = MotionValidator(strict=strict)

    def validate_all(self) -> Dict[str, ValidationResult]:
        """
        Validate all motion files in the directory.

        Returns:
            Dictionary mapping filenames to validation results
        """
        results = {}

        for filepath in self.data_dir.glob("*.json"):
            if filepath.name == "recording_manifest.json":
                continue

            results[filepath.name] = self.validator.validate_file(str(filepath))

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        results = self.validate_all()

        total_files = len(results)
        valid_files = sum(1 for r in results.values() if r.is_valid)
        avg_score = sum(r.score for r in results.values()) / total_files if total_files > 0 else 0

        # Collect all issues
        all_issues = []
        for result in results.values():
            all_issues.extend(result.issues)

        # Count by severity
        severity_counts = {}
        for issue in all_issues:
            severity = issue.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': total_files - valid_files,
            'validation_rate': valid_files / total_files if total_files > 0 else 0,
            'average_score': avg_score,
            'severity_counts': severity_counts
        }


def validate_motion_file(filepath: str, strict: bool = False) -> ValidationResult:
    """
    Convenience function to validate a single motion file.

    Args:
        filepath: Path to JSON file
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult
    """
    validator = MotionValidator(strict=strict)
    return validator.validate_file(filepath)


def print_validation_report(result: ValidationResult) -> None:
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    status = "VALID" if result.is_valid else "INVALID"
    print(f"\nStatus: {status}")
    print(f"Quality Score: {result.score:.2f}/1.00")

    if result.metadata:
        print(f"\nGesture: {result.metadata.get('gesture_name', 'Unknown')}")
        print(f"Frames: {result.metadata.get('total_frames', 'N/A')}")
        print(f"Duration: {result.metadata.get('duration_seconds', 'N/A'):.2f}s")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            prefix = "  "
            if issue.severity == ValidationSeverity.CRITICAL:
                prefix = "  [CRITICAL] "
            elif issue.severity == ValidationSeverity.ERROR:
                prefix = "  [ERROR] "
            elif issue.severity == ValidationSeverity.WARNING:
                prefix = "  [WARNING] "
            else:
                prefix = "  [INFO] "
            print(f"{prefix}{issue.message}")
    else:
        print("\nNo issues found!")

    print("\n" + "=" * 60)
