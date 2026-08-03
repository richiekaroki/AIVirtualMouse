"""
Hand Motion Interpretation Pipeline

A real-time motion capture and analysis system designed as foundational
infrastructure for sign language translation and accessibility technology.

Core modules:
    - detection: Hand landmark detection (MediaPipe/cvzone)
    - descriptor: Structured motion representation
    - analyzer: Offline analysis and visualization

Applications:
    - apps.cursor: Gesture-based cursor control
    - apps.cursor_enhanced: Enhanced UI version
    - apps.record: Single gesture recording
    - apps.batch_record: Dataset creation
    - apps.analyze_dataset: Dataset analysis
"""

from hand_motion.descriptor import MotionDescriptor
from hand_motion.descriptor import PRIMITIVE_MAP

__version__ = "0.7.0"
__all__ = ["MotionDescriptor", "handDetector", "MotionAnalyzer", "GestureComparator"]


def __getattr__(name: str):
    """Lazy imports for heavy dependencies (cv2, matplotlib)."""
    if name == "handDetector":
        from hand_motion.detection import handDetector
        return handDetector
    elif name == "HandDetector":
        from hand_motion.detection import HandDetector
        return HandDetector
    elif name == "MotionAnalyzer":
        from hand_motion.analyzer import MotionAnalyzer
        return MotionAnalyzer
    elif name == "GestureComparator":
        from hand_motion.analyzer import GestureComparator
        return GestureComparator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
