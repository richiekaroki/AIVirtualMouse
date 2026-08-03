"""
Hand Motion Interpretation Pipeline

A real-time motion capture and analysis system designed as foundational
infrastructure for sign language translation and accessibility technology.

Core modules:
    - detection: Hand landmark detection (MediaPipe/cvzone)
    - descriptor: Structured motion representation
    - analyzer: Offline analysis and visualization
    - validation: Dataset quality checks
    - export: Multi-format data export
    - ai: Gesture recognition and translation
    - pose: Body tracking (MediaPipe Pose)
    - face: Facial expressions (MediaPipe Face Mesh)
    - gpu: GPU acceleration support
    - database: SQLite storage for datasets
    - gloss: Gloss annotation tools
    - animation: 3D animation export

Applications:
    - apps.cursor: Gesture-based cursor control
    - apps.cursor_enhanced: Enhanced UI version
    - apps.record: Single gesture recording
    - apps.batch_record: Dataset creation
    - apps.analyze_dataset: Dataset analysis
    - web: Browser-based interface
"""

from hand_motion.descriptor import MotionDescriptor
from hand_motion.descriptor import PRIMITIVE_MAP

__version__ = "0.8.0"
__all__ = [
    "MotionDescriptor",
    "handDetector",
    "HandDetector",
    "MotionAnalyzer",
    "GestureComparator",
    "MotionValidator",
    "MotionExporter",
    "GestureRecognizer",
    "InferenceEngine",
    "GestureTranslator",
    "PoseDetector",
    "FaceDetector",
    "GPUManager",
    "MotionDatabase",
    "GlossAnnotator",
    "AnimationExporter"
]


def __getattr__(name: str):
    """Lazy imports for heavy dependencies (cv2, matplotlib, torch)."""
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
    elif name == "MotionValidator":
        from hand_motion.validation import MotionValidator
        return MotionValidator
    elif name == "MotionExporter":
        from hand_motion.export import MotionExporter
        return MotionExporter
    elif name == "GestureRecognizer":
        from hand_motion.ai import GestureRecognizer
        return GestureRecognizer
    elif name == "InferenceEngine":
        from hand_motion.ai import InferenceEngine
        return InferenceEngine
    elif name == "GestureTranslator":
        from hand_motion.ai import GestureTranslator
        return GestureTranslator
    elif name == "PoseDetector":
        from hand_motion.pose import PoseDetector
        return PoseDetector
    elif name == "FaceDetector":
        from hand_motion.face import FaceDetector
        return FaceDetector
    elif name == "GPUManager":
        from hand_motion.gpu import GPUManager
        return GPUManager
    elif name == "MotionDatabase":
        from hand_motion.database import MotionDatabase
        return MotionDatabase
    elif name == "GlossAnnotator":
        from hand_motion.gloss import GlossAnnotator
        return GlossAnnotator
    elif name == "AnimationExporter":
        from hand_motion.animation import AnimationExporter
        return AnimationExporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
