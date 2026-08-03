# Design Document: Hand Motion Interpretation Pipeline

## Overview

This document describes the architectural design and technical decisions behind the Hand Motion Interpretation Pipeline - a real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.

## Table of Contents

- [Architecture Principles](#architecture-principles)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)
- [Testing Strategy](#testing-strategy)
- [Future Extensibility](#future-extensibility)

---

## Architecture Principles

### 1. Motion as Structured Data

The core insight driving this architecture: **hand gestures should be treated as structured linguistic data**, not just visual patterns. This enables:

- Multiple downstream applications from the same capture
- Machine learning training data generation
- Animation system integration
- Real-time translation pipelines

### 2. Pipeline Architecture

```
Capture → Structure → Multiple Outputs
```

A single motion capture feeds into multiple use cases:
- Real-time cursor control
- JSON export for ML training
- Analysis and visualization
- Future: Animation, translation

### 3. Separation of Concerns

Each module has a single, well-defined responsibility:
- **Detection**: Raw landmark extraction
- **Descriptor**: Structured representation
- **Analyzer**: Offline analysis
- **Apps**: User-facing applications

### 4. Offline-First for Analysis

Analysis tools work on saved JSON files, enabling:
- Batch processing
- Reproducible analysis
- Dataset validation
- Comparison tools

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  cursor.py │ cursor_enhanced.py │ record.py │ batch_record.py│
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  ABSTRACTION LAYER                           │
│            MotionDescriptor (Core Module)                    │
│  • Handshape classification    • Velocity tracking          │
│  • Primitive detection         • Coordinate normalization   │
│  • Feature extraction          • History management         │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  CAPTURE LAYER                               │
│            HandDetector (MediaPipe/cvzone)                   │
│  • 21-point tracking           • Confidence filtering       │
│  • Multi-hand support          • Real-time processing       │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
hand_motion/
├── __init__.py          # Package exports, lazy loading
├── detection.py         # MediaPipe wrapper
│   └── depends on: cvzone, mediapipe, opencv
├── descriptor.py        # Core abstraction
│   └── depends on: stdlib only (math, time, json)
├── analyzer.py          # Offline analysis
│   └── depends on: matplotlib, numpy, descriptor
└── apps/
    ├── cursor.py        # Basic cursor control
    ├── cursor_enhanced.py # Enhanced UI
    ├── record.py        # Single recording
    ├── batch_record.py  # Dataset creation
    └── analyze_dataset.py # Dataset analysis
```

---

## Core Components

### 1. HandDetector (`detection.py`)

**Purpose**: Extract 21-point hand landmarks from video frames.

**Design Decisions**:
- Wraps cvzone's HandDetector (MediaPipe-based) for compatibility
- Provides typed interface with proper error handling
- Supports multi-hand detection with configurable confidence

**Key Types**:
```python
LandmarkList = List[List[float]]  # [[id, x, y], ...]
BoundingBox = List[int]           # [xmin, ymin, xmax, ymax]
```

**Performance Notes**:
- MediaPipe runs at ~30 FPS on standard hardware
- GPU acceleration available when CUDA is configured
- Detection confidence threshold affects accuracy vs. speed tradeoff

### 2. MotionDescriptor (`descriptor.py`)

**Purpose**: Convert raw landmarks into structured, reusable motion data.

**Core Data Structure**:
```python
{
    'timestamp': float,           # Unix timestamp
    'relative_time': float,       # Seconds since recording start
    'frame_num': int,             # Frame index
    'hand': str,                  # 'left' or 'right'
    'fingers_extended': List[int], # [thumb, index, middle, ring, pinky]
    'finger_count': int,          # Number of extended fingers
    'handshape_code': str,        # Binary string like '11000'
    'landmarks': Dict,            # Key landmark positions
    'features': Dict,             # Extracted features
    'primitive': str,             # Classified primitive
    'velocity': Optional[Dict],   # Velocity information
    'normalized': Optional[Dict]  # Normalized coordinates
}
```

**Classification System**:
- Uses lookup table (`PRIMITIVE_MAP`) for common gestures
- Handles special cases (OK_SIGN vs PINCH_READY)
- Falls back to `UNKNOWN_XXXXX` for unrecognized patterns

**Optimizations**:
- `__slots__` for memory efficiency
- Constants for landmark indices
- Reuse of calculated values within frame

### 3. MotionAnalyzer (`analyzer.py`)

**Purpose**: Offline analysis and visualization of recorded motion sequences.

**Capabilities**:
- Statistical summaries
- Trajectory visualization
- Primitive timeline plotting
- Velocity profiling
- Gesture comparison

**Design Decisions**:
- Works on saved JSON files (offline-first)
- Generates publication-quality plots
- Supports batch analysis for datasets

---

## Data Flow

### Real-Time Capture Flow

```
Video Frame
    │
    ▼
HandDetector.findHands()
    │
    ▼
HandDetector.findPosition()
    │
    ▼
HandDetector.fingersUp()
    │
    ▼
MotionDescriptor.create_descriptor()
    │
    ├──► Hand classification (left/right)
    ├──► Primitive classification
    ├──► Feature extraction
    ├──► Velocity calculation
    └──► History management
         │
         ▼
    Structured Descriptor
         │
         ├──► Real-time display
         ├──► Recording buffer
         └──► JSON export (on save)
```

### JSON Export Structure

```json
{
    "metadata": {
        "gesture_name": "wave",
        "recorded_at": "2026-01-15T10:00:00",
        "duration_seconds": 2.5,
        "total_frames": 75,
        "average_fps": 30.0,
        "primitives_used": ["OPEN_HAND", "POINT"],
        "custom": {}
    },
    "frames": [
        {
            "timestamp": 1000000000.0,
            "relative_time": 0.0,
            "frame_num": 0,
            "hand": "right",
            "fingers_extended": [1, 1, 1, 1, 1],
            "finger_count": 5,
            "handshape_code": "11111",
            "landmarks": { ... },
            "features": { ... },
            "primitive": "OPEN_HAND",
            "velocity": null
        }
    ]
}
```

---

## Design Decisions

### 1. Why MediaPipe via cvzone?

**Decision**: Use cvzone's wrapper around MediaPipe.

**Rationale**:
- MediaPipe is Google's production-grade hand tracking
- cvzone simplifies integration and provides Python 3.13+ compatibility
- Abstracts away MediaPipe's complex initialization
- Maintains access to raw landmarks when needed

### 2. Why Structured Descriptors?

**Decision**: Create intermediate structured representation.

**Rationale**:
- Decouples capture from consumption
- Enables multiple output formats
- Facilitates ML training data generation
- Makes analysis tools reusable

### 3. Why JSON for Export?

**Decision**: Use JSON for motion data storage.

**Rationale**:
- Human-readable and debuggable
- Wide library support (Python, JS, etc.)
- Schema flexibility for extensions
- Easy to version control

### 4. Why Primitive Classification?

**Decision**: Classify hand configurations into discrete primitives.

**Rationale**:
- Simplifies temporal pattern recognition
- Enables linguistic annotation
- Reduces data dimensionality
- Maps to sign language parameters

---

## Performance Considerations

### Memory Management

- **History pruning**: Configurable `max_history` prevents unbounded growth
- **`__slots__`**: Reduces memory footprint per descriptor instance
- **Lazy loading**: Heavy dependencies (cv2, matplotlib) loaded on demand

### CPU Optimization

- **Lookup tables**: Primitive classification uses O(1) dictionary lookup
- **Cached calculations**: Landmarks extracted once per frame
- **Efficient statistics**: Uses `Counter` instead of manual counting

### GPU Utilization

- MediaPipe supports GPU acceleration when available
- OpenCV operations can use CUDA with proper configuration
- Matplotlib rendering is CPU-only (acceptable for offline analysis)

### Profiling Recommendations

```python
# Profile descriptor creation
python -m cProfile -s cumtime your_script.py

# Memory profiling
python -m memory_profiler your_script.py

# Line-by-line profiling
kernprof -l -v your_script.py
```

---

## Testing Strategy

### Test Pyramid

```
        ┌─────────────┐
        │   E2E Tests │  (10%)
        │  (manual)   │
        ├─────────────┤
        │ Integration │  (30%)
        │   Tests     │
        ├─────────────┤
        │    Unit     │  (60%)
        │   Tests     │
        └─────────────┘
```

### Test Coverage

- **descriptor.py**: 95%+ coverage
- **detection.py**: 85%+ coverage (mocked MediaPipe)
- **analyzer.py**: 80%+ coverage (mocked matplotlib)

### Testing Patterns Used

1. **Parametrized tests** for primitive classification
2. **Mocking** for external dependencies (cv2, matplotlib)
3. **Fixtures** for common test data
4. **Edge case testing** for error conditions
5. **Integration testing** for module interactions

---

## Future Extensibility

### Planned Extensions

1. **MediaPipe Pose Integration**
   - Body position tracking
   - Two-handed coordination
   - Full upper body capture

2. **MediaPipe Face Mesh**
   - Facial expressions
   - Non-manual markers
   - Lip reading foundation

3. **3D Animation Output**
   - Blender integration
   - Three.js visualization
   - Motion retargeting

4. **ML Translation Pipeline**
   - Sequence-to-sequence models
   - Gloss annotation
   - Real-time translation

### Extension Points

- **New primitives**: Add to `PRIMITIVE_MAP`
- **New features**: Extend `_calculate_*` methods
- **New outputs**: Create new app modules
- **New analysis**: Extend `MotionAnalyzer` class

### Backward Compatibility

- `handDetector` alias maintained for existing code
- JSON format versioning via metadata
- Deprecation warnings for breaking changes

---

## Conclusion

The Hand Motion Interpretation Pipeline is designed as foundational infrastructure for sign language technology. Its modular architecture, structured data approach, and extensibility make it suitable for:

- Research and experimentation
- Dataset creation and validation
- Real-time accessibility applications
- Animation and visualization systems

The key insight - treating motion as structured, reusable data - enables the same capture system to serve multiple downstream applications, from cursor control to sign language translation.

---

*Document Version: 1.0*
*Last Updated: August 2026*
