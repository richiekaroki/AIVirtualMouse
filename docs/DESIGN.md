# Design Document: Hand Motion Interpretation Pipeline

## Overview

This document describes the architectural design and technical decisions behind the Hand Motion Interpretation Pipeline — a real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.

## Architecture Principles

### 1. Motion as Structured Data

The core insight: **hand gestures should be treated as structured linguistic data**, not just visual patterns.

### 2. Pipeline Architecture

```
Capture -> Structure -> Multiple Outputs
```

A single motion capture feeds into multiple use cases:
- Browser-based real-time gesture recognition
- JSON export for ML training
- Analysis and visualization
- Future: Animation, translation

### 3. Separation of Concerns

- **Detection**: Raw landmark extraction
- **Descriptor**: Structured representation + classification
- **AI**: ML-based gesture classification
- **Web**: Browser interface + real-time streaming
- **Analyzer**: Offline analysis

### 4. Dual Classification

- **Rule-based**: Instant lookup from 32 finger combinations + motion patterns
- **ML-based**: RandomForest trained on recorded gestures (64% accuracy)

---

## System Architecture

```
+------------------------------------------+
|          BROWSER / DESKTOP               |
|  getUserMedia -> base64 JPEG -> WebSocket |
+------------------+-----------------------+
                   |
+------------------v-----------------------+
|         Flask + SocketIO Server          |
|  handle_process_frame -> MediaPipe       |
+------------------+-----------------------+
                   |
       +-----------+-----------+
       |           |           |
+------v---+ +-----v----+ +---v----------+
| Rule     | | ML       | | Motion       |
| 32 combos| | Forest   | | Circle/Wave  |
+------+---+ +-----+----+ +---+----------+
       +-----------+-----------+
                   |
+------------------v-----------------------+
|         Browser Canvas + UI              |
|  Skeleton + Gesture display + Library    |
+------------------------------------------+
```

---

## Core Components

### 1. HandDetector (detection.py)

- Wraps cvzone's HandDetector (MediaPipe-based)
- `staticMode=True` by default — prevents timestamp crashes
- Thread-safe via `_mediapipe_lock` in web server
- Suppresses C++ stderr warnings via `_Suppress_stderr`

### 2. MotionDescriptor (descriptor.py)

- `PRIMITIVE_MAP`: 32 finger combinations (all 2^5)
- Motion detection: Circle, Wave, Swipe (position history)
- Special cases: OK_SIGN vs PINCH, THUMBS_UP angle validation

### 3. LandmarkClassifier (ai/landmark_classifier.py)

- RandomForest (100 trees, max_depth=10)
- 78 features extracted from 21 landmarks
- Trained on 15 gestures from motion_data/
- 64% accuracy on test set

### 4. Web Server (web/app.py)

- `process_frame`: Receives base64 JPEG, runs MediaPipe, returns results
- `play_recording`: Streams recorded gestures frame-by-frame
- `_mediapipe_lock`: Serializes all MediaPipe calls
- `_camera_streams`: Per-connection state tracking

---

## Data Flow

### Real-Time Capture (Browser)

```
Camera -> base64 JPEG -> WebSocket -> Server
  -> cv2.imdecode -> cv2.flip
  -> MediaPipe Hands -> 21 landmarks
  -> Rule: PRIMITIVE_MAP -> gesture
  -> ML: LandmarkClassifier -> ml_gesture
  -> WebSocket -> Browser canvas + UI
```

### Recording Playback

```
Client: emit('play_recording', {filename})
Server: Read JSON, flatten landmarks
  -> Stream frame by frame (30 FPS)
  -> Client: drawSkeleton() + updateGesture()
```

---

## Design Decisions

### Why MediaPipe via cvzone?

- MediaPipe is Google's production-grade hand tracking
- cvzone simplifies integration and provides Python 3.13+ compatibility
- Maintains access to raw landmarks when needed

### Why Dual Classification?

- Rule-based is instant and explainable (0ms latency)
- ML-based improves with more training data
- Showing both lets users compare and build trust

### Why WebSocket (not HTTP polling)?

- Sub-100ms round-trip for real-time feel
- Bidirectional: server can push frames for playback
- Single persistent connection reduces overhead

### Why Flask + SocketIO?

- Simple, well-documented, production-ready
- gevent async mode handles concurrent connections
- Easy to deploy on Render with gunicorn

---

## Performance Considerations

### Thread Safety

- `_mediapipe_lock` prevents non-monotonic timestamp errors
- `_camera_streams` dict tracks per-connection state
- `_Suppress_stderr` prevents C++ warning spam from blocking

### Memory Management

- `max_history` parameter prunes motion history
- `__slots__` on HandDetector reduces memory per instance
- JPEG quality (0.6) balances quality vs bandwidth

### Browser Optimization

- 15 FPS capture (not 30) reduces CPU + bandwidth
- Canvas overlay (not video processing) for skeleton drawing
- requestAnimationFrame not needed — setInterval is sufficient

---

## Testing Strategy

### Test Coverage (142 tests)

- **detection.py**: 8 tests (mocked MediaPipe)
- **descriptor.py**: 46 tests (parametrized primitives)
- **analyzer.py**: 16 tests (mocked matplotlib)
- **batch.py**: 15 tests (mocked cv2)
- **pose.py**: 8 tests (mocked MediaPipe)
- **Other modules**: 49 tests

### Testing Patterns

1. Parametrized tests for 32 primitive combinations
2. Mocking for external dependencies (cv2, MediaPipe, matplotlib)
3. Fixtures for common test data
4. Edge case testing for error conditions

---

## Future Extensibility

### Planned Extensions

1. **MediaPipe Pose** — Body position, two-handed coordination
2. **MediaPipe Face Mesh** — Facial expressions, non-manual markers
3. **3D Animation** — Blender/Three.js integration, motion retargeting
4. **ML Translation** — Sequence-to-sequence models, gloss annotation

### Extension Points

- **New primitives**: Add to `PRIMITIVE_MAP`
- **New features**: Extend `_calculate_*` methods
- **New gestures**: Add recordings to `motion_data/`, retrain classifier
- **New outputs**: Create new app modules or web endpoints

---

*Document Version: 2.0*
*Last Updated: August 2026*
