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

- **Detection**: Raw landmark extraction + handedness + multi-hand
- **Descriptor**: Structured representation + classification
- **AI**: ML-based gesture classification + continuous recognition + DTW + fingerspelling
- **Web**: Browser interface + real-time streaming + face mesh
- **Analyzer**: Offline analysis

### 4. Dual Classification

- **Rule-based**: Instant lookup from 32 finger combinations + motion patterns
- **ML-based**: RandomForest trained on recorded gestures (64% accuracy)
- Both run simultaneously, results shown side-by-side

### 5. AI Feature Stack

12 integrated AI features form the recognition pipeline:

| # | Feature | Module | Purpose |
|---|---------|--------|---------|
| 1 | Handedness labeling | `detection.py` | Left/right hand identification |
| 2 | Two-hand detection | `detection.py` | Multi-hand landmark extraction |
| 3 | Temporal smoothing | `inference_engine.py` | EMA confidence smoothing |
| 4 | Confidence calibration | `landmark_classifier.py` | Platt scaling on RF outputs |
| 5 | Data augmentation | `augmentation.py` | Rotation, scale, noise, time warp |
| 6 | Non-manual markers | `face.py` | Facial expressions for sign language |
| 7 | Continuous CSLR | `continuous_recognizer.py` | CTC/attention gloss decoding |
| 8 | DTW dictionary | `dtw_dictionary.py` | Template matching recognition |
| 9 | Fingerspelling | `fingerspelling.py` | A-Z letter detection |
| 10 | HID output | `hid_output.py` | Gesture-to-keyboard/mouse |
| 11 | Gloss-to-text NLP | `translator.py` | Seq2Seq attention translation |
| 12 | Transformer CSLR | `transformer_cslr.py` | ViT-based recognition |

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
|  + FaceDetector (non-manual markers)     |
+------------------+-----------------------+
                   |
       +-----------+-----------+-----------+
       |           |           |           |
+------v---+ +-----v----+ +---v--------+ +v----------+
| Rule     | | ML       | | Face       | | Handedness|
| 32 combos| | Forest+  | | EAR/MAR/   | | Left/Right|
| + motion | | Platt    | | Brows/Head | | + 2 hands |
+------+---+ +-----+----+ +---+--------+ +---+------+
       +-----------+-----------+-----------+
                   |
+------------------v-----------------------+
|         Browser Canvas + UI              |
|  Skeleton + Gesture + Face + Library     |
+------------------------------------------+
```

---

## Core Components

### 1. HandDetector (detection.py)

- Wraps cvzone's HandDetector (MediaPipe-based)
- `staticMode=True` by default — prevents timestamp crashes
- Thread-safe via `_mediapipe_lock` in web server
- Suppresses C++ stderr warnings via `_Suppress_stderr`
- **Multi-hand**: `getHandsCount()`, `findPosition(handNo=N)`
- **Handedness**: `getHandedness()` using wrist-vs-MCP heuristic
- **Finger detection**: `fingersUp(handNo=N)` for any hand

### 2. MotionDescriptor (descriptor.py)

- `PRIMITIVE_MAP`: 32 finger combinations (all 2^5)
- Motion detection: Circle, Wave, Swipe (position history)
- Special cases: OK_SIGN vs PINCH, THUMBS_UP angle validation

### 3. LandmarkClassifier (ai/landmark_classifier.py)

- RandomForest (100 trees, max_depth=10)
- 78 features extracted from 21 landmarks
- Platt scaling for calibrated confidence scores
- Data augmentation: rotation, scale, noise, time warp, mirror
- Trained on 15 gestures from motion_data/

### 4. FaceDetector (face.py)

- 468-point MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) — blink detection
- Mouth Aspect Ratio (MAR) — mouth open/close
- Eyebrow height — questioning expressions
- Head orientation — yaw, pitch, roll estimation
- Facial expression classification for non-manual markers

### 5. Web Server (web/app.py)

- `process_frame`: Receives base64 JPEG, runs MediaPipe + Face, returns results
- `play_recording`: Streams recorded gestures frame-by-frame
- `_mediapipe_lock`: Serializes all MediaPipe calls
- `_camera_streams`: Per-connection state tracking
- Security: CORS restricted, security headers, non-root Docker

### 6. AI Modules

- **ContinuousRecognizer** (`continuous_recognizer.py`): CTC + attention decoding for variable-length gloss sequences
- **DTWDictionary** (`dtw_dictionary.py`): Template-based sign matching with Sakoe-Chiba band constraint
- **FingerspellingDetector** (`fingerspelling.py`): ASL A-Z letter recognition from handshape geometry
- **HIDController** (`hid_output.py`): Gesture-to-keyboard/mouse mapping via pynput/pyautogui
- **NeuralGlossTranslator** (`translator.py`): Seq2Seq attention model for gloss-to-text
- **TransformerCSLR** (`transformer_cslr.py`): ViT architecture replacing CNN+LSTM

---

## Data Flow

### Real-Time Capture (Browser)

```
Camera -> base64 JPEG -> WebSocket -> Server
  -> cv2.imdecode -> cv2.flip
  -> FaceDetector -> facial expression markers
  -> MediaPipe Hands -> N hands x 21 landmarks
  -> Handedness: Left/Right per hand
  -> Rule: PRIMITIVE_MAP -> gesture
  -> ML: LandmarkClassifier (Platt-calibrated) -> ml_gesture
  -> WebSocket -> Browser canvas + UI + Face chips
```

### Continuous Recognition Flow

```
Landmark sequence -> FrameBuffer -> TransformerCSLR/ContinuousRecognizer
  -> CTC/Attention decoder -> gloss tokens
  -> NeuralGlossTranslator -> natural language text
```

### Template Matching Flow

```
Landmark sequence -> DTWDictionary.recognize()
  -> DTW distance to all templates
  -> Best match with confidence score
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

### Completed Features (v0.9.0)

1. **Handedness labeling** — Left/right hand identification via landmark heuristic
2. **Two-hand detection** — Multi-hand support with per-hand rendering
3. **Temporal smoothing** — EMA on confidence values for stable output
4. **Confidence calibration** — Platt scaling on Random Forest probabilities
5. **Data augmentation** — Rotation, scaling, noise, time warping, mirror
6. **Non-manual markers** — Face mesh wired into pipeline (EAR, MAR, eyebrows)
7. **Continuous CSLR** — CTC + attention decoder for variable-length glosses
8. **DTW dictionary** — Template matching with Sakoe-Chiba band
9. **Fingerspelling** — ASL A-Z letter detection from handshape geometry
10. **HID output** — Gesture-to-keyboard/mouse mapping
11. **Gloss-to-text NLP** — Seq2Seq attention translation model
12. **Transformer CSLR** — ViT architecture replacing CNN+LSTM

### Future Extensions

1. **BERT/Transformer NLP** — Pre-trained language model for gloss-to-text
2. **3D Animation** — Blender/Three.js integration, motion retargeting
3. **Federated Learning** — Privacy-preserving multi-user model training
4. **Mobile Deployment** — TFLite/ONNX for on-device inference

---

*Document Version: 3.0*
*Last Updated: August 2026*
