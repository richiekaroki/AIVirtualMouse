# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-08-04 - Web Interface & ML Classification

### Added
- **Browser webcam support** via WebSocket (no desktop app required)
- **ML gesture classifier** (RandomForest, 78 features, 15 gestures, 64% accuracy)
- **32 gesture primitives** — all finger combinations mapped (was 8)
- **Motion gesture detection** — circle, wave, and swipe recognition
- **Flask + SocketIO web server** with real-time frame processing
- **Web UI** — responsive split-screen layout with skeleton overlay
- **Recording library** — browse, select, and play back recorded gestures
- **Render deployment** — Docker + gunicorn + gevent-websocket
- **Dual classification** — rule-based + ML shown side-by-side
- **MediaPipe timestamp fix** — `staticMode=True` prevents non-monotonic crashes
- **Thread-safe MediaPipe** — `_mediapipe_lock` serializes all calls
- **stderr suppression** — `_Suppress_stderr` context manager for C++ warnings

### Fixed
- Recordings API path — resolved `DATA_DIR` to project root
- Playback landmark format — nested `[[id,x,y]]` flattened to `[x1,y1]`
- Classifier training path — now resolves to project root correctly

### Changed
- Version bumped to 0.8.0
- Tests expanded from 79 to 142 (all passing)
- UI redesigned for clarity — removed clutter, recordings front-and-center
- `.dockerignore` — removed `motion_data/*.json` exclusion that blocked recordings
- `Dockerfile` — added `models/` directory copy
- `wsgi.py` — uses auto-resolved data directory

## [0.7.0] - 2026-08-01 - Package Reorganization

### Changed
- Reorganized into `src/hand_motion/` package structure
- Fixed blocking `input()` → CLI args
- Replaced `print()` with `logging` across core modules
- Added motion history pruning via `max_history` parameter
- Updated CI workflow to use new import paths
- Fixed path separators for cross-platform compatibility

## [0.6.0] - 2026-01-12 - Quality Dataset Complete

### Added
- Comprehensive gesture dataset (45 recordings, 15 unique gestures)
- `batch_record.py` — Guided recording script with quality checks
- `analyze_dataset.py` — Dataset analysis and ranking tool
- Recording manifest with metadata
- Best attempt identification and visualization

### Dataset
- Static handshapes: point, fist, open_hand, thumbs_up, peace
- Dynamic movements: wave, circle, swipe_right
- Transitions: open_close, point_fist
- Directional: push_forward, pull_back, point_up
- Complex: ok_sign, pinch_release

### Metrics
- Average quality score: 0.87/1.00
- Average FPS: 30.0
- Average duration: 2.5 seconds
- High quality recordings: 85%+

## [0.4.5] - 2026-01-12 - Enhanced UI

### Added
- Split-screen interface (640x480 video + 400px info panel)
- Hand skeleton overlay with color-coded joints
- Real-time primitive sequence timeline (last 10 primitives)
- Color-coded primitive blocks for visual tracking
- Recording progress indicator with timer
- Session statistics display
- Professional color scheme and typography
- Visual recording indicator (blinking red circle)

## [0.1.0] - 2024-12 - Initial Release

### Added
- Real-time hand tracking using MediaPipe
- 21-point hand landmark detection
- Gesture-based cursor control
- Finger state detection (which fingers are extended)
- Distance-based gesture recognition
- Motion smoothing for cursor movement
- Modular HandTrackingModule class
- Click detection via two-finger gesture

### Technical Details
- 30fps real-time processing
- Coordinate normalization and mapping
- Frame reduction for performance optimization
- Smoothing algorithm for natural cursor movement

---

## Project Evolution Summary

**v0.1.0** (Dec 2024) — Initial gesture-based cursor control
**v0.2.0** (Jan 2026) — Conceptual reframe toward sign language
**v0.3.0** (Jan 2026) — Motion descriptor abstraction layer
**v0.4.0** (Jan 2026) — Recording mode and JSON export
**v0.4.5** (Jan 2026) — Enhanced UI with split-screen layout
**v0.5.0** (Jan 2026) — Motion analysis and visualization tools
**v0.6.0** (Jan 2026) — Quality dataset complete
**v0.7.0** (Aug 2026) — Package reorganization
**v0.8.0** (Aug 2026) — Web interface, ML classification, Render deploy

**Next:** v1.0.0 — Animation integration
