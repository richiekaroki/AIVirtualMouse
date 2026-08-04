# Hand Motion Interpretation Pipeline

### From Gesture Recognition to Sign Language Infrastructure

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)]()
[![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple.svg)]()

> **A real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.**

Originally built for gesture-based cursor control, this project evolved into a full motion interpretation pipeline that treats hand gestures as **structured linguistic data**, making it suitable for sign language research, animation systems, and accessibility applications.

## What's New in v0.8.0

- **Browser Webcam Support**: Real-time gesture recognition via WebSocket — no desktop app required
- **ML Gesture Classifier**: RandomForest model trained on 15 gestures (78 features, 64% accuracy)
- **32 Gesture Primitives**: All finger combinations mapped (was 8)
- **Motion Gesture Detection**: Circle, wave, and swipe recognition from position history
- **Web UI**: Clean, responsive interface with live skeleton overlay
- **Render Deployment**: One-click deploy at `https://hand-motion-pipeline.onrender.com`
- **142 Unit Tests**: Comprehensive coverage across all modules

---

## Table of Contents

- [Project Vision](#project-vision)
- [Technical Architecture](#technical-architecture)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Author](#author)

---

## Project Vision

Sign language translation isn't about generating smooth animations — it's about **preserving meaning through motion**.

This project is built on four principles:

- **Motion as data**, not just visuals
- **Sequences over frames** (signs are temporal)
- **Semantic accuracy over visual similarity**
- **Reusable pipeline architecture** (capture → structure → multiple outputs)

---

## Technical Architecture

```text
┌────────────────────────────────────────────┐
│          BROWSER / DESKTOP                 │
│  Camera → base64 JPEG → WebSocket          │
└───────────────┬────────────────────────────┘
                │
┌───────────────▼────────────────────────────┐
│          Flask + SocketIO Server           │
│  MediaPipe Hands → 21 Landmarks           │
└───────────────┬────────────────────────────┘
                │
        ┌───────┼────────────┐
        │       │            │
┌───────▼───┐ ┌─▼──────────┐ ┌▼─────────────┐
│ Rule-Based│ │ ML Random  │ │ Motion       │
│ 32 combos │ │ Forest     │ │ Circle/Wave/ │
│ + motion  │ │ 15 gestures│ │ Swipe        │
└───────┬───┘ └─┬──────────┘ └┬─────────────┘
        └───────┼────────────┘
                │
┌───────────────▼────────────────────────────┐
│          Browser Canvas                    │
│  Skeleton overlay + Gesture display        │
└────────────────────────────────────────────┘
```

---

## Key Features

### Browser Webcam Support
- Works on any device with a camera (laptop, phone, tablet)
- No installation required — just open the URL
- Real-time at 15 FPS over WebSocket
- HTTPS on Render (required for webcam access)

### Dual Gesture Classification
- **Rule-based**: 32 finger combinations + motion patterns (instant)
- **ML-based**: RandomForest classifier trained on 15 gestures (64% accuracy)
- Both run simultaneously, results shown side-by-side

### Motion Detection
- **Circle**: Radius + angular analysis from index tip positions
- **Wave**: X-axis oscillation detection over 15 frames
- **Swipe**: Unidirectional movement detection

### Web UI
- Responsive split-screen layout (camera + panel)
- Live hand skeleton overlay
- Recording library with playback
- Session statistics

---

## Quick Start

### Python Version

**Python 3.11+ required**

### Installation

```bash
git clone https://github.com/richiekaroki/AIVirtualMouse.git
cd AIVirtualMouse
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .
```

### Run Locally

```bash
# Start the web server
python wsgi.py

# Open in browser
# http://localhost:8000
```

### Run on Render

Push to GitHub → Render auto-deploys via `render.yaml`

Live at: `https://hand-motion-pipeline.onrender.com`

---

## Usage

### 1. Browser Webcam (Recommended)

```bash
python wsgi.py
# Open http://localhost:8000
# Click Start to activate camera
# Allow camera access when prompted
```

### 2. Desktop App (Enhanced UI)

```bash
python -m hand_motion.apps.cursor_enhanced
```

**Controls:**
- **R** – Start recording
- **S** – Stop and save
- **C** – Cancel
- **Q** – Quit

### 3. Batch Dataset Recording

```bash
python -m hand_motion.apps.batch_record
```

### 4. Motion Analysis

```bash
python -m hand_motion.analyzer motion_data/circle_sample.json
python -m hand_motion.analyzer motion_data/circle_sample.json --plot trajectory
python -m hand_motion.analyzer motion_data/circle_sample.json motion_data/fist_sample.json --compare
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v --tb=short
```

---

## Project Structure

```
AIVirtualMouse/
├── src/hand_motion/
│   ├── __init__.py
│   ├── detection.py              # Hand tracking (MediaPipe/cvzone)
│   ├── descriptor.py             # Core abstraction (32 primitives)
│   ├── analyzer.py               # Analysis toolkit
│   ├── animation.py              # 3D animation export
│   ├── batch.py                  # Batch processing
│   ├── database.py               # SQLite storage
│   ├── export.py                 # Multi-format export
│   ├── face.py                   # Face mesh (MediaPipe)
│   ├── gloss.py                  # Gloss annotation
│   ├── gpu.py                    # GPU acceleration
│   ├── pose.py                   # Body tracking (MediaPipe Pose)
│   ├── validation.py             # Dataset quality checks
│   ├── video_export.py           # Video export
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── landmark_classifier.py    # ML classifier (RandomForest)
│   │   ├── gesture_recognizer.py     # CNN+LSTM architecture
│   │   ├── inference_engine.py       # Frame buffering + sliding window
│   │   └── translator.py             # Gesture translation
│   ├── web/
│   │   ├── __init__.py
│   │   ├── __main__.py               # Entry point
│   │   ├── app.py                    # Flask + SocketIO server
│   │   ├── static/                   # Static assets
│   │   └── templates/index.html      # Web UI
│   └── apps/
│       ├── __init__.py
│       ├── cursor.py                 # Basic cursor control
│       ├── cursor_enhanced.py        # Enhanced UI version
│       ├── record.py                 # Single gesture recording
│       ├── batch_record.py           # Dataset creation
│       └── analyze_dataset.py        # Dataset analysis
├── models/
│   └── landmark_classifier.pkl   # Trained ML model
├── motion_data/
│   └── *.json                    # 15 sample recordings
├── tests/
│   └── test_*.py                 # 142 tests
├── examples/                     # Usage examples
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
├── Dockerfile                    # Render deployment
├── render.yaml                   # Render blueprint
├── requirements.txt              # Core + web dependencies
├── pyproject.toml                # Package config
├── wsgi.py                       # Gunicorn entry point
├── LICENSE
└── README.md
```

---

## Testing

**142 tests** covering:

| Module | Tests | Coverage |
|--------|-------|----------|
| detection.py | 8 | 85%+ |
| descriptor.py | 46 | 95%+ |
| analyzer.py | 16 | 80%+ |
| batch.py | 15 | 80%+ |
| pose.py | 8 | 80%+ |
| Other | 49 | 90%+ |

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_descriptor.py -v

# Quick run
python -m pytest tests/ -x -q
```

---

## Dataset

- **15 sample recordings** in `motion_data/`
- **2 seconds each** at 30 FPS
- **Gestures**: circle, fist, grab, ok_sign, open_hand, peace, pinch, point, push, swipe_left, swipe_right, three_fingers, thumbs_up, two_fingers, wave

---

## Roadmap

### Phase 1 — Extended Motion Capture
- MediaPipe Pose (body tracking)
- MediaPipe Face Mesh (expressions)
- Two-handed coordination

### Phase 2 — Linguistic Layer
- Gloss annotation tools
- Co-articulation modeling
- Deaf community validation

### Phase 3 — Animation Output
- 3D rig integration (Blender / Three.js)
- Motion retargeting

### Phase 4 — Translation Pipeline
- Text → gloss (NLP)
- Gloss → motion synthesis
- Real-time deployment

---

## Contributing

Contributions, suggestions, and feedback are welcome — especially in:

- Sign language expertise (especially Kenyan Sign Language)
- 3D animation and rigging
- ML/NLP translation systems
- Accessibility research

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

---

## Author

**Richard Kabue Karoki**
Nairobi, Kenya
[karokirichard522@gmail.com](mailto:karokirichard522@gmail.com)

[GitHub](https://github.com/richiekaroki) | [LinkedIn](https://linkedin.com/in/richard-karoki-007)

**Education:** B.Sc. in Computer Technology, JKUAT (2024)
**Experience:** 4+ years in backend/systems engineering, computer vision, and accessibility tech

---

## Project Status

- **Status:** Active Development
- **Version:** 0.8.0
- **Last Updated:** August 2026
- **Test Status:** 142 tests passing
- **Deploy:** [https://hand-motion-pipeline.onrender.com](https://hand-motion-pipeline.onrender.com)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with care for accessibility and linguistic preservation**

</div>
