# Hand Motion Interpretation Pipeline

### From Gesture Recognition to Sign Language Infrastructure

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen.svg)]()
[![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple.svg)]()

> **A real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.**

Originally built for gesture-based cursor control, this project evolved into a full motion interpretation pipeline that treats hand gestures as **structured linguistic data**, making it suitable for sign language research, animation systems, and accessibility applications.

## What's New in v0.9.0

- **12 AI Features**: Complete sign language recognition pipeline
- **Handedness Labeling**: Left/right hand identification in real-time
- **Two-Hand Detection**: Multi-hand support with distinct skeleton colors
- **Non-Manual Markers**: Face mesh (eyes, mouth, eyebrows, head) wired into UI
- **Continuous CSLR**: CTC/attention decoding for variable-length gloss sequences
- **DTW Dictionary**: Template-based sign matching
- **Fingerspelling A-Z**: Individual letter detection from handshape
- **HID Output**: Gesture-to-keyboard/mouse control
- **Gloss-to-Text NLP**: Seq2Seq attention translation model
- **Transformer CSLR**: Vision Transformer architecture
- **Temporal Smoothing**: EMA confidence values
- **Platt Calibration**: Calibrated ML confidence scores
- **Data Augmentation**: Rotation, scale, noise, time warping

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
│  MediaPipe Hands → N hands × 21 landmarks │
│  + Face Mesh → non-manual markers         │
└───────────────┬────────────────────────────┘
                │
    ┌───────────┼──────────┬──────────┐
    │           │          │          │
┌───▼───┐  ┌───▼──┐  ┌───▼───┐  ┌──▼──────┐
│ Rule  │  │ ML   │  │ Face  │  │ Handed  │
│ Based │  │ RF+  │  │ EAR/  │  │ Left/   │
│ 32    │  │ Platt│  │ MAR   │  │ Right   │
│ combos│  │ Cal. │  │ Brows │  │ +2 hands│
└───┬───┘  └───┬──┘  └───┬───┘  └──┬──────┘
    └───────────┼─────────┼─────────┘
                │
┌───────────────▼────────────────────────────┐
│          AI Modules                       │
│  Continuous CSLR · DTW · Fingerspelling   │
│  HID Output · NLP Translator · ViT        │
└───────────────┬────────────────────────────┘
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

### 12 AI Features

| # | Feature | What It Does |
|---|---------|-------------|
| 1 | Handedness labeling | Shows Left/Right badge per hand |
| 2 | Two-hand detection | Renders both hands with distinct colors |
| 3 | Temporal smoothing | EMA-smoothed confidence values |
| 4 | Confidence calibration | Platt-scaled RF probabilities |
| 5 | Data augmentation | Rotation, scale, noise, time warp |
| 6 | Non-manual markers | Face mesh (eyes, mouth, eyebrows, head) |
| 7 | Continuous CSLR | CTC/attention gloss decoding |
| 8 | DTW dictionary | Template matching recognition |
| 9 | Fingerspelling | ASL A-Z letter detection |
| 10 | HID output | Gesture-to-keyboard/mouse |
| 11 | Gloss-to-text NLP | Seq2Seq attention translation |
| 12 | Transformer CSLR | Vision Transformer architecture |

### Motion Detection
- **Circle**: Radius + angular analysis from index tip positions
- **Wave**: X-axis oscillation detection over 15 frames
- **Swipe**: Unidirectional movement detection

### Web UI
- Responsive split-screen layout (camera + panel)
- Live hand skeleton overlay (amber/purple for two hands)
- Face expression chips (Face, Eyes, Mouth, Brow)
- Recording library with playback
- Session statistics

---

## User Guide

### Getting Started

Open the app in any browser with a camera. Click **Start** and allow camera access. A green **Live** badge confirms the connection — you'll see your hand skeleton appear instantly.

### What You'll See

| Element | What It Shows |
|---------|--------------|
| **Skeleton overlay** | Amber = primary hand, Purple = second hand |
| **Handedness badge** | Blue "Left" or Purple "Right" based on hand orientation |
| **Finger dots** (T I M R P) | Light up yellow as each finger extends |
| **Gesture name** | Rule-based classification from finger positions |
| **ML gesture** | Machine learning prediction with confidence % |
| **Face chips** | Eyes open/closed, mouth open/closed, eyebrow raise, expression |
| **Second hand** | Auto-appears when two hands are in frame |

### Controls

- **Start** — activate camera feed
- **Stop** — disconnect camera
- **Rec** — start/stop recording a gesture to the library
- **Capture** — take a screenshot
- **Play** — replay a selected recording from the library

### Tips

- Keep your hand 30–60 cm from the camera for best tracking
- Good lighting improves landmark detection
- The face section appears automatically when your face is visible
- Two-hand gestures show both skeletons with different colors

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
│   ├── detection.py              # Hand tracking (MediaPipe/cvzone) + handedness + multi-hand
│   ├── descriptor.py             # Core abstraction (32 primitives)
│   ├── analyzer.py               # Analysis toolkit
│   ├── animation.py              # 3D animation export
│   ├── batch.py                  # Batch processing
│   ├── database.py               # SQLite storage
│   ├── export.py                 # Multi-format export
│   ├── face.py                   # Face mesh (468 landmarks, EAR/MAR/brows/head)
│   ├── gloss.py                  # Gloss annotation
│   ├── gpu.py                    # GPU acceleration
│   ├── pose.py                   # Body tracking (MediaPipe Pose)
│   ├── validation.py             # Dataset quality checks
│   ├── video_export.py           # Video export
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── landmark_classifier.py    # ML classifier (RandomForest + Platt scaling)
│   │   ├── gesture_recognizer.py     # CNN+LSTM architecture
│   │   ├── inference_engine.py       # Frame buffering + EMA smoothing
│   │   ├── translator.py             # Gesture translation + Seq2Seq NLP
│   │   ├── augmentation.py           # Landmark data augmentation
│   │   ├── continuous_recognizer.py  # CTC/attention continuous CSLR
│   │   ├── dtw_dictionary.py         # DTW template matching
│   │   ├── fingerspelling.py         # ASL A-Z letter detection
│   │   ├── hid_output.py             # BLE/USB HID gesture-to-input
│   │   └── transformer_cslr.py       # Vision Transformer CSLR
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
├── docs/                         # Documentation + DESIGN.md
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

### Completed (v0.9.0)
- 12 AI features: handedness, two-hand, smoothing, calibration, augmentation, face, CSLR, DTW, fingerspelling, HID, NLP, Transformer
- Face mesh integration for non-manual markers
- Security hardening (CORS, headers, non-root Docker)
- SEO/accessibility improvements

### Next
- BERT/Transformer NLP for gloss-to-text
- Federated learning for multi-user training
- Mobile TFLite/ONNX deployment
- 3D animation export (Blender/Three.js)

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
- **Version:** 0.9.0
- **Last Updated:** August 2026
- **Test Status:** 142 tests passing
- **AI Features:** 12 integrated
- **Deploy:** [https://hand-motion-pipeline.onrender.com](https://hand-motion-pipeline.onrender.com)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with care for accessibility and linguistic preservation**

</div>
