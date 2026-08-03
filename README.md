# 🖐️ Hand Motion Interpretation Pipeline

### From Gesture Recognition to Sign Language Infrastructure

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-79%20passing-brightgreen.svg)]()
[![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple.svg)]()

> **A real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.**

Originally built for gesture-based cursor control, this project evolved into a full motion interpretation pipeline that treats hand gestures as **structured linguistic data**, making it suitable for sign language research, animation systems, and accessibility applications.

## What's New in v0.7.0

- **Enhanced Type Safety**: Complete type hints across all modules
- **Improved Error Handling**: Graceful handling of edge cases
- **79 Unit Tests**: Comprehensive test coverage (95%+ for core modules)
- **Performance Optimizations**: Memory-efficient `__slots__`, optimized statistics
- **CV Best Practices**: Proper resource cleanup, input validation
- **Documentation**: Architecture design document (DESIGN.md)

---

## 📑 Table of Contents

- [Project Vision](#-project-vision)
- [Technical Architecture](#️-technical-architecture)
- [Key Features](#-key-features)
- [Why This Matters for Sign Language](#-why-this-matters-for-sign-language)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Dataset Overview](#-dataset-overview)
- [Key Learnings](#-key-learnings)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Project Vision

Sign language translation isn't about generating smooth animations — it's about **preserving meaning through motion**.

This project is built on four principles:

- **Motion as data**, not just visuals
- **Sequences over frames** (signs are temporal)
- **Semantic accuracy over visual similarity**
- **Reusable pipeline architecture** (capture → structure → multiple outputs)

**Key insight:** The same motion capture system can power animation, training datasets, validation tools, and real-time translation — if the architecture treats motion as structured, reusable data.

---

## 🏗️ Technical Architecture

```text
┌────────────────────────────────────────────┐
│          MOTION CAPTURE LAYER              │
│  MediaPipe Hands + OpenCV Processing       │
└───────────────┬────────────────────────────┘
                │
┌───────────────▼────────────────────────────┐
│        ABSTRACTION LAYER (CORE)            │
│  MotionDescriptor → Structured Motion Data │
│  • Handshape classification                │
│  • Location tracking                       │
│  • Velocity & trajectory analysis          │
│  • Primitive detection                     │
└───────────────┬────────────────────────────┘
                │
        ┌───────┼────────┬────────┬────────┐
        │       │        │        │        │
┌───────▼───┐ ┌─▼──────┐ ┌▼──────┐ ┌▼──────┐
│ Cursor    │ │ JSON   │ │ Visual │ │ Future│
│ Control   │ │ Export │ │ Analysis│ │Animation│
└───────────┘ └────────┘ └────────┘ └────────┘
```

---

## ✨ Key Features

### 🎥 Real-Time Motion Capture

- 21-point hand tracking at ~30 FPS (MediaPipe)
- Coordinate normalization for cross-device consistency
- Motion smoothing and occlusion handling

### 🧠 Motion Descriptor Abstraction

- Converts landmarks → structured linguistic motion data
- Classifies gesture primitives (POINT, FIST, OPEN_HAND, etc.)
- Tracks temporal sequences, velocity, and transitions

### 💾 Data Pipeline

- JSON export for ML training and animation systems
- Metadata + timestamps + quality metrics
- Batch recording and dataset validation tools

### 📊 Analysis & Visualization

- Trajectory plots
- Primitive timelines
- Velocity profiles
- Gesture comparison tools

### 🎨 Professional UI

- Split-screen interface (video + analytics)
- Live skeleton overlay
- Recording controls and session statistics

---

## 🔬 Why This Matters for Sign Language

Sign languages are built on four core parameters:

| Parameter   | How This System Captures It              | Status         |
| ----------- | ---------------------------------------- | -------------- |
| Handshape   | Finger states, landmark relationships    | ✅ Implemented |
| Location    | Normalized 2D coordinates                | ✅ Implemented |
| Movement    | Velocity, trajectory, temporal sequences | ✅ Implemented |
| Orientation | Landmark directions                      | ⚠️ Partial     |

**Planned extensions:**

- Facial expressions (non-manual markers) → MediaPipe Face
- Body position context → MediaPipe Pose
- Two-handed coordination

---

## 🚀 Quick Start

### 🐍 Python Version Requirement

**Python 3.11 is required.**  
(MediaPipe does not currently support Python 3.12+.)

```bash
py -3.11 --version
```

### Installation

```bash
git clone https://github.com/richiekaroki/AIVirtualMouse.git
cd AIVirtualMouse
py -3.11 -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### ▶️ Usage

#### 1. Real-Time Capture (Enhanced UI)

```bash
python -m hand_motion.apps.cursor_enhanced
```

**Controls:**

- **R** – Start recording
- **S** – Stop and save
- **C** – Cancel
- **Q** – Quit

#### 2. Batch Dataset Recording

```bash
python -m hand_motion.apps.batch_record
```

#### 3. Motion Analysis

```bash
python -m hand_motion.analyzer motion_data/gesture.json
python -m hand_motion.analyzer motion_data/gesture.json --plot trajectory
python -m hand_motion.analyzer motion_data/gesture.json --output plots/
python -m hand_motion.analyzer gesture1.json gesture2.json --compare
```

#### 4. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=hand_motion

# Run specific test file
python -m pytest tests/test_motion_descriptor.py -v
```

---

## 🧪 Testing

The project includes comprehensive unit tests with **79 test cases** covering:

- **MotionDescriptor**: Core motion data structure (95%+ coverage)
- **HandDetector**: Landmark detection (85%+ coverage)
- **MotionAnalyzer**: Analysis tools (80%+ coverage)

### Test Categories

| Category | Tests | Coverage |
|----------|-------|----------|
| Unit Tests | 60 | 90%+ |
| Integration Tests | 15 | 80%+ |
| Edge Cases | 4 | 100% |

### Running Tests

```bash
# Quick test run
python -m pytest tests/ -v

# With detailed output
python -m pytest tests/ -v --tb=long

# Generate coverage report
python -m pytest tests/ --cov=hand_motion --cov-report=html
```

---

## 📁 Project Structure

```
AIVirtualMouse/
├── src/
│   └── hand_motion/
│       ├── __init__.py              # Package init
│       ├── detection.py             # Hand tracking (MediaPipe/cvzone)
│       ├── descriptor.py            # Core abstraction ⭐
│       ├── analyzer.py              # Analysis toolkit ⭐
│       └── apps/
│           ├── cursor.py            # Basic cursor control
│           ├── cursor_enhanced.py   # Enhanced UI ⭐
│           ├── record.py            # Single gesture recording
│           ├── batch_record.py      # Dataset creation
│           └── analyze_dataset.py   # Dataset analysis
├── tests/
│   └── test_motion_descriptor.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── blog-post.md
│   └── site/                        # GitHub Pages
├── motion_data/
│   └── *.json
├── analysis_plots/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview

- **45 total recordings** (15 gestures × 3 attempts)
- **Average FPS:** 30
- **Average duration:** 2.5 seconds
- **Quality score:** 0.87 / 1.00

**Gesture categories include:**

- Static handshapes (fist, open hand, point, thumbs up)
- Dynamic movements (wave, circle, swipe)
- Directional motions (push, pull, point up)
- Transitions and complex gestures

---

## 🎓 Key Learnings

### Co-articulation is the hardest problem

Signs blend into each other — context matters.

### Sequences matter more than frames

A sign is a motion pattern, not a pose.

### Abstraction enables scalability

MotionDescriptor unlocked multiple downstream uses.

### Velocity and timing carry meaning

Motion dynamics affect semantics, not just appearance.

---

## 🔮 Roadmap

### Phase 1 — Extended Motion Capture

- MediaPipe Pose (body tracking)
- MediaPipe Face Mesh (expressions)
- Two-handed coordination
- 3D orientation extraction

### Phase 2 — Linguistic Layer

- Gloss annotation tools
- Co-articulation modeling
- Deaf community validation framework

### Phase 3 — Animation Output

- 3D rig integration (Blender / Three.js)
- Motion retargeting
- Keyframe generation

### Phase 4 — Translation Pipeline

- Text → gloss (NLP)
- Gloss → motion synthesis
- Real-time deployment

---

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome — especially in:

- Sign language expertise (especially Kenyan Sign Language)
- 3D animation and rigging
- ML/NLP translation systems
- Accessibility research

---

## 👤 Author

**Richard Kabue Karoki**  
📍 Nairobi, Kenya  
📧 <karokirichard522@gmail.com>

🌐 [GitHub](https://github.com/richiekaroki)  
🔗 [LinkedIn](https://linkedin.com/in/richard-karoki-007)

**Education:** B.Sc. in Computer Technology, JKUAT (2024)  
**Experience:** 4+ years in backend/systems engineering, computer vision, and accessibility tech

---

## 🌟 Project Status

- **Status:** Active Development
- **Version:** 0.7.0
- **Last Updated:** August 2026
- **Test Status:** 79 tests passing
- **Next Milestone:** Animation integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for accessibility and linguistic preservation**

⭐ Star this repo if you find it useful!

</div>
