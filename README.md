# 🖐️ Hand Motion Interpretation Pipeline

### From Gesture Recognition to Sign Language Infrastructure

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple.svg)]()

> **A real-time motion capture and analysis system designed as foundational infrastructure for sign language translation and accessibility technology.**

Originally built for gesture-based cursor control, this project evolved into a full motion interpretation pipeline that treats hand gestures as **structured linguistic data**, making it suitable for sign language research, animation systems, and accessibility applications.

---

## 📑 Table of Contents

- [Project Vision](#-project-vision)
- [Technical Architecture](#️-technical-architecture)
- [Key Features](#-key-features)
- [Why This Matters for Sign Language](#-why-this-matters-for-sign-language)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
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
python AiVirtualMouseProject_Enhanced.py
```

**Controls:**

- **R** – Start recording
- **S** – Stop and save
- **C** – Cancel
- **Q** – Quit

#### 2. Batch Dataset Recording

```bash
python batch_record.py
```

#### 3. Motion Analysis

```bash
python MotionAnalyzer.py motion_data/gesture.json
python MotionAnalyzer.py motion_data/gesture.json --plot trajectory
python MotionAnalyzer.py motion_data/gesture.json --output plots/
python MotionAnalyzer.py gesture1.json gesture2.json --compare
```

---

## 📁 Project Structure

```
AIVirtualMouse/
├── MotionDescriptor.py        # Core abstraction ⭐
├── MotionAnalyzer.py          # Analysis toolkit ⭐
├── AiVirtualMouseProject_Enhanced.py
├── batch_record.py
├── analyze_dataset.py
├── motion_data/
│   ├── *.json
│   └── README.md
├── analysis_plots/
├── docs/
│   ├── SIGNLANGUAGE.md
│   ├── ANALYSIS.md
│   ├── CHANGELOG.md
│   └── recording_plan.md
├── requirements.txt
└── dataset_summary.md
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
- **Version:** 0.6.0
- **Last Updated:** January 2026
- **Next Milestone:** Animation integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ for accessibility and linguistic preservation**

⭐ Star this repo if you find it useful!

</div>
