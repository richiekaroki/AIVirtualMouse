# Hand Motion Interpretation Pipeline

![Status](https://img.shields.io/badge/status-evolving%20toward%20sign%20language-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange)
![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple)

> **Project Evolution:** Originally built for gesture-based cursor control, this project is being extended toward sign language motion capture and accessibility infrastructure.  
> [Read about the sign language extension →](SIGNLANGUAGE.md)

---

## 🎯 Current Focus: Motion as Linguistic Data

This system captures real-time hand motion using MediaPipe and OpenCV. While it currently demonstrates cursor control, the underlying technology—landmark extraction, temporal smoothing, and gesture recognition—forms the foundation for sign language translation systems.

**Key insight:** Sign languages are built from precise combinations of handshape, location, movement, and orientation. This system captures all of these as structured data.

### Why This Matters for Accessibility

Sign language translation requires treating motion as **linguistic infrastructure**, not just visual patterns. This project approaches gesture recognition as a data pipeline problem: capture → structure → multiple outputs (cursor control, JSON export, animation, validation).

---

## Requirements

- **Python 3.11** (Python 3.13+ not yet supported by MediaPipe)
- See `requirements.txt` for dependencies

### Why Python 3.11?

MediaPipe (required for hand tracking) currently supports Python 3.8-3.11.
Python 3.13 support is in development.

### Installation

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎨 Enhanced User Interface

The project includes two versions:

### Standard Version (`AiVirtualMouseProject.py`)

- Basic OpenCV interface
- Essential information display
- Cursor control + recording

### Enhanced Version (`AiVirtualMouseProject_Enhanced.py`)

- **Split-screen layout**: 640x480 video + 400px analysis panel
- **Hand skeleton overlay**: Visual representation of tracked landmarks
- **Primitive timeline**: Last 10 gestures shown as colored blocks
- **Real-time analysis**: Motion metrics updated live
- **Session statistics**: Track gestures and frames
- **Professional design**: Clean typography and color scheme

**Comparison:**

| Feature             | Standard | Enhanced     |
| ------------------- | -------- | ------------ |
| Video feed          | ✅       | ✅           |
| Hand tracking       | ✅       | ✅           |
| Skeleton overlay    | ❌       | ✅           |
| Primitive timeline  | ❌       | ✅           |
| Split-screen layout | ❌       | ✅           |
| Session stats       | ❌       | ✅           |
| Recording timer     | ❌       | ✅           |
| Visual polish       | Basic    | Professional |

**Usage:**

```bash
# Standard version
python AiVirtualMouseProject.py

# Enhanced version (recommended for demos)
python AiVirtualMouseProject_Enhanced.py
```

---

## ✅ Checkpoint: Enhanced UI Complete

### **Files Created:**

- ✅ `AiVirtualMouseProject_Enhanced.py` - Professional UI version
- ✅ `screenshots/` - Before/after comparison (optional)

### **Features Added:**

1. **Split-Screen Layout**
   - 640x480 video feed
   - 400px info panel
   - Total: 1040x480 window

2. **Visual Enhancements**
   - Hand skeleton overlay (yellow lines, orange joints)
   - Color-coded primitives timeline
   - Blinking recording indicator
   - Professional color scheme
   - Section headers with backgrounds

3. **Information Architecture**
   - Recording status header
   - Recent sequence (10 primitives)
   - Motion analysis section
   - Session statistics
   - Control instructions

4. **User Experience**
   - Better visual hierarchy
   - Clearer information organization
   - Professional appearance
   - Suitable for demos and presentations

---

### **Capabilities Added:**

1. **Trajectory Analysis**
   - 2D hand path visualization
   - Start/end markers
   - Direction indicators
   - Position over time plots

2. **Primitive Timeline**
   - Temporal sequence visualization
   - Color-coded primitives
   - Transition analysis

3. **Velocity Profiling**
   - Speed over time
   - X/Y components
   - Mean/max/min statistics

4. **Hand Openness Tracking**
   - Open/closed state over time
   - Handshape transitions

5. **Statistical Analysis**
   - Comprehensive summaries
   - Primitive distribution
   - Quality metrics

6. **Comparison Tools**
   - Side-by-side gesture comparison
   - Consistency validation

7. **Export Capabilities**
   - Save plots as PNG
   - Batch processing
   - Documentation generation

---

## 📊 Progress Tracking

Day 1: Conceptual Reframe ████████████████████ 100% ✅
Day 2: Motion Descriptor ████████████████████ 100% ✅
Day 3: Recording + Export ████████████████████ 100% ✅
Day 3.5: Enhanced UI ████████████████████ 100% ✅
Day 4: Visualization ████████████████████ 100% ✅ COMPLETE
Day 5: Record Gestures ░░░░░░░░░░░░░░░░░░░░ 0%
Day 6: Full README Rewrite ░░░░░░░░░░░░░░░░░░░░ 0%
Day 7: Demo + Polish ░░░░░░░░░░░░░░░░░░░░ 0%

Overall Progress: ████████████████░░░░ 71%

---

## About

Real-time hand motion capture and interpretation system built with MediaPipe and OpenCV. Demonstrates:

- ✅ 21-point hand landmark tracking at 30fps
- ✅ Gesture-based cursor control (index finger = move, two fingers = click)
- ✅ Motion smoothing and coordinate normalization
- ✅ Modular architecture (HandTrackingModule)
- 🔄 Being extended for sign language motion capture ([details](SIGNLANGUAGE.md))

**Technical Foundation:** Computer vision pipeline that captures hand motion as structured data, enabling multiple downstream applications beyond cursor control.
