# Hand Motion Interpretation Pipeline

### From Gesture Recognition to Sign Language Infrastructure

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Focus](https://img.shields.io/badge/focus-accessibility%20tech-purple.svg)]()

> **A real-time motion capture and analysis system designed to serve as foundational infrastructure for sign language translation and accessibility technology.**

Originally developed for gesture-based cursor control, this project has evolved into a comprehensive motion interpretation pipeline that treats hand gestures as structured, linguistic data—making it suitable for sign language research, animation systems, and accessibility applications.

---

## 🎯 Project Vision

Sign language translation isn't about recognizing pretty gestures—it's about **preserving meaning through motion**. This project approaches the problem as **linguistic infrastructure**:

- **Motion as data**, not just visual input
- **Sequences over frames** (signs are temporal, not static)
- **Semantic accuracy** over visual similarity
- **Reusable pipeline** architecture (capture → structure → multiple outputs)

**Key insight:** The same motion capture that drives a cursor can drive animation systems, training datasets, validation tools, and real-time translation—if the architecture treats motion as structured, reusable data.

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MOTION CAPTURE LAYER                     │
│  MediaPipe Hand Tracking (21 landmarks, 30fps)             │
│  OpenCV Processing | Coordinate Normalization              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  ABSTRACTION LAYER                          │
│  MotionDescriptor: Converts landmarks → structured data    │
│  • Handshape classification (finger states)                │
│  • Location tracking (coordinates)                          │
│  • Movement analysis (velocity, trajectory)                 │
│  • Primitive detection (building blocks of signs)          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬────────────┐
        │            │            │            │
┌───────▼───┐  ┌────▼─────┐ ┌───▼──────┐ ┌──▼────────┐
│  Cursor   │  │   JSON   │ │  Visual  │ │  Future:  │
│  Control  │  │  Export  │ │  Analysis│ │ Animation │
│           │  │ (Training│ │  Plots   │ │  Systems  │
│           │  │   Data)  │ │          │ │           │
└───────────┘  └──────────┘ └──────────┘ └───────────┘
```

**Why this matters:** Decoupling motion capture from output enables the same data to serve multiple purposes—essential for production sign language systems.

---

## ✨ Key Features

### 🎥 Real-Time Motion Capture

- **21-point hand tracking** using MediaPipe at 30fps
- **Coordinate normalization** for cross-resolution compatibility
- **Motion smoothing** and interpolation
- **Occlusion handling** with frame dropout recovery

### 🧠 Motion Descriptor Abstraction

- **Structured representation** of hand state (not raw coordinates)
- **Gesture primitive classification**: POINT, FIST, OPEN_HAND, PEACE_V, etc.
- **Temporal sequence tracking** (motion history over time)
- **Velocity analysis** (speed and direction)
- **Hand openness metrics** (transition tracking)

### 💾 Data Pipeline

- **JSON export** of motion sequences for training/analysis
- **Reusable format** with metadata and timestamps
- **Quality metrics** (FPS, frame count, primitive distribution)
- **Batch processing** support

### 📊 Analysis & Visualization

- **Trajectory plotting** (2D hand path + position over time)
- **Primitive timeline** visualization (temporal sequences)
- **Velocity profiling** (movement dynamics)
- **Hand openness tracking** (handshape transitions)
- **Statistical analysis** (comprehensive metrics)
- **Comparison tools** (side-by-side gesture analysis)

### 🎨 Professional UI

- **Split-screen interface** (video + analysis panel)
- **Hand skeleton overlay** (visual landmark representation)
- **Real-time primitive display**
- **Recording progress indicators**
- **Session statistics**

---

## 🔬 Why This Matters for Sign Language

Sign languages are complete, complex languages built from four core parameters:

| Parameter       | How This System Captures It              | Status         |
| --------------- | ---------------------------------------- | -------------- |
| **Handshape**   | Finger states, landmark relationships    | ✅ Implemented |
| **Location**    | 2D coordinates (normalized)              | ✅ Implemented |
| **Movement**    | Velocity, trajectory, temporal sequences | ✅ Implemented |
| **Orientation** | Landmark directions (implicit)           | ⚠️ Partial     |

**Additional requirements:**

- **Non-manual markers** (facial expressions) → 📋 Planned via MediaPipe Face
- **Body position** (signs relative to body) → 📋 Planned via MediaPipe Pose
- **Two-handed coordination** → 📋 Planned

**Current capability:** This system captures 3 of 4 core parameters as structured data, providing a foundation for sign language motion analysis.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Installation

```bash
# Clone repository
git clone https://github.com/richiekaroki/AIVirtualMouse.git
cd AIVirtualMouse

# Install requirements
pip install opencv-python mediapipe numpy pyautogui matplotlib

# Verify installation
python -c "import cv2, mediapipe; print('✓ Ready')"
```

### Basic Usage

#### 1. Real-Time Capture (Enhanced UI)

```bash
python AiVirtualMouseProject_Enhanced.py
```

**Controls:**

- **R** - Start recording a gesture sequence
- **S** - Stop recording and save to JSON
- **C** - Cancel recording
- **Q** - Quit application

**Mouse control mode** (when not recording):

- Index finger up → Move cursor
- Index + middle up → Click

#### 2. Batch Recording (Guided Dataset Creation)

```bash
python batch_record.py
```

Guides you through recording 15 pre-defined gestures with quality checks.

#### 3. Motion Analysis

```bash
# Analyze single gesture
python MotionAnalyzer.py motion_data/gesture.json

# Generate specific plot
python MotionAnalyzer.py motion_data/gesture.json --plot trajectory

# Save all plots
python MotionAnalyzer.py motion_data/gesture.json --output plots/

# Compare two gestures
python MotionAnalyzer.py gesture1.json gesture2.json --compare
```

#### 4. Dataset Analysis

```bash
python analyze_dataset.py
```

Analyzes all recorded gestures, ranks by quality, generates summary report.

---

## 📁 Project Structure

```
AIVirtualMouse/
├── AiVirtualMouseProject.py          # Original version (backward compatible)
├── AiVirtualMouseProject_Enhanced.py # Professional UI version
├── HandTrackingModule.py             # Core hand detection (MediaPipe wrapper)
├── MotionDescriptor.py               # Motion abstraction layer ⭐
├── MotionAnalyzer.py                 # Offline analysis toolkit ⭐
├── batch_record.py                   # Guided recording script
├── analyze_dataset.py                # Dataset quality analysis
├── record_gesture.py                 # Quick single-gesture recording
├── test_motion_descriptor.py         # Unit tests
│
├── motion_data/                      # Recorded gesture sequences
│   ├── *.json                        # Individual recordings
│   ├── recording_manifest.json       # Recording metadata
│   └── README.md                     # Data format documentation
│
├── analysis_plots/                   # Generated visualizations
│   ├── *_trajectory.png
│   ├── *_primitives.png
│   └── ...
│
├── examples/                         # Example analyses
│   └── sample_analysis.md
│
├── docs/
│   ├── README.md                     # This file
│   ├── SIGNLANGUAGE.md              # Sign language context ⭐
│   ├── ANALYSIS.md                   # Analysis guide ⭐
│   ├── CHANGELOG.md                  # Version history
│   └── recording_plan.md             # Dataset strategy
│
├── requirements.txt                  # Python dependencies
└── dataset_summary.md               # Dataset analysis report
```

**⭐ Key innovations:** MotionDescriptor (abstraction), MotionAnalyzer (visualization), documentation (linguistic context)

---

## 💡 Usage Examples

### Example 1: Recording a Gesture Sequence

```python
from MotionDescriptor import MotionDescriptor
import cv2
from HandTrackingModule import handDetector

# Initialize
cap = cv2.VideoCapture(0)
detector = handDetector()
motion_descriptor = MotionDescriptor()

# Record for 3 seconds
import time
start_time = time.time()

while time.time() - start_time < 3:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        descriptor = motion_descriptor.create_descriptor(lmList, fingers)

        # Now you have structured motion data
        print(f"Primitive: {descriptor['primitive']}")
        print(f"Velocity: {descriptor['velocity']}")

# Save sequence
motion_descriptor.save_sequence("my_gesture.json", "wave")

cap.release()
```

### Example 2: Analyzing Recorded Motion

```python
from MotionAnalyzer import MotionAnalyzer

# Load and analyze
analyzer = MotionAnalyzer("motion_data/wave_123.json")

# Print statistics
analyzer.print_summary()

# Generate visualizations
analyzer.plot_trajectory()
analyzer.plot_primitives_timeline()
analyzer.plot_velocity_profile()

# Or generate all plots at once
analyzer.generate_all_plots(output_dir="analysis/")
```

### Example 3: Comparing Two Attempts

```python
from MotionAnalyzer import GestureComparator

# Compare
comparator = GestureComparator(
    "motion_data/wave_attempt1.json",
    "motion_data/wave_attempt2.json"
)

comparator.compare_statistics()
comparator.compare_trajectories()
```

---

## 📊 Sample Dataset

The repository includes a curated dataset of 15 gestures across 5 categories:

### Static Handshapes

- **point** - Index finger extended
- **fist** - All fingers closed
- **open_hand** - All fingers extended
- **thumbs_up** - Thumb extended upward
- **peace** - V-sign (index + middle)

### Dynamic Movements

- **wave** - Side-to-side motion
- **circle** - Circular hand motion
- **swipe_right** - Left-to-right sweep

### Transitions

- **open_close** - Repeated opening/closing
- **point_fist** - Alternating between states

### Directional

- **push_forward** - Away from body
- **pull_back** - Toward body
- **point_up** - Upward pointing

### Complex

- **ok_sign** - Thumb-index circle
- **pinch_release** - Pinching motion

**Dataset statistics:**

- 45 total recordings (3 attempts × 15 gestures)
- Average FPS: 30
- Average duration: 2.5 seconds
- Quality score: 0.87/1.00

See `dataset_summary.md` for detailed analysis.

---

## 🎓 What I Learned

### 1. **Co-articulation is the Hard Problem**

Sign language doesn't have clear boundaries between signs. Your hand position at the end of "hello" affects how "goodbye" starts. Current smoothing helps, but true co-articulation requires linguistic context, not just motion continuity.

**Approach:** Temporal sequence capture (implemented) is the foundation. Next step: contextual models that understand sign grammar.

### 2. **Sequences Matter More Than Frames**

A "wave" isn't just "open hand"—it's "open hand + sideways motion + repetition." Static handshape recognition is necessary but insufficient.

**Solution:** MotionDescriptor tracks temporal sequences, not just instantaneous states. Motion history enables sequence-based analysis.

### 3. **Data Structure Determines Capability**

Moving from "landmarks → action" to "landmarks → descriptor → outputs" unlocked multiple use cases from a single capture.

**Lesson:** Abstraction layers aren't overhead—they're enablers. The motion descriptor makes this system extensible.

### 4. **Primitive Classification is Harder Than Expected**

`[0,1,0,0,0]` is "POINT" but `[0,1,0,0,0]` with different wrist orientation might be a different sign entirely. Current classification is finger-based; sign language needs full 3D orientation.

**Current limitation:** Handshape detection captures extension but not curl, orientation, or 3D positioning.

### 5. **Velocity and Timing are Linguistically Significant**

Fast vs. slow movement can change meaning. Analyzing velocity profiles revealed that motion dynamics carry semantic information.

**Implementation:** Velocity tracking added to motion descriptors. Analysis tools visualize speed patterns.

### 6. **Real-time ≠ Production Ready**

30fps capture with live primitive detection works well for demos but needs additional validation for production sign language systems.

**Gap:** Missing facial expressions (grammatical markers), body position context, and deaf community validation.

---

## 🔮 Future Roadmap

### Phase 1: Extended Motion Capture _(Planned - Q1 2026)_

- [ ] MediaPipe Pose integration (upper body tracking)
- [ ] MediaPipe Face Mesh (facial expression capture)
- [ ] Two-handed gesture coordination
- [ ] 3D orientation extraction from landmarks

### Phase 2: Linguistic Layer _(Planned - Q2 2026)_

- [ ] Sign language gloss annotation tools
- [ ] Co-articulation modeling
- [ ] Linguistic validation framework
- [ ] Collaboration with deaf community

### Phase 3: Animation Output _(Planned - Q2 2026)_

- [ ] 3D rig integration (Blender/Three.js)
- [ ] Motion retargeting across character systems
- [ ] Keyframe generation from motion data
- [ ] Real-time animation preview

### Phase 4: Translation Pipeline _(Future)_

- [ ] Text-to-gloss translation (NLP)
- [ ] Gloss-to-motion synthesis
- [ ] Real-time translation system
- [ ] Production deployment

---

## 🤝 Contributing

This project is currently a personal research project, but contributions, suggestions, and feedback are welcome!

**Areas where contributions would be valuable:**

- Sign language expertise (especially Kenyan Sign Language)
- 3D animation and rigging knowledge
- ML/NLP for translation systems
- Deaf community connections for validation
- Documentation improvements

**To contribute:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## 📚 Documentation

### Core Documentation

- **[SIGNLANGUAGE.md](SIGNLANGUAGE.md)** - Sign language linguistic context and project evolution
- **[ANALYSIS.md](ANALYSIS.md)** - Complete guide to motion analysis tools
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and development timeline
- **[recording_plan.md](recording_plan.md)** - Dataset recording strategy

### Technical Reference

- **[MotionDescriptor.py](MotionDescriptor.py)** - Heavily documented source (450+ lines)
- **[MotionAnalyzer.py](MotionAnalyzer.py)** - Analysis toolkit source (600+ lines)
- **[motion_data/README.md](motion_data/README.md)** - Data format specification

### Examples

- **[examples/sample_analysis.md](examples/sample_analysis.md)** - Complete analysis walkthrough
- **[dataset_summary.md](dataset_summary.md)** - Dataset analysis report

---

## 🛠️ Technical Details

### Performance

- **Capture Rate:** 30 fps (consistent)
- **Latency:** < 33ms (real-time)
- **Landmarks:** 21 per hand
- **Smoothing:** Configurable interpolation
- **Memory:** < 100MB typical usage

### Data Format

Motion sequences are exported as JSON:

```json
{
  "metadata": {
    "gesture_name": "wave",
    "recorded_at": "2026-01-11T14:30:15",
    "duration_seconds": 2.5,
    "total_frames": 75,
    "average_fps": 30.0,
    "primitives_used": ["OPEN_HAND", "FIST"]
  },
  "frames": [
    {
      "timestamp": 1704567890.123,
      "frame_num": 0,
      "primitive": "OPEN_HAND",
      "fingers_extended": [1, 1, 1, 1, 1],
      "landmarks": {
        "wrist": {"x": 320, "y": 240},
        "index_tip": {"x": 350, "y": 180},
        ...
      },
      "velocity": {
        "magnitude": 45.2,
        "vx": 30.1,
        "vy": -33.5
      }
    },
    ...
  ]
}
```

### Dependencies

- **opencv-python** (4.5+) - Video capture and processing
- **mediapipe** (0.8+) - Hand tracking ML model
- **numpy** (1.19+) - Numerical operations
- **pyautogui** (0.9.50+) - Mouse control
- **matplotlib** (3.3+) - Visualization and plotting

---

## 🎯 Use Cases

This system is suitable for:

### ✅ Research & Development

- Sign language motion analysis
- Gesture recognition research
- HCI (Human-Computer Interaction) studies
- Accessibility technology prototyping

### ✅ Training Data Generation

- ML model training datasets
- Animation reference data
- Validation datasets for sign language systems

### ✅ Education

- Teaching gesture-based interfaces
- Demonstrating motion capture pipelines
- Computer vision education
- Sign language documentation

### ✅ Accessibility Applications

- Sign language translation systems (with extensions)
- Gesture-based controls for assistive technology
- Communication aids

---

## ⚠️ Limitations & Considerations

### Current Limitations

1. **Single-hand tracking** - No two-handed coordination yet
2. **2D coordinates only** - Depth (Z-axis) less reliable
3. **No facial tracking** - Missing non-manual markers
4. **No body context** - Signs relative to body not captured
5. **Primitive classification** - Handshape detection incomplete

### Not Suitable For

- ❌ Production sign language translation (needs extensions)
- ❌ Medical-grade hand tracking (not calibrated)
- ❌ Security applications (not designed for authentication)
- ❌ Real-time gaming (latency not optimized for gaming)

### Ethical Considerations

- **Deaf community involvement:** Sign language systems should be built **with** deaf people, not just **for** them
- **Cultural respect:** Sign languages are complete languages, not "gestures"
- **Accuracy requirements:** Motion similarity ≠ semantic correctness
- **Data privacy:** Recorded gestures may contain identifiable characteristics

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026 Richard Kabue Karoki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Author

**Richard Kabue Karoki**

- Email: <karokirichard522@gmail.com>
- Location: Nairobi, Kenya
- LinkedIn: [linkedin.com/in/richard-karoki-007](https://linkedin.com/in/richard-karoki-007)
- GitHub: [github.com/richiekaroki](https://github.com/richiekaroki)

**Education:** B.Sc. in Computer Technology, Jomo Kenyatta University of Agriculture and Technology (JKUAT), 2024

**Experience:**

- 4+ years backend and systems engineering
- Computer vision and ML projects
- Accessibility technology focus

---

## 🙏 Acknowledgments

### Technology

- **MediaPipe** (Google) - Hand tracking ML model
- **OpenCV** - Computer vision library
- **Python community** - Excellent ecosystem

### Inspiration

This project is inspired by the need for better sign language accessibility technology. The deaf community deserves translation systems that:

- Respect the linguistic complexity of their language
- Preserve semantic meaning, not just visual similarity
- Are built **with** deaf community input, not just **for** them

### References

- Sign language linguistics research
- Accessibility technology best practices
- Motion capture and animation systems

---

## 📞 Contact & Support

### For Technical Questions

- Open an issue on GitHub
- Email: <karokirichard522@gmail.com>

### For Collaboration

Interested in using this for sign language research? Building on this for accessibility tech? Want to contribute? Reach out!

### For Sign Language Expertise

If you're a member of the deaf community or a sign language expert and have feedback on this approach, I'd love to hear from you. This system should serve your community.

---

## 🌟 Project Status

**Current Phase:** Research & Development  
**Status:** Active Development  
**Version:** 0.6.0 (Dataset Complete)  
**Last Updated:** January 2026

**Recent Updates:**

- ✅ Enhanced UI with split-screen layout
- ✅ Comprehensive analysis toolkit
- ✅ Quality gesture dataset (45 recordings)
- ✅ Complete documentation suite

**Next Milestone:** Animation integration (Q1 2026)

---

<div align="center">

**If this project interests you, consider:**

- ⭐ Starring the repository
- 🔀 Forking for your own research
- 📧 Reaching out to collaborate
- 💬 Providing feedback

**Built with care for accessibility 🤝 | Respecting sign language as a complete language | Open to collaboration**

</div>

---

_This project treats motion as linguistic data, not just visual patterns. Every design decision prioritizes semantic accuracy and accessibility over visual effects._

_Last generated: January 2026_
