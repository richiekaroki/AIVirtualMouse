# Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│                                                                 │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐  │
│  │   Camera     │ →  │  MediaPipe  │ →  │   21 Landmarks   │  │
│  │  (OpenCV)    │    │   Hands     │    │   (x, y, z)      │  │
│  └──────────────┘    └─────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                             │
│                                                                 │
│  HandTrackingModule.py                                          │
│  ├── findHands()        - Detect and draw landmarks            │
│  ├── findPosition()     - Extract landmark coordinates         │
│  ├── fingersUp()        - Determine finger states              │
│  └── findDistance()     - Calculate landmark distances         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   ABSTRACTION LAYER                             │
│                                                                 │
│  MotionDescriptor.py                                            │
│  ├── create_descriptor() - Convert landmarks → structured data │
│  │   ├── Handshape encoding                                    │
│  │   ├── Primitive classification                              │
│  │   ├── Velocity calculation                                  │
│  │   └── Feature extraction                                    │
│  ├── get_motion_sequence() - Retrieve temporal history         │
│  └── save_sequence() - Export to JSON                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
┌───────────────────▼──┐    ┌───────────▼──────────────────────┐
│   REAL-TIME OUTPUT   │    │      OFFLINE OUTPUT              │
│                      │    │                                  │
│  • Cursor Control    │    │  • JSON Files (training data)   │
│  • Visual Feedback   │    │  • Analysis Plots               │
│  • Primitive Display │    │  • Statistical Reports          │
│  • Live Recording    │    │  • Comparison Tools             │
└──────────────────────┘    └──────────────────────────────────┘
```

## Data Flow

### 1. Capture Phase

```
Camera Frame (640x480)
     ↓
MediaPipe Processing
     ↓
21 Hand Landmarks [(id, x, y), ...]
     ↓
Finger States [thumb, index, middle, ring, pinky]
```

### 2. Structuring Phase

```
Raw Landmarks + Finger States
     ↓
MotionDescriptor.create_descriptor()
     ↓
Structured Motion Descriptor:
{
  timestamp: float,
  primitive: str,
  fingers_extended: [int],
  landmarks: {landmark: {x, y}},
  velocity: {vx, vy, magnitude},
  features: {openness, span, ...}
}
```

### 3. Output Phase

```
Motion Descriptor
     ↓
   ┌─┴─┐
   │   │
   ↓   ↓
Mouse   JSON Export
Action  ↓
        MotionAnalyzer
        ↓
        Visualizations
```

## Module Responsibilities

| Module                 | Responsibility         | Input        | Output                   |
| ---------------------- | ---------------------- | ------------ | ------------------------ |
| **HandTrackingModule** | Raw landmark detection | Video frames | Landmarks, finger states |
| **MotionDescriptor**   | Data structuring       | Landmarks    | Motion descriptors       |
| **AiVirtualMouse...**  | Application logic      | Descriptors  | UI, recording, control   |
| **MotionAnalyzer**     | Offline analysis       | JSON files   | Plots, statistics        |

## Key Design Decisions

### 1. Abstraction Layer (MotionDescriptor)

**Why:** Decouples motion capture from output actions.  
**Benefit:** Same data drives cursor, JSON, analysis, future animation.  
**Trade-off:** Added complexity, but essential for extensibility.

### 2. JSON Export Format

**Why:** Human-readable, version-controllable, language-agnostic.  
**Benefit:** Easy to process with any tool, shareable datasets.  
**Trade-off:** Larger file size than binary, but transparency matters more.

### 3. Primitive Classification

**Why:** Provides semantic labels for motion states.  
**Benefit:** Makes data understandable, enables sequence analysis.  
**Trade-off:** Current implementation is simplistic, needs refinement.

### 4. Temporal Sequences

**Why:** Signs are temporal, not static.  
**Benefit:** Captures motion history, enables co-articulation analysis.  
**Trade-off:** Memory usage grows with history length.

## Performance Characteristics

- **Latency:** < 33ms (30fps)
- **Memory:** ~50-100MB typical
- **CPU:** Moderate (MediaPipe optimized)
- **Storage:** ~5KB per second of recording (JSON)

## Extensibility Points

Future developers can extend:

1. **New Primitives:** Add to `MotionDescriptor._classify_primitive()`
2. **New Features:** Add to `MotionDescriptor._calculate_*()` methods
3. **New Outputs:** Consume motion descriptors in new applications
4. **New Analysis:** Add visualization methods to `MotionAnalyzer`

## Sign Language Integration Path

```
Current System (v0.6.0)
     ↓
+ MediaPipe Pose (body context)
     ↓
+ MediaPipe Face (facial expressions)
     ↓
+ Two-Hand Coordination
     ↓
+ 3D Animation System
     ↓
+ Deaf Community Validation
     ↓
Production Sign Language System
```

Each layer builds on the foundation established by the motion descriptor architecture.

---

_This architecture prioritizes extensibility, semantic accuracy, and respect for sign language as a complete linguistic system._
