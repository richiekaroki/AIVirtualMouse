---
title: "I Built a Hand Motion Pipeline for Sign Language — Here's What I Learned"
published: false
description: "How I turned a gesture-controlled mouse into sign language infrastructure using MediaPipe, Python, and a three-layer architecture."
tags: python, accessibility, opencv, machinelearning
canonical_url: https://richiekaroki.github.io/AIVirtualMouse/
cover_image: https://dev-to-uploads.s3.amazonaws.com/uploads/articles/your-cover-image.png
---

# I Built a Hand Motion Pipeline for Sign Language — Here's What I Learned

**TL;DR:** I started building a gesture-controlled mouse cursor. It evolved into a full motion interpretation pipeline treating hand gestures as structured linguistic data — infrastructure for sign language translation.

---

## The Problem I Didn't Plan to Solve

Last year I built a simple Python script: move your index finger to control the mouse cursor, pinch to click. It was a weekend project. Fun, but not useful.

Then I asked: *what if this could do more than control a cursor?*

Sign language is motion. Handshapes, locations, movements, orientations — it's all captured by a webcam. But there's no clean pipeline that takes raw hand landmarks and turns them into structured, reusable data.

So I built one.

---

## The Architecture That Changed Everything

The key insight: **decouple motion capture from output actions.**

Most gesture projects work like this:

```
Camera → Detect Gesture → Do One Thing
```

I wanted this:

```
Camera → Detect Landmarks → Structure as Data → Do Anything
```

Here's what that looks like:

```text
┌──────────────────────────────────────┐
│        MOTION CAPTURE LAYER          │
│  MediaPipe Hands + OpenCV            │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│      ABSTRACTION LAYER (Core)        │
│  MotionDescriptor                    │
│  • Handshape classification          │
│  • Location tracking                 │
│  • Velocity & trajectory             │
│  • Primitive detection               │
└──────────────┬───────────────────────┘
               │
       ┌───────┼────────┬────────┬────────┐
       │       │        │        │        │
  ┌────▼───┐ ┌─▼──────┐ ┌▼──────┐ ┌▼──────┐
  │Cursor  │ │JSON    │ │Plots  │ │Future │
  │Control │ │Export  │ │       │ │3D Anim│
  └────────┘ └────────┘ └───────┘ └───────┘
```

The `MotionDescriptor` is the core. It takes 21 hand landmarks from MediaPipe and converts them into a structured dictionary:

```python
{
    "timestamp": 1735814400.123,
    "relative_time": 0.5,
    "hand": "right",
    "handshape_code": "01000",       # thumb=0, index=1, middle=0, ring=0, pinky=0
    "primitive": "POINT",            # classified gesture primitive
    "features": {
        "pinch_distance": 42.3,
        "hand_openness": 0.20,
        "hand_span": 156.7,
        "palm_center": {"x": 320.0, "y": 240.0}
    },
    "velocity": {
        "vx": 45.2,
        "vy": -12.8,
        "magnitude": 47.0,
        "direction": -0.28
    },
    "normalized": { ... }            # coordinates mapped to [0, 1]
}
```

Same data, four outputs: cursor control, JSON files, analysis plots, future animation.

---

## What I Built

### The Motion Descriptor

This is the abstraction that makes everything else possible. It classifies hand positions into gesture primitives:

| Fingers Extended | Primitive | What It Means |
|-----------------|-----------|---------------|
| `[0,1,0,0,0]` | `POINT` | Index finger only |
| `[1,1,1,1,1]` | `OPEN_HAND` | All fingers up |
| `[0,0,0,0,0]` | `FIST` | No fingers up |
| `[1,0,0,0,0]` | `THUMBS_UP` | Thumb only |
| `[0,1,1,0,0]` | `PEACE_V` | Index + middle |

These are building blocks — not complete signs. Real sign language needs sequences, context, and facial expressions. But you have to start somewhere.

### Real-Time Capture

The Enhanced UI shows everything happening in real-time:

- **Split screen**: camera feed on the left, analytics on the right
- **Skeleton overlay**: hand landmarks drawn with connections
- **Primitive timeline**: last 10 classified primitives in a scrolling list
- **Recording indicator**: red border, timer, frame count

### Data Pipeline

Every recording saves a JSON file with:

```json
{
    "metadata": {
        "gesture_name": "wave",
        "recorded_at": "2026-01-02T14:30:00",
        "duration_seconds": 2.5,
        "total_frames": 75,
        "average_fps": 30.0,
        "primitives_used": ["OPEN_HAND", "POINT"]
    },
    "frames": [ ... ]  // Array of motion descriptors
}
```

This is the kind of structured data ML models need. Not just "wave detected" — a full temporal sequence with timestamps, velocities, and intermediate states.

### Analysis Toolkit

After recording, I can generate:

- **Trajectory plots**: 2D path of the hand over time
- **Velocity profiles**: speed magnitude and direction
- **Primitive timelines**: when gestures transition
- **Distribution charts**: which primitives appear most

---

## The Hardest Parts

### 1. Co-articulation

Signs blend into each other. When you transition from POINT to OPEN_HAND, there's a frame where it's neither. My classifier handles this with an `UNKNOWN_*` fallback, but real sign language needs temporal context to resolve ambiguity.

**Solution (planned):** Sliding window classification that considers the last N frames, not just the current one.

### 2. Left vs Right Hand

MediaPipe detects both hands but doesn't label them. I added heuristic detection based on the relative position of the wrist and index finger MCP joint:

```python
def _detect_hand(self, lmList):
    wrist_x = lmList[0][1]
    index_mcp_x = lmList[5][1]
    return "right" if index_mcp_x > wrist_x else "left"
```

This works for self-view (mirrored) but needs refinement for production.

### 3. Memory Growth

`motion_history` is an unbounded list. A 10-minute recording at 30 FPS = 18,000 frames in memory. I added a `max_history` parameter that prunes old frames automatically:

```python
md = MotionDescriptor(max_history=1000)  # Keep last 1000 frames
```

### 4. Blocking Input

The original code called `input()` inside the video loop to ask for a gesture name. This froze the entire GUI. Fixed it with CLI arguments:

```bash
python -m hand_motion.apps.cursor_enhanced --gesture wave
```

---

## Why This Matters for Sign Language

Sign languages are built on four core parameters:

| Parameter | How I Capture It | Status |
|-----------|-----------------|--------|
| **Handshape** | Finger extension states, handshape codes | Implemented |
| **Location** | Normalized 2D coordinates | Implemented |
| **Movement** | Velocity vectors, trajectory analysis | Implemented |
| **Orientation** | Landmark directions | Partial |

The missing pieces:

- **Facial expressions** (non-manual markers) → MediaPipe Face Mesh
- **Body position** → MediaPipe Pose
- **Two-handed coordination** → Multi-hand tracking
- **3D orientation** → Depth estimation

These are all planned in my roadmap. The architecture is ready — the MotionDescriptor just needs more input channels.

---

## What's Next

**Phase 1 — Extended Capture:** MediaPipe Pose, Face Mesh, two-handed coordination

**Phase 2 — Linguistic Layer:** Gloss annotation tools, co-articulation modeling, deaf community validation

**Phase 3 — Animation:** Blender/Three.js rig integration, motion retargeting

**Phase 4 — Translation:** Text → gloss → motion synthesis pipeline

---

## Try It Yourself

```bash
git clone https://github.com/richiekaroki/AIVirtualMouse.git
cd AIVirtualMouse
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m hand_motion.apps.cursor_enhanced
```

**Controls:** R = record, S = stop & save, C = cancel, Q = quit

Requirements: Python 3.11+, webcam, Windows (PyAutoGUI is Windows-only for cursor control — the core pipeline works cross-platform).

---

## Key Takeaways

1. **Abstraction enables scale.** The MotionDescriptor turned a weekend project into a multi-output pipeline. Same capture, different uses.

2. **Sequences matter more than frames.** A sign isn't a pose — it's a motion pattern. Build for temporal data from day one.

3. **Structured data beats binary classification.** "Wave detected" is less useful than a full motion sequence with timestamps, velocities, and primitive transitions.

4. **Accessibility tech needs community input.** I'm building infrastructure, not a product. The deaf community should drive what matters.

---

## Links

- **GitHub:** [richiekaroki/AIVirtualMouse](https://github.com/richiekaroki/AIVirtualMouse)
- **Demo site:** [richiekaroki.github.io/AIVirtualMouse](https://richiekaroki.github.io/AIVirtualMouse/)
- **Author:** [Richard Kabue Karoki](https://github.com/richiekaroki) — Nairobi, Kenya

---

*Built with ❤️ for accessibility and linguistic preservation.*
