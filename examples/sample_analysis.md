# Sample Analysis: "Point" Gesture

This document shows a complete analysis of a recorded "point" gesture.

## Recording Details

- **Gesture**: point
- **Duration**: 2.3 seconds
- **Frames**: 69
- **FPS**: 30.0
- **Recorded**: 2026-01-11

## Statistical Summary

### Primitive Distribution

- **POINT**: 69 frames (100.0%)

The gesture maintained a consistent pointing handshape throughout, with no transitions to other primitives. This indicates a stable, held gesture.

### Velocity Profile

- **Mean velocity**: 12.5 px/s
- **Max velocity**: 45.2 px/s
- **Min velocity**: 1.1 px/s
- **Std deviation**: 8.3 px/s

Low mean velocity with small standard deviation indicates a relatively static gesture with minimal movement—consistent with a held pointing position.

### Hand Openness

- **Mean**: 0.20
- **Range**: 0.20 - 0.20

Constant openness of 0.20 (one finger extended, four fingers closed) perfectly matches the pointing handshape.

## Visual Analysis

### Trajectory

The hand remained relatively stationary in screen space, with minor drift due to natural hand tremor. The starting and ending positions are within 20 pixels of each other, confirming this is a static gesture rather than a directional motion.

### Movement Pattern

Minimal movement throughout the sequence. Small oscillations visible in position-over-time plot represent natural hand steadiness, not intentional motion.

## Sign Language Interpretation

This recording captures a **static point gesture** with:

- ✅ Correct handshape (index finger extended)
- ✅ Stable hold (no unintended movement)
- ✅ Consistent form throughout duration

**Semantic accuracy**: High. The captured motion accurately represents a pointing gesture.

**Quality assessment**: Excellent recording suitable for training data.

---

Generated using MotionAnalyzer.py\_
