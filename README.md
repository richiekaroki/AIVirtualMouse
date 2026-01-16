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

## About

Real-time hand motion capture and interpretation system built with MediaPipe and OpenCV. Demonstrates:

- ✅ 21-point hand landmark tracking at 30fps
- ✅ Gesture-based cursor control (index finger = move, two fingers = click)
- ✅ Motion smoothing and coordinate normalization
- ✅ Modular architecture (HandTrackingModule)
- 🔄 Being extended for sign language motion capture ([details](SIGNLANGUAGE.md))

**Technical Foundation:** Computer vision pipeline that captures hand motion as structured data, enabling multiple downstream applications beyond cursor control.
