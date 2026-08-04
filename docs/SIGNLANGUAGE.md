# Sign Language Motion Capture - Project Evolution

## Overview

This project is evolving from a gesture-based cursor control system into sign language motion capture infrastructure. The core technology — real-time hand tracking using MediaPipe — directly transfers to capturing and structuring sign language gestures.

## Why This Project Matters for Sign Language

Sign languages are complete, complex languages with their own grammar and structure. They are built from precise combinations of:

### The Four Core Parameters of Sign Language

1. **Handshape** — Configuration of fingers and thumb
   - _Captured by:_ MediaPipe's 21 hand landmarks
   - _Status:_ Fully implemented (32 gesture primitives)

2. **Location** — Where the sign is performed (in space relative to body)
   - _Captured by:_ 2D coordinates of hand landmarks
   - _Status:_ Implemented with coordinate normalization

3. **Movement** — How the hand transitions through space
   - _Captured by:_ Temporal sequences of landmark positions
   - _Status:_ Implemented (velocity, trajectory, circle/wave/swipe detection)

4. **Orientation** — Direction palm/fingers face
   - _Captured by:_ Relationships between landmarks
   - _Status:_ Implicit in landmark data, not explicitly extracted

## Current Capabilities (v0.8.0)

### What This System Already Does

- Real-time hand tracking at 15 FPS via browser WebSocket
- 21 landmark extraction per hand with MediaPipe
- 32 gesture primitives (all finger combinations mapped)
- Motion gesture detection (circle, wave, swipe)
- ML gesture classification (RandomForest, 15 gestures, 64% accuracy)
- Dual classification (rule-based + ML) shown side-by-side
- JSON export of motion sequences
- Web-based interface (no installation required)
- Recording library with playback
- 142 unit tests covering all modules

### Integration Status

**Implemented:**
- Handshape parameter (finger states, 32 primitives)
- Location parameter (2D coordinates, normalized)
- Movement parameter (velocity, trajectory, circle/wave/swipe)
- Motion descriptor abstraction (structured data representation)
- JSON export pipeline (training data generation)
- ML gesture classifier (RandomForest, 78 features)
- Web-based real-time interface
- Render deployment (HTTPS, webcam access)

**Planned:**
- Orientation parameter (3D landmark analysis)
- MediaPipe Pose integration (body position context)
- MediaPipe Face Mesh (facial expressions/non-manual markers)
- Two-handed coordination tracking
- Animation system integration (3D rigs, keyframes)
- Deaf community validation framework
- Sign language gloss annotation tools

## Technical Challenges

### Challenge 1: Co-articulation

Signs don't have clear boundaries. The end position of one sign affects the start of the next.

**Approach:** Need linguistic context, not just motion continuity.

### Challenge 2: Semantic Accuracy vs. Visual Similarity

Two gestures can look similar but have different meanings based on subtle differences in orientation, location, or timing.

**Approach:** Require validation with deaf community, not just computer vision metrics.

### Challenge 3: Data Scarcity

Limited labeled sign language datasets, especially for African sign languages (Kenyan Sign Language).

**Approach:** Build data collection pipeline first, then ML models.

### Challenge 4: Real-time Performance

Translation must be fast enough for conversation (< 100ms latency).

**Approach:** Optimize pipeline, use efficient models, parallel processing.

## Project Roadmap

### Phase 1: Foundation (Complete)

- Real-time hand tracking
- Landmark extraction
- Basic gesture recognition
- Motion descriptor abstraction
- 32 gesture primitives

### Phase 2: Data Infrastructure (Complete)

- JSON export of motion sequences
- Gesture primitive classification
- Temporal sequence recording
- Motion visualization tools
- ML gesture classifier

### Phase 3: Web Platform (Complete)

- Browser-based webcam support
- Real-time WebSocket streaming
- Web UI with recording library
- Render deployment

### Phase 4: Extended Motion Capture (Planned)

- MediaPipe Pose integration
- Facial landmark tracking
- Two-handed gesture handling

### Phase 5: Translation Pipeline (Future)

- Sign language dataset collection
- ML model training (CNN+LSTM)
- Real-time translation pipeline
- Deaf community validation

## Applications

This motion capture pipeline supports:

1. **Sign Language Translation Systems**
   - Text/speech to sign language animation
   - Real-time translation for accessibility

2. **Sign Language Education**
   - Teaching tools with motion feedback
   - Standardized sign databases

3. **Research and Documentation**
   - Linguistic analysis of sign languages
   - Preservation of regional variations

4. **Accessibility Technology**
   - Video call sign language interpretation
   - Public service announcements in sign language

## Recommended Use Cases

1. Sign language motion research and analysis
2. Training data collection for ML models
3. Gesture recognition system prototyping
4. Accessibility technology development
5. Sign language translation (with extensions)

---

*For the complete technical system, see the main [README.md](README.md)*

## Contact

Richard Kabue Karoki
[karokirichard522@gmail.com](mailto:karokirichard522@gmail.com)
Nairobi, Kenya

---

*Last Updated: August 2026*
