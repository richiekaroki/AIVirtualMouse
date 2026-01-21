# Motion Analysis Guide

This guide explains how to analyze recorded gesture sequences using the `MotionAnalyzer.py` tool.

## Overview

The Motion Analyzer provides comprehensive offline analysis of recorded gestures:

- **Trajectory plots**: Visualize hand movement paths
- **Primitive timelines**: See how gestures evolve over time
- **Velocity profiles**: Analyze motion speed and dynamics
- **Hand openness**: Track open/closed hand transitions
- **Statistical summaries**: Numerical analysis of gesture properties
- **Comparison tools**: Side-by-side analysis of different attempts

## Basic Usage

### Analyze a Single Gesture

```bash
python MotionAnalyzer.py motion_data/your_gesture.json
```

This will:

1. Print statistical summary to console
2. Display all 5 visualization plots
3. Wait for you to close each plot before showing the next

### Generate Specific Plot Only

```bash
# Trajectory only
python MotionAnalyzer.py motion_data/gesture.json --plot trajectory

# Primitives only
python MotionAnalyzer.py motion_data/gesture.json --plot primitives

# Velocity only
python MotionAnalyzer.py motion_data/gesture.json --plot velocity

# Hand openness only
python MotionAnalyzer.py motion_data/gesture.json --plot openness

# Distribution only
python MotionAnalyzer.py motion_data/gesture.json --plot distribution
```

### Save Plots to Files

```bash
# Save all plots to directory
python MotionAnalyzer.py motion_data/gesture.json --output analysis_plots/

# This creates:
#   analysis_plots/gesture_trajectory.png
#   analysis_plots/gesture_primitives.png
#   analysis_plots/gesture_velocity.png
#   analysis_plots/gesture_openness.png
#   analysis_plots/gesture_distribution.png
```

### Compare Two Gestures

```bash
python MotionAnalyzer.py gesture1.json gesture2.json --compare
```

Useful for:

- Comparing different attempts at the same sign
- Analyzing variation between signers
- Checking consistency over time

### Understanding the Plots

1. Trajectory Plot
   What it shows: The 2D path your hand took during the gesture.
   Key insights:

Green circle = Starting position
Red X = Ending position
Blue arrows = Direction of movement
Right panel = X and Y coordinates over time

Sign language relevance: Movement is one of the 4 core parameters. The trajectory shows spatial relationships and motion direction.
Example interpretation:

Straight line = directional sign (pointing, pushing away)
Circular path = circular motion sign
Back-and-forth = repetitive motion

2. Primitive Timeline
   What it shows: Which motion primitive was detected at each point in time.
   Key insights:

Horizontal axis = time progression
Vertical axis = different primitives
Color = specific primitive type
Transitions = where handshape changed

Sign language relevance: Signs are sequences of primitives. The transitions between primitives often carry meaning.
Example interpretation:

Flat horizontal line = held steady handshape
Frequent changes = dynamic gesture
Repeated pattern = possible sign repetition

3. Velocity Profile
   What it shows: How fast your hand was moving throughout the gesture.
   Top panel:

Blue line = Speed (magnitude)
Filled area = Visual emphasis
Red dashed line = Average speed

Bottom panel:

Red line = Horizontal (X) velocity
Green line = Vertical (Y) velocity
Zero line = No movement in that direction

Sign language relevance: Movement speed is linguistically significant. Fast vs. slow can change meaning.
Example interpretation:

High velocity peak = quick motion (snap, flick)
Low flat velocity = slow, deliberate movement
Multiple peaks = repetitive motion

4. Hand Openness
   What it shows: How open (1.0) or closed (0.0) your hand was over time.
   Key insights:

Purple line = Openness value
Reference lines at 0.0, 0.5, 1.0
Smooth transitions vs. sharp changes

Sign language relevance: Handshape transitions are critical. Gradual vs. sudden changes matter.
Example interpretation:

Stays near 1.0 = open hand throughout
Stays near 0.0 = fist throughout
Transitions 1.0 → 0.0 = closing hand
Oscillates = opening and closing repeatedly

5. Primitive Distribution
   What it shows: How much time was spent in each primitive state.
   Left panel (bar chart):

Height = number of frames
Numbers on bars = exact frame count

Right panel (pie chart):

- Percentage breakdown
- Visual proportion

Sign language relevance: Dominant primitives indicate the core handshape of the sign.
Example interpretation:

One dominant primitive = static sign
Even distribution = transitional gesture
Two alternating = binary motion (open/close)

Statistical Summary
When you run the analyzer, it prints:
==================================================================
Motion Analysis: wave
==================================================================

Metadata:
Recorded: 2026-01-11T14:30:15.123456
Duration: 2.50 seconds
Frames: 75
Average FPS: 30.0

Primitive Analysis:
Unique primitives: 2
OPEN_HAND : 65 frames ( 86.7%)
FIST : 10 frames ( 13.3%)

Velocity Analysis:
Mean velocity: 45.23 px/s
Max velocity: 120.45 px/s
Min velocity: 2.10 px/s
Std deviation: 28.34 px/s

Hand Openness:
Mean: 0.87
Range: 0.00 - 1.00

==================================================================
What to look for:

Duration: Should match your intended gesture length
FPS: Should be ~30 (consistent capture rate)
Primitives: Are they what you expected?
Velocity: High variance = dynamic gesture, low variance = steady
Openness: Matches your hand state?
