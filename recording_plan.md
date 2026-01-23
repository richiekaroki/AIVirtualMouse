# Gesture Recording Plan

## Objective

Create a high-quality dataset of 15 gestures with 3 clean attempts each (45 total recordings).

## Gesture List

### 1. Static Handshapes (5 gestures)

- **point**: Index finger extended, others closed
- **fist**: All fingers closed
- **open_hand**: All fingers extended
- **thumbs_up**: Thumb extended, others closed
- **peace**: Index and middle extended (V-sign)

### 2. Dynamic Movements (3 gestures)

- **wave**: Open hand moving side-to-side
- **circle**: Hand making circular motion
- **swipe_right**: Hand moving left to right

### 3. Transitions (2 gestures)

- **open_close**: Hand opening and closing repeatedly
- **point_fist**: Alternating between point and fist

### 4. Directional (3 gestures)

- **push_forward**: Hand moving away from body
- **pull_back**: Hand moving toward body
- **point_up**: Pointing upward motion

### 5. Complex (2 gestures)

- **ok_sign**: Thumb and index forming circle
- **pinch_release**: Thumb and index pinching and releasing

## Recording Guidelines

### Before Recording

- [ ] Good lighting (no shadows on hand)
- [ ] Clear background
- [ ] Camera stable and at proper distance
- [ ] Hand fully in frame throughout gesture

### During Recording

- [ ] Hold static gestures for 2-3 seconds
- [ ] Perform dynamic gestures smoothly
- [ ] Complete transitions fully
- [ ] Keep hand in camera view

### After Recording

- [ ] Review JSON file generated
- [ ] Check primitive classification
- [ ] Verify frame count (60+ for most gestures)
- [ ] Analyze with MotionAnalyzer.py

### Quality Criteria

✅ **Good Recording:**

- Consistent FPS (~30)
- No tracking loss
- Clear primitive detection
- Smooth motion (for dynamic gestures)

❌ **Discard if:**

- Hand leaves frame
- Tracking jumps/glitches
- Too short (< 1.5 seconds)
- Wrong gesture performed

## Recording Schedule

**Session 1: Static Handshapes (30 min)**

- Record each gesture 3 times
- Analyze after each set
- Re-record if quality issues

**Session 2: Dynamic & Transitions (30 min)**

- Record each gesture 3 times
- Focus on smooth motion
- Check velocity profiles

**Session 3: Directional & Complex (30 min)**

- Record each gesture 3 times
- Verify semantic accuracy
- Final quality check

**Session 4: Analysis & Cleanup (30 min)**

- Analyze all recordings
- Generate plots for best attempts
- Document findings

## Expected Outputs

```
motion_data/
├── point_1.json, point_2.json, point_3.json
├── fist_1.json, fist_2.json, fist_3.json
├── open_hand_1.json, open_hand_2.json, open_hand_3.json
├── ... (15 gestures × 3 attempts = 45 files)

analysis_plots/
├── point_1_trajectory.png
├── point_1_primitives.png
├── ... (best attempts visualized)

dataset_summary.md (generated after completion)
```

## Success Metrics

- ✅ 45 clean recordings (15 gestures × 3 attempts)
- ✅ Average FPS > 28
- ✅ < 5% discard rate
- ✅ All primitives represented
- ✅ Comprehensive analysis plots
