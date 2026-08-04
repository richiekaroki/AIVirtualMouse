"""Generate sample gesture recordings for demo purposes."""

import json
import math
import random
import time
from datetime import datetime
from pathlib import Path


GESTURE_PATTERNS = {
    "point": {"fingers": [0, 1, 0, 0, 0], "center_x": 320, "center_y": 240},
    "fist": {"fingers": [0, 0, 0, 0, 0], "center_x": 320, "center_y": 240},
    "open_hand": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
    "peace": {"fingers": [0, 1, 1, 0, 0], "center_x": 320, "center_y": 240},
    "thumbs_up": {"fingers": [1, 0, 0, 0, 0], "center_x": 320, "center_y": 200},
    "wave": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
    "pinch": {"fingers": [0, 1, 0, 0, 0], "center_x": 320, "center_y": 240},
    "swipe_right": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
    "swipe_left": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
    "circle": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
    "grab": {"fingers": [0, 0, 0, 0, 0], "center_x": 320, "center_y": 240},
    "two_fingers": {"fingers": [0, 1, 1, 0, 0], "center_x": 320, "center_y": 240},
    "three_fingers": {"fingers": [0, 1, 1, 1, 0], "center_x": 320, "center_y": 240},
    "ok_sign": {"fingers": [1, 1, 0, 0, 0], "center_x": 320, "center_y": 240},
    "push": {"fingers": [1, 1, 1, 1, 1], "center_x": 320, "center_y": 240},
}


def _generate_landmarks(center_x, center_y, fingers, frame_w=640, frame_h=480):
    """Generate 21 hand landmarks for a given finger configuration."""
    random.seed()
    jitter = lambda: random.uniform(-3, 3)

    wrist_x = center_x + jitter()
    wrist_y = center_y + 80 + jitter()

    thumb_base = (wrist_x - 40 + jitter(), wrist_y - 20 + jitter())
    index_base = (wrist_x - 15 + jitter(), wrist_y - 50 + jitter())
    middle_base = (wrist_x + 5 + jitter(), wrist_y - 55 + jitter())
    ring_base = (wrist_x + 25 + jitter(), wrist_y - 50 + jitter())
    pinky_base = (wrist_x + 40 + jitter(), wrist_y - 35 + jitter())

    lm = [
        [0, wrist_x, wrist_y],
        [1, thumb_base[0], thumb_base[1]],
        [2, thumb_base[0] - 15, thumb_base[1] - 25],
        [3, thumb_base[0] - 25, thumb_base[1] - 45],
        [4, thumb_base[0] - 30, thumb_base[1] - 65],
        [5, index_base[0], index_base[1]],
        [6, index_base[0] - 5, index_base[1] - 30],
        [7, index_base[0] - 5, index_base[1] - 55],
        [8, index_base[0] - 5, index_base[1] - 75],
        [9, middle_base[0], middle_base[1]],
        [10, middle_base[0], middle_base[1] - 30],
        [11, middle_base[0], middle_base[1] - 55],
        [12, middle_base[0], middle_base[1] - 75],
        [13, ring_base[0], ring_base[1]],
        [14, ring_base[0] + 5, ring_base[1] - 28],
        [15, ring_base[0] + 5, ring_base[1] - 50],
        [16, ring_base[0] + 5, ring_base[1] - 68],
        [17, pinky_base[0], pinky_base[1]],
        [18, pinky_base[0] + 8, pinky_base[1] - 18],
        [19, pinky_base[0] + 8, pinky_base[1] - 32],
        [20, pinky_base[0] + 8, pinky_base[1] - 42],
    ]

    tips_extended = {
        4: fingers[0],
        8: fingers[1],
        12: fingers[2],
        16: fingers[3],
        20: fingers[4],
    }
    for tip_id, extended in tips_extended.items():
        if not extended:
            lm[tip_id][1] = lm[tip_id - 1][1] + jitter()
            lm[tip_id][2] = lm[tip_id - 1][2] + 8 + jitter()

    return lm


def _motion_path(gesture, n_frames):
    cx, cy = 320, 240
    if gesture == "wave":
        return [(cx + 60 * math.sin(i * 0.5), cy) for i in range(n_frames)]
    if gesture.startswith("swipe"):
        direction = 1 if "right" in gesture else -1
        return [(cx + direction * i * 4, cy) for i in range(n_frames)]
    if gesture == "circle":
        return [(cx + 60 * math.cos(i * 0.3), cy + 60 * math.sin(i * 0.3)) for i in range(n_frames)]
    if gesture == "push":
        return [(cx, cy + i * 2) for i in range(n_frames)]
    if gesture == "grab":
        return [(cx, cy) for _ in range(n_frames)]
    return [(cx + random.uniform(-2, 2), cy + random.uniform(-2, 2)) for _ in range(n_frames)]


def generate_recording(gesture_name, duration_s=2.0, fps=30):
    n_frames = int(duration_s * fps)
    pattern = GESTURE_PATTERNS.get(gesture_name, GESTURE_PATTERNS["open_hand"])
    fingers = pattern["fingers"]
    path = _motion_path(gesture_name, n_frames)

    start_ts = time.time()
    frames = []
    for i in range(n_frames):
        cx, cy = path[i]
        lm = _generate_landmarks(cx, cy, fingers)

        open_count = sum(fingers)
        thumb_tip = lm[4]
        index_tip = lm[8]
        pinch_dist = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
        palm_x = (lm[0][1] + lm[9][1]) / 2
        palm_y = (lm[0][2] + lm[9][2]) / 2

        vel = {"dx": 0.0, "dy": 0.0, "magnitude": 0.0}
        if i > 0:
            prev = frames[-1]
            dt = 1.0 / fps
            dx = palm_x - prev["features"]["palm_center"]["x"]
            dy = palm_y - prev["features"]["palm_center"]["y"]
            vel = {"dx": dx, "dy": dy, "magnitude": math.hypot(dx, dy) / dt}

        frame = {
            "timestamp": start_ts + i / fps,
            "frame_index": i,
            "landmarks": lm,
            "finger_states": fingers,
            "handshape_code": "".join(str(f) for f in fingers),
            "hand": "right",
            "keypoints": {
                "wrist": {"x": lm[0][1], "y": lm[0][2]},
                "thumb_tip": {"x": lm[4][1], "y": lm[4][2]},
                "index_tip": {"x": lm[8][1], "y": lm[8][2]},
                "middle_tip": {"x": lm[12][1], "y": lm[12][2]},
                "ring_tip": {"x": lm[16][1], "y": lm[16][2]},
                "pinky_tip": {"x": lm[20][1], "y": lm[20][2]},
            },
            "features": {
                "pinch_distance": round(pinch_dist, 2),
                "hand_openness": open_count / 5.0,
                "hand_span": round(math.hypot(lm[4][1] - lm[20][1], lm[4][2] - lm[20][2]), 2),
                "palm_center": {"x": round(palm_x, 2), "y": round(palm_y, 2)},
            },
            "primitive": gesture_name.upper(),
            "velocity": vel,
        }
        frames.append(frame)

    return {
        "metadata": {
            "gesture_name": gesture_name,
            "recorded_at": datetime.now().isoformat(),
            "duration_seconds": duration_s,
            "total_frames": n_frames,
            "average_fps": fps,
            "primitives_used": [gesture_name.upper()],
            "custom": {"sample": True, "generated": True},
        },
        "frames": frames,
    }


def main():
    out_dir = Path("motion_data")
    out_dir.mkdir(exist_ok=True)

    for gesture in GESTURE_PATTERNS:
        data = generate_recording(gesture, duration_s=2.0, fps=30)
        fname = f"{gesture}_sample.json"
        with open(out_dir / fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Created {fname} ({data['metadata']['total_frames']} frames)")

    print(f"\nGenerated {len(GESTURE_PATTERNS)} sample recordings in {out_dir}/")


if __name__ == "__main__":
    main()
