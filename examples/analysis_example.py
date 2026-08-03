"""
Analysis Example - MotionAnalyzer Usage

This script demonstrates how to analyze recorded motion data:
1. Loading JSON files
2. Generating visualizations
3. Comparing gestures
"""

import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hand_motion.analyzer import MotionAnalyzer, GestureComparator


def create_sample_data():
    """Create sample motion data for analysis."""
    data = {
        "metadata": {
            "gesture_name": "wave",
            "recorded_at": "2026-01-15T10:00:00",
            "duration_seconds": 2.0,
            "total_frames": 60,
            "average_fps": 30.0,
            "primitives_used": ["OPEN_HAND", "POINT"]
        },
        "frames": []
    }

    # Generate sample frames
    import math
    for i in range(60):
        t = i / 30.0  # Time in seconds
        frame = {
            "timestamp": 1000000000 + t,
            "relative_time": t,
            "frame_num": i,
            "hand": "right",
            "fingers_extended": [1, 1, 1, 1, 1] if i % 10 < 5 else [0, 1, 0, 0, 0],
            "finger_count": 5 if i % 10 < 5 else 1,
            "handshape_code": "11111" if i % 10 < 5 else "01000",
            "landmarks": {
                "wrist": {"x": 300 + 50 * math.sin(t * 3), "y": 400},
                "thumb_tip": {"x": 250 + 50 * math.sin(t * 3), "y": 350},
                "index_tip": {"x": 350 + 50 * math.sin(t * 3), "y": 200},
                "middle_tip": {"x": 400 + 50 * math.sin(t * 3), "y": 180},
                "ring_tip": {"x": 450 + 50 * math.sin(t * 3), "y": 200},
                "pinky_tip": {"x": 500 + 50 * math.sin(t * 3), "y": 220}
            },
            "features": {
                "pinch_distance": 50.0,
                "hand_openness": 1.0 if i % 10 < 5 else 0.2,
                "hand_span": 150.0,
                "palm_center": {"x": 375 + 50 * math.sin(t * 3), "y": 350}
            },
            "primitive": "OPEN_HAND" if i % 10 < 5 else "POINT",
            "velocity": {
                "vx": 50 * math.cos(t * 3) * 3,
                "vy": 0,
                "magnitude": abs(50 * math.cos(t * 3) * 3),
                "direction": 0 if math.cos(t * 3) > 0 else math.pi
            } if i > 0 else None
        }
        data["frames"].append(frame)

    return data


def main():
    """Demonstrate motion analysis."""
    print("=" * 60)
    print("Motion Analysis Example")
    print("=" * 60)

    # Create sample data
    os.makedirs('motion_data', exist_ok=True)
    filepath = 'motion_data/wave_example.json'

    print("\n1. Creating sample motion data...")
    data = create_sample_data()
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"   Saved to: {filepath}")

    # Load and analyze
    print("\n2. Loading and analyzing...")
    analyzer = MotionAnalyzer(filepath)

    # Print summary
    print("\n3. Summary:")
    analyzer.print_summary()

    # Generate plots
    print("\n4. Generating plots...")
    output_dir = 'analysis_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    print("   Generating all plots...")
    analyzer.generate_all_plots(output_dir=output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete! Check analysis_plots/ directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
