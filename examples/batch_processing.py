"""
Batch Processing Example - Multi-file Analysis

This script demonstrates how to process multiple motion files:
1. Batch analysis of JSON files
2. Dataset statistics
3. Quality metrics
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hand_motion.analyzer import MotionAnalyzer


def create_sample_dataset():
    """Create a sample dataset with multiple gestures."""
    os.makedirs('motion_data', exist_ok=True)

    gestures = ["point", "fist", "open_hand", "wave"]
    for gesture in gestures:
        for attempt in range(1, 4):
            data = {
                "metadata": {
                    "gesture_name": gesture,
                    "recorded_at": "2026-01-15T10:00:00",
                    "duration_seconds": 2.0,
                    "total_frames": 60,
                    "average_fps": 30.0,
                    "primitives_used": [gesture.upper()]
                },
                "frames": [
                    {
                        "timestamp": 1000000000 + i * 0.033,
                        "relative_time": i * 0.033,
                        "frame_num": i,
                        "hand": "right",
                        "fingers_extended": [1, 1, 1, 1, 1] if gesture == "open_hand" else [0, 1, 0, 0, 0],
                        "finger_count": 5 if gesture == "open_hand" else 1,
                        "handshape_code": "11111" if gesture == "open_hand" else "01000",
                        "landmarks": {
                            "wrist": {"x": 300, "y": 400},
                            "thumb_tip": {"x": 250, "y": 350},
                            "index_tip": {"x": 350, "y": 200},
                            "middle_tip": {"x": 400, "y": 180},
                            "ring_tip": {"x": 450, "y": 200},
                            "pinky_tip": {"x": 500, "y": 220}
                        },
                        "features": {
                            "pinch_distance": 50.0,
                            "hand_openness": 1.0 if gesture == "open_hand" else 0.2,
                            "hand_span": 150.0,
                            "palm_center": {"x": 375, "y": 350}
                        },
                        "primitive": gesture.upper(),
                        "velocity": {
                            "vx": 10.0,
                            "vy": 5.0,
                            "magnitude": 11.18,
                            "direction": 0.46
                        } if i > 0 else None
                    }
                    for i in range(60)
                ]
            }

            filepath = f'motion_data/{gesture}_{attempt}.json'
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

    return gestures


def analyze_dataset(data_dir: str):
    """Analyze all motion files in a directory."""
    results = []

    for filepath in Path(data_dir).glob("*.json"):
        try:
            analyzer = MotionAnalyzer(str(filepath))
            stats = analyzer.data['metadata']

            results.append({
                'file': filepath.name,
                'gesture': stats['gesture_name'],
                'frames': stats['total_frames'],
                'fps': stats['average_fps'],
                'duration': stats['duration_seconds']
            })
        except Exception as e:
            print(f"  Error analyzing {filepath}: {e}")

    return results


def main():
    """Demonstrate batch processing."""
    print("=" * 60)
    print("Batch Processing Example")
    print("=" * 60)

    # Create sample dataset
    print("\n1. Creating sample dataset...")
    gestures = create_sample_dataset()
    print(f"   Created {len(gestures)} gesture types")

    # Analyze dataset
    print("\n2. Analyzing dataset...")
    results = analyze_dataset('motion_data')

    # Print summary
    print("\n3. Dataset Summary:")
    print(f"   Total files: {len(results)}")
    print(f"   Gesture types: {len(set(r['gesture'] for r in results))}")

    # Group by gesture
    gesture_counts = {}
    for r in results:
        gesture = r['gesture']
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

    print("\n   Files per gesture:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"     {gesture}: {count}")

    print("\n" + "=" * 60)
    print("Batch processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
