"""
Basic Usage Example - Hand Motion Interpretation Pipeline

This script demonstrates the core functionality of the pipeline:
1. Creating motion descriptors from hand landmarks
2. Classifying gesture primitives
3. Saving motion data to JSON
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hand_motion import MotionDescriptor


def create_sample_landmarks():
    """Create sample hand landmarks for demonstration."""
    # Simulated 21-point hand landmarks [id, x, y]
    return [
        [0, 300, 400],   # Wrist
        [1, 280, 360],   # Thumb CMC
        [2, 260, 320],   # Thumb MCP
        [3, 240, 280],   # Thumb IP
        [4, 220, 240],   # Thumb Tip
        [5, 320, 280],   # Index MCP
        [6, 340, 240],   # Index PIP
        [7, 360, 200],   # Index DIP
        [8, 380, 160],   # Index Tip
        [9, 350, 280],   # Middle MCP
        [10, 370, 240],  # Middle PIP
        [11, 390, 200],  # Middle DIP
        [12, 410, 160],  # Middle Tip
        [13, 380, 300],  # Ring MCP
        [14, 400, 260],  # Ring PIP
        [15, 420, 220],  # Ring DIP
        [16, 440, 180],  # Ring Tip
        [17, 400, 320],  # Pinky MCP
        [18, 420, 280],  # Pinky PIP
        [19, 440, 240],  # Pinky DIP
        [20, 460, 200],  # Pinky Tip
    ]


def main():
    """Demonstrate basic pipeline usage."""
    print("=" * 60)
    print("Hand Motion Interpretation Pipeline - Basic Usage")
    print("=" * 60)

    # Initialize motion descriptor
    md = MotionDescriptor(max_history=100)

    # Create sample landmarks
    landmarks = create_sample_landmarks()

    # Simulate finger states (index finger up, others down)
    fingers = [0, 1, 0, 0, 0]  # [thumb, index, middle, ring, pinky]

    # Create descriptor
    print("\n1. Creating motion descriptor...")
    descriptor = md.create_descriptor(
        landmarks,
        fingers,
        frame_shape=(480, 640)
    )

    if descriptor:
        print(f"   Primitive: {descriptor['primitive']}")
        print(f"   Handshape: {descriptor['handshape_code']}")
        print(f"   Hand: {descriptor['hand']}")

    # Add more frames to build history
    print("\n2. Building motion history...")
    for i in range(5):
        # Slightly move the hand
        moved_landmarks = [[lm[0], lm[1] + i * 5, lm[2]] for lm in landmarks]
        md.create_descriptor(moved_landmarks, fingers)

    print(f"   History length: {len(md.motion_history)}")

    # Get statistics
    print("\n3. Motion statistics:")
    stats = md.get_statistics()
    print(f"   Total frames: {stats['total_frames']}")
    print(f"   Duration: {stats['duration_seconds']:.2f}s")
    print(f"   Primitives: {stats['primitive_counts']}")

    # Save to JSON
    print("\n4. Saving to JSON...")
    os.makedirs('motion_data', exist_ok=True)
    filepath = 'motion_data/example_gesture.json'
    if md.save_sequence(filepath, "example_gesture"):
        print(f"   Saved to: {filepath}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
