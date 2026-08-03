"""
Guided batch recording for dataset creation.

Features:
- Predefined gesture list (15 gestures, 5 categories)
- Countdown timers
- Quality checks
- Recording manifest
"""

import cv2
import os
import time
import json
import argparse

from hand_motion.detection import handDetector
from hand_motion.descriptor import MotionDescriptor

GESTURE_CATEGORIES = {
    "Static Handshapes": [
        ("point", "Index finger extended, others closed"),
        ("fist", "All fingers closed"),
        ("open_hand", "All fingers extended and spread"),
        ("thumbs_up", "Thumb extended upward, others closed"),
        ("peace", "Index and middle extended (V-sign)"),
    ],
    "Dynamic Movements": [
        ("wave", "Open hand moving side-to-side"),
        ("circle", "Hand making circular motion in air"),
        ("swipe_right", "Hand moving smoothly left to right"),
    ],
    "Transitions": [
        ("open_close", "Hand opening and closing repeatedly"),
        ("point_fist", "Alternating between point and fist"),
    ],
    "Directional": [
        ("push_forward", "Hand moving away from body"),
        ("pull_back", "Hand moving toward body"),
        ("point_up", "Index finger pointing upward"),
    ],
    "Complex": [
        ("ok_sign", "Thumb and index forming circle, others extended"),
        ("pinch_release", "Thumb and index pinching together and releasing"),
    ]
}

RECORDING_DURATION = 3
COUNTDOWN_TIME = 3
ATTEMPTS_PER_GESTURE = 3


def countdown(seconds, message="Starting in"):
    for i in range(seconds, 0, -1):
        print(f"\r{message} {i}...", end='', flush=True)
        time.sleep(1)
    print(f"\r{message} NOW!     ")


def record_single_gesture(gesture_name, description, attempt_num, cap, detector, motion_descriptor):
    print(f"\n{'=' * 70}")
    print(f"Recording: {gesture_name} (Attempt {attempt_num}/{ATTEMPTS_PER_GESTURE})")
    print(f"Description: {description}")
    print(f"{'=' * 70}")

    print("\nGet ready...")
    countdown(COUNTDOWN_TIME, "Starting in")

    motion_descriptor.clear_history()
    start_time = time.time()
    frame_count = 0

    print("RECORDING... (Press 'q' to cancel)")

    while (time.time() - start_time) < RECORDING_DURATION:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        elapsed = time.time() - start_time
        remaining = RECORDING_DURATION - elapsed

        if len(lmList) != 0:
            fingers = detector.fingersUp()
            descriptor = motion_descriptor.create_descriptor(lmList, fingers)
            frame_count += 1

            cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(img, "RECORDING", (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, f"Time: {remaining:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"Frames: {frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if descriptor:
                cv2.putText(img, f"Primitive: {descriptor['primitive']}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            bar_width = 400
            progress = elapsed / RECORDING_DURATION
            filled_width = int(bar_width * progress)
            cv2.rectangle(img, (10, 460), (10 + bar_width, 480), (100, 100, 100), -1)
            cv2.rectangle(img, (10, 460), (10 + filled_width, 480), (0, 255, 0), -1)
        else:
            cv2.putText(img, "NO HAND DETECTED", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Batch Recording", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None

    if frame_count < 20:
        print(f"\nRecording too short ({frame_count} frames). Minimum 20 required.")
        return None

    timestamp = int(time.time())
    filename = f"motion_data/{gesture_name}_{attempt_num}_{timestamp}.json"
    motion_descriptor.save_sequence(filename, gesture_name,
                                    metadata={'attempt': attempt_num, 'description': description})

    stats = motion_descriptor.get_statistics()
    print(f"\nComplete! Frames: {stats['total_frames']}, FPS: {stats['average_fps']:.1f}")

    if stats['average_fps'] < 25:
        print("Warning: Low FPS")
    if stats['total_frames'] < 30:
        print("Warning: Too few frames")

    return filename


def main():
    parser = argparse.ArgumentParser(description="Batch gesture recording")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.7.0")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Batch Gesture Recording - Dataset Creation")
    print("=" * 70)

    total_gestures = sum(len(g) for g in GESTURE_CATEGORIES.values())
    total_recordings = total_gestures * ATTEMPTS_PER_GESTURE

    print(f"\n{total_gestures} gestures x {ATTEMPTS_PER_GESTURE} attempts = {total_recordings} recordings")
    print(f"~{RECORDING_DURATION}s per recording")

    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')

    cap = cv2.VideoCapture(args.camera)
    cap.set(3, 640)
    cap.set(4, 480)

    detector = handDetector()
    motion_descriptor = MotionDescriptor()

    completed_recordings = []
    current_num = 0

    try:
        for category_name, gestures in GESTURE_CATEGORIES.items():
            print(f"\n### {category_name} ###")

            for gesture_name, description in gestures:
                current_num += 1
                print(f"\n[{current_num}/{total_gestures}] {gesture_name}")

                for attempt in range(1, ATTEMPTS_PER_GESTURE + 1):
                    filename = record_single_gesture(
                        gesture_name, description, attempt,
                        cap, detector, motion_descriptor
                    )
                    if filename:
                        completed_recordings.append(filename)
                    if attempt < ATTEMPTS_PER_GESTURE:
                        time.sleep(2)

                if current_num < total_gestures:
                    choice = input("\nContinue? (y/n): ").lower()
                    if choice == 'n':
                        break
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    manifest = {
        'session_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_recordings': len(completed_recordings),
        'recordings': completed_recordings,
        'categories': {cat: [g[0] for g in gestures] for cat, gestures in GESTURE_CATEGORIES.items()}
    }

    with open('motion_data/recording_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(completed_recordings)}/{total_recordings} recordings completed")
    print("Manifest saved to motion_data/recording_manifest.json")


if __name__ == "__main__":
    main()
