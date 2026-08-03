"""
Quick single-gesture recording with timed capture.

Usage:
    python -m hand_motion.apps.record
"""

import cv2
import os
import time
import argparse

from hand_motion.detection import handDetector
from hand_motion.descriptor import MotionDescriptor


def main():
    parser = argparse.ArgumentParser(description="Record a single gesture")
    parser.add_argument("--name", "-n", type=str, help="Gesture name")
    parser.add_argument("--duration", "-d", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.7.0")
    args = parser.parse_args()

    gesture_name = args.name or input("Enter gesture name: ").strip()
    if not gesture_name:
        print("Invalid name. Exiting.")
        return

    duration = args.duration
    print(f"Recording '{gesture_name}' for {duration} seconds...")
    time.sleep(3)

    cap = cv2.VideoCapture(args.camera)
    detector = handDetector()
    motion_descriptor = MotionDescriptor()

    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')

    start_time = time.time()
    frame_count = 0

    print("RECORDING...")

    while time.time() - start_time < duration:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            fingers = detector.fingersUp()
            descriptor = motion_descriptor.create_descriptor(lmList, fingers)
            frame_count += 1

            remaining = duration - (time.time() - start_time)
            cv2.putText(img, f"Recording: {remaining:.1f}s", (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, f"Frames: {frame_count}", (10, 80),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            if descriptor:
                cv2.putText(img, f"Primitive: {descriptor['primitive']}", (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

        cv2.imshow("Recording", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        timestamp = int(time.time())
        filename = f"motion_data/{gesture_name}_{timestamp}.json"
        motion_descriptor.save_sequence(filename, gesture_name)
        print(f"Saved {frame_count} frames to {filename}")
    else:
        print("No frames recorded.")


if __name__ == "__main__":
    main()
