"""
Gesture-based cursor control with motion recording.

Controls:
    Mouse Mode (default):
        - Index finger up = move cursor
        - Index + middle up = click
    Recording Mode:
        - Press 'r' = start recording
        - Press 's' = stop and save
        - Press 'c' = cancel recording
        - Press 'q' = quit application
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import time
import pyautogui
import sys
import logging
import argparse

from hand_motion.detection import handDetector
from hand_motion.descriptor import MotionDescriptor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Motion Interpretation Pipeline")
    parser.add_argument("--gesture", "-g", type=str, default=None,
                        help="Gesture name to record (skips interactive prompt)")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--version", "-v", action="version", version="%(prog)s 0.7.0")
    return parser.parse_args()


def get_gesture_name(cli_name=None):
    if cli_name:
        return cli_name
    return f"gesture_{int(time.time())}"


def draw_recording_indicator(img, recording, gesture_name=None, frame_count=0):
    if recording:
        cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
        cv2.putText(img, "RECORDING", (55, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        if gesture_name:
            cv2.putText(img, f"Gesture: {gesture_name}", (10, 120),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f"Frames: {frame_count}", (10, 150),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        cv2.circle(img, (30, 30), 15, (0, 255, 0), 2)
        cv2.putText(img, "READY", (55, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


def draw_primitive_info(img, descriptor, hCam):
    if descriptor:
        primitive = descriptor['primitive']
        cv2.putText(img, f"Primitive: {primitive}", (10, hCam - 90),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        handshape = descriptor['handshape_code']
        cv2.putText(img, f"Handshape: {handshape}", (10, hCam - 60),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        openness = descriptor['features']['hand_openness']
        cv2.putText(img, f"Openness: {openness:.2f}", (10, hCam - 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        if descriptor['velocity']:
            vel = descriptor['velocity']['magnitude']
            cv2.putText(img, f"Velocity: {vel:.1f}", (10, hCam - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)


def draw_controls_help(img, wCam):
    help_text = ["Controls:", "R - Start recording", "S - Stop & save",
                 "C - Cancel recording", "Q - Quit"]
    for i, text in enumerate(help_text):
        color = (200, 200, 200) if i == 0 else (150, 150, 150)
        cv2.putText(img, text, (wCam - 180, 180 + i * 25),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)


def main():
    args = parse_args()
    wCam, hCam = args.width, args.height
    frameR = 100
    smoothening = 5

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = True

    pTime = 0
    plocX, plocY = 0, 0

    cap = cv2.VideoCapture(args.camera)
    cap.set(3, wCam)
    cap.set(4, hCam)

    if not cap.isOpened():
        logger.error("Cannot open camera %d", args.camera)
        sys.exit(1)

    detector = handDetector(detectionCon=0.7)
    motion_descriptor = MotionDescriptor()
    wScr, hScr = pyautogui.size()

    recording_mode = False
    gesture_name = None
    frame_count = 0

    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')

    print("\n" + "=" * 60)
    print("Hand Motion Interpretation Pipeline")
    print("=" * 60)
    print("\nControls:")
    print("  R - Start recording | S - Stop & save | C - Cancel | Q - Quit")
    print("  Index finger up -> Move cursor | Index + Middle up -> Click")
    print("=" * 60 + "\n")

    while True:
        success, img = cap.read()
        if not success:
            logger.warning("Failed to capture frame")
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            fingers = detector.fingersUp()
            descriptor = motion_descriptor.create_descriptor(lmList, fingers, frame_shape=(hCam, wCam))
            draw_primitive_info(img, descriptor, hCam)

            if not recording_mode:
                cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    pyautogui.moveTo(wScr - clocX, clocY)
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    plocX, plocY = clocX, clocY

                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(8, 12, img)
                    if length < 40:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()
                        time.sleep(0.05)
            else:
                frame_count += 1
                cv2.rectangle(img, (0, 0), (wCam, hCam), (0, 0, 255), 5)
        else:
            if not recording_mode:
                cv2.putText(img, "No hand detected", (10, hCam - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        draw_recording_indicator(img, recording_mode, gesture_name, frame_count)
        draw_controls_help(img, wCam)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Hand Motion Pipeline", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording_mode:
            gesture_name = get_gesture_name(args.gesture)
            recording_mode = True
            frame_count = 0
            motion_descriptor.clear_history()
            logger.info("Recording '%s'... (Press 'S' to stop, 'C' to cancel)", gesture_name)

        elif key == ord('s') and recording_mode:
            recording_mode = False
            if frame_count > 0:
                timestamp = int(time.time())
                filename = f"motion_data/{gesture_name}_{timestamp}.json"
                motion_descriptor.save_sequence(filename, gesture_name)
                stats = motion_descriptor.get_statistics()
                logger.info("Saved: %s | %.2fs | %d frames", filename,
                            stats['duration_seconds'], stats['total_frames'])
            gesture_name = None
            frame_count = 0

        elif key == ord('c') and recording_mode:
            recording_mode = False
            motion_descriptor.clear_history()
            logger.info("Recording cancelled")
            gesture_name = None
            frame_count = 0

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if motion_descriptor.motion_history:
        stats = motion_descriptor.get_statistics()
        logger.info("Session: %d frames, %.2fs, %d primitives",
                     stats['total_frames'], stats['duration_seconds'],
                     stats['unique_primitives'])


if __name__ == "__main__":
    main()
