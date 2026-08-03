"""
Enhanced UI version of the cursor control application.

Features:
- Split-screen layout (video + analysis panel)
- Hand skeleton overlay
- Primitive sequence timeline
- Recording progress indicator
- Session statistics
"""

import cv2
import numpy as np
import time
import pyautogui
import os
import sys
import logging
import argparse
from collections import deque

from hand_motion.detection import handDetector
from hand_motion.descriptor import MotionDescriptor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Hand Motion Pipeline - Enhanced UI")
    parser.add_argument("--gesture", "-g", type=str, default=None,
                        help="Gesture name to record")
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


# Colors (BGR)
COLOR_PRIMARY = (255, 140, 0)
COLOR_SUCCESS = (0, 255, 0)
COLOR_DANGER = (0, 0, 255)
COLOR_INFO = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG_DARK = (40, 40, 40)
COLOR_BG_LIGHT = (60, 60, 60)
COLOR_SKELETON = (0, 255, 255)


def create_info_panel(width, height):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = COLOR_BG_DARK
    return panel


def draw_section_header(panel, text, y_pos, color=COLOR_PRIMARY):
    cv2.rectangle(panel, (0, y_pos), (panel.shape[1], y_pos + 35), COLOR_BG_LIGHT, -1)
    cv2.putText(panel, text, (15, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return y_pos + 40


def draw_recording_header(panel, recording, elapsed_time):
    if recording:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 50), COLOR_DANGER, -1)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(panel, (30, 25), 12, (255, 255, 255), -1)
        cv2.putText(panel, "RECORDING", (55, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        cv2.putText(panel, f"{elapsed_time:.1f}s", (panel.shape[1] - 80, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    else:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 50), COLOR_SUCCESS, -1)
        cv2.circle(panel, (30, 25), 12, (255, 255, 255), 2)
        cv2.putText(panel, "READY", (55, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    return 55


def draw_primitive_timeline(panel, primitive_history, y_start):
    y_pos = draw_section_header(panel, "RECENT SEQUENCE", y_start)
    if not primitive_history:
        cv2.putText(panel, "No motion detected", (15, y_pos + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return y_pos + 80

    primitive_colors = {
        'POINT': (255, 140, 0), 'OPEN_HAND': (0, 255, 0), 'FIST': (0, 0, 255),
        'PEACE_V': (255, 0, 255), 'THUMBS_UP': (0, 255, 255), 'PINCH_READY': (255, 255, 0),
    }

    for i, primitive in enumerate(reversed(list(primitive_history))):
        y = y_pos + i * 30
        color = primitive_colors.get(primitive, (150, 150, 150))
        cv2.rectangle(panel, (15, y), (135, y + 25), color, -1)
        cv2.putText(panel, primitive[:12], (145, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)

    return y_pos + len(primitive_history) * 30 + 20


def draw_motion_analysis(panel, descriptor, y_start):
    if not descriptor:
        return y_start
    y_pos = draw_section_header(panel, "MOTION ANALYSIS", y_start)

    primitive = descriptor['primitive']
    prim_color = COLOR_SUCCESS if primitive != "UNKNOWN" else (150, 150, 150)
    cv2.putText(panel, "Primitive", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, primitive, (15, y_pos + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prim_color, 2)
    y_pos += 50

    cv2.putText(panel, "Handshape", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, descriptor['handshape_code'], (15, y_pos + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    y_pos += 50

    openness = descriptor['features']['hand_openness']
    cv2.putText(panel, "Openness", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"{openness:.2f}", (15, y_pos + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_INFO, 2)
    y_pos += 50

    if descriptor['velocity']:
        vel = descriptor['velocity']['magnitude']
        cv2.putText(panel, "Velocity", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"{vel:.1f} px/s", (15, y_pos + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        y_pos += 50

    return y_pos


def draw_session_stats(panel, stats, y_start):
    y_pos = draw_section_header(panel, "SESSION STATS", y_start)
    elapsed = time.time() - stats['session_start']

    cv2.putText(panel, f"Gestures: {stats['gestures_recorded']}",
                (15, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    cv2.putText(panel, f"Total Frames: {stats['total_frames']}",
                (15, y_pos + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    cv2.putText(panel, f"Session Time: {int(elapsed)}s",
                (15, y_pos + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
    return y_pos + 95


def draw_controls_help(panel, y_start):
    y_pos = draw_section_header(panel, "CONTROLS", y_start, COLOR_INFO)
    for key, desc in [("R", "Start recording"), ("S", "Stop & save"),
                      ("C", "Cancel recording"), ("Q", "Quit application")]:
        cv2.rectangle(panel, (15, y_pos), (45, y_pos + 25), COLOR_PRIMARY, -1)
        cv2.putText(panel, key, (22, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        cv2.putText(panel, desc, (55, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 35
    return y_pos


def draw_hand_skeleton(img, lmList):
    if len(lmList) < 21:
        return
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)
    ]
    for start_idx, end_idx in connections:
        if start_idx < len(lmList) and end_idx < len(lmList):
            pt1 = (lmList[start_idx][1], lmList[start_idx][2])
            pt2 = (lmList[end_idx][1], lmList[end_idx][2])
            cv2.line(img, pt1, pt2, COLOR_SKELETON, 2)
    for lm in lmList:
        cv2.circle(img, (lm[1], lm[2]), 5, COLOR_PRIMARY, -1)
        cv2.circle(img, (lm[1], lm[2]), 7, COLOR_SKELETON, 2)


def main():
    args = parse_args()
    CAMERA_WIDTH, CAMERA_HEIGHT = args.width, args.height
    PANEL_WIDTH = 400
    frameR = 100
    smoothening = 10

    pTime = 0
    plocX, plocY = 0, 0

    cap = cv2.VideoCapture(args.camera)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open camera %d", args.camera)
        sys.exit(1)

    detector = handDetector()
    motion_descriptor = MotionDescriptor()
    wScr, hScr = pyautogui.size()

    recording_mode = False
    gesture_name = None
    recording_start_time = None
    primitive_history = deque(maxlen=10)
    session_stats = {'gestures_recorded': 0, 'total_frames': 0, 'session_start': time.time()}

    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        info_panel = create_info_panel(PANEL_WIDTH, CAMERA_HEIGHT)

        img = detector.findHands(img, draw=False)
        lmList, bbox = detector.findPosition(img, draw=False)

        current_descriptor = None

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            fingers = detector.fingersUp()
            current_descriptor = motion_descriptor.create_descriptor(
                lmList, fingers, frame_shape=(CAMERA_HEIGHT, CAMERA_WIDTH)
            )

            if current_descriptor:
                primitive_history.append(current_descriptor['primitive'])
                session_stats['total_frames'] += 1

            draw_hand_skeleton(img, lmList)

            if not recording_mode:
                cv2.rectangle(img, (frameR, frameR), (CAMERA_WIDTH - frameR, CAMERA_HEIGHT - frameR),
                              COLOR_PRIMARY, 2)
                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (frameR, CAMERA_WIDTH - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, CAMERA_HEIGHT - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    pyautogui.moveTo(wScr - clocX, clocY)
                    cv2.circle(img, (x1, y1), 12, COLOR_PRIMARY, -1)
                    plocX, plocY = clocX, clocY

                if fingers[1] == 1 and fingers[2] == 1:
                    length, img, lineInfo = detector.findDistance(8, 12, img, draw=False)
                    if length < 40:
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, COLOR_SUCCESS, -1)
                        pyautogui.click()
                        time.sleep(0.2)
            else:
                cv2.rectangle(img, (0, 0), (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1), COLOR_DANGER, 5)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.rectangle(img, (5, 5), (100, 35), (0, 0, 0), -1)
        cv2.putText(img, f"FPS: {int(fps)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SUCCESS, 2)

        y_position = 0
        elapsed_time = time.time() - recording_start_time if recording_start_time else 0
        y_position = draw_recording_header(info_panel, recording_mode, elapsed_time)
        y_position = draw_primitive_timeline(info_panel, primitive_history, y_position)
        y_position = draw_motion_analysis(info_panel, current_descriptor, y_position)
        y_position = draw_session_stats(info_panel, session_stats, y_position)
        y_position = draw_controls_help(info_panel, y_position)

        combined_frame = np.hstack([img, info_panel])
        cv2.imshow("Hand Motion Pipeline - Enhanced", combined_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording_mode:
            gesture_name = get_gesture_name(args.gesture)
            recording_mode = True
            recording_start_time = time.time()
            motion_descriptor.clear_history()
            logger.info("Recording '%s'...", gesture_name)

        elif key == ord('s') and recording_mode:
            recording_mode = False
            recording_start_time = None
            if motion_descriptor.motion_history:
                timestamp = int(time.time())
                filename = f"motion_data/{gesture_name}_{timestamp}.json"
                motion_descriptor.save_sequence(filename, gesture_name)
                session_stats['gestures_recorded'] += 1
                logger.info("Saved to %s", filename)
            gesture_name = None

        elif key == ord('c') and recording_mode:
            recording_mode = False
            recording_start_time = None
            motion_descriptor.clear_history()
            logger.info("Recording cancelled")
            gesture_name = None

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    logger.info("Gestures recorded: %d | Frames: %d | Duration: %ds",
                session_stats['gestures_recorded'], session_stats['total_frames'],
                int(time.time() - session_stats['session_start']))


if __name__ == "__main__":
    main()
