"""
Hand Landmark Detection Module

Wraps cvzone's HandDetector (MediaPipe-based) for 21-point hand tracking.
Provides finger state detection and inter-landmark distance calculations.
"""

import math
import time
import cv2
from cvzone.HandTrackingModule import HandDetector as CvzoneHandDetector


class handDetector:
    """Hand tracking detector compatible with Python 3.13+"""

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.detector = CvzoneHandDetector(
            detectionCon=detectionCon,
            maxHands=maxHands
        )

        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        """Detect hands in image and optionally draw landmarks."""
        hands, img = self.detector.findHands(img, draw=draw)
        self.results = hands
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Extract landmark positions from detected hand.

        Returns:
            lmList: [[id, x, y], ...] - 21 landmarks per hand
            bbox: [xmin, ymin, xmax, ymax] - bounding box
        """
        self.lmList = []
        bbox = []

        if self.results and len(self.results) > handNo:
            hand = self.results[handNo]
            lmList = hand["lmList"]
            self.lmList = [[i, lm[0], lm[1]] for i, lm in enumerate(lmList)]

            if "bbox" in hand:
                bbox_xywh = hand["bbox"]
                bbox = [
                    bbox_xywh[0],
                    bbox_xywh[1],
                    bbox_xywh[0] + bbox_xywh[2],
                    bbox_xywh[1] + bbox_xywh[3]
                ]

        return self.lmList, bbox

    def fingersUp(self):
        """Detect which fingers are extended (0=down, 1=up). Returns [thumb, index, middle, ring, pinky]."""
        fingers = []

        if not self.results or len(self.results) == 0:
            return fingers

        hand = self.results[0]
        fingers = self.detector.fingersUp(hand)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Calculate distance between two landmarks.

        Returns:
            length: Euclidean distance
            img: Image with optional drawings
            lineInfo: [x1, y1, x2, y2, cx, cy]
        """
        if len(self.lmList) == 0:
            return 0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


if __name__ == "__main__":
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            fingers = detector.fingersUp()
            cv2.putText(img, f"Fingers: {fingers}", (10, 110),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracking Test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
