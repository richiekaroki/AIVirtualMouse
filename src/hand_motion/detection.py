"""
Hand Landmark Detection Module

Wraps cvzone's HandDetector (MediaPipe-based) for 21-point hand tracking.
Provides finger state detection and inter-landmark distance calculations.
"""

import math
import time
import logging
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector as CvzoneHandDetector

logger = logging.getLogger(__name__)

# Type aliases
LandmarkList = List[List[float]]
BoundingBox = List[int]


class HandDetector:
    """
    Hand tracking detector compatible with Python 3.13+

    Attributes:
        mode: Detection mode
        maxHands: Maximum number of hands to detect
        detectionCon: Detection confidence threshold
        trackCon: Tracking confidence threshold
        tipIds: Landmark IDs for fingertips
    """

    __slots__ = (
        'mode', 'maxHands', 'detectionCon', 'trackCon',
        'detector', 'tipIds', 'results', 'lmList', 'handedness'
    )

    def __init__(
        self,
        mode: bool = True,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5
    ) -> None:
        """
        Initialize hand detector.

        Args:
            mode: Static image mode (True) or video mode (False).
                  True treats each frame independently (no timestamp dependency).
                  False requires monotonically increasing timestamps.
            maxHands: Maximum number of hands to detect (1-2)
            detectionCon: Detection confidence threshold (0.0-1.0)
            trackCon: Tracking confidence threshold (0.0-1.0)
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.detector = CvzoneHandDetector(
            staticMode=mode,
            detectionCon=detectionCon,
            maxHands=maxHands
        )

        self.tipIds: List[int] = [4, 8, 12, 16, 20]
        self.results: Optional[List[Any]] = None
        self.lmList: LandmarkList = []
        self.handedness: str = "Unknown"

    def findHands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Detect hands in image and optionally draw landmarks.

        Args:
            img: Input image (BGR format)
            draw: Whether to draw landmarks on image

        Returns:
            Image with optional landmark drawings

        Raises:
            ValueError: If image is None or invalid
        """
        if img is None or img.size == 0:
            logger.warning("Invalid image provided to findHands")
            return img

        hands, img = self.detector.findHands(img, draw=draw)
        self.results = hands
        return img

    def findPosition(
        self,
        img: np.ndarray,
        handNo: int = 0,
        draw: bool = True
    ) -> Tuple[LandmarkList, BoundingBox]:
        """
        Extract landmark positions from detected hand.

        Args:
            img: Input image
            handNo: Hand index (0 for first detected hand)
            draw: Whether to draw landmarks on image

        Returns:
            Tuple of (landmarks, bounding_box)
            - landmarks: List of [id, x, y] for 21 points
            - bounding_box: [xmin, ymin, xmax, ymax]
        """
        self.lmList = []
        bbox: BoundingBox = []

        if img is None or img.size == 0:
            logger.warning("Invalid image provided to findPosition")
            return self.lmList, bbox

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

    def getHandedness(self, handNo: int = 0) -> str:
        """
        Determine handedness (Left/Right) from landmark positions.

        Uses a heuristic based on the relative positions of the wrist (landmark 0)
        and the middle finger MCP (landmark 9). Works best with a mirrored
        (selfie) camera view.

        Args:
            handNo: Hand index to check

        Returns:
            "Left" or "Right" (or "Unknown" if detection fails)
        """
        if not self.results or len(self.results) <= handNo:
            return "Unknown"

        hand = self.results[handNo]
        lm_list = hand.get("lmList", [])

        if len(lm_list) < 10:
            return "Unknown"

        wrist_x = lm_list[0][0]
        middle_mcp_x = lm_list[9][0]

        if wrist_x < middle_mcp_x:
            self.handedness = "Right"
        else:
            self.handedness = "Left"

        return self.handedness

    def fingersUp(self, handNo: int = 0) -> List[int]:
        """
        Detect which fingers are extended.

        Args:
            handNo: Hand index to check (default 0)

        Returns:
            List of 5 binary values [thumb, index, middle, ring, pinky]
            where 1 = extended, 0 = closed
        """
        if not self.results or len(self.results) <= handNo:
            return []

        hand = self.results[handNo]
        fingers = self.detector.fingersUp(hand)
        return fingers

    def getHandsCount(self) -> int:
        """Return the number of detected hands."""
        if not self.results:
            return 0
        return len(self.results)

    def getHandednessAll(self) -> List[str]:
        """Return handedness for all detected hands."""
        if not self.results:
            return []
        return [self.getHandedness(i) for i in range(len(self.results))]

    def findDistance(
        self,
        p1: int,
        p2: int,
        img: np.ndarray,
        draw: bool = True,
        r: int = 15,
        t: int = 3
    ) -> Tuple[float, np.ndarray, BoundingBox]:
        """
        Calculate distance between two landmarks.

        Args:
            p1: First landmark index
            p2: Second landmark index
            img: Input image
            draw: Whether to draw visualization
            r: Circle radius for drawing
            t: Line thickness for drawing

        Returns:
            Tuple of (distance, image, line_info)
            - distance: Euclidean distance in pixels
            - image: Image with optional drawings
            - line_info: [x1, y1, x2, y2, cx, cy]
        """
        if len(self.lmList) == 0:
            return 0.0, img, [0, 0, 0, 0, 0, 0]

        # Validate landmark indices
        if p1 >= len(self.lmList) or p2 >= len(self.lmList):
            logger.warning("Invalid landmark indices: %d, %d", p1, p2)
            return 0.0, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), t)
            cv2.circle(img, (int(x1), int(y1)), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(x2), int(y2)), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(cx), int(cy)), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


# Backward compatibility alias
handDetector = HandDetector


if __name__ == "__main__":
    pTime = 0
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)

    detector = HandDetector()

    try:
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
