"""
HandTrackingModule - Real-time Hand Landmark Detection

Originally built for gesture-based cursor control, this module provides the
foundational layer for sign language motion capture. Uses cvzone (MediaPipe wrapper)
for Python 3.13+ compatibility while maintaining the same 21-point hand tracking.

Core Parameters Captured:
1. Handshape - via landmark relationships (finger positions)
2. Location - via 2D coordinates (x, y)
3. Movement - via temporal sequences (tracking over time)
4. Orientation - via landmark directions (wrist to fingertips)

This module is motion-capture agnostic: it extracts landmarks, which can
drive cursor control, animation systems, JSON logging, or ML models.

Key Design Principle: Separate motion capture from output action.

Technical Note: Uses cvzone.HandTrackingModule (MediaPipe wrapper) for
Python 3.13 compatibility. API is identical to original MediaPipe implementation.
"""

import math
import time
import cv2
from cvzone.HandTrackingModule import HandDetector as CvzoneHandDetector


class handDetector:
    """
    Hand tracking detector compatible with Python 3.13+
    
    This wraps cvzone's HandDetector to maintain the same API as the original
    MediaPipe implementation, allowing seamless switching between backends.
    """
    
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        """
        Initialize hand detector.
        
        Args:
            mode: Static image mode (not used with cvzone, kept for API compatibility)
            maxHands: Maximum number of hands to detect
            detectionCon: Minimum detection confidence [0.0, 1.0]
            trackCon: Minimum tracking confidence (not used with cvzone)
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Initialize cvzone hand detector
        self.detector = CvzoneHandDetector(
            detectionCon=detectionCon,
            maxHands=maxHands
        )
        
        # Fingertip landmark IDs (same as MediaPipe)
        self.tipIds = [4, 8, 12, 16, 20]
        
        # State variables
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        """
        Detect hands in image and optionally draw landmarks.
        
        Args:
            img: Input image (BGR format from OpenCV)
            draw: Whether to draw hand landmarks on image
            
        Returns:
            img: Image with optional hand landmarks drawn
        """
        # Find hands using cvzone
        hands, img = self.detector.findHands(img, draw=draw)
        self.results = hands
        
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Extract landmark positions from detected hand.
        
        Returns structured list of landmarks with their 2D coordinates.
        Critical for sign language: these coordinates are the raw data
        from which handshape, location, and movement are derived.
        
        Args:
            img: Input image
            handNo: Which hand to process (0 for first detected)
            draw: Whether to draw landmarks (handled by findHands in cvzone)
            
        Returns:
            lmList: [[id, x, y], ...] - 21 landmarks per hand
            bbox: [xmin, ymin, xmax, ymax] - bounding box
        """
        self.lmList = []
        bbox = []
        
        if self.results and len(self.results) > handNo:
            hand = self.results[handNo]
            
            # Get landmark list from cvzone format
            lmList = hand["lmList"]
            
            # Convert to expected format: [[id, x, y], ...]
            self.lmList = [[i, lm[0], lm[1]] for i, lm in enumerate(lmList)]
            
            # Get bounding box [x, y, w, h] and convert to [xmin, ymin, xmax, ymax]
            if "bbox" in hand:
                bbox_xywh = hand["bbox"]
                bbox = [
                    bbox_xywh[0],  # xmin
                    bbox_xywh[1],  # ymin
                    bbox_xywh[0] + bbox_xywh[2],  # xmax
                    bbox_xywh[1] + bbox_xywh[3]   # ymax
                ]
        
        return self.lmList, bbox

    def fingersUp(self):
        """
        Detect which fingers are extended (0 = down, 1 = up).
        
        Returns [thumb, index, middle, ring, pinky]
        
        This is a basic handshape classifier. In sign language, handshape
        is one of the four critical parameters. This method captures finger
        extension but doesn't capture:
        - Finger curl/bend amount
        - 3D orientation
        - Relative finger positions
        
        Future: More sophisticated handshape classification needed for
        full sign language vocabulary.
        
        Returns:
            fingers: List of 5 binary values [thumb, index, middle, ring, pinky]
        """
        fingers = []
        
        if not self.results or len(self.results) == 0:
            return fingers
        
        # Get fingers up from cvzone
        hand = self.results[0]
        fingers = self.detector.fingersUp(hand)
        
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Calculate distance between two landmarks.
        
        Args:
            p1: First landmark ID
            p2: Second landmark ID
            img: Image to draw on
            draw: Whether to draw line and circles
            r: Circle radius
            t: Line thickness
            
        Returns:
            length: Euclidean distance between points
            img: Image with optional drawings
            lineInfo: [x1, y1, x2, y2, cx, cy] - point coordinates and center
        """
        if len(self.lmList) == 0:
            return 0, img, [0, 0, 0, 0, 0, 0]
        
        # Get landmark coordinates
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw if requested
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        # Calculate distance
        length = math.hypot(x2 - x1, y2 - y1)
        
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    """Test the hand detector"""
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    print("\n" + "="*60)
    print("Hand Tracking Test (cvzone-based)")
    print("="*60)
    print("\nCompatible with Python 3.13+")
    print("Uses cvzone (MediaPipe wrapper) for hand tracking")
    print("\nPress 'q' to quit\n")
    
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        # Find hands
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        
        # Display info if hand detected
        if len(lmList) != 0:
            # Show finger states
            fingers = detector.fingersUp()
            cv2.putText(img, f"Fingers: {fingers}", (10, 110), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Show landmark count
            cv2.putText(img, f"Landmarks: {len(lmList)}", (10, 150), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # Calculate and display FPS
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
    
    print("\n" + "="*60)
    print("Test complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()