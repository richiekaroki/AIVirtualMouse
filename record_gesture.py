"""
Quick Gesture Recording Script

Usage:
    python record_gesture.py
    
This script provides a simplified interface for recording gestures
without the mouse control functionality.
"""

import cv2
from HandTrackingModule import handDetector
from MotionDescriptor import MotionDescriptor
import os
import time

def main():
    print("\n" + "="*60)
    print("Gesture Recording Tool")
    print("="*60)
    
    gesture_name = input("\nEnter gesture name: ").strip()
    if not gesture_name:
        print("✗ Invalid name. Exiting.")
        return
    
    duration = input("Recording duration (seconds, default 3): ").strip()
    duration = int(duration) if duration.isdigit() else 3
    
    print(f"\n✓ Will record '{gesture_name}' for {duration} seconds")
    print("✓ Get ready... Starting in 3 seconds")
    time.sleep(3)
    
    # Initialize
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    motion_descriptor = MotionDescriptor()
    
    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')
    
    start_time = time.time()
    frame_count = 0
    
    print("✓ RECORDING...")
    
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
            
            # Visual feedback
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
    
    # Save
    if frame_count > 0:
        timestamp = int(time.time())
        filename = f"motion_data/{gesture_name}_{timestamp}.json"
        motion_descriptor.save_sequence(filename, gesture_name)
        print(f"✓ Saved {frame_count} frames to {filename}")
    else:
        print("✗ No frames recorded.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()