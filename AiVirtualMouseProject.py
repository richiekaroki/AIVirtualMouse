"""
Hand Motion Interpretation Pipeline - Main Application

Originally: Gesture-based cursor control
Now: Motion capture + cursor control + sequence recording

This demonstrates the power of the MotionDescriptor abstraction:
- Same motion capture drives multiple outputs
- Can switch between cursor control and recording modes
- Motion data can be exported for training/analysis

Controls:
    Mouse Mode (default):
        - Index finger up = move cursor
        - Index + middle up = click
    
    Recording Mode:
        - Press 'r' = start recording
        - Press 's' = stop and save
        - Press 'c' = cancel recording
        - Press 'q' = quit application

Author: Richard Kabue Karoki
Extended: January 2026 for sign language infrastructure
"""

import cv2
import numpy as np
import HandTrackingModule as htm
from MotionDescriptor import MotionDescriptor
import time
import pyautogui
import os

############################
# Configuration
############################
wCam, hCam = 640, 480
frameR = 100  # frame reduction for mouse control area
smoothening = 10

############################
# Initialize
############################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector()
motion_descriptor = MotionDescriptor()

wScr, hScr = pyautogui.size()  # Get actual screen size

# Recording state
recording_mode = False
gesture_name = None
frame_count = 0

# Create motion_data directory if it doesn't exist
if not os.path.exists('motion_data'):
    os.makedirs('motion_data')
    print("✓ Created motion_data/ directory")

############################
# Helper Functions
############################

def draw_recording_indicator(img, recording, gesture_name=None, frame_count=0):
    """Draw recording status on screen"""
    if recording:
        # Red recording indicator
        cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
        cv2.putText(img, "RECORDING", (55, 40), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        if gesture_name:
            cv2.putText(img, f"Gesture: {gesture_name}", (10, 120), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        
        cv2.putText(img, f"Frames: {frame_count}", (10, 150), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        # Green ready indicator
        cv2.circle(img, (30, 30), 15, (0, 255, 0), 2)
        cv2.putText(img, "READY", (55, 40), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

def draw_primitive_info(img, descriptor):
    """Display current motion primitive and features"""
    if descriptor:
        # Primitive name
        primitive = descriptor['primitive']
        cv2.putText(img, f"Primitive: {primitive}", (10, hCam - 90), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        
        # Handshape code
        handshape = descriptor['handshape_code']
        cv2.putText(img, f"Handshape: {handshape}", (10, hCam - 60), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        
        # Hand openness
        openness = descriptor['features']['hand_openness']
        cv2.putText(img, f"Openness: {openness:.2f}", (10, hCam - 30), 
                   cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        
        # Velocity (if available)
        if descriptor['velocity']:
            vel = descriptor['velocity']['magnitude']
            cv2.putText(img, f"Velocity: {vel:.1f}", (10, hCam - 5), 
                       cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

def draw_controls_help(img):
    """Draw control instructions"""
    help_text = [
        "Controls:",
        "R - Start recording",
        "S - Stop & save",
        "C - Cancel recording",
        "Q - Quit"
    ]
    
    y_offset = 180
    for i, text in enumerate(help_text):
        color = (200, 200, 200) if i == 0 else (150, 150, 150)
        cv2.putText(img, text, (wCam - 180, y_offset + i*25), 
                   cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)

############################
# Main Loop
############################

print("\n" + "="*60)
print("Hand Motion Interpretation Pipeline")
print("="*60)
print("\nMode: Cursor Control + Motion Recording")
print("\nControls:")
print("  R - Start recording a gesture sequence")
print("  S - Stop recording and save to JSON")
print("  C - Cancel recording without saving")
print("  Q - Quit application")
print("\nMouse Control:")
print("  Index finger up → Move cursor")
print("  Index + Middle up → Click")
print("\n" + "="*60 + "\n")

while True:
    # 1. Capture frame
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    img = cv2.flip(img, 1)  # Mirror image for natural interaction
    
    # 2. Detect hands
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    # 3. Process if hand detected
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        
        # Get finger states
        fingers = detector.fingersUp()
        
        # Create motion descriptor
        descriptor = motion_descriptor.create_descriptor(
            lmList, 
            fingers,
            frame_shape=(hCam, wCam)
        )
        
        # Display primitive info
        draw_primitive_info(img, descriptor)
        
        # ==========================================
        # CURSOR CONTROL MODE (if not recording)
        # ==========================================
        if not recording_mode:
            # Draw mouse control area
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                         (255, 0, 255), 2)
            
            # 4. Only Index Finger: Move Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert coordinates to screen space
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # Smooth cursor movement
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                # Move mouse
                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
            
            # 5. Both Index and Middle Fingers: Click Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                
                # Click if distance is short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]),
                             15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.2)  # Prevent multiple clicks
        
        # ==========================================
        # RECORDING MODE
        # ==========================================
        else:
            frame_count += 1
            # Visual feedback during recording
            cv2.rectangle(img, (0, 0), (wCam, hCam), (0, 0, 255), 5)
    
    else:
        # No hand detected
        if not recording_mode:
            cv2.putText(img, "No hand detected", (10, hCam - 30), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    # ==========================================
    # UI OVERLAYS
    # ==========================================
    
    # Draw recording indicator
    draw_recording_indicator(img, recording_mode, gesture_name, frame_count)
    
    # Draw controls help
    draw_controls_help(img)
    
    # Draw FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), 
               cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    # ==========================================
    # DISPLAY
    # ==========================================
    cv2.imshow("Hand Motion Pipeline", img)
    
    # ==========================================
    # KEYBOARD CONTROLS
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    
    # R - Start Recording
    if key == ord('r') and not recording_mode:
        gesture_name = input("\n→ Enter gesture name: ").strip()
        if gesture_name:
            recording_mode = True
            frame_count = 0
            motion_descriptor.clear_history()
            print(f"✓ Recording '{gesture_name}'... (Press 'S' to stop, 'C' to cancel)")
        else:
            print("✗ Invalid gesture name. Recording cancelled.")
    
    # S - Stop and Save Recording
    elif key == ord('s') and recording_mode:
        recording_mode = False
        
        if frame_count > 0:
            # Generate filename
            timestamp = int(time.time())
            filename = f"motion_data/{gesture_name}_{timestamp}.json"
            
            # Save sequence
            motion_descriptor.save_sequence(filename, gesture_name)
            
            # Show statistics
            stats = motion_descriptor.get_statistics()
            print(f"\n✓ Recording saved successfully!")
            print(f"  File: {filename}")
            print(f"  Duration: {stats['duration_seconds']:.2f}s")
            print(f"  Frames: {stats['total_frames']}")
            print(f"  FPS: {stats['average_fps']:.1f}")
            print(f"  Primitives: {', '.join(stats['primitive_counts'].keys())}")
            print()
        else:
            print("✗ No frames recorded. Recording cancelled.")
        
        gesture_name = None
        frame_count = 0
    
    # C - Cancel Recording
    elif key == ord('c') and recording_mode:
        recording_mode = False
        motion_descriptor.clear_history()
        print(f"✗ Recording of '{gesture_name}' cancelled.")
        gesture_name = None
        frame_count = 0
    
    # Q - Quit
    elif key == ord('q'):
        print("\nShutting down...")
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Session Summary")
print("="*60)

# Show final statistics if there's motion history
if motion_descriptor.motion_history:
    stats = motion_descriptor.get_statistics()
    print(f"Total motion captured: {stats['total_frames']} frames")
    print(f"Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"Unique primitives seen: {stats['unique_primitives']}")
    print(f"Primitives: {', '.join(stats['primitive_counts'].keys())}")
else:
    print("No motion data recorded this session.")

print("\n✓ Application closed successfully.")
print("="*60 + "\n")