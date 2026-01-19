"""
Hand Motion Interpretation Pipeline - Enhanced UI Version

Professional interface for motion capture with:
- Split-screen layout (video + analysis panel)
- Hand skeleton overlay
- Recent primitive sequence timeline
- Recording progress indicator
- Session statistics
- Clean, organized information display

Author: Richard Kabue Karoki
Enhanced UI: January 2026
"""

import cv2
import numpy as np
import HandTrackingModule as htm
from MotionDescriptor import MotionDescriptor
import time
import pyautogui
import os
from collections import deque

############################
# Configuration
############################
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PANEL_WIDTH = 400  # Info panel width
TOTAL_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
TOTAL_HEIGHT = CAMERA_HEIGHT

# Colors (BGR format)
COLOR_PRIMARY = (255, 140, 0)      # Orange
COLOR_SUCCESS = (0, 255, 0)        # Green
COLOR_DANGER = (0, 0, 255)         # Red
COLOR_INFO = (255, 255, 0)         # Cyan
COLOR_TEXT = (255, 255, 255)       # White
COLOR_BG_DARK = (40, 40, 40)       # Dark gray
COLOR_BG_LIGHT = (60, 60, 60)      # Light gray
COLOR_SKELETON = (0, 255, 255)     # Yellow

# Mouse control
frameR = 100
smoothening = 10

############################
# Initialize
############################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, CAMERA_WIDTH)
cap.set(4, CAMERA_HEIGHT)

detector = htm.handDetector()
motion_descriptor = MotionDescriptor()

wScr, hScr = pyautogui.size()

# Recording state
recording_mode = False
gesture_name = None
recording_start_time = None

# Primitive sequence tracking (last 10)
primitive_history = deque(maxlen=10)

# Session statistics
session_stats = {
    'gestures_recorded': 0,
    'total_frames': 0,
    'session_start': time.time()
}

# Create output directory
if not os.path.exists('motion_data'):
    os.makedirs('motion_data')

############################
# UI Helper Functions
############################

def create_info_panel(width, height):
    """Create blank info panel with dark background"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = COLOR_BG_DARK
    return panel

def draw_section_header(panel, text, y_pos, color=COLOR_PRIMARY):
    """Draw a section header"""
    cv2.rectangle(panel, (0, y_pos), (panel.shape[1], y_pos + 35), 
                  COLOR_BG_LIGHT, -1)
    cv2.putText(panel, text, (15, y_pos + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return y_pos + 40

def draw_info_row(panel, label, value, y_pos, value_color=COLOR_TEXT):
    """Draw an information row"""
    cv2.putText(panel, label, (15, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, str(value), (15, y_pos + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, value_color, 2)
    return y_pos + 50

def draw_recording_header(panel, recording, gesture_name, elapsed_time):
    """Draw recording status header"""
    if recording:
        # Red recording header
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 50), COLOR_DANGER, -1)
        
        # Blinking red circle
        if int(time.time() * 2) % 2 == 0:  # Blink every 0.5s
            cv2.circle(panel, (30, 25), 12, (255, 255, 255), -1)
        
        cv2.putText(panel, "RECORDING", (55, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
        
        # Timer
        timer_text = f"{elapsed_time:.1f}s"
        cv2.putText(panel, timer_text, (panel.shape[1] - 80, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    else:
        # Green ready header
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 50), COLOR_SUCCESS, -1)
        cv2.circle(panel, (30, 25), 12, (255, 255, 255), 2)
        cv2.putText(panel, "READY", (55, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    
    return 55

def draw_primitive_timeline(panel, primitive_history, y_start):
    """Draw recent primitive sequence as colored blocks"""
    y_pos = draw_section_header(panel, "RECENT SEQUENCE", y_start)
    
    if not primitive_history:
        cv2.putText(panel, "No motion detected", (15, y_pos + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return y_pos + 80
    
    # Color mapping for primitives
    primitive_colors = {
        'POINT': (255, 140, 0),
        'OPEN_HAND': (0, 255, 0),
        'FIST': (0, 0, 255),
        'PEACE_V': (255, 0, 255),
        'THUMBS_UP': (0, 255, 255),
        'PINCH_READY': (255, 255, 0),
    }
    
    block_height = 25
    block_spacing = 5
    
    for i, primitive in enumerate(reversed(list(primitive_history))):
        y = y_pos + i * (block_height + block_spacing)
        
        # Get color for primitive
        color = primitive_colors.get(primitive, (150, 150, 150))
        
        # Draw colored block
        bar_width = 120
        cv2.rectangle(panel, (15, y), (15 + bar_width, y + block_height), 
                     color, -1)
        
        # Draw primitive name
        text = primitive[:12]  # Truncate if too long
        cv2.putText(panel, text, (15 + bar_width + 10, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
    
    return y_pos + len(primitive_history) * (block_height + block_spacing) + 20

def draw_motion_analysis(panel, descriptor, y_start):
    """Draw current motion analysis"""
    if not descriptor:
        return y_start
    
    y_pos = draw_section_header(panel, "MOTION ANALYSIS", y_start)
    
    # Primitive
    primitive = descriptor['primitive']
    prim_color = COLOR_SUCCESS if primitive != "UNKNOWN" else (150, 150, 150)
    y_pos = draw_info_row(panel, "Primitive", primitive, y_pos, prim_color)
    
    # Handshape
    handshape = descriptor['handshape_code']
    y_pos = draw_info_row(panel, "Handshape", handshape, y_pos)
    
    # Openness
    openness = descriptor['features']['hand_openness']
    y_pos = draw_info_row(panel, "Hand Open", f"{openness:.2f}", y_pos, COLOR_INFO)
    
    # Velocity (if available)
    if descriptor['velocity']:
        vel = descriptor['velocity']['magnitude']
        y_pos = draw_info_row(panel, "Velocity", f"{vel:.1f} px/s", y_pos, COLOR_PRIMARY)
    
    return y_pos

def draw_session_stats(panel, stats, y_start):
    """Draw session statistics"""
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
    """Draw control instructions"""
    y_pos = draw_section_header(panel, "CONTROLS", y_start, COLOR_INFO)
    
    controls = [
        ("R", "Start recording"),
        ("S", "Stop & save"),
        ("C", "Cancel recording"),
        ("Q", "Quit application")
    ]
    
    for key, desc in controls:
        cv2.rectangle(panel, (15, y_pos), (45, y_pos + 25), COLOR_PRIMARY, -1)
        cv2.putText(panel, key, (22, y_pos + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
        cv2.putText(panel, desc, (55, y_pos + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 35
    
    return y_pos

def draw_hand_skeleton(img, lmList):
    """Draw hand skeleton overlay"""
    if len(lmList) < 21:
        return
    
    # Define connections (MediaPipe hand connections)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(lmList) and end_idx < len(lmList):
            start_point = (lmList[start_idx][1], lmList[start_idx][2])
            end_point = (lmList[end_idx][1], lmList[end_idx][2])
            cv2.line(img, start_point, end_point, COLOR_SKELETON, 2)
    
    # Draw landmarks
    for lm in lmList:
        cv2.circle(img, (lm[1], lm[2]), 5, COLOR_PRIMARY, -1)
        cv2.circle(img, (lm[1], lm[2]), 7, COLOR_SKELETON, 2)

def draw_fps(img, fps):
    """Draw FPS counter"""
    cv2.rectangle(img, (5, 5), (100, 35), (0, 0, 0), -1)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SUCCESS, 2)

def draw_mouse_control_zone(img):
    """Draw mouse control zone indicator"""
    cv2.rectangle(img, (frameR, frameR), 
                 (CAMERA_WIDTH - frameR, CAMERA_HEIGHT - frameR),
                 COLOR_PRIMARY, 2)
    cv2.putText(img, "Mouse Zone", (frameR + 10, frameR + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)

############################
# Main Loop
############################

print("\n" + "="*70)
print("Hand Motion Interpretation Pipeline - Enhanced UI")
print("="*70)
print("\nFeatures:")
print("  • Split-screen interface (video + analysis)")
print("  • Hand skeleton overlay")
print("  • Real-time primitive sequence")
print("  • Recording progress indicator")
print("  • Session statistics")
print("\nControls:")
print("  R - Start recording | S - Stop & save | C - Cancel | Q - Quit")
print("="*70 + "\n")

while True:
    # Capture frame
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    
    # Create info panel
    info_panel = create_info_panel(PANEL_WIDTH, TOTAL_HEIGHT)
    
    # Detect hands
    img = detector.findHands(img, draw=False)  # We'll draw our own skeleton
    lmList, bbox = detector.findPosition(img, draw=False)
    
    current_descriptor = None
    
    # Process if hand detected
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        fingers = detector.fingersUp()
        
        # Create motion descriptor
        current_descriptor = motion_descriptor.create_descriptor(
            lmList, fingers, frame_shape=(CAMERA_HEIGHT, CAMERA_WIDTH)
        )
        
        # Update primitive history
        if current_descriptor:
            primitive_history.append(current_descriptor['primitive'])
            session_stats['total_frames'] += 1
        
        # Draw hand skeleton overlay
        draw_hand_skeleton(img, lmList)
        
        # Mouse control (if not recording)
        if not recording_mode:
            draw_mouse_control_zone(img)
            
            # Index finger only: Move
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, CAMERA_WIDTH - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, CAMERA_HEIGHT - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 12, COLOR_PRIMARY, -1)
                plocX, plocY = clocX, clocY
            
            # Both fingers: Click
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img, draw=False)
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, COLOR_SUCCESS, -1)
                    pyautogui.click()
                    time.sleep(0.2)
        else:
            # Recording mode - draw border
            cv2.rectangle(img, (0, 0), (CAMERA_WIDTH-1, CAMERA_HEIGHT-1), 
                         COLOR_DANGER, 5)
    
    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    # Draw FPS on video
    draw_fps(img, fps)
    
    # Build info panel
    y_position = 0
    
    # Recording header
    elapsed_time = time.time() - recording_start_time if recording_start_time else 0
    y_position = draw_recording_header(info_panel, recording_mode, gesture_name, elapsed_time)
    
    # Recent sequence timeline
    y_position = draw_primitive_timeline(info_panel, primitive_history, y_position)
    
    # Motion analysis
    y_position = draw_motion_analysis(info_panel, current_descriptor, y_position)
    
    # Session stats
    y_position = draw_session_stats(info_panel, session_stats, y_position)
    
    # Controls
    y_position = draw_controls_help(info_panel, y_position)
    
    # Combine video and info panel
    combined_frame = np.hstack([img, info_panel])
    
    # Display
    cv2.imshow("Hand Motion Pipeline - Enhanced", combined_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    # R - Start Recording
    if key == ord('r') and not recording_mode:
        gesture_name = input("\n→ Enter gesture name: ").strip()
        if gesture_name:
            recording_mode = True
            recording_start_time = time.time()
            motion_descriptor.clear_history()
            print(f"✓ Recording '{gesture_name}'... (Press 'S' to stop)")
    
    # S - Stop and Save
    elif key == ord('s') and recording_mode:
        recording_mode = False
        recording_start_time = None
        
        if motion_descriptor.motion_history:
            timestamp = int(time.time())
            filename = f"motion_data/{gesture_name}_{timestamp}.json"
            motion_descriptor.save_sequence(filename, gesture_name)
            session_stats['gestures_recorded'] += 1
            print(f"✓ Saved to {filename}")
        
        gesture_name = None
    
    # C - Cancel
    elif key == ord('c') and recording_mode:
        recording_mode = False
        recording_start_time = None
        motion_descriptor.clear_history()
        print(f"✗ Recording cancelled")
        gesture_name = None
    
    # Q - Quit
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("Session Summary")
print("="*70)
print(f"Gestures recorded: {session_stats['gestures_recorded']}")
print(f"Total frames captured: {session_stats['total_frames']}")
print(f"Session duration: {int(time.time() - session_stats['session_start'])}s")
print("="*70 + "\n")