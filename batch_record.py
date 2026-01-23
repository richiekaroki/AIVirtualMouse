"""
Batch Gesture Recording Script

Guides you through recording a complete gesture dataset with:
- Predefined gesture list
- Quality checks
- Automatic naming
- Progress tracking

Usage:
    python batch_record.py
    
This script will guide you through recording each gesture with
countdown timers and quality feedback.

Author: Richard Kabue Karoki
Created: January 2026
"""

import cv2
from HandTrackingModule import handDetector
from MotionDescriptor import MotionDescriptor
import os
import time
import json

# Predefined gesture list
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

# Recording parameters
RECORDING_DURATION = 3  # seconds per gesture
COUNTDOWN_TIME = 3      # seconds before recording starts
ATTEMPTS_PER_GESTURE = 3


def countdown(seconds, message="Starting in"):
    """Display countdown with large text"""
    for i in range(seconds, 0, -1):
        print(f"\r{message} {i}...", end='', flush=True)
        time.sleep(1)
    print(f"\r{message} NOW!     ")


def record_single_gesture(gesture_name, description, attempt_num, cap, detector, motion_descriptor):
    """Record a single gesture attempt"""
    print(f"\n{'='*70}")
    print(f"Recording: {gesture_name} (Attempt {attempt_num}/{ATTEMPTS_PER_GESTURE})")
    print(f"Description: {description}")
    print(f"Duration: {RECORDING_DURATION} seconds")
    print(f"{'='*70}")
    
    print("\nGet ready to perform the gesture...")
    countdown(COUNTDOWN_TIME, "Starting in")
    
    # Clear motion history
    motion_descriptor.clear_history()
    
    start_time = time.time()
    frame_count = 0
    recording = True
    
    print("🔴 RECORDING... (Press 'q' to cancel)")
    
    while recording and (time.time() - start_time) < RECORDING_DURATION:
        success, img = cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        
        elapsed = time.time() - start_time
        remaining = RECORDING_DURATION - elapsed
        
        # Visual feedback
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            descriptor = motion_descriptor.create_descriptor(lmList, fingers)
            frame_count += 1
            
            # Show recording status
            cv2.circle(img, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(img, "RECORDING", (55, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(img, f"Time: {remaining:.1f}s", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(img, f"Frames: {frame_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if descriptor:
                cv2.putText(img, f"Primitive: {descriptor['primitive']}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Progress bar
            bar_width = 400
            bar_height = 20
            progress = elapsed / RECORDING_DURATION
            filled_width = int(bar_width * progress)
            
            cv2.rectangle(img, (10, 460), (10 + bar_width, 460 + bar_height), (100, 100, 100), -1)
            cv2.rectangle(img, (10, 460), (10 + filled_width, 460 + bar_height), (0, 255, 0), -1)
        else:
            cv2.putText(img, "NO HAND DETECTED", (10, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        cv2.imshow("Batch Recording", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✗ Recording cancelled by user")
            return None
    
    # Save recording
    if frame_count < 20:
        print(f"\n✗ Recording too short ({frame_count} frames). Minimum 20 frames required.")
        return None
    
    timestamp = int(time.time())
    filename = f"motion_data/{gesture_name}_{attempt_num}_{timestamp}.json"
    motion_descriptor.save_sequence(filename, gesture_name, 
                                   metadata={'attempt': attempt_num, 'description': description})
    
    # Quality check
    stats = motion_descriptor.get_statistics()
    
    print(f"\n✓ Recording complete!")
    print(f"  File: {filename}")
    print(f"  Frames: {stats['total_frames']}")
    print(f"  FPS: {stats['average_fps']:.1f}")
    print(f"  Primitives: {', '.join(stats['primitive_counts'].keys())}")
    
    # Quality assessment
    quality_issues = []
    
    if stats['average_fps'] < 25:
        quality_issues.append("Low FPS (< 25)")
    
    if stats['total_frames'] < 30:
        quality_issues.append("Too few frames (< 30)")
    
    if quality_issues:
        print(f"\n⚠️  Quality warnings: {', '.join(quality_issues)}")
        retry = input("  Retry this attempt? (y/n): ").lower()
        if retry == 'y':
            os.remove(filename)
            return record_single_gesture(gesture_name, description, attempt_num, 
                                        cap, detector, motion_descriptor)
    
    return filename


def main():
    """Main batch recording workflow"""
    print("\n" + "="*70)
    print("Batch Gesture Recording - Dataset Creation")
    print("="*70)
    
    # Count total gestures
    total_gestures = sum(len(gestures) for gestures in GESTURE_CATEGORIES.values())
    total_recordings = total_gestures * ATTEMPTS_PER_GESTURE
    
    print(f"\nYou will record:")
    print(f"  • {total_gestures} unique gestures")
    print(f"  • {ATTEMPTS_PER_GESTURE} attempts per gesture")
    print(f"  • {total_recordings} total recordings")
    print(f"  • ~{RECORDING_DURATION} seconds per recording")
    print(f"  • Estimated time: {int(total_recordings * (RECORDING_DURATION + COUNTDOWN_TIME + 5) / 60)} minutes")
    
    print("\nTips for good recordings:")
    print("  ✓ Ensure good lighting")
    print("  ✓ Keep hand fully in frame")
    print("  ✓ Perform gestures smoothly")
    print("  ✓ Hold static gestures steady")
    
    input("\nPress ENTER to begin...")
    
    # Initialize
    if not os.path.exists('motion_data'):
        os.makedirs('motion_data')
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    detector = handDetector()
    motion_descriptor = MotionDescriptor()
    
    # Track progress
    completed_recordings = []
    current_gesture_num = 0
    
    try:
        # Record each category
        for category_name, gestures in GESTURE_CATEGORIES.items():
            print(f"\n{'#'*70}")
            print(f"Category: {category_name}")
            print(f"{'#'*70}")
            
            for gesture_name, description in gestures:
                current_gesture_num += 1
                
                print(f"\n[{current_gesture_num}/{total_gestures}] Gesture: {gesture_name}")
                
                # Record multiple attempts
                for attempt in range(1, ATTEMPTS_PER_GESTURE + 1):
                    filename = record_single_gesture(
                        gesture_name, description, attempt,
                        cap, detector, motion_descriptor
                    )
                    
                    if filename:
                        completed_recordings.append(filename)
                    
                    if attempt < ATTEMPTS_PER_GESTURE:
                        print("\nPrepare for next attempt...")
                        time.sleep(2)
                
                print(f"\n✓ Completed all attempts for '{gesture_name}'")
                
                # Option to continue or pause
                if current_gesture_num < total_gestures:
                    choice = input("\nContinue to next gesture? (y/n/p=pause): ").lower()
                    if choice == 'n':
                        raise KeyboardInterrupt
                    elif choice == 'p':
                        input("Paused. Press ENTER to continue...")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording session interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "="*70)
    print("Recording Session Summary")
    print("="*70)
    print(f"Total recordings: {len(completed_recordings)}")
    print(f"Target: {total_recordings}")
    print(f"Completion: {len(completed_recordings)/total_recordings*100:.1f}%")
    
    # Save manifest
    manifest = {
        'session_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_recordings': len(completed_recordings),
        'recordings': completed_recordings,
        'categories': {cat: [g[0] for g in gestures] 
                      for cat, gestures in GESTURE_CATEGORIES.items()}
    }
    
    with open('motion_data/recording_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Manifest saved to motion_data/recording_manifest.json")
    print("\nNext steps:")
    print("  1. Review recordings with MotionAnalyzer.py")
    print("  2. Generate visualizations for best attempts")
    print("  3. Document findings in dataset_summary.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()