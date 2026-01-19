# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - Sign Language Extension

### Planned

- Motion descriptor abstraction layer
- JSON export of gesture sequences
- Gesture primitive classification system
- Temporal sequence recording
- Motion visualization tools
- MediaPipe Pose integration for full body tracking

## [1.0.0] - 2024-12 - Initial Release

### Added

- Real-time hand tracking using MediaPipe
- 21-point hand landmark detection
- Gesture-based cursor control
- Finger state detection (which fingers are extended)
- Distance-based gesture recognition
- Motion smoothing for cursor movement
- Modular HandTrackingModule class
- Click detection via two-finger gesture

### Technical Details

- 30fps real-time processing
- Coordinate normalization and mapping
- Frame reduction for performance optimization
- Smoothing algorithm for natural cursor movement

## [0.4.5] - 2026-01-12 - Enhanced UI

### Added

- Split-screen interface (640x480 video + 400px info panel)
- Hand skeleton overlay with color-coded joints
- Real-time primitive sequence timeline (last 10 primitives)
- Color-coded primitive blocks for visual tracking
- Recording progress indicator with timer
- Session statistics display
- Professional color scheme and typography
- Visual recording indicator (blinking red circle)
- Organized section headers
- Mouse control zone indicator

### Improved

- Information organization and readability
- Visual feedback during recording
- Real-time motion analysis display
- FPS counter positioning
- Overall UI/UX polish

### Technical

- Separate enhanced version (AiVirtualMouseProject_Enhanced.py)
- Modular UI helper functions
- Color constants for consistent theming
- Deque-based primitive history tracking

## Project Evolution

**Phase 1** (Dec 2024): Gesture-based cursor control  
**Phase 2** (Jan 2026): Sign language motion infrastructure  
**Phase 3** (Future): Full translation pipeline with deaf community validation
