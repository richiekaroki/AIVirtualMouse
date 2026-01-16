# Motion Data Directory

This directory stores recorded motion sequences in JSON format.

## File Naming Convention

Files are automatically named: `{gesture_name}_{timestamp}.json`

Example: `wave_1704567890.json`

## File Structure

Each JSON file contains:

```json
{
  "metadata": {
    "gesture_name": "wave",
    "recorded_at": "2026-01-11T10:30:00",
    "duration_seconds": 2.5,
    "total_frames": 75,
    "average_fps": 30.0
  },
  "frames": [
    {
      "timestamp": 1704567890.123,
      "primitive": "OPEN_HAND",
      "landmarks": {...},
      "velocity": {...}
    },
    ...
  ]
}
```

## Usage

These files can be used for:

- Training ML models
- Motion analysis and debugging
- Animation retargeting
- Validation with deaf signers
- Building sign language datasets

## Git Note

This directory is tracked but specific JSON files may be excluded
from version control (see `.gitignore`).
