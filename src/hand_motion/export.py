"""
Export Formats Module

Provides export functionality for motion data:
- CSV export
- Parquet export (if pandas available)
- COCO format export
- Custom format support
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


class MotionExporter:
    """
    Export motion data to various formats.
    """

    def __init__(self, output_dir: str = "exports"):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(
        self,
        filepath: str,
        output_name: Optional[str] = None,
        include_landmarks: bool = True
    ) -> str:
        """
        Export motion data to CSV format.

        Args:
            filepath: Path to motion JSON file
            output_name: Output filename (without extension)
            include_landmarks: Whether to include landmark coordinates

        Returns:
            Path to exported CSV file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        if output_name is None:
            output_name = Path(filepath).stem

        csv_path = self.output_dir / f"{output_name}.csv"

        # Flatten frame data for CSV
        rows = []
        for frame in data.get('frames', []):
            row = {
                'timestamp': frame.get('timestamp'),
                'relative_time': frame.get('relative_time'),
                'frame_num': frame.get('frame_num'),
                'hand': frame.get('hand'),
                'finger_count': frame.get('finger_count'),
                'handshape_code': frame.get('handshape_code'),
                'primitive': frame.get('primitive'),
            }

            # Add finger states
            fingers = frame.get('fingers_extended', [])
            if len(fingers) >= 5:
                row['thumb_up'] = fingers[0]
                row['index_up'] = fingers[1]
                row['middle_up'] = fingers[2]
                row['ring_up'] = fingers[3]
                row['pinky_up'] = fingers[4]

            # Add features
            features = frame.get('features', {})
            row['pinch_distance'] = features.get('pinch_distance')
            row['hand_openness'] = features.get('hand_openness')
            row['hand_span'] = features.get('hand_span')

            # Add palm center
            palm = features.get('palm_center', {})
            row['palm_x'] = palm.get('x')
            row['palm_y'] = palm.get('y')

            # Add velocity
            velocity = frame.get('velocity')
            if velocity:
                row['velocity_magnitude'] = velocity.get('magnitude')
                row['velocity_vx'] = velocity.get('vx')
                row['velocity_vy'] = velocity.get('vy')

            # Add landmarks if requested
            if include_landmarks:
                landmarks = frame.get('landmarks', {})
                for lm_name, coords in landmarks.items():
                    row[f'lm_{lm_name}_x'] = coords.get('x')
                    row[f'lm_{lm_name}_y'] = coords.get('y')

            rows.append(row)

        # Write CSV
        if rows:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info("Exported CSV to: %s", csv_path)
        return str(csv_path)

    def export_parquet(
        self,
        filepath: str,
        output_name: Optional[str] = None
    ) -> str:
        """
        Export motion data to Parquet format.

        Args:
            filepath: Path to motion JSON file
            output_name: Output filename (without extension)

        Returns:
            Path to exported Parquet file

        Raises:
            ImportError: If pandas/pyarrow not installed
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Parquet export requires pandas and pyarrow. "
                "Install with: pip install pandas pyarrow"
            )

        with open(filepath, 'r') as f:
            data = json.load(f)

        if output_name is None:
            output_name = Path(filepath).stem

        parquet_path = self.output_dir / f"{output_name}.parquet"

        # Flatten frame data
        rows = []
        for frame in data.get('frames', []):
            row = {
                'timestamp': frame.get('timestamp'),
                'relative_time': frame.get('relative_time'),
                'frame_num': frame.get('frame_num'),
                'hand': frame.get('hand'),
                'finger_count': frame.get('finger_count'),
                'handshape_code': frame.get('handshape_code'),
                'primitive': frame.get('primitive'),
            }

            # Add features
            features = frame.get('features', {})
            row['pinch_distance'] = features.get('pinch_distance')
            row['hand_openness'] = features.get('hand_openness')
            row['hand_span'] = features.get('hand_span')

            # Add velocity
            velocity = frame.get('velocity')
            if velocity:
                row['velocity_magnitude'] = velocity.get('magnitude')

            rows.append(row)

        # Create DataFrame and export
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)

        logger.info("Exported Parquet to: %s", parquet_path)
        return str(parquet_path)

    def export_coco(
        self,
        filepath: str,
        output_name: Optional[str] = None
    ) -> str:
        """
        Export motion data to COCO format for ML training.

        Args:
            filepath: Path to motion JSON file
            output_name: Output filename (without extension)

        Returns:
            Path to exported COCO JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        if output_name is None:
            output_name = Path(filepath).stem

        coco_path = self.output_dir / f"{output_name}_coco.json"

        # Build COCO structure
        coco_data = {
            'info': {
                'description': f"Motion data: {data.get('metadata', {}).get('gesture_name', 'Unknown')}",
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'categories': [],
            'images': [],
            'annotations': []
        }

        # Create category from gesture name
        gesture_name = data.get('metadata', {}).get('gesture_name', 'unknown')
        category = {
            'id': 1,
            'name': gesture_name,
            'supercategory': 'gesture'
        }
        coco_data['categories'].append(category)

        # Add image entry
        image = {
            'id': 1,
            'file_name': Path(filepath).stem,
            'width': 640,
            'height': 480
        }
        coco_data['images'].append(image)

        # Process frames
        for i, frame in enumerate(data.get('frames', [])):
            # Create annotation for each frame
            annotation = {
                'id': i + 1,
                'image_id': 1,
                'category_id': 1,
                'frame_num': frame.get('frame_num', i),
                'relative_time': frame.get('relative_time', 0),
                'primitive': frame.get('primitive', 'UNKNOWN'),
                'hand': frame.get('hand', 'unknown'),
                'fingers_extended': frame.get('fingers_extended', []),
                'handshape_code': frame.get('handshape_code', '')
            }

            # Add keypoints (flattened landmarks)
            landmarks = frame.get('landmarks', {})
            keypoints = []
            for lm_name in ['wrist', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']:
                lm = landmarks.get(lm_name, {})
                keypoints.extend([lm.get('x', 0), lm.get('y', 0), 1])  # x, y, visibility

            annotation['keypoints'] = keypoints
            annotation['num_keypoints'] = len(keypoints) // 3

            # Add features
            features = frame.get('features', {})
            annotation['features'] = {
                'hand_openness': features.get('hand_openness', 0),
                'pinch_distance': features.get('pinch_distance', 0),
                'hand_span': features.get('hand_span', 0)
            }

            coco_data['annotations'].append(annotation)

        # Write COCO JSON
        with open(coco_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info("Exported COCO to: %s", coco_path)
        return str(coco_path)

    def export_all_formats(
        self,
        filepath: str,
        output_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export motion data to all available formats.

        Args:
            filepath: Path to motion JSON file
            output_name: Output filename (without extension)

        Returns:
            Dictionary mapping format names to output paths
        """
        results = {}

        # CSV (always available)
        try:
            csv_path = self.export_csv(filepath, output_name)
            results['csv'] = csv_path
        except Exception as e:
            logger.error("CSV export failed: %s", e)

        # Parquet (optional)
        try:
            parquet_path = self.export_parquet(filepath, output_name)
            results['parquet'] = parquet_path
        except ImportError:
            logger.info("Parquet export skipped (pandas not installed)")
        except Exception as e:
            logger.error("Parquet export failed: %s", e)

        # COCO (always available)
        try:
            coco_path = self.export_coco(filepath, output_name)
            results['coco'] = coco_path
        except Exception as e:
            logger.error("COCO export failed: %s", e)

        return results


def export_motion_data(
    filepath: str,
    output_dir: str = "exports",
    formats: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Convenience function to export motion data.

    Args:
        filepath: Path to motion JSON file
        output_dir: Output directory
        formats: List of formats to export ('csv', 'parquet', 'coco', 'all')

    Returns:
        Dictionary mapping format names to output paths
    """
    exporter = MotionExporter(output_dir)

    if formats is None or 'all' in formats:
        return exporter.export_all_formats(filepath)

    results = {}
    for fmt in formats:
        if fmt == 'csv':
            results['csv'] = exporter.export_csv(filepath)
        elif fmt == 'parquet':
            results['parquet'] = exporter.export_parquet(filepath)
        elif fmt == 'coco':
            results['coco'] = exporter.export_coco(filepath)
        else:
            logger.warning("Unknown format: %s", fmt)

    return results
