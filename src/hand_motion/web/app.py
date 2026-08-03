"""
Flask Web Application

Provides browser-based interface for:
- Real-time gesture visualization
- Recording and playback
- Analysis and export
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, jsonify, request, send_file
import logging

logger = logging.getLogger(__name__)


def create_app(
    static_folder: str = None,
    template_folder: str = None,
    data_dir: str = "motion_data"
) -> Flask:
    """
    Create Flask application.

    Args:
        static_folder: Static files directory
        template_folder: Templates directory
        data_dir: Motion data directory

    Returns:
        Flask application
    """
    # Default paths
    base_dir = Path(__file__).parent
    if static_folder is None:
        static_folder = str(base_dir / 'static')
    if template_folder is None:
        template_folder = str(base_dir / 'templates')

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder
    )

    app.config['DATA_DIR'] = Path(data_dir)
    app.config['DATA_DIR'].mkdir(parents=True, exist_ok=True)

    # Register routes
    register_routes(app)

    return app


def register_routes(app: Flask) -> None:
    """Register all routes."""

    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')

    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'version': '0.7.0',
            'name': 'Hand Motion Interpretation Pipeline'
        })

    @app.route('/api/recordings')
    def list_recordings():
        """List all recorded motions."""
        data_dir = app.config['DATA_DIR']
        recordings = []

        for filepath in data_dir.glob("*.json"):
            if filepath.name == "recording_manifest.json":
                continue

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                metadata = data.get('metadata', {})
                recordings.append({
                    'filename': filepath.name,
                    'gesture': metadata.get('gesture_name', 'Unknown'),
                    'frames': metadata.get('total_frames', 0),
                    'duration': metadata.get('duration_seconds', 0),
                    'fps': metadata.get('average_fps', 0)
                })
            except Exception as e:
                logger.error("Error reading %s: %s", filepath, e)

        return jsonify({
            'recordings': recordings,
            'count': len(recordings)
        })

    @app.route('/api/recording/<filename>')
    def get_recording(filename: str):
        """Get a specific recording."""
        filepath = app.config['DATA_DIR'] / filename

        if not filepath.exists():
            return jsonify({'error': 'Recording not found'}), 404

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analyze/<filename>')
    def analyze_recording(filename: str):
        """Analyze a recording."""
        filepath = app.config['DATA_DIR'] / filename

        if not filepath.exists():
            return jsonify({'error': 'Recording not found'}), 404

        try:
            from hand_motion.analyzer import MotionAnalyzer
            analyzer = MotionAnalyzer(str(filepath))
            stats = analyzer.data['metadata']

            return jsonify({
                'gesture': stats.get('gesture_name'),
                'frames': stats.get('total_frames'),
                'duration': stats.get('duration_seconds'),
                'fps': stats.get('average_fps'),
                'primitives': stats.get('primitives_used', [])
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/validate/<filename>')
    def validate_recording(filename: str):
        """Validate a recording file."""
        filepath = app.config['DATA_DIR'] / filename

        if not filepath.exists():
            return jsonify({'error': 'Recording not found'}), 404

        try:
            from hand_motion.validation import validate_motion_file
            result = validate_motion_file(str(filepath))

            return jsonify({
                'is_valid': result.is_valid,
                'score': result.score,
                'issues': [
                    {
                        'severity': issue.severity.value,
                        'message': issue.message
                    }
                    for issue in result.issues
                ]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/stats')
    def get_stats():
        """Get overall dataset statistics."""
        data_dir = app.config['DATA_DIR']
        total_files = len(list(data_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in data_dir.glob("*.json"))

        return jsonify({
            'total_recordings': total_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'data_directory': str(data_dir)
        })

    @app.route('/api/gestures')
    def list_gestures():
        """List available gesture types."""
        gestures = [
            'point', 'fist', 'open_hand', 'peace', 'thumbs_up',
            'wave', 'pinch', 'swipe', 'circle', 'push'
        ]
        return jsonify({'gestures': gestures})


def run_app(
    host: str = '0.0.0.0',
    port: int = 8000,
    debug: bool = False,
    data_dir: str = "motion_data"
) -> None:
    """
    Run the web application.

    Args:
        host: Host address
        port: Port number
        debug: Enable debug mode
        data_dir: Motion data directory
    """
    app = create_app(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("Hand Motion Interpretation Pipeline - Web Interface")
    print("=" * 60)
    print(f"\nServer running at: http://localhost:{port}")
    print(f"API docs: http://localhost:{port}/api/health")
    print(f"\nData directory: {data_dir}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(host=host, port=port, debug=debug)
