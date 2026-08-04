"""
Flask Web Application with WebSocket Support

Provides browser-based interface for:
- Real-time gesture visualization via WebSocket
- Recording and playback
- Analysis and export
- Live camera feed with gesture recognition
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import json
import sys
import time
import base64
import threading
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import logging

logger = logging.getLogger(__name__)

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")


def create_app(
    static_folder: str = None,
    template_folder: str = None,
    data_dir: str = "motion_data",
) -> Flask:
    """Create Flask application with SocketIO."""
    base_dir = Path(__file__).parent
    if static_folder is None:
        static_folder = str(base_dir / "static")
    if template_folder is None:
        template_folder = str(base_dir / "templates")

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder,
    )
    app.config["DATA_DIR"] = Path(data_dir)
    app.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "hand-motion-dev-key")

    socketio.init_app(app)

    register_routes(app)
    register_socket_handlers(app)

    return app


def register_routes(app: Flask) -> None:
    """Register HTTP routes."""

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "healthy",
            "version": "0.8.0",
            "name": "Hand Motion Interpretation Pipeline",
            "websocket": True,
        })

    @app.route("/api/recordings")
    def list_recordings():
        data_dir = app.config["DATA_DIR"]
        recordings = []
        for filepath in data_dir.glob("*.json"):
            if filepath.name == "recording_manifest.json":
                continue
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                recordings.append({
                    "filename": filepath.name,
                    "gesture": meta.get("gesture_name", "Unknown"),
                    "frames": meta.get("total_frames", 0),
                    "duration": meta.get("duration_seconds", 0),
                    "fps": meta.get("average_fps", 0),
                })
            except Exception as e:
                logger.error("Error reading %s: %s", filepath, e)
        return jsonify({"recordings": recordings, "count": len(recordings)})

    @app.route("/api/recording/<filename>")
    def get_recording(filename: str):
        filepath = app.config["DATA_DIR"] / filename
        if not filepath.exists():
            return jsonify({"error": "Recording not found"}), 404
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/analyze/<filename>")
    def analyze_recording(filename: str):
        filepath = app.config["DATA_DIR"] / filename
        if not filepath.exists():
            return jsonify({"error": "Recording not found"}), 404
        try:
            from hand_motion.analyzer import MotionAnalyzer
            analyzer = MotionAnalyzer(str(filepath))
            stats = analyzer.data["metadata"]
            return jsonify({
                "gesture": stats.get("gesture_name"),
                "frames": stats.get("total_frames"),
                "duration": stats.get("duration_seconds"),
                "fps": stats.get("average_fps"),
                "primitives": stats.get("primitives_used", []),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/validate/<filename>")
    def validate_recording(filename: str):
        filepath = app.config["DATA_DIR"] / filename
        if not filepath.exists():
            return jsonify({"error": "Recording not found"}), 404
        try:
            from hand_motion.validation import validate_motion_file
            result = validate_motion_file(str(filepath))
            return jsonify({
                "is_valid": result.is_valid,
                "score": result.score,
                "issues": [
                    {"severity": issue.severity.value, "message": issue.message}
                    for issue in result.issues
                ],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/stats")
    def get_stats():
        data_dir = app.config["DATA_DIR"]
        total_files = len(list(data_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in data_dir.glob("*.json"))
        return jsonify({
            "total_recordings": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_directory": str(data_dir),
        })

    @app.route("/api/gestures")
    def list_gestures():
        gestures = [
            "point", "fist", "open_hand", "peace", "thumbs_up",
            "wave", "pinch", "swipe", "circle", "push",
        ]
        return jsonify({"gestures": gestures})


# Global state for camera streaming
_camera_streams: Dict[str, Dict[str, Any]] = {}

# Lazily initialized detectors per-thread
_detector = None
_descriptor = None
_server_frame_counter = 0
_mediapipe_lock = threading.Lock()


class _Suppress_stderr:
    """Suppress C++ level MediaPipe/TensorFlow warnings written to stderr."""
    def __enter__(self):
        self._original = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._original


def _get_detector():
    global _detector, _descriptor
    if _detector is None:
        from hand_motion.detection import HandDetector
        from hand_motion.descriptor import MotionDescriptor
        _detector = HandDetector(detectionCon=0.7)
        _descriptor = MotionDescriptor()
    return _detector, _descriptor


def register_socket_handlers(app: Flask) -> None:
    """Register WebSocket event handlers."""

    @socketio.on("connect")
    def handle_connect():
        logger.info("Client connected: %s", request.sid)
        emit("connected", {"status": "ok", "sid": request.sid})

    @socketio.on("disconnect")
    def handle_disconnect():
        sid = request.sid
        if sid in _camera_streams:
            _camera_streams[sid]["active"] = False
            del _camera_streams[sid]
        logger.info("Client disconnected: %s", sid)

    @socketio.on("process_frame")
    def handle_process_frame(data):
        """Receive a frame from the browser, run MediaPipe, return gesture results."""
        global _server_frame_counter
        try:
            import cv2
            import numpy as np

            img_b64 = data.get("image")
            width = data.get("width", 640)
            height = data.get("height", 480)
            client_ts = data.get("client_ts", 0)

            if not img_b64:
                emit("frame_result", {"error": "No image data"})
                return

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                emit("frame_result", {"error": "Failed to decode image"})
                return

            detector, descriptor = _get_detector()

            img = cv2.flip(img, 1)
            with _mediapipe_lock:
                with _Suppress_stderr():
                    img = detector.findHands(img, draw=False)
                    lm_list, bbox = detector.findPosition(img, draw=False)

            _server_frame_counter += 1

            result = {
                "frame": _server_frame_counter,
                "client_ts": client_ts,
                "hands_detected": bool(lm_list and len(lm_list) > 0),
                "landmarks": [],
                "gesture": None,
                "confidence": 0,
                "fingers": [],
                "features": None,
                "velocity": None,
            }

            if lm_list and len(lm_list) != 0:
                fingers = detector.fingersUp()
                descriptor.create_descriptor(lm_list, fingers, frame_shape=(height, width))

                flat = []
                for lm in lm_list:
                    flat.extend([lm[1], lm[2]])

                last_desc = descriptor.motion_history[-1] if descriptor.motion_history else None

                result.update({
                    "landmarks": flat,
                    "fingers": fingers,
                    "gesture": last_desc["primitive"] if last_desc else None,
                    "features": last_desc["features"] if last_desc else None,
                    "velocity": last_desc["velocity"] if last_desc else None,
                })

            emit("frame_result", result)

        except Exception as e:
            logger.error("Frame processing error: %s", e)
            emit("frame_result", {"error": str(e)})

    @socketio.on("start_camera")
    def handle_start_camera(data):
        """Start streaming camera feed with gesture recognition (server-side camera)."""
        sid = request.sid
        camera_index = data.get("camera", 0)
        width = data.get("width", 640)
        height = data.get("height", 480)

        if sid in _camera_streams and _camera_streams[sid].get("active"):
            emit("camera_error", {"error": "Camera already streaming"})
            return

        _camera_streams[sid] = {"active": True, "camera": camera_index}

        def stream_worker():
            try:
                import cv2

                cap = cv2.VideoCapture(camera_index)
                cap.set(3, width)
                cap.set(4, height)

                if not cap.isOpened():
                    socketio.emit("camera_error", {"error": f"Cannot open camera {camera_index}"}, room=sid)
                    return

                detector, descriptor = _get_detector()

                socketio.emit("camera_started", {"width": width, "height": height}, room=sid)

                frame_count = 0
                while _camera_streams.get(sid, {}).get("active", False):
                    ret, img = cap.read()
                    if not ret:
                        break

                    img = cv2.flip(img, 1)
                    with _mediapipe_lock:
                        with _Suppress_stderr():
                            img = detector.findHands(img, draw=False)
                            lm_list, bbox = detector.findPosition(img, draw=False)

                    gesture_data = {
                        "frame": frame_count,
                        "timestamp": time.time(),
                        "hands_detected": bool(lm_list and len(lm_list) > 0),
                    }

                    if lm_list and len(lm_list) != 0:
                        fingers = detector.fingersUp()
                        descriptor.create_descriptor(lm_list, fingers, frame_shape=(height, width))

                        landmarks_flat = []
                        for lm in lm_list:
                            landmarks_flat.extend([lm[1], lm[2]])

                        last_desc = descriptor.motion_history[-1] if descriptor.motion_history else None

                        gesture_data.update({
                            "landmarks": landmarks_flat,
                            "fingers": fingers,
                            "bbox": bbox,
                            "gesture": last_desc["primitive"] if last_desc else None,
                            "features": last_desc["features"] if last_desc else None,
                            "velocity": last_desc["velocity"] if last_desc else None,
                        })

                        _, img_encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        gesture_data["image"] = base64.b64encode(img_encoded).decode("utf-8")

                    socketio.emit("gesture_frame", gesture_data, room=sid)
                    frame_count += 1

                    socketio.sleep(0.03)

                cap.release()
                socketio.emit("camera_stopped", {}, room=sid)

            except Exception as e:
                logger.error("Camera stream error: %s", e)
                socketio.emit("camera_error", {"error": str(e)}, room=sid)

        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()

    @socketio.on("stop_camera")
    def handle_stop_camera():
        """Stop camera streaming."""
        sid = request.sid
        if sid in _camera_streams:
            _camera_streams[sid]["active"] = False
        emit("camera_stopping", {})

    @socketio.on("play_recording")
    def handle_play_recording(data):
        """Stream a recorded gesture file frame by frame."""
        sid = request.sid
        filename = data.get("filename")
        speed = data.get("speed", 1.0)

        if not filename:
            emit("playback_error", {"error": "No filename provided"})
            return

        data_dir = Path("motion_data")
        filepath = data_dir / filename
        if not filepath.exists():
            emit("playback_error", {"error": "File not found"})
            return

        def play_worker():
            try:
                with open(filepath, "r") as f:
                    recording = json.load(f)

                frames = recording.get("frames", [])
                metadata = recording.get("metadata", {})

                socketio.emit("playback_started", {
                    "filename": filename,
                    "gesture": metadata.get("gesture_name"),
                    "total_frames": len(frames),
                }, room=sid)

                for i, frame in enumerate(frames):
                    if not _camera_streams.get(sid, {}).get("active", True):
                        break

                    socketio.emit("playback_frame", {
                        "frame_index": i,
                        "landmarks": frame.get("landmarks", []),
                        "primitive": frame.get("primitive"),
                        "finger_states": frame.get("finger_states"),
                        "features": frame.get("features"),
                        "velocity": frame.get("velocity"),
                    }, room=sid)

                    delay = 1.0 / 30.0 / speed
                    socketio.sleep(delay)

                socketio.emit("playback_finished", {}, room=sid)

            except Exception as e:
                socketio.emit("playback_error", {"error": str(e)}, room=sid)

        _camera_streams[sid] = {"active": True}
        thread = threading.Thread(target=play_worker, daemon=True)
        thread.start()

    @socketio.on("stop_playback")
    def handle_stop_playback():
        sid = request.sid
        if sid in _camera_streams:
            _camera_streams[sid]["active"] = False

    @socketio.on("classify_landmarks")
    def handle_classify(data):
        """Classify a set of landmarks using the ML classifier."""
        try:
            from hand_motion.ai.landmark_classifier import LandmarkClassifier
            classifier = LandmarkClassifier()
            landmarks = data.get("landmarks", [])

            if len(landmarks) == 42:
                lm_list = [[i // 2, landmarks[i], landmarks[i + 1]] for i in range(0, 42, 2)]
            elif len(landmarks) == 63:
                lm_list = [[landmarks[i * 3], landmarks[i * 3 + 1], landmarks[i * 3 + 2]]
                           for i in range(21)]
            else:
                emit("classification_result", {"error": "Expected 42 or 63 values"})
                return

            result = classifier.predict(lm_list)
            emit("classification_result", result)
        except Exception as e:
            emit("classification_result", {"error": str(e)})


def run_app(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    data_dir: str = "motion_data",
) -> None:
    """Run the web application with WebSocket support."""
    app = create_app(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("Hand Motion Interpretation Pipeline - Web Interface")
    print("=" * 60)
    print(f"\nServer running at: http://localhost:{port}")
    print(f"API docs: http://localhost:{port}/api/health")
    print(f"WebSocket: ws://localhost:{port}")
    print(f"\nData directory: {data_dir}")
    print("\nFeatures:")
    print("  - Real-time gesture streaming via WebSocket")
    print("  - Camera feed with skeleton overlay")
    print("  - Recording playback and analysis")
    print("  - ML gesture classification")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
