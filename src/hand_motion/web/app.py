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

from flask import Flask, render_template, jsonify, request, make_response
from flask_socketio import SocketIO, emit
import logging

logger = logging.getLogger(__name__)

_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://hand-motion-pipeline.onrender.com,http://localhost:*,http://127.0.0.1:*"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
# Use wildcard if any entry contains a wildcard pattern
if any("*" in o for o in ALLOWED_ORIGINS):
    ALLOWED_ORIGINS = "*"
socketio = SocketIO(cors_allowed_origins=ALLOWED_ORIGINS, async_mode="threading")


def _safe_filename(filename: str, data_dir: Path) -> Path | None:
    """Sanitize a filename and return resolved path if valid, None otherwise."""
    if not filename:
        return None
    name = Path(filename).name
    if ".." in name or "/" in name or "\\" in name:
        return None
    if not name.endswith(".json"):
        return None
    resolved = (data_dir / name).resolve()
    if not str(resolved).startswith(str(data_dir.resolve())):
        return None
    return resolved


def _security_headers(response):
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def create_app(
    static_folder: str = None,
    template_folder: str = None,
    data_dir: str = None,
) -> Flask:
    """Create Flask application with SocketIO."""
    base_dir = Path(__file__).parent
    project_root = base_dir.parent.parent.parent
    if data_dir is None:
        data_dir = str(project_root / "motion_data")
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

    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        if os.environ.get("FLASK_ENV") == "production":
            raise RuntimeError("SECRET_KEY environment variable is required in production")
        secret_key = "hand-motion-dev-key"
    app.config["SECRET_KEY"] = secret_key

    app.after_request(_security_headers)

    socketio.init_app(app)

    register_routes(app)
    register_socket_handlers(app)

    return app


def register_routes(app: Flask) -> None:
    """Register HTTP routes."""

    @app.route("/")
    def index():
        from flask import request
        base_url = request.host_url.rstrip("/")
        return render_template("index.html", base_url=base_url)

    @app.route("/robots.txt")
    def robots_txt():
        from flask import Response, request
        base = request.host_url.rstrip("/")
        content = f"User-agent: *\nAllow: /\nDisallow: /api/\nSitemap: {base}/sitemap.xml\n"
        return Response(content, mimetype="text/plain")

    @app.route("/sitemap.xml")
    def sitemap_xml():
        from flask import Response, request
        base = request.host_url.rstrip("/")
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>{base}/</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base}/api/health</loc>
    <changefreq>daily</changefreq>
    <priority>0.3</priority>
  </url>
</urlset>"""
        return Response(xml, mimetype="application/xml")

    @app.route("/api/health")
    def health():
        from hand_motion import __version__
        return jsonify({
            "status": "healthy",
            "version": __version__,
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
        filepath = _safe_filename(filename, app.config["DATA_DIR"])
        if filepath is None or not filepath.exists():
            return jsonify({"error": "Recording not found"}), 404
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": "Failed to read recording"}), 500

    @app.route("/api/analyze/<filename>")
    def analyze_recording(filename: str):
        filepath = _safe_filename(filename, app.config["DATA_DIR"])
        if filepath is None or not filepath.exists():
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
            return jsonify({"error": "Failed to analyze recording"}), 500

    @app.route("/api/validate/<filename>")
    def validate_recording(filename: str):
        filepath = _safe_filename(filename, app.config["DATA_DIR"])
        if filepath is None or not filepath.exists():
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
            return jsonify({"error": "Failed to validate recording"}), 500

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

    @app.route("/api/diagnostics")
    def get_diagnostics():
        return jsonify({
            "architecture": "client_side_detection",
            "frames_received": _server_frame_counter,
            "active_streams": len([s for s in _camera_streams.values() if s.get("active")]),
            "active_recordings": len([s for s in _recording_sessions.values() if s.get("active")]),
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
_recording_sessions: Dict[str, Dict[str, Any]] = {}

# Server frame counter (from client-side detection)
_server_frame_counter = 0


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
        """Receive landmarks from client-side MediaPipe, store for recording/playback."""
        global _server_frame_counter
        try:
            landmarks = data.get("landmarks")
            gesture = data.get("gesture")
            fingers = data.get("fingers")
            confidence = data.get("confidence", 0)
            width = data.get("width", 640)
            height = data.get("height", 480)
            client_ts = data.get("client_ts", 0)

            if not landmarks:
                return

            _server_frame_counter += 1

            sid = request.sid
            if sid in _recording_sessions and _recording_sessions[sid].get("active"):
                _recording_sessions[sid]["frames"].append({
                    "landmarks": landmarks,
                    "gesture": gesture,
                    "fingers": fingers,
                    "confidence": confidence,
                    "timestamp": client_ts,
                })

        except Exception as e:
            logger.error("Frame processing error: %s", e)

    @socketio.on("play_recording")
    def handle_play_recording(data):
        """Stream a recorded gesture file frame by frame."""
        sid = request.sid
        filename = data.get("filename")
        speed = data.get("speed", 1.0)

        if not filename:
            emit("playback_error", {"error": "No filename provided"})
            return

        filepath = _safe_filename(filename, app.config["DATA_DIR"])
        if filepath is None or not filepath.exists():
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

                    raw_lm = frame.get("landmarks", [])
                    if raw_lm and isinstance(raw_lm[0], list):
                        flat = []
                        for pt in raw_lm:
                            flat.extend([pt[1], pt[2]])
                        landmarks = flat
                    else:
                        landmarks = raw_lm

                    socketio.emit("playback_frame", {
                        "frame_index": i,
                        "landmarks": landmarks,
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
    data_dir: str = None,
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
