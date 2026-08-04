"""
Video Export Module

Renders annotated videos with hand skeleton overlays and gesture labels.
Supports recording playback, real-time capture, and batch export.

Usage:
    exporter = VideoExporter()
    exporter.render_recording("motion_data/point_sample.json", "output/point_annotated.mp4")
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

COLOR_SKELETON = (0, 255, 255)
COLOR_LANDMARKS = (0, 140, 255)
COLOR_JOINTS = (255, 255, 255)
COLOR_LABEL_BG = (40, 40, 40)
COLOR_LABEL_TEXT = (255, 255, 255)
COLOR_FPS = (0, 255, 0)
COLOR_VELOCITY = (255, 0, 255)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGERTIP_IDS = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}


class VideoExporter:
    """Export motion recordings as annotated videos."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_recording(
        self,
        json_path: str,
        output_path: Optional[str] = None,
        fps: int = 30,
        show_skeleton: bool = True,
        show_labels: bool = True,
        show_velocity: bool = True,
        show_timeline: bool = True,
        canvas_width: int = 960,
    ) -> str:
        """
        Render a gesture recording as an annotated video.

        Args:
            json_path: Path to recording JSON file
            output_path: Output video path (auto-generated if None)
            fps: Output video FPS
            show_skeleton: Draw hand skeleton
            show_labels: Show gesture label overlay
            show_velocity: Show velocity arrow
            show_timeline: Show primitive timeline at bottom
            canvas_width: Width of output canvas

        Returns:
            Path to rendered video
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        frames = data.get("frames", [])
        gesture_name = metadata.get("gesture_name", "unknown")

        if not frames:
            raise ValueError("No frames in recording")

        sample_lm = frames[0].get("landmarks", [])
        if len(sample_lm) >= 2:
            max_x = max(abs(lm[1]) for lm in sample_lm if len(lm) >= 2)
            max_y = max(abs(lm[2]) for lm in sample_lm if len(lm) >= 2)
        else:
            max_x, max_y = 640, 480

        cam_w = max(640, int(max_x * 1.2))
        cam_h = max(480, int(max_y * 1.2))

        if output_path is None:
            stem = Path(json_path).stem
            output_path = str(self.output_dir / f"{stem}_annotated.mp4")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        panel_width = canvas_width - cam_w
        if panel_width < 250:
            canvas_width = cam_w + 300
            panel_width = 300

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, cam_h))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        prev_palm = None
        primitives_seen = []

        for i, frame in enumerate(frames):
            lm_list = frame.get("landmarks", [])
            primitive = frame.get("primitive", "UNKNOWN")
            velocity = frame.get("velocity", {})
            finger_states = frame.get("finger_states", [])

            canvas = np.zeros((cam_h, canvas_width, 3), dtype=np.uint8)
            cam_view = canvas[:, :cam_w]
            cam_view[:] = (30, 30, 30)

            if len(lm_list) >= 21:
                if show_skeleton:
                    self._draw_skeleton(cam_view, lm_list)
                if show_velocity and velocity:
                    palm = frame.get("features", {}).get("palm_center", {})
                    if palm and prev_palm:
                        self._draw_velocity_arrow(
                            cam_view,
                            (int(prev_palm["x"]), int(prev_palm["y"])),
                            (int(palm["x"]), int(palm["y"])),
                        )
                    prev_palm = palm

            if show_labels:
                self._draw_label_overlay(
                    canvas[:, cam_w:], gesture_name, primitive,
                    finger_states, velocity, frame, cam_h, panel_width,
                )

            if show_timeline:
                if primitive not in primitives_seen or (
                    primitives_seen and primitives_seen[-1] != primitive
                ):
                    if primitive not in primitives_seen:
                        primitives_seen.append(primitive)
                self._draw_timeline(canvas, primitives_seen, primitive, cam_w, canvas_width, cam_h)

            frame_ts = frame.get("timestamp", 0)
            frame_idx = frame.get("frame_index", i)
            cv2.putText(canvas, f"Frame {frame_idx}", (10, cam_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            writer.write(canvas)

        writer.release()
        logger.info("Rendered %d frames to %s", len(frames), output_path)
        return output_path

    def render_camera_feed(
        self,
        output_path: str,
        camera_index: int = 0,
        duration_s: float = 5.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
    ) -> str:
        """
        Capture from camera and render annotated video in real-time.

        Args:
            output_path: Output video path
            camera_index: Camera device index
            duration_s: Recording duration in seconds
            fps: Target FPS
            resolution: (width, height)

        Returns:
            Path to rendered video
        """
        from hand_motion.detection import HandDetector
        from hand_motion.descriptor import MotionDescriptor

        cap = cv2.VideoCapture(camera_index)
        cap.set(3, resolution[0])
        cap.set(4, resolution[1])

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        detector = HandDetector(detectionCon=0.7)
        descriptor = MotionDescriptor()

        w, h = resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        start = time.time()
        frame_count = 0

        try:
            while time.time() - start < duration_s:
                ret, img = cap.read()
                if not ret:
                    break

                img = cv2.flip(img, 1)
                img = detector.findHands(img, draw=False)
                lm_list, _ = detector.findPosition(img, draw=False)

                if len(lm_list) != 0:
                    fingers = detector.fingersUp()
                    descriptor.create_descriptor(lm_list, fingers, frame_shape=(h, w))
                    self._draw_skeleton(img, lm_list)

                elapsed = time.time() - start
                cv2.putText(img, f"{elapsed:.1f}s / {duration_s:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_FPS, 2)
                cv2.putText(img, f"Frames: {frame_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FPS, 2)

                writer.write(img)
                frame_count += 1
        finally:
            cap.release()
            writer.release()

        logger.info("Captured %d frames to %s", frame_count, output_path)
        return output_path

    def batch_render(
        self,
        data_dir: str,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Render all recordings in a directory.

        Args:
            data_dir: Input directory with JSON files
            output_dir: Output directory for videos
            **kwargs: Additional args for render_recording

        Returns:
            List of output video paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        data_path = Path(data_dir)
        outputs = []

        for json_file in sorted(data_path.glob("*.json")):
            if json_file.name == "recording_manifest.json":
                continue
            try:
                out = self.render_recording(str(json_file), **kwargs)
                outputs.append(out)
            except Exception as e:
                logger.error("Failed to render %s: %s", json_file, e)

        logger.info("Batch rendered %d/%d videos", len(outputs), len(list(data_path.glob("*.json"))))
        return outputs

    def _draw_skeleton(self, img: np.ndarray, lm_list: List[List[float]]) -> None:
        """Draw hand skeleton with connections and joints."""
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(lm_list) and end_idx < len(lm_list):
                pt1 = (int(lm_list[start_idx][1]), int(lm_list[start_idx][2]))
                pt2 = (int(lm_list[end_idx][1]), int(lm_list[end_idx][2]))
                cv2.line(img, pt1, pt2, COLOR_SKELETON, 2)

        for lm in lm_list:
            x, y = int(lm[1]), int(lm[2])
            cv2.circle(img, (x, y), 4, COLOR_JOINTS, -1)
            cv2.circle(img, (x, y), 6, COLOR_LANDMARKS, 1)

        for tip_id, name in FINGERTIP_IDS.items():
            if tip_id < len(lm_list):
                x, y = int(lm_list[tip_id][1]), int(lm_list[tip_id][2])
                cv2.circle(img, (x, y), 7, (0, 255, 0), 2)

    def _draw_velocity_arrow(
        self, img: np.ndarray, from_pt: Tuple[int, int], to_pt: Tuple[int, int]
    ) -> None:
        """Draw velocity direction arrow."""
        cv2.arrowedLine(img, from_pt, to_pt, COLOR_VELOCITY, 2, tipLength=0.3)

    def _draw_label_overlay(
        self, panel: np.ndarray, gesture_name: str, primitive: str,
        finger_states: List[int], velocity: Dict, frame: Dict,
        height: int, width: int,
    ) -> None:
        """Draw information overlay on the side panel."""
        panel[:] = (40, 40, 40)

        y = 30
        cv2.putText(panel, "GESTURE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        cv2.putText(panel, gesture_name.upper(), (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 40

        cv2.putText(panel, "PRIMITIVE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        y += 25
        color = (0, 255, 0) if primitive != "UNKNOWN" else (150, 150, 150)
        cv2.putText(panel, primitive, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 40

        if finger_states:
            cv2.putText(panel, "FINGERS", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 25
            labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for j, (label, state) in enumerate(zip(labels, finger_states)):
                c = (0, 255, 0) if state else (100, 100, 100)
                txt = f"{label}: {'UP' if state else 'DOWN'}"
                cv2.putText(panel, txt, (15, y + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            y += len(finger_states) * 22 + 15

        features = frame.get("features", {})
        if features:
            cv2.putText(panel, "FEATURES", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 25
            openness = features.get("hand_openness", 0)
            cv2.putText(panel, f"Openness: {openness:.2f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LABEL_TEXT, 1)
            y += 20
            pinch = features.get("pinch_distance", 0)
            cv2.putText(panel, f"Pinch: {pinch:.1f}px", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LABEL_TEXT, 1)
            y += 20
            span = features.get("hand_span", 0)
            cv2.putText(panel, f"Span: {span:.1f}px", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_LABEL_TEXT, 1)
            y += 30

        if velocity and velocity.get("magnitude", 0) > 0:
            cv2.putText(panel, "VELOCITY", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            y += 25
            mag = velocity["magnitude"]
            cv2.putText(panel, f"{mag:.1f} px/s", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VELOCITY, 2)

    def _draw_timeline(
        self, canvas: np.ndarray, primitives: List[str], current: str,
        cam_w: int, canvas_w: int, canvas_h: int,
    ) -> None:
        """Draw primitive timeline bar at the bottom."""
        bar_h = 40
        bar_y = canvas_h - bar_h
        cv2.rectangle(canvas, (0, bar_y), (canvas_w, canvas_h), (50, 50, 50), -1)

        n = len(primitives)
        if n == 0:
            return

        seg_w = canvas_w // max(n, 1)
        colors = {
            "POINT": (255, 140, 0), "OPEN_HAND": (0, 255, 0), "FIST": (0, 0, 255),
            "PEACE": (255, 0, 255), "THUMBS_UP": (0, 255, 255), "PINCH": (255, 255, 0),
            "WAVE": (128, 0, 255), "SWIPE_RIGHT": (0, 128, 255), "SWIPE_LEFT": (255, 128, 0),
            "CIRCLE": (0, 255, 128), "GRAB": (128, 128, 0), "TWO_FINGERS": (0, 128, 128),
            "THREE_FINGERS": (128, 0, 128), "OK_SIGN": (200, 200, 0), "PUSH": (200, 0, 200),
        }

        for i, prim in enumerate(primitives):
            x1 = i * seg_w
            x2 = (i + 1) * seg_w if i < n - 1 else canvas_w
            c = colors.get(prim, (150, 150, 150))
            if prim == current:
                c = tuple(min(255, ch + 60) for ch in c)
            cv2.rectangle(canvas, (x1, bar_y), (x2, canvas_h), c, -1)
            cv2.rectangle(canvas, (x1, bar_y), (x2, canvas_h), (200, 200, 200), 1)

            txt = prim[:6]
            tx = x1 + 5
            ty = bar_y + 25
            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
