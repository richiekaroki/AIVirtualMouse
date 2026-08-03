"""
3D Animation Export

Export motion data for 3D animation tools:
- Blender Python format
- Three.js JSON format
- BVH motion capture format
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


class AnimationExporter:
    """
    Export motion data to various 3D animation formats.
    """

    def __init__(self):
        self.exporters = {
            'blender': self._export_blender,
            'threejs': self._export_threejs,
            'bvh': self._export_bvh
        }

    def export(
        self,
        frames: List[Dict],
        format: str = 'blender',
        output_path: str = 'animation',
        fps: int = 30
    ) -> str:
        """
        Export animation to specified format.

        Args:
            frames: List of frame data dictionaries
            format: Export format (blender, threejs, bvh)
            output_path: Output file path (without extension)
            fps: Frames per second

        Returns:
            Path to exported file
        """
        exporter = self.exporters.get(format)
        if not exporter:
            raise ValueError(f"Unsupported format: {format}")

        return exporter(frames, output_path, fps)

    def _export_blender(
        self,
        frames: List[Dict],
        output_path: str,
        fps: int
    ) -> str:
        """
        Export as Blender Python script.

        Creates a Python script that can be run in Blender to recreate the animation.
        """
        filepath = f"{output_path}_blender.py"

        lines = [
            'import bpy',
            'import math',
            '',
            '# Clear existing objects',
            'bpy.ops.object.select_all(action="SELECT")',
            'bpy.ops.object.delete()',
            '',
            '# Create armature',
            'bpy.ops.object.armature_add(enter_editmode=True)',
            'armature = bpy.context.object',
            'bones = armature.data.bones',
            '',
            '# Hand bone',
            'bpy.ops.armature.primitive_bone_add()',
            'hand_bone = bpy.context.active_object',
            'hand_bone.name = "Hand"',
            '',
            '# Animation setup',
            'scene = bpy.context.scene',
            f'scene.render.fps = {fps}',
            f'scene.frame_start = 1',
            f'scene.frame_end = {len(frames)}',
            '',
            '# Keyframe data',
        ]

        for i, frame in enumerate(frames):
            timestamp = frame.get('timestamp_ms', 0) / 1000
            hand = frame.get('hand_landmarks', {})

            if isinstance(hand, dict) and 'landmarks' in hand:
                landmarks = hand['landmarks']
                if landmarks:
                    # Get wrist position (landmark 0)
                    wrist = landmarks[0]
                    x = wrist.get('x', 0)
                    y = wrist.get('y', 0)
                    z = wrist.get('z', 0)

                    lines.append(f'# Frame {i + 1}')
                    lines.append(f'scene.frame_set({i + 1})')
                    lines.append(f'hand_bone.location = ({x}, {y}, {z})')
                    lines.append(f'hand_bone.keyframe_insert(data_path="location", frame={i + 1})')
                    lines.append('')

        lines.extend([
            '',
            '# Set interpolation to linear',
            'for action in bpy.data.actions:',
            '    for fcurve in action.fcurves:',
            '        for keyframe in fcurve.keyframe_points:',
            '            keyframe.interpolation = "LINEAR"',
            '',
            'print("Animation imported successfully!")'
        ])

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported Blender script: {filepath}")
        return filepath

    def _export_threejs(
        self,
        frames: List[Dict],
        output_path: str,
        fps: int
    ) -> str:
        """
        Export as Three.js compatible JSON.

        Creates a JSON file compatible with THREE.AnimationClip.
        """
        filepath = f"{output_path}_threejs.json"

        # Build animation clip
        tracks = []
        times = []
        positions = []
        quaternions = []

        for i, frame in enumerate(frames):
            timestamp = frame.get('timestamp_ms', 0) / 1000
            hand = frame.get('hand_landmarks', {})

            times.append(timestamp)

            if isinstance(hand, dict) and 'landmarks' in hand:
                landmarks = hand['landmarks']
                if landmarks:
                    wrist = landmarks[0]
                    positions.extend([
                        wrist.get('x', 0),
                        wrist.get('y', 0),
                        wrist.get('z', 0)
                    ])
                    # Default quaternion (no rotation)
                    quaternions.extend([0, 0, 0, 1])
                else:
                    positions.extend([0, 0, 0])
                    quaternions.extend([0, 0, 0, 1])
            else:
                positions.extend([0, 0, 0])
                quaternions.extend([0, 0, 0, 1])

        # Position track
        tracks.append({
            'name': '.position',
            'type': 'VectorKeyframeTrack',
            'times': times,
            'values': positions
        })

        # Quaternion track
        tracks.append({
            'name': '.quaternion',
            'type': 'QuaternionKeyframeTrack',
            'times': times,
            'values': quaternions
        })

        animation_clip = {
            'name': 'HandMotion',
            'duration': times[-1] if times else 0,
            'tracks': tracks
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(animation_clip, f, indent=2)

        logger.info(f"Exported Three.js animation: {filepath}")
        return filepath

    def _export_bvh(
        self,
        frames: List[Dict],
        output_path: str,
        fps: int
    ) -> str:
        """
        Export as BVH (Biovision Hierarchy) motion capture format.

        Standard format for motion capture data exchange.
        """
        filepath = f"{output_path}_motion.bvh"

        lines = [
            'HIERARCHY',
            'ROOT Hips',
            '{',
            '\tOFFSET 0.0 0.0 0.0',
            '\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation',
            '\tJOINT LeftHand',
            '\t{',
            '\t\tOFFSET 0.15 0.0 0.0',
            '\t\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation',
            '\t\tEnd Site',
            '\t\t{',
            '\t\t\tOFFSET 0.1 0.0 0.0',
            '\t\t}',
            '\t}',
            '\tJOINT RightHand',
            '\t{',
            '\t\tOFFSET -0.15 0.0 0.0',
            '\t\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation',
            '\t\tEnd Site',
            '\t\t{',
            '\t\t\tOFFSET -0.1 0.0 0.0',
            '\t\t}',
            '\t}',
            '}',
            f'MOTION',
            f'Frames: {len(frames)}',
            f'Frame Time: {1.0/fps:.6f}'
        ]

        for frame in frames:
            hand = frame.get('hand_landmarks', {})
            values = [0.0] * 12  # 2 joints × 6 channels

            if isinstance(hand, dict) and 'landmarks' in hand:
                landmarks = hand['landmarks']
                if landmarks and len(landmarks) >= 21:
                    # Left hand (average of finger tips)
                    left_x = np.mean([landmarks[i].get('x', 0) for i in [4, 8, 12, 16, 20]])
                    left_y = np.mean([landmarks[i].get('y', 0) for i in [4, 8, 12, 16, 20]])
                    left_z = np.mean([landmarks[i].get('z', 0) for i in [4, 8, 12, 16, 20]])

                    # Right hand
                    right_x = np.mean([landmarks[i].get('x', 0) for i in [4, 8, 12, 16, 20]])
                    right_y = np.mean([landmarks[i].get('y', 0) for i in [4, 8, 12, 16, 20]])
                    right_z = np.mean([landmarks[i].get('z', 0) for i in [4, 8, 12, 16, 20]])

                    # Set values
                    values[0] = left_x
                    values[1] = left_y
                    values[2] = left_z
                    values[6] = right_x
                    values[7] = right_y
                    values[8] = right_z

            lines.append('\t'.join(f'{v:.6f}' for v in values))

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Exported BVH motion: {filepath}")
        return filepath

    def create_threejs_viewer(self, animation_path: str, output_path: str = 'viewer.html'):
        """
        Create an HTML viewer for Three.js animation.

        Args:
            animation_path: Path to Three.js JSON animation
            output_path: Output HTML file path
        """
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Hand Motion Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: Arial, sans-serif; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
        }}
        #controls {{
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        button {{
            margin: 0 5px;
            padding: 5px 15px;
        }}
    </style>
</head>
<body>
    <div id="info">Hand Motion Viewer</div>
    <div id="controls">
        <button id="playBtn">Play</button>
        <button id="pauseBtn">Pause</button>
        <button id="resetBtn">Reset</button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Create hand representation
        const handGeometry = new THREE.SphereGeometry(0.05, 16, 16);
        const handMaterial = new THREE.MeshBasicMaterial({{ color: 0x00ff00 }});
        const handMesh = new THREE.Mesh(handGeometry, handMaterial);
        scene.add(handMesh);

        // Create trail
        const trailMaterial = new THREE.LineBasicMaterial({{ color: 0x0088ff }});
        const trailGeometry = new THREE.BufferGeometry();
        const trailLine = new THREE.Line(trailGeometry, trailMaterial);
        scene.add(trailLine);

        // Load animation
        let animationData = null;
        fetch('{animation_path}')
            .then(r => r.json())
            .then(data => {{
                animationData = data;
                console.log('Animation loaded:', data.name);
            }});

        // Camera position
        camera.position.z = 1;

        // Trail points
        const trailPoints = [];
        const maxTrailPoints = 100;

        // Animation state
        let isPlaying = false;
        let currentTime = 0;

        function animate() {{
            requestAnimationFrame(animate);

            if (isPlaying && animationData) {{
                currentTime += 0.016; // ~60fps

                // Find current frame
                const times = animationData.tracks[0].times;
                const positions = animationData.tracks[0].values;

                let frameIndex = 0;
                for (let i = 0; i < times.length; i++) {{
                    if (times[i] <= currentTime) {{
                        frameIndex = i;
                    }} else {{
                        break;
                    }}
                }}

                // Update hand position
                const x = positions[frameIndex * 3];
                const y = positions[frameIndex * 3 + 1];
                const z = positions[frameIndex * 3 + 2];

                handMesh.position.set(x, y, z);

                // Update trail
                trailPoints.push(new THREE.Vector3(x, y, z));
                if (trailPoints.length > maxTrailPoints) {{
                    trailPoints.shift();
                }}

                if (trailPoints.length > 1) {{
                    trailGeometry.setFromPoints(trailPoints);
                }}

                // Loop animation
                if (currentTime > animationData.duration) {{
                    currentTime = 0;
                    trailPoints.length = 0;
                }}
            }}

            renderer.render(scene, camera);
        }}

        // Controls
        document.getElementById('playBtn').onclick = () => isPlaying = true;
        document.getElementById('pauseBtn').onclick = () => isPlaying = false;
        document.getElementById('resetBtn').onclick = () => {{
            currentTime = 0;
            trailPoints.length = 0;
            handMesh.position.set(0, 0, 0);
        }};

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        animate();
    </script>
</body>
</html>'''

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Created Three.js viewer: {output_path}")
        return output_path
