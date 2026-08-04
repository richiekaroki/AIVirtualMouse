"""
Lightweight Gesture Classifier

ML-based gesture recognition using scikit-learn on MediaPipe hand landmarks.
No GPU required — works with the project's existing recording format.

Usage:
    classifier = LandmarkClassifier()
    classifier.train_from_directory("motion_data/")
    result = classifier.predict(landmarks)
"""

import json
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("scikit-learn not installed. ML classifier unavailable.")


def _landmarks_to_features(lm_list: List[List[float]]) -> np.ndarray:
    """
    Convert 21 hand landmarks to a fixed-size feature vector.

    Features (total 78 dims):
    - 42: normalized x,y offsets of 21 points from wrist
    - 10: distances from each fingertip to wrist
    - 5: fingertip-to-fingertip distances (thumb-index, index-middle, etc.)
    - 10: angles between consecutive finger chains
    - 5: finger extension ratios (tip-to-MCP / wrist-to-MCP)
    - 6: palm geometry (width, height, area approx)
    """
    if len(lm_list) < 21:
        return np.zeros(78, dtype=np.float32)

    wrist = np.array([lm_list[0][1], lm_list[0][2]], dtype=np.float32)

    points = np.array([[lm[1], lm[2]] for lm in lm_list[:21]], dtype=np.float32)

    offsets = (points - wrist).flatten() / 300.0

    tip_ids = [4, 8, 12, 16, 20]
    tip_to_wrist = [math.hypot(points[t][0] - wrist[0], points[t][1] - wrist[1]) / 300.0 for t in tip_ids]

    tip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)]
    tip_to_tip = [math.hypot(points[a][0] - points[b][0], points[a][1] - points[b][1]) / 300.0 for a, b in tip_pairs]

    def angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return math.acos(np.clip(cos_a, -1, 1))

    chain_angles = [
        angle(points[0], points[5], points[8]),
        angle(points[0], points[9], points[12]),
        angle(points[0], points[13], points[16]),
        angle(points[0], points[17], points[20]),
        angle(points[5], points[9], points[13]),
    ]
    chain_angles2 = [
        angle(points[5], points[6], points[8]),
        angle(points[9], points[10], points[12]),
        angle(points[13], points[14], points[16]),
        angle(points[17], points[18], points[20]),
        angle(points[0], points[17], points[20]),
    ]

    mcp_ids = [2, 5, 9, 13, 17]
    finger_ext = []
    for tip, mcp in zip(tip_ids, mcp_ids):
        wrist_to_mcp = math.hypot(points[mcp][0] - wrist[0], points[mcp][1] - wrist[1]) + 1e-8
        wrist_to_tip = math.hypot(points[tip][0] - wrist[0], points[tip][1] - wrist[1])
        finger_ext.append(wrist_to_tip / wrist_to_mcp)

    palm_w = math.hypot(points[5][0] - points[17][0], points[5][1] - points[17][1]) / 300.0
    palm_h = math.hypot(points[0][0] - points[9][0], points[0][1] - points[9][1]) / 300.0
    palm_area = palm_w * palm_h
    palm_diag = math.hypot(points[5][0] - points[17][0], points[5][1] - points[17][1]) / 300.0
    palm_center_x = (points[5][0] + points[17][0]) / 2 / 640.0
    palm_center_y = (points[5][1] + points[17][1]) / 2 / 480.0

    features = np.concatenate([
        np.array(offsets, dtype=np.float32),
        np.array(tip_to_wrist, dtype=np.float32),
        np.array(tip_to_tip, dtype=np.float32),
        np.array(chain_angles + chain_angles2, dtype=np.float32),
        np.array(finger_ext, dtype=np.float32),
        np.array([palm_w, palm_h, palm_area, palm_diag, palm_center_x, palm_center_y], dtype=np.float32),
    ])

    return features


class LandmarkClassifier:
    """
    Lightweight ML gesture classifier operating on MediaPipe landmarks.

    Uses scikit-learn (RandomForest + GradientBoosting ensemble) with
    hand-crafted geometric features — no GPU, no deep learning.
    """

    DEFAULT_MODEL_PATH = "models/landmark_classifier.pkl"

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.gesture_names: List[str] = []

        if SKLEARN_AVAILABLE:
            self.pipeline = Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1,
                )),
            ])
        else:
            self.pipeline = None

        if os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def extract_features(self, lm_list: List[List[float]]) -> np.ndarray:
        """Extract feature vector from 21 hand landmarks."""
        return _landmarks_to_features(lm_list)

    def train_from_directory(self, data_dir: str, min_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Train classifier from a directory of recording JSON files.

        Args:
            data_dir: Path to directory containing gesture JSON files
            min_confidence: Minimum confidence threshold for including frames

        Returns:
            Training report with accuracy and gesture counts
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not installed"}

        X_all = []
        y_all = []

        data_path = Path(data_dir)
        for json_file in data_path.glob("*.json"):
            if json_file.name == "recording_manifest.json":
                continue
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                gesture = data.get("metadata", {}).get("gesture_name", "unknown")
                frames = data.get("frames", [])

                for frame in frames:
                    lm = frame.get("landmarks", [])
                    if len(lm) < 21:
                        continue
                    features = self.extract_features(lm)
                    X_all.append(features)
                    y_all.append(gesture)
            except Exception as e:
                logger.warning("Skipping %s: %s", json_file, e)

        if not X_all:
            return {"error": "No training data found"}

        X = np.array(X_all)
        y = np.array(y_all)

        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.gesture_names = list(self.label_encoder.classes_)

        cv_folds = min(3, min(Counter(y).values()))
        scores = cross_val_score(self.pipeline, X, y_encoded, cv=cv_folds, scoring="accuracy")

        self.pipeline.fit(X, y_encoded)
        self.is_trained = True

        report = {
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "n_samples": len(X),
            "n_classes": len(set(y)),
            "gestures": dict(Counter(y)),
            "feature_dim": X.shape[1],
        }
        logger.info("Trained classifier: %.1f%% accuracy (%d samples, %d classes)",
                     report["accuracy_mean"] * 100, report["n_samples"], report["n_classes"])

        return report

    def predict(self, lm_list: List[List[float]]) -> Dict[str, Any]:
        """
        Predict gesture from hand landmarks.

        Args:
            lm_list: List of 21 landmarks [id, x, y]

        Returns:
            Dict with 'gesture', 'confidence', 'probabilities'
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return self._rule_based_predict(lm_list)

        features = self.extract_features(lm_list).reshape(1, -1)
        pred_encoded = self.pipeline.predict(features)[0]
        probs = self.pipeline.predict_proba(features)[0]

        gesture = self.label_encoder.inverse_transform([pred_encoded])[0]
        confidence = float(np.max(probs))

        prob_dict = {
            self.label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }

        return {
            "gesture": gesture,
            "confidence": confidence,
            "probabilities": prob_dict,
            "method": "ml",
        }

    def _rule_based_predict(self, lm_list: List[List[float]]) -> Dict[str, Any]:
        """Fallback rule-based prediction when no ML model is available."""
        if len(lm_list) < 21:
            return {"gesture": "unknown", "confidence": 0.0, "method": "rule"}

        tip_ids = [4, 8, 12, 16, 20]
        mcp_ids = [2, 5, 9, 13, 17]
        fingers = []
        for tip, mcp in zip(tip_ids, mcp_ids):
            extended = 1 if lm_list[tip][2] < lm_list[mcp][2] else 0
            fingers.append(extended)

        finger_sum = sum(fingers)
        thumb_index_dist = math.hypot(
            lm_list[4][1] - lm_list[8][1], lm_list[4][2] - lm_list[8][2]
        )

        if finger_sum == 0:
            gesture = "fist"
        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "thumbs_up"
        elif fingers == [0, 1, 0, 0, 0]:
            gesture = "point"
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "peace"
        elif finger_sum == 5:
            gesture = "open_hand"
        elif thumb_index_dist < 30:
            gesture = "pinch"
        else:
            gesture = "open_hand"

        return {"gesture": gesture, "confidence": 0.7, "method": "rule"}

    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            logger.warning("No trained model to save")
            return

        path = filepath or self.model_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        data = {
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "gesture_names": self.gesture_names,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Model saved to %s", path)

    def load_model(self, filepath: Optional[str] = None) -> bool:
        """Load trained model from disk."""
        path = filepath or self.model_path
        if not os.path.exists(path):
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.pipeline = data["pipeline"]
            self.label_encoder = data["label_encoder"]
            self.gesture_names = data["gesture_names"]
            self.is_trained = True
            logger.info("Loaded model from %s (%d gestures)", path, len(self.gesture_names))
            return True
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False
