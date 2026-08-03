"""
Gesture Recognition Model

AI-powered gesture classification using:
- CNN for spatial features (handshape)
- LSTM for temporal features (motion patterns)
- Attention mechanism for important timesteps
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed. AI features will use fallback.")


class HandShapeCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    CNN for extracting spatial features from hand landmarks.
    """

    def __init__(self, input_dim: int = 63, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            return

        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features, sequence)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return x


class MotionLSTM(nn.Module if TORCH_AVAILABLE else object):
    """
    LSTM for capturing temporal patterns in motion sequences.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2):
        if not TORCH_AVAILABLE:
            return

        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, sequence, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return context, lstm_out


class GestureRecognizer(nn.Module if TORCH_AVAILABLE else object):
    """
    Complete gesture recognition model.

    Architecture:
    1. CNN extracts spatial features from hand landmarks
    2. LSTM captures temporal motion patterns
    3. Classifier predicts gesture category
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_dim: int = 63,  # 21 landmarks * 3 coordinates
        sequence_length: int = 30,
        hidden_dim: int = 256
    ):
        if not TORCH_AVAILABLE:
            self.num_classes = num_classes
            self.available = False
            return

        super().__init__()
        self.available = True
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Feature extraction
        self.cnn = HandShapeCNN(input_dim, hidden_dim // 2)
        self.lstm = MotionLSTM(hidden_dim // 2, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Logits of shape (batch, num_classes)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        batch_size, seq_len, features = x.shape

        # Reshape for CNN: (batch, features, sequence)
        x_cnn = x.permute(0, 2, 1)
        cnn_features = self.cnn(x_cnn)

        # Expand CNN features for LSTM input
        cnn_features = cnn_features.unsqueeze(1).expand(-1, seq_len, -1)

        # LSTM for temporal features
        temporal_features, _ = self.lstm(cnn_features)

        # Classification
        logits = self.classifier(temporal_features[:, -1, :])

        return logits

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Predict gesture from numpy array.

        Args:
            x: Input array of shape (sequence, features) or (batch, sequence, features)

        Returns:
            Dictionary with prediction results
        """
        if not TORCH_AVAILABLE:
            return self._fallback_predict(x)

        # Convert to tensor
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x_tensor = torch.FloatTensor(x)

        self.eval()
        with torch.no_grad():
            logits = self.forward(x_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        return {
            'class_id': pred_class,
            'confidence': confidence,
            'probabilities': probs[0].numpy().tolist()
        }

    def _fallback_predict(self, x: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction when PyTorch not available."""
        # Simple heuristic based on finger count
        if x.ndim >= 2:
            # Use last frame's features
            last_frame = x[-1] if x.ndim == 2 else x[0, -1]
            # Estimate finger count from landmark positions
            finger_count = min(5, max(0, int(np.mean(last_frame[:5]) * 5)))
        else:
            finger_count = 0

        return {
            'class_id': finger_count,
            'confidence': 0.5,
            'probabilities': [0.1] * self.num_classes,
            'fallback': True
        }

    def save_model(self, filepath: str) -> None:
        """Save model weights."""
        if not TORCH_AVAILABLE:
            logger.warning("Cannot save model: PyTorch not available")
            return

        torch.save({
            'state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'sequence_length': self.sequence_length
        }, filepath)
        logger.info("Model saved to: %s", filepath)

    def load_model(self, filepath: str) -> None:
        """Load model weights."""
        if not TORCH_AVAILABLE:
            logger.warning("Cannot load model: PyTorch not available")
            return

        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        logger.info("Model loaded from: %s", filepath)


class GestureClassifier:
    """
    High-level gesture classifier with pre-defined gestures.
    """

    # Default gesture classes
    DEFAULT_GESTURES = [
        'point', 'fist', 'open_hand', 'peace', 'thumbs_up',
        'wave', 'pinch', 'swipe', 'circle', 'push'
    ]

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize gesture classifier.

        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.gestures = self.DEFAULT_GESTURES
        self.model = GestureRecognizer(
            num_classes=len(self.gestures),
            input_dim=63,
            sequence_length=30
        )

        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)

    def classify(self, landmarks_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Classify a sequence of hand landmarks.

        Args:
            landmarks_sequence: Array of shape (sequence_length, 63)

        Returns:
            Dictionary with classification results
        """
        result = self.model.predict(landmarks_sequence)

        # Map class id to gesture name
        class_id = result['class_id']
        if 0 <= class_id < len(self.gestures):
            gesture_name = self.gestures[class_id]
        else:
            gesture_name = 'unknown'

        return {
            'gesture': gesture_name,
            'confidence': result['confidence'],
            'class_id': class_id,
            'all_probs': dict(zip(self.gestures, result['probabilities']))
        }

    def train(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        epochs: int = 50,
        learning_rate: float = 0.001
    ) -> Dict[str, List[float]]:
        """
        Train the gesture classifier.

        Args:
            train_data: Training data of shape (n_samples, sequence_length, features)
            train_labels: Training labels of shape (n_samples)
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            Dictionary with training history
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot train: PyTorch not available")
            return {'error': 'PyTorch not available'}

        # Prepare data
        X = torch.FloatTensor(train_data)
        y = torch.LongTensor(train_labels)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history = {'loss': [], 'accuracy': []}

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()

            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={accuracy:.4f}")

        return history
