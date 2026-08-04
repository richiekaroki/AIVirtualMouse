"""
Transformer-based Continuous Sign Language Recognition

Replaces the CNN+LSTM backbone with a Vision Transformer (ViT) for
sign language recognition. Uses temporal patching of landmark sequences
and self-attention for global context modeling.

Usage:
    model = TransformerCSLR(num_classes=100)
    output = model.forward(landmarks_sequence)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PatchEmbedding(nn.Module if TORCH_AVAILABLE else object):
    """
    Embeds landmark frames as patches for transformer input.

    Each frame (21 landmarks * 2 coords = 42 dims) is projected
    to embed_dim via a linear layer.
    """

    def __init__(self, input_dim: int = 42, embed_dim: int = 128, patch_size: int = 1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, num_patches, embed_dim)
        """
        B, T, D = x.shape
        if self.patch_size > 1:
            # Group consecutive frames into patches
            num_patches = T // self.patch_size
            x = x[:, :num_patches * self.patch_size]
            x = x.reshape(B, num_patches, self.patch_size * D)
        else:
            num_patches = T

        x = self.proj(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module if TORCH_AVAILABLE else object):
    """Learnable positional encoding for temporal sequences."""

    def __init__(self, max_len: int = 200, embed_dim: int = 128):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pos_embed[:, :x.size(1)]


class TransformerBlock(nn.Module if TORCH_AVAILABLE else object):
    """Standard transformer encoder block with pre-norm."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ff(normed)
        return x


class TransformerCSLR(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer-based Continuous Sign Language Recognizer.

    Architecture:
    1. Patch embedding: project landmark frames to embeddings
    2. Positional encoding: add temporal position info
    3. Transformer encoder: self-attention over sequence
    4. CTC head: per-frame token predictions
    """

    def __init__(
        self,
        input_dim: int = 42,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        num_classes: int = 100,
        max_seq_len: int = 200,
        patch_size: int = 1,
        dropout: float = 0.1,
    ):
        if not TORCH_AVAILABLE:
            return
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch + positional encoding
        self.patch_embed = PatchEmbedding(input_dim, embed_dim, patch_size)
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # CTC head (per-frame predictions)
        self.ctc_head = nn.Linear(embed_dim, num_classes)

        # Classification head (for single-gesture mode)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "ctc",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) landmark sequences
            mode: "ctc" for per-frame predictions, "cls" for single classification

        Returns:
            "ctc": (batch, seq_len, num_classes) logits
            "cls": (batch, num_classes) logits
        """
        B, T, _ = x.shape

        # Patch embedding + positional encoding
        patches = self.patch_embed(x)
        tokens = self.pos_encoding(patches)
        tokens = self.dropout(tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Transformer encoder
        for layer in self.encoder:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        if mode == "cls":
            # Use CLS token for single-gesture classification
            cls_out = tokens[:, 0]
            return self.cls_head(cls_out)
        else:
            # Per-frame CTC predictions (skip CLS token)
            frame_out = tokens[:, 1:]
            return self.ctc_head(frame_out)

    def predict(self, x: np.ndarray, mode: str = "ctc") -> Dict[str, Any]:
        """
        Predict from numpy array.

        Args:
            x: (seq_len, input_dim) or (batch, seq_len, input_dim)
            mode: "ctc" or "cls"

        Returns:
            Prediction dict
        """
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}

        if x.ndim == 2:
            x = x.unsqueeze(0)

        x_tensor = torch.FloatTensor(x)
        self.eval()

        with torch.no_grad():
            logits = self.forward(x_tensor, mode=mode)

        if mode == "cls":
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
            return {"class_id": pred_class, "confidence": confidence, "mode": "cls"}
        else:
            # CTC: greedy decode
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            # Collapse repeats and remove blanks
            tokens = []
            prev = None
            for p in preds[0]:
                p = p.item()
                if p != 0 and p != prev:
                    tokens.append(p)
                prev = p
            return {"token_ids": tokens, "mode": "ctc"}

    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss for training."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)
        return nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, targets, input_lengths, target_lengths)

    def save_model(self, path: str):
        if not TORCH_AVAILABLE:
            return
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "embed_dim": self.embed_dim,
                "num_classes": self.num_classes,
            }
        }, path)

    def load_model(self, path: str):
        if not TORCH_AVAILABLE:
            return
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])
