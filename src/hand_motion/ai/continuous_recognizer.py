"""
Continuous Sign Language Recognition

CTC and attention-based decoding for variable-length gloss sequences.
Extends the CNN+LSTM backbone for sequence-to-sequence prediction.

Usage:
    recognizer = ContinuousRecognizer(vocab_size=100)
    glosses = recognizer.decode(landmarks_sequence)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed. Continuous recognizer unavailable.")


class CTCDecoder:
    """
    Beam search and greedy CTC decoding.

    Collapses repeated tokens and removes blanks from model output.
    """

    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def greedy_decode(self, logits: np.ndarray) -> List[int]:
        """
        Greedy CTC decoding: argmax + collapse repeats + remove blanks.

        Args:
            logits: Array of shape (T, vocab_size)

        Returns:
            List of token IDs
        """
        preds = np.argmax(logits, axis=-1)
        collapsed = []
        prev = None
        for p in preds:
            if p != self.blank_id and p != prev:
                collapsed.append(int(p))
            prev = p
        return collapsed

    def beam_search_decode(
        self,
        logits: np.ndarray,
        beam_width: int = 5,
        lm_weight: float = 0.3,
    ) -> Tuple[List[int], float]:
        """
        Beam search CTC decoding with optional language model.

        Args:
            logits: Array of shape (T, vocab_size)
            beam_width: Number of beams to keep
            lm_weight: Weight for language model score (0 = no LM)

        Returns:
            (best_sequence, score)
        """
        T, V = logits.shape
        log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

        # Each beam: (prefix_tuple, last_blank, score)
        beams = [((), True, 0.0)]

        for t in range(T):
            new_beams = {}
            for prefix, last_blank, score in beams:
                for v in range(V):
                    new_score = score + log_probs[t, v]
                    if v == self.blank_id:
                        key = prefix
                        new_beams[key] = max(new_beams.get(key, (None, None, -np.inf)), (prefix, True, new_score), key=lambda x: x[2])
                    elif last_blank or (prefix and prefix[-1] == v):
                        key = prefix + (v,) if not last_blank or (prefix and prefix[-1] != v) else prefix
                        if prefix and prefix[-1] == v and not last_blank:
                            key = prefix
                        else:
                            key = prefix + (v,)
                        new_beams[key] = max(new_beams.get(key, (None, None, -np.inf)), (key, False, new_score), key=lambda x: x[2])
                    else:
                        key = prefix + (v,)
                        new_beams[key] = max(new_beams.get(key, (None, None, -np.inf)), (key, False, new_score), key=lambda x: x[2])

            # Prune to beam_width
            beams = sorted(new_beams.values(), key=lambda x: x[2], reverse=True)[:beam_width]

        if not beams:
            return [], 0.0

        best = beams[0]
        return list(best[0]), best[2]


class ContinuousCTCDecoder(nn.Module if TORCH_AVAILABLE else object):
    """
    CTC head on top of existing CNN+LSTM backbone.

    Produces per-timestep logits for CTC loss and greedy/beam decoding.
    """

    def __init__(self, hidden_dim: int = 256, vocab_size: int = 100):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, time, hidden_dim)

        Returns:
            (batch, time, vocab_size) log-probs (after log_softmax)
        """
        logits = self.proj(lstm_out)
        return F.log_softmax(logits, dim=-1)


class AttentionDecoder(nn.Module if TORCH_AVAILABLE else object):
    """
    Attention-based decoder for sequence-to-sequence gloss prediction.
    """

    def __init__(self, hidden_dim: int = 256, vocab_size: int = 100, num_layers: int = 1, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        encoder_out: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        max_len: int = 50,
    ) -> torch.Tensor:
        """
        Teacher-forced during training, auto-regressive during inference.

        Args:
            encoder_out: (batch, time, hidden_dim)
            target_seq: (batch, seq_len) ground truth (None = inference)
            max_len: Max decoding length during inference

        Returns:
            (batch, max_len, vocab_size) logits
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # Initial hidden state from encoder mean
        context = encoder_out.mean(dim=1, keepdim=True)  # (batch, 1, hidden)

        if target_seq is not None:
            # Teacher forcing
            embeds = self.embedding(target_seq)  # (batch, tgt_len, hidden)
            lstm_out, _ = self.rnn(embeds)
            logits = self.output_proj(lstm_out)
            return logits
        else:
            # Auto-regressive inference
            outputs = []
            input_tok = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            for _ in range(max_len):
                embeds = self.embedding(input_tok)
                lstm_out, _ = self.rnn(embeds)
                logits = self.output_proj(lstm_out)
                outputs.append(logits)
                input_tok = logits.argmax(dim=-1)
            return torch.cat(outputs, dim=1)


class ContinuousRecognizer:
    """
    High-level API for continuous sign language recognition.

    Combines CNN+LSTM backbone with CTC or attention decoding.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        hidden_dim: int = 256,
        input_dim: int = 63,
        decoder_type: str = "ctc",
    ):
        """
        Args:
            vocab_size: Number of gloss tokens (including blank)
            hidden_dim: LSTM hidden dimension
            input_dim: Feature dimension per frame
            decoder_type: "ctc" or "attention"
        """
        self.vocab_size = vocab_size
        self.decoder_type = decoder_type
        self.ctc_decoder = CTCDecoder(blank_id=0)
        self.gloss_vocab: Dict[int, str] = {}

        if TORCH_AVAILABLE:
            from hand_motion.ai.gesture_recognizer import HandShapeCNN, MotionLSTM
            self.cnn = HandShapeCNN(input_dim, hidden_dim // 2)
            self.lstm = MotionLSTM(hidden_dim // 2, hidden_dim)

            if decoder_type == "ctc":
                self.decoder = ContinuousCTCDecoder(hidden_dim, vocab_size)
            else:
                self.decoder = AttentionDecoder(hidden_dim, vocab_size)
        else:
            self.cnn = None
            self.lstm = None
            self.decoder = None

    def set_gloss_vocab(self, vocab: Dict[int, str]):
        """Set the gloss vocabulary mapping token_id -> string."""
        self.gloss_vocab = vocab

    def decode_sequence(self, logits: np.ndarray) -> str:
        """
        Decode model output logits to a gloss string.

        Args:
            logits: (time, vocab_size) array

        Returns:
            Space-separated gloss string
        """
        if self.decoder_type == "ctc":
            token_ids = self.ctc_decoder.greedy_decode(logits)
        else:
            token_ids = list(np.argmax(logits, axis=-1))
            # Remove repeats for attention output
            collapsed = []
            prev = None
            for t in token_ids:
                if t != prev and t != 0:
                    collapsed.append(int(t))
                prev = t
            token_ids = collapsed

        glosses = [self.gloss_vocab.get(tid, f"<{tid}>") for tid in token_ids]
        return " ".join(glosses)

    def predict(self, landmarks_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Run continuous recognition on a landmark sequence.

        Args:
            landmarks_sequence: (time, features) numpy array

        Returns:
            Dict with 'glosses', 'token_ids', 'raw_logits'
        """
        if not TORCH_AVAILABLE or self.cnn is None:
            return {"glosses": "", "token_ids": [], "fallback": True}

        import torch

        x = torch.FloatTensor(landmarks_sequence)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        with torch.no_grad():
            x_cnn = x.permute(0, 2, 1)
            cnn_feat = self.cnn(x_cnn)
            cnn_expanded = cnn_feat.unsqueeze(1).expand(-1, x.size(1), -1)
            lstm_out, _ = self.lstm(cnn_expanded)
            logits = self.decoder(lstm_out)

        logits_np = logits.squeeze(0).cpu().numpy()
        gloss_str = self.decode_sequence(logits_np)
        token_ids = self.ctc_decoder.greedy_decode(logits_np) if self.decoder_type == "ctc" else list(np.argmax(logits_np, axis=-1))

        return {
            "glosses": gloss_str,
            "token_ids": token_ids,
        }

    def compute_ctc_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss for training.

        Args:
            logits: (batch, time, vocab_size) log-probs
            targets: (batch, max_target_len) token IDs
            input_lengths: (batch,) actual lengths of inputs
            target_lengths: (batch,) actual lengths of targets

        Returns:
            Scalar CTC loss
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CTC loss")

        log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, batch, V)
        return nn.CTCLoss(blank=0, zero_infinity=True)(
            log_probs, targets, input_lengths, target_lengths
        )
