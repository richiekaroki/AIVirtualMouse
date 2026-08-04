"""
Gesture Translation Pipeline

Translates gesture sequences into text or gloss notation:
- Sequence-to-sequence translation
- Gloss notation generation
- Context-aware translation
- Attention-based neural NLP for gloss-to-text
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GlossTranslator:
    """
    Translates gesture sequences to gloss notation (sign language transcription).
    """

    # ASL gloss symbols
    GLOSS_SYMBOLS = {
        'point': 'POINT',
        'fist': 'FIST',
        'open_hand': 'OPEN',
        'peace': 'PEACE',
        'thumbs_up': 'THUMBS',
        'wave': 'WAVE',
        'pinch': 'PINCH',
        'swipe': 'SWIPE',
        'circle': 'CIRCLE',
        'push': 'PUSH',
        'pull': 'PULL',
        'uncertain': '?'
    }

    def __init__(self):
        self.symbol_map = self.GLOSS_SYMBOLS

    def translate_sequence(
        self,
        gesture_sequence: List[str],
        timestamps: Optional[List[float]] = None
    ) -> str:
        """
        Translate a sequence of gestures to gloss notation.

        Args:
            gesture_sequence: List of gesture names
            timestamps: Optional timestamps for duration markers

        Returns:
            Gloss notation string
        """
        if not gesture_sequence:
            return ""

        gloss_parts = []
        prev_gesture = None
        repeat_count = 0

        for i, gesture in enumerate(gesture_sequence):
            symbol = self.symbol_map.get(gesture, gesture.upper())

            # Handle repetition
            if gesture == prev_gesture:
                repeat_count += 1
            else:
                if repeat_count > 1:
                    gloss_parts[-1] = f"{gloss_parts[-1]}x{repeat_count}"
                repeat_count = 1
                gloss_parts.append(symbol)

            prev_gesture = gesture

        # Handle final repetition
        if repeat_count > 1 and gloss_parts:
            gloss_parts[-1] = f"{gloss_parts[-1]}x{repeat_count}"

        return " ".join(gloss_parts)


class TextTranslator:
    """
    Translates gesture sequences to natural language text.
    """

    # Gesture to text mappings
    GLOSS_TO_TEXT = {
        'point': 'pointing',
        'fist': 'closed fist',
        'open_hand': 'open hand',
        'peace': 'peace sign',
        'thumbs_up': 'thumbs up',
        'wave': 'waving',
        'pinch': 'pinching',
        'swipe': 'swiping',
        'circle': 'circling',
        'push': 'pushing',
        'pull': 'pulling'
    }

    # Common gesture sequences to phrases
    SEQUENCE_PHRASES = {
        ('open_hand', 'wave'): 'hello',
        ('point', 'circle'): 'select all',
        ('fist', 'open_hand'): 'grab and release',
        ('thumbs_up', 'point'): 'good, look at this',
        ('pinch', 'swipe'): 'zoom'
    }

    def __init__(self):
        self.gloss_to_text = self.GLOSS_TO_TEXT
        self.sequence_phrases = self.SEQUENCE_PHRASES

    def translate_sequence(self, gesture_sequence: List[str]) -> str:
        """
        Translate gesture sequence to text.

        Args:
            gesture_sequence: List of gesture names

        Returns:
            Natural language text
        """
        if not gesture_sequence:
            return ""

        # Check for known sequences
        for seq_len in range(min(3, len(gesture_sequence)), 1, -1):
            for i in range(len(gesture_sequence) - seq_len + 1):
                subseq = tuple(gesture_sequence[i:i + seq_len])
                if subseq in self.sequence_phrases:
                    return self.sequence_phrases[subseq]

        # Fall back to individual gesture translations
        texts = []
        for gesture in gesture_sequence:
            text = self.gloss_to_text.get(gesture, gesture)
            texts.append(text)

        return " and ".join(texts)


class GestureTranslator:
    """
    Main translation interface for gesture sequences.
    """

    def __init__(self, gloss_dict_path: Optional[str] = None):
        """
        Initialize translator.

        Args:
            gloss_dict_path: Path to custom gloss dictionary JSON
        """
        self.gloss_translator = GlossTranslator()
        self.text_translator = TextTranslator()

        # Load custom dictionary if provided
        if gloss_dict_path and os.path.exists(gloss_dict_path):
            self._load_custom_dict(gloss_dict_path)

    def _load_custom_dict(self, path: str) -> None:
        """Load custom gloss dictionary."""
        try:
            with open(path, 'r') as f:
                custom_dict = json.load(f)

            # Update symbol maps
            self.gloss_translator.symbol_map.update(custom_dict.get('gloss', {}))
            self.text_translator.gloss_to_text.update(custom_dict.get('text', {}))

            logger.info("Loaded custom dictionary from: %s", path)
        except Exception as e:
            logger.error("Failed to load custom dictionary: %s", e)

    def translate(
        self,
        gesture_sequence: List[str],
        format: str = 'text',
        timestamps: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Translate gesture sequence.

        Args:
            gesture_sequence: List of gesture names
            format: Output format ('text', 'gloss', 'json')
            timestamps: Optional timestamps

        Returns:
            Translation result dictionary
        """
        if not gesture_sequence:
            return {
                'text': '',
                'gloss': '',
                'gestures': [],
                'confidence': 0.0
            }

        # Generate translations
        gloss = self.gloss_translator.translate_sequence(gesture_sequence, timestamps)
        text = self.text_translator.translate_sequence(gesture_sequence)

        # Calculate gesture frequency
        gesture_counts = Counter(gesture_sequence)
        total = len(gesture_sequence)
        gesture_freq = {g: c / total for g, c in gesture_counts.items()}

        # Calculate confidence based on consistency
        unique_gestures = len(gesture_counts)
        confidence = 1.0 / unique_gestures if unique_gestures > 0 else 0.0

        result = {
            'text': text,
            'gloss': gloss,
            'gestures': gesture_sequence,
            'gesture_frequency': gesture_freq,
            'confidence': confidence,
            'sequence_length': len(gesture_sequence)
        }

        if format == 'json':
            return result
        elif format == 'gloss':
            return {'gloss': gloss}
        else:
            return {'text': text}

    def translate_realtime(
        self,
        current_gesture: str,
        history: List[str],
        context_window: int = 5
    ) -> Dict[str, Any]:
        """
        Translate in real-time with context.

        Args:
            current_gesture: Current gesture
            history: Recent gesture history
            context_window: Number of past gestures to consider

        Returns:
            Real-time translation result
        """
        # Get context window
        context = history[-context_window:] if len(history) > context_window else history
        context.append(current_gesture)

        # Translate
        result = self.translate(context, format='json')

        # Add real-time info
        result['is_stable'] = len(set(context[-3:])) == 1 if len(context) >= 3 else False
        result['gesture_count'] = len(set(context))

        return result


class SignLanguageDictionary:
    """
    Dictionary of sign language gloss notations.
    """

    def __init__(self, dict_path: Optional[str] = None):
        self.entries = {}

        if dict_path and os.path.exists(dict_path):
            self.load(dict_path)
        else:
            self._load_default()

    def _load_default(self) -> None:
        """Load default dictionary entries."""
        self.entries = {
            'hello': {
                'gloss': 'HELLO',
                'gestures': ['open_hand', 'wave'],
                'category': 'greeting'
            },
            'thank_you': {
                'gloss': 'THANK-YOU',
                'gestures': ['open_hand', 'point'],
                'category': 'politeness'
            },
            'yes': {
                'gloss': 'YES',
                'gestures': ['fist', 'nod'],
                'category': 'affirmation'
            },
            'no': {
                'gloss': 'NO',
                'gestures': ['fist', 'swipe'],
                'category': 'negation'
            },
            'please': {
                'gloss': 'PLEASE',
                'gestures': ['open_hand', 'circle'],
                'category': 'politeness'
            },
            'help': {
                'gloss': 'HELP',
                'gestures': ['fist', 'open_hand', 'push'],
                'category': 'request'
            },
            'stop': {
                'gloss': 'STOP',
                'gestures': ['open_hand', 'push'],
                'category': 'command'
            }
        }

    def save(self, path: str) -> None:
        """Save dictionary to JSON."""
        with open(path, 'w') as f:
            json.dump(self.entries, f, indent=2)
        logger.info("Dictionary saved to: %s", path)

    def load(self, path: str) -> None:
        """Load dictionary from JSON."""
        with open(path, 'r') as f:
            self.entries = json.load(f)
        logger.info("Dictionary loaded from: %s", path)

    def lookup(self, gesture_sequence: List[str]) -> Optional[str]:
        """Look up gesture sequence in dictionary."""
        for word, entry in self.entries.items():
            if entry.get('gestures') == gesture_sequence:
                return word
        return None

    def add_entry(self, word: str, gloss: str, gestures: List[str], category: str = 'custom') -> None:
        """Add a new dictionary entry."""
        self.entries[word] = {
            'gloss': gloss,
            'gestures': gestures,
            'category': category
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search dictionary by word or gloss."""
        results = []
        query = query.lower()

        for word, entry in self.entries.items():
            if query in word.lower() or query in entry.get('gloss', '').lower():
                results.append({
                    'word': word,
                    **entry
                })

        return results


# ── Neural Seq2Seq Gloss-to-Text Model ──────────────────────────────────

class GlossEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Encoder: embeds gloss tokens and produces hidden states."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len) token IDs
        Returns:
            outputs: (batch, seq_len, hidden_dim)
            hidden: (batch, hidden_dim)
        """
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        # Combine bidirectional hidden
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = torch.tanh(self.fc(hidden))
        return outputs, hidden


class GlossAttention(nn.Module if TORCH_AVAILABLE else object):
    """Bahdanau attention over encoder outputs."""

    def __init__(self, hidden_dim: int):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_dim) decoder hidden
            encoder_outputs: (batch, src_len, hidden_dim)
        Returns:
            attention weights: (batch, src_len)
        """
        src_len = encoder_outputs.size(1)
        hidden_expanded = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden_expanded, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class TextDecoder(nn.Module if TORCH_AVAILABLE else object):
    """Decoder: generates text tokens one at a time with attention."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.3):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = GlossAttention(hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2 + embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        """
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))  # (batch, 1, embed)

        attn_weights = self.attention(hidden, encoder_outputs)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden*2)

        rnn_input = torch.cat([embedded, context], dim=2)  # (batch, 1, embed + hidden*2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=1))
        return prediction, hidden, attn_weights


class Seq2SeqTranslator(nn.Module if TORCH_AVAILABLE else object):
    """
    Full encoder-decoder model for gloss-to-text translation.
    """

    PAD_TOKEN = 0
    SOS_TOKEN = 1
    EOS_TOKEN = 2

    def __init__(self, src_vocab: int, tgt_vocab: int, embed_dim: int = 64, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.encoder = GlossEncoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = TextDecoder(tgt_vocab, embed_dim, hidden_dim)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: (batch, src_len) source token IDs
            tgt: (batch, tgt_len) target token IDs
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            outputs: (batch, tgt_len, tgt_vocab_size) logits
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab).to(src.device)
        encoder_outputs, hidden = self.encoder(src)

        input_token = tgt[:, 0]  # SOS token

        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = prediction

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = prediction.argmax(dim=1)

        return outputs


class NeuralGlossTranslator:
    """
    High-level wrapper for the seq2seq gloss-to-text model.

    Provides train/translate interface with vocabulary management.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.src_vocab: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.tgt_vocab: Dict[str, int] = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.src_idx_to_token: Dict[int, str] = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx_to_token: Dict[int, str] = {v: k for k, v in self.tgt_vocab.items()}

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _build_vocab(self, sequences: List[List[str]], min_count: int = 1):
        """Build vocabulary from token sequences."""
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        for token, count in counter.items():
            if count >= min_count and token not in self.src_vocab:
                idx = len(self.src_vocab)
                self.src_vocab[token] = idx
                self.src_idx_to_token[idx] = token

    def _encode(self, tokens: List[str], vocab: Dict[str, int]) -> List[int]:
        return [vocab.get(t, vocab["<sos>"]) for t in tokens]

    def _decode(self, indices: List[int]) -> List[str]:
        tokens = []
        for idx in indices:
            if idx == self.tgt_vocab["<eos>"]:
                break
            if idx not in (self.tgt_vocab["<pad>"], self.tgt_vocab["<sos>"]):
                tokens.append(self.tgt_idx_to_token.get(idx, "<unk>"))
        return tokens

    def translate(self, gloss_sequence: List[str]) -> str:
        """
        Translate a gloss sequence to text using the neural model.
        Falls back to rule-based translation if model not available.
        """
        if not TORCH_AVAILABLE or self.model is None:
            return " ".join(gloss_sequence)

        src_tokens = self._encode(gloss_sequence, self.src_vocab)
        src_tensor = torch.LongTensor([src_tokens])

        self.model.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)

            input_token = torch.LongTensor([self.tgt_vocab["<sos>"]])
            decoded_tokens = []

            for _ in range(50):
                prediction, hidden, _ = self.model.decoder(input_token, hidden, encoder_outputs)
                top1 = prediction.argmax(dim=1)
                decoded_tokens.append(top1.item())
                input_token = top1
                if top1.item() == self.tgt_vocab["<eos>"]:
                    break

        result_tokens = self._decode(decoded_tokens)
        return " ".join(result_tokens)

    def save_model(self, path: str):
        if not TORCH_AVAILABLE or self.model is None:
            return
        torch.save({
            "state_dict": self.model.state_dict(),
            "src_vocab": self.src_vocab,
            "tgt_vocab": self.tgt_vocab,
        }, path)

    def load_model(self, path: str):
        if not TORCH_AVAILABLE:
            return
        checkpoint = torch.load(path, map_location="cpu")
        self.src_vocab = checkpoint["src_vocab"]
        self.tgt_vocab = checkpoint["tgt_vocab"]
        self.src_idx_to_token = {v: k for k, v in self.src_vocab.items()}
        self.tgt_idx_to_token = {v: k for k, v in self.tgt_vocab.items()}

        self.model = Seq2SeqTranslator(len(self.src_vocab), len(self.tgt_vocab))
        self.model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded neural translator from %s", path)
