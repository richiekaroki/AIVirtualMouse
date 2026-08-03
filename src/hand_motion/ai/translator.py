"""
Gesture Translation Pipeline

Translates gesture sequences into text or gloss notation:
- Sequence-to-sequence translation
- Gloss notation generation
- Context-aware translation
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


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
