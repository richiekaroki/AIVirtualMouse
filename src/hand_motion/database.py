"""
Database Storage for Datasets

SQLite-based storage for:
- Motion data persistence
- Dataset management
- Annotation storage
- Query support
"""

import json
import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MotionDatabase:
    """
    SQLite database for motion capture data.
    """

    def __init__(self, db_path: str = ':memory:'):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file (':memory:' for in-memory)
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Recordings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                source_file TEXT,
                duration_ms REAL,
                frame_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        # Frames table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER,
                frame_index INTEGER,
                timestamp_ms REAL,
                hand_landmarks TEXT,
                pose_landmarks TEXT,
                face_landmarks TEXT,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
            )
        ''')

        # Annotations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER,
                frame_start INTEGER,
                frame_end INTEGER,
                label TEXT NOT NULL,
                category TEXT,
                confidence REAL DEFAULT 1.0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
            )
        ''')

        # Datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                recording_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Dataset recordings junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_recordings (
                dataset_id INTEGER,
                recording_id INTEGER,
                split TEXT DEFAULT 'train',
                PRIMARY KEY (dataset_id, recording_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
            )
        ''')

        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_recording ON frames(recording_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_annotations_recording ON annotations(recording_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_annotations_label ON annotations(label)')

        self.conn.commit()

    def add_recording(
        self,
        name: str,
        source_file: Optional[str] = None,
        duration_ms: Optional[float] = None,
        frame_count: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a recording to the database.

        Args:
            name: Recording name
            source_file: Source video file path
            duration_ms: Duration in milliseconds
            frame_count: Number of frames
            metadata: Additional metadata

        Returns:
            Recording ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO recordings (name, source_file, duration_ms, frame_count, metadata)
               VALUES (?, ?, ?, ?, ?)''',
            (name, source_file, duration_ms, frame_count, json.dumps(metadata) if metadata else None)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_frame(
        self,
        recording_id: int,
        frame_index: int,
        timestamp_ms: float,
        hand_landmarks: Optional[List[Dict]] = None,
        pose_landmarks: Optional[List[Dict]] = None,
        face_landmarks: Optional[List[Dict]] = None
    ) -> int:
        """
        Add a frame to the database.

        Args:
            recording_id: Parent recording ID
            frame_index: Frame index
            timestamp_ms: Timestamp in milliseconds
            hand_landmarks: Hand landmark data
            pose_landmarks: Pose landmark data
            face_landmarks: Face landmark data

        Returns:
            Frame ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO frames (recording_id, frame_index, timestamp_ms,
               hand_landmarks, pose_landmarks, face_landmarks)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (
                recording_id,
                frame_index,
                timestamp_ms,
                json.dumps(hand_landmarks) if hand_landmarks else None,
                json.dumps(pose_landmarks) if pose_landmarks else None,
                json.dumps(face_landmarks) if face_landmarks else None
            )
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_frames_batch(
        self,
        recording_id: int,
        frames: List[Dict[str, Any]]
    ) -> int:
        """
        Add multiple frames in a batch.

        Args:
            recording_id: Parent recording ID
            frames: List of frame dictionaries

        Returns:
            Number of frames added
        """
        cursor = self.conn.cursor()
        data = []
        for frame in frames:
            data.append((
                recording_id,
                frame.get('frame_index', 0),
                frame.get('timestamp_ms', 0),
                json.dumps(frame.get('hand_landmarks')) if frame.get('hand_landmarks') else None,
                json.dumps(frame.get('pose_landmarks')) if frame.get('pose_landmarks') else None,
                json.dumps(frame.get('face_landmarks')) if frame.get('face_landmarks') else None
            ))

        cursor.executemany(
            '''INSERT INTO frames (recording_id, frame_index, timestamp_ms,
               hand_landmarks, pose_landmarks, face_landmarks)
               VALUES (?, ?, ?, ?, ?, ?)''',
            data
        )
        self.conn.commit()
        return len(data)

    def add_annotation(
        self,
        recording_id: int,
        label: str,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        category: Optional[str] = None,
        confidence: float = 1.0,
        notes: Optional[str] = None
    ) -> int:
        """
        Add an annotation to the database.

        Args:
            recording_id: Parent recording ID
            label: Annotation label
            frame_start: Start frame
            frame_end: End frame
            category: Annotation category
            confidence: Confidence score
            notes: Additional notes

        Returns:
            Annotation ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            '''INSERT INTO annotations (recording_id, frame_start, frame_end,
               label, category, confidence, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (recording_id, frame_start, frame_end, label, category, confidence, notes)
        )
        self.conn.commit()
        return cursor.lastrowid

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None
    ) -> int:
        """
        Create a new dataset.

        Args:
            name: Dataset name
            description: Dataset description

        Returns:
            Dataset ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO datasets (name, description) VALUES (?, ?)',
            (name, description)
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_recording_to_dataset(
        self,
        dataset_id: int,
        recording_id: int,
        split: str = 'train'
    ):
        """
        Add a recording to a dataset.

        Args:
            dataset_id: Dataset ID
            recording_id: Recording ID
            split: Data split (train/val/test)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO dataset_recordings (dataset_id, recording_id, split) VALUES (?, ?, ?)',
            (dataset_id, recording_id, split)
        )

        # Update recording count
        cursor.execute(
            '''UPDATE datasets SET recording_count = (
                SELECT COUNT(*) FROM dataset_recordings WHERE dataset_id = ?
            ), updated_at = CURRENT_TIMESTAMP WHERE id = ?''',
            (dataset_id, dataset_id)
        )
        self.conn.commit()

    def get_recording(self, recording_id: int) -> Optional[Dict]:
        """
        Get a recording by ID.

        Args:
            recording_id: Recording ID

        Returns:
            Recording dictionary or None
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM recordings WHERE id = ?', (recording_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_frames(
        self,
        recording_id: int,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None
    ) -> List[Dict]:
        """
        Get frames for a recording.

        Args:
            recording_id: Recording ID
            frame_start: Start frame index
            frame_end: End frame index

        Returns:
            List of frame dictionaries
        """
        cursor = self.conn.cursor()

        query = 'SELECT * FROM frames WHERE recording_id = ?'
        params = [recording_id]

        if frame_start is not None:
            query += ' AND frame_index >= ?'
            params.append(frame_start)
        if frame_end is not None:
            query += ' AND frame_index <= ?'
            params.append(frame_end)

        query += ' ORDER BY frame_index'

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_annotations(
        self,
        recording_id: Optional[int] = None,
        label: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Get annotations with optional filters.

        Args:
            recording_id: Filter by recording ID
            label: Filter by label
            category: Filter by category

        Returns:
            List of annotation dictionaries
        """
        cursor = self.conn.cursor()
        query = 'SELECT * FROM annotations WHERE 1=1'
        params = []

        if recording_id is not None:
            query += ' AND recording_id = ?'
            params.append(recording_id)
        if label:
            query += ' AND label = ?'
            params.append(label)
        if category:
            query += ' AND category = ?'
            params.append(category)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def search_by_label(self, label: str) -> List[Dict]:
        """
        Search for recordings with a specific label.

        Args:
            label: Label to search for

        Returns:
            List of recording dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT r.* FROM recordings r
            JOIN annotations a ON r.id = a.recording_id
            WHERE a.label = ?
        ''', (label,))
        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Statistics dictionary
        """
        cursor = self.conn.cursor()

        stats = {}
        cursor.execute('SELECT COUNT(*) FROM recordings')
        stats['recordings'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM frames')
        stats['frames'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM annotations')
        stats['annotations'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM datasets')
        stats['datasets'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT label) FROM annotations')
        stats['unique_labels'] = cursor.fetchone()[0]

        return stats

    def export_recording_frames(self, recording_id: int) -> np.ndarray:
        """
        Export recording frames as numpy array.

        Args:
            recording_id: Recording ID

        Returns:
            Numpy array of frame data
        """
        frames = self.get_frames(recording_id)
        if not frames:
            return np.array([])

        frame_data = []
        for frame in frames:
            hand = json.loads(frame['hand_landmarks']) if frame['hand_landmarks'] else []
            pose = json.loads(frame['pose_landmarks']) if frame['pose_landmarks'] else []
            face = json.loads(frame['face_landmarks']) if frame['face_landmarks'] else []

            frame_data.append({
                'timestamp_ms': frame['timestamp_ms'],
                'hand_landmarks': hand,
                'pose_landmarks': pose,
                'face_landmarks': face
            })

        return np.array(frame_data, dtype=object)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
