"""
SQLite database layer for storing images, face detections, embeddings, and clusters.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Database:
    """Manages the SQLite database for photo faces and clustering."""

    def __init__(self, db_path: Path | str = "face_clusters.db"):
        self.db_path = str(Path(db_path).resolve())
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    box_x1 INTEGER NOT NULL,
                    box_y1 INTEGER NOT NULL,
                    box_x2 INTEGER NOT NULL,
                    box_y2 INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    cluster_id INTEGER DEFAULT -1,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
                CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
            """)

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy float32 embedding vector to raw bytes."""
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        """Deserialize raw bytes back to numpy float32 vector."""
        return np.frombuffer(blob, dtype=np.float32)

    def insert_image(self, file_path: str) -> int:
        """Insert or retrieve an image ID given its file path."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO images (file_path) VALUES (?)",
                (file_path,)
            )
            cursor.execute(
                "SELECT id FROM images WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return int(row["id"])

    def insert_face(
        self,
        image_id: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        embedding: np.ndarray,
        cluster_id: int = -1,
    ) -> int:
        """Insert a detected face with its embedding."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        emb_blob = self.serialize_embedding(embedding)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO faces (image_id, box_x1, box_y1, box_x2, box_y2, confidence, embedding, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(image_id), x1, y1, x2, y2, float(confidence), emb_blob, int(cluster_id))
            )
            return cursor.lastrowid or 0

    def get_all_unclustered_or_all_faces(self) -> List[Dict[str, Any]]:
        """Retrieve all face records and their deserialized embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.image_id, f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                       f.confidence, f.embedding, f.cluster_id, i.file_path
                FROM faces f
                JOIN images i ON f.image_id = i.id
                ORDER BY f.id ASC
            """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r["id"],
                    "image_id": r["image_id"],
                    "bbox": (int(r["box_x1"]), int(r["box_y1"]), int(r["box_x2"]), int(r["box_y2"])),
                    "confidence": float(r["confidence"]),
                    "embedding": self.deserialize_embedding(r["embedding"]),
                    "cluster_id": int(r["cluster_id"]),
                    "file_path": r["file_path"],
                })
            return results

    def update_face_clusters(self, face_ids: List[int], cluster_ids: List[int]) -> None:
        """Update cluster IDs for a batch of faces."""
        if len(face_ids) != len(cluster_ids):
            raise ValueError("face_ids and cluster_ids must have the same length")
        with self._get_connection() as conn:
            conn.executemany(
                "UPDATE faces SET cluster_id = ? WHERE id = ?",
                [(int(c_id), int(f_id)) for f_id, c_id in zip(face_ids, cluster_ids)]
            )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as cnt FROM images")
            images_scanned = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM faces")
            faces_detected = cursor.fetchone()["cnt"]

            # Count clusters (excluding -1 unclustered/noise)
            cursor.execute("""
                SELECT COUNT(DISTINCT cluster_id) as cnt
                FROM faces
                WHERE cluster_id >= 0
            """)
            num_clusters = cursor.fetchone()["cnt"]

            # Per cluster breakdown
            cursor.execute("""
                SELECT
                    cluster_id,
                    COUNT(faces.id) as face_count,
                    COUNT(DISTINCT images.id) as image_count
                FROM faces
                JOIN images ON faces.image_id = images.id
                WHERE cluster_id >= 0
                GROUP BY cluster_id
                ORDER BY face_count DESC, cluster_id ASC
            """)
            cluster_rows = cursor.fetchall()
            clusters = [
                {
                    "cluster_id": r["cluster_id"],
                    "faces_count": r["face_count"],
                    "images_count": r["image_count"],
                }
                for r in cluster_rows
            ]

            # Unclustered / noise count
            cursor.execute("""
                SELECT COUNT(*) as cnt FROM faces WHERE cluster_id < 0
            """)
            unclustered_count = cursor.fetchone()["cnt"]

            return {
                "images_scanned": images_scanned,
                "faces_detected": faces_detected,
                "num_clusters": num_clusters,
                "clusters": clusters,
                "unclustered_faces": unclustered_count,
            }

    def inspect_cluster(self, cluster_id: int) -> Dict[str, Any]:
        """Retrieve all details and associated image paths for a given cluster."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    f.id as face_id,
                    f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                    f.confidence,
                    i.id as image_id,
                    i.file_path
                FROM faces f
                JOIN images i ON f.image_id = i.id
                WHERE f.cluster_id = ?
                ORDER BY i.file_path ASC, f.id ASC
            """, (cluster_id,))
            rows = cursor.fetchall()

            faces = []
            image_paths = set()
            for r in rows:
                image_paths.add(r["file_path"])
                faces.append({
                    "face_id": r["face_id"],
                    "image_id": r["image_id"],
                    "file_path": r["file_path"],
                    "bbox": (int(r["box_x1"]), int(r["box_y1"]), int(r["box_x2"]), int(r["box_y2"])),
                    "confidence": float(r["confidence"]),
                })

            return {
                "cluster_id": cluster_id,
                "total_faces": len(faces),
                "total_images": len(image_paths),
                "image_paths": sorted(list(image_paths)),
                "faces": faces,
            }

    def clear_all(self) -> None:
        """Clear all records from images and faces."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM faces;")
            conn.execute("DELETE FROM images;")
