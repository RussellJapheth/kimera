"""
SQLite database layer for storing images, face detections, embeddings, clusters, and named people.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}

LOW_QUALITY_FACE_MIN_SIDE = 48
LOW_QUALITY_FACE_MIN_CONF = 0.70


def format_file_size(size_bytes: int | float | None) -> str:
    """Format byte count into human-readable size string (e.g. '12.4 MB')."""
    if not size_bytes or size_bytes <= 0:
        return "0 B"
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} B"
            elif unit == "KB":
                return f"{size:.1f} KB"
            elif unit == "MB":
                return f"{size:.1f} MB"
            else:
                return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{int(size_bytes)} B"


def format_duration(duration_sec: float | int | None) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if not duration_sec or duration_sec <= 0:
        return ""
    total_sec = int(round(float(duration_sec)))
    hours = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


class Database:
    """Manages the SQLite database for photo faces, clustering, favorites, and people."""

    def __init__(self, db_path: Path | str = "face_clusters.db"):
        self.db_path = str(Path(db_path).resolve())
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite tables and apply migrations safely in order."""
        with self._get_connection() as conn:
            # 1. Ensure core tables exist
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_favorite INTEGER DEFAULT 0,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    mtime REAL DEFAULT 0.0,
                    content_hash TEXT DEFAULT '',
                    duration REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    cover_face_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    person_id INTEGER DEFAULT NULL,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_exclusions (
                    face_id INTEGER NOT NULL,
                    person_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (face_id, person_id),
                    FOREIGN KEY (face_id) REFERENCES faces(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS image_tags (
                    image_id INTEGER NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (image_id, tag_id),
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS media_embeddings (
                    image_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    model TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
                );
            """)

            # 2. Run column migrations for older database schemas
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(images)")
            image_cols = {col["name"] for col in cursor.fetchall()}
            if "is_favorite" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN is_favorite INTEGER DEFAULT 0")
            if "width" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN width INTEGER DEFAULT 0")
            if "height" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN height INTEGER DEFAULT 0")
            if "file_size" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN file_size INTEGER DEFAULT 0")
            if "mtime" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN mtime REAL DEFAULT 0.0")
            if "content_hash" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN content_hash TEXT DEFAULT ''")
            if "duration" not in image_cols:
                cursor.execute("ALTER TABLE images ADD COLUMN duration REAL DEFAULT 0.0")

            cursor.execute("PRAGMA table_info(faces)")
            face_cols = {col["name"] for col in cursor.fetchall()}
            if "person_id" not in face_cols:
                cursor.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER DEFAULT NULL")

            # 3. Create indices after verifying all columns exist
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(is_favorite);
                CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path);
                CREATE INDEX IF NOT EXISTS idx_images_file_size ON images(file_size);
                CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);
                CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
                CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
                CREATE INDEX IF NOT EXISTS idx_person_exclusions_face ON person_exclusions(face_id);
                CREATE INDEX IF NOT EXISTS idx_person_exclusions_person ON person_exclusions(person_id);
                CREATE INDEX IF NOT EXISTS idx_image_tags_image ON image_tags(image_id);
                CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id);
                CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
                CREATE INDEX IF NOT EXISTS idx_media_embeddings_image ON media_embeddings(image_id);
            """)

            # 4. Migrate and collapse any legacy video keyframe entries (e.g. "...#t=1.23s") to single clean video files
            cursor.execute("SELECT id, file_path, width, height, is_favorite FROM images WHERE file_path LIKE '%#t=%'")
            legacy_video_rows = cursor.fetchall()
            if legacy_video_rows:
                for row in legacy_video_rows:
                    old_id = row["id"]
                    old_path = row["file_path"]
                    clean_path = old_path.split("#t=")[0]

                    cursor.execute("SELECT id FROM images WHERE file_path = ?", (clean_path,))
                    clean_row = cursor.fetchone()
                    if clean_row:
                        clean_id = clean_row["id"]
                    else:
                        cursor.execute(
                            "INSERT INTO images (file_path, width, height, is_favorite) VALUES (?, ?, ?, ?)",
                            (clean_path, row["width"] or 0, row["height"] or 0, row["is_favorite"] or 0)
                        )
                        clean_id = cursor.lastrowid

                    cursor.execute("UPDATE faces SET image_id = ? WHERE image_id = ?", (clean_id, old_id))
                    cursor.execute("DELETE FROM images WHERE id = ?", (old_id,))

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize numpy float32 embedding vector to raw bytes."""
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize_embedding(blob: bytes) -> np.ndarray:
        """Deserialize raw bytes back to numpy float32 vector."""
        return np.frombuffer(blob, dtype=np.float32)

    def insert_image(
        self,
        file_path: str,
        width: int = 0,
        height: int = 0,
        file_size: int = 0,
        mtime: float = 0.0,
        content_hash: str = "",
        duration: float = 0.0,
    ) -> int:
        """Insert or retrieve an image ID given its file path and metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO images (file_path, width, height, file_size, mtime, content_hash, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (file_path, int(width), int(height), int(file_size), float(mtime), content_hash, float(duration))
            )
            if (width > 0 and height > 0) or file_size > 0 or duration > 0.0:
                cursor.execute(
                    """
                    UPDATE images 
                    SET width = COALESCE(NULLIF(?, 0), width),
                        height = COALESCE(NULLIF(?, 0), height),
                        file_size = COALESCE(NULLIF(?, 0), file_size),
                        mtime = COALESCE(NULLIF(?, 0.0), mtime),
                        content_hash = COALESCE(NULLIF(?, ''), content_hash),
                        duration = COALESCE(NULLIF(?, 0.0), duration)
                    WHERE file_path = ?
                    """,
                    (int(width), int(height), int(file_size), float(mtime), content_hash, float(duration), file_path)
                )
            cursor.execute(
                "SELECT id FROM images WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return int(row["id"])

    def get_image_file_meta(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve fast stat/hash metadata for an indexed file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, file_path, file_size, mtime, content_hash, width, height, duration FROM images WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def get_all_images_file_meta_map(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve a dictionary mapping file_path -> file_meta for all indexed files."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, file_size, mtime, content_hash, width, height, duration FROM images")
            rows = cursor.fetchall()
            return {str(row["file_path"]): dict(row) for row in rows}

    def get_hash_records(self) -> List[Dict[str, Any]]:
        """Return id, file_path, file_size, mtime, content_hash for every indexed file."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, file_path, file_size, mtime, content_hash FROM images ORDER BY id ASC"
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_duplicate_groups(self) -> Dict[str, Any]:
        """
        Group indexed media by content fingerprint and return only groups with 2+ files.
        The first indexed file (lowest id) is designated as the "keep" record.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, file_path, file_size, content_hash FROM images WHERE content_hash != '' ORDER BY id ASC"
            )
            rows = cursor.fetchall()

        by_hash: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            rec = dict(r)
            by_hash.setdefault(rec["content_hash"], []).append(rec)

        raw_groups = [g for g in by_hash.values() if len(g) > 1]
        raw_groups.sort(key=lambda g: (-len(g), g[0]["file_path"].lower()))

        groups = []
        total_duplicates = 0
        for g in raw_groups:
            keep = g[0]
            dups = g[1:]
            total_duplicates += len(dups)
            groups.append({
                "content_hash": keep["content_hash"],
                "size": len(g),
                "keep": {**keep, "filename": Path(keep["file_path"]).name},
                "duplicates": [
                    {**d, "filename": Path(d["file_path"]).name}
                    for d in dups
                ],
            })

        return {
            "total_groups": len(groups),
            "total_duplicates": total_duplicates,
            "total_files": len(rows),
            "groups": groups,
        }


    def update_image_meta(
        self,
        image_id: int,
        file_size: int,
        mtime: float,
        content_hash: str,
        width: int = 0,
        height: int = 0,
        duration: float = 0.0,
    ) -> None:
        """Update file stat and hash metadata for an image record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if width > 0 and height > 0:
                cursor.execute(
                    """
                    UPDATE images 
                    SET file_size = ?, mtime = ?, content_hash = ?, width = ?, height = ?,
                        duration = COALESCE(NULLIF(?, 0.0), duration)
                    WHERE id = ?
                    """,
                    (int(file_size), float(mtime), content_hash, int(width), int(height), float(duration), int(image_id))
                )
            else:
                cursor.execute(
                    """
                    UPDATE images 
                    SET file_size = ?, mtime = ?, content_hash = ?,
                        duration = COALESCE(NULLIF(?, 0.0), duration)
                    WHERE id = ?
                    """,
                    (int(file_size), float(mtime), content_hash, float(duration), int(image_id))
                )

    def clear_faces_for_image(self, image_id: int) -> None:
        """Clear detected face records for a modified image prior to re-scanning."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM faces WHERE image_id = ?", (int(image_id),))

    def delete_missing_images(self, existing_file_paths: set[str]) -> int:
        """Delete image and face records for files no longer present on disk."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path FROM images")
            rows = cursor.fetchall()
            deleted_count = 0
            for r in rows:
                if r["file_path"] not in existing_file_paths and not Path(r["file_path"]).exists():
                    cursor.execute("DELETE FROM images WHERE id = ?", (r["id"],))
                    deleted_count += 1
            return deleted_count

    def insert_face(
        self,
        image_id: int,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        embedding: np.ndarray,
        cluster_id: int = -1,
        person_id: Optional[int] = None,
    ) -> int:
        """Insert a detected face with its embedding."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        emb_blob = self.serialize_embedding(embedding)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO faces (image_id, box_x1, box_y1, box_x2, box_y2, confidence, embedding, cluster_id, person_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(image_id), x1, y1, x2, y2, float(confidence), emb_blob, int(cluster_id), person_id)
            )
            return cursor.lastrowid or 0

    def toggle_favorite(self, image_id: int) -> bool:
        """Toggle favorite state of an image. Returns new favorite state (True/False)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT is_favorite FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                return False
            new_val = 0 if row["is_favorite"] else 1
            cursor.execute("UPDATE images SET is_favorite = ? WHERE id = ?", (new_val, image_id))
            return bool(new_val)

    def get_image(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve single image metadata with detected faces."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
            row = cursor.fetchone()
            if not row:
                return None
            img = dict(row)
            img["is_favorite"] = bool(img.get("is_favorite", 0))

            file_path_str = img["file_path"]
            p = Path(file_path_str)
            img["filename"] = p.name
            img["is_video"] = p.suffix.lower() in VIDEO_EXTENSIONS
            img["file_size"] = int(img.get("file_size") or 0)
            img["duration"] = float(img.get("duration") or 0.0)

            # Lazy backfill file_size or duration if missing
            updated = False
            if img["file_size"] <= 0 and p.is_file():
                try:
                    img["file_size"] = p.stat().st_size
                    updated = True
                except Exception:
                    pass

            if img["is_video"] and img["duration"] <= 0.0 and p.is_file():
                try:
                    import cv2
                    cap = cv2.VideoCapture(file_path_str)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                        cap.release()
                        if fps > 0 and frames > 0:
                            img["duration"] = float(frames / fps)
                            updated = True
                except Exception:
                    pass

            if updated:
                cursor.execute(
                    "UPDATE images SET file_size = ?, duration = ? WHERE id = ?",
                    (img["file_size"], img["duration"], image_id)
                )

            img["formatted_size"] = format_file_size(img["file_size"])
            img["formatted_duration"] = format_duration(img["duration"])
            ext = p.suffix.lstrip(".").upper()
            img["extension"] = ext
            img["file_format"] = f"{ext} Video" if img["is_video"] else (f"{ext} Image" if ext else "Image")

            # Fetch associated faces and deduplicate by person/cluster
            cursor.execute("""
                SELECT f.id as face_id, f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                       f.confidence, f.cluster_id, f.person_id, p.name as person_name
                FROM faces f
                LEFT JOIN people p ON f.person_id = p.id
                WHERE f.image_id = ?
                ORDER BY f.confidence DESC
            """, (image_id,))
            
            seen_people: Dict[str, Dict[str, Any]] = {}
            for fr in cursor.fetchall():
                person_id = fr["person_id"]
                cluster_id = int(fr["cluster_id"])
                
                # Key on person_id if named, cluster_id if clustered, or face_id if unclustered
                if person_id is not None:
                    key = f"p_{person_id}"
                elif cluster_id >= 0:
                    key = f"c_{cluster_id}"
                else:
                    key = f"f_{fr['face_id']}"

                name = fr["person_name"] or (f"Person #{cluster_id}" if cluster_id >= 0 else "Unknown")

                if key not in seen_people:
                    seen_people[key] = {
                        "face_id": fr["face_id"],
                        "bbox": (int(fr["box_x1"]), int(fr["box_y1"]), int(fr["box_x2"]), int(fr["box_y2"])),
                        "confidence": float(fr["confidence"]),
                        "cluster_id": cluster_id,
                        "person_id": person_id,
                        "person_name": name,
                        "appearance_count": 1,
                    }
                else:
                    seen_people[key]["appearance_count"] += 1

            # Priority: Named faces first, then cluster faces, then confidence
            sorted_faces = sorted(
                seen_people.values(),
                key=lambda f: (0 if f["person_id"] is not None else (1 if f["cluster_id"] >= 0 else 2), -f["confidence"])
            )
            img["faces"] = sorted_faces
            img["face_count"] = len(sorted_faces)
            img["tags"] = self.get_image_tags(image_id)
            return img

    def get_images(
        self,
        filter_type: str = "all",  # 'all', 'favorites', 'person', 'unclustered'
        person_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        folder_path: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "date",  # 'date', 'name', 'faces', 'size'
        sort_order: str = "desc",  # 'desc', 'asc'
        page: int = 1,
        limit: int = 60,
        tag_id: Optional[int] = None,
        exclude_duplicates: bool = False,
    ) -> Dict[str, Any]:
        """Query images with filtering, search, sorting, folder filtering, and pagination."""
        offset = max(0, (page - 1) * limit)
        params: List[Any] = []
        where_clauses: List[str] = []

        if exclude_duplicates:
            where_clauses.append(
                "(i.content_hash = '' OR i.id = (SELECT MIN(mi.id) FROM images mi WHERE mi.content_hash = i.content_hash AND mi.content_hash != ''))"
            )

        if filter_type == "favorites":
            where_clauses.append("i.is_favorite = 1")

        if person_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE person_id = ?)")
            params.append(person_id)
        elif cluster_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE cluster_id = ?)")
            params.append(cluster_id)

        if tag_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM image_tags WHERE tag_id = ?)")
            params.append(tag_id)

        if folder_path:
            norm_folder = str(Path(folder_path).resolve())
            where_clauses.append("(i.file_path LIKE ? OR i.file_path LIKE ?)")
            params.extend([f"{norm_folder}/%", f"{norm_folder}\\%"])

        if search:
            search_like = f"%{search.strip()}%"
            where_clauses.append(
                "(i.file_path LIKE ? OR i.id IN (SELECT f.image_id FROM faces f JOIN people p ON f.person_id = p.id WHERE p.name LIKE ?))"
            )
            params.extend([search_like, search_like])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        # Determine sorting
        order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"
        if sort_by == "name":
            order_sql = f"ORDER BY i.file_path {order_dir}, i.id {order_dir}"
        elif sort_by == "faces":
            order_sql = f"ORDER BY face_count {order_dir}, i.id {order_dir}"
        elif sort_by == "size":
            order_sql = f"ORDER BY i.file_size {order_dir}, i.id {order_dir}"
        else:  # default date
            order_sql = f"ORDER BY i.scanned_at {order_dir}, i.id {order_dir}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            count_query = f"SELECT COUNT(DISTINCT i.id) as total FROM images i {where_sql}"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()["total"]

            query = f"""
                SELECT i.id, i.file_path, i.scanned_at, i.is_favorite, i.width, i.height, i.file_size, i.duration,
                       COUNT(DISTINCT COALESCE(f.person_id, f.cluster_id, f.id)) as face_count
                FROM images i
                LEFT JOIN faces f ON i.id = f.image_id
                {where_sql}
                GROUP BY i.id
                {order_sql}
                LIMIT ? OFFSET ?
            """
            cursor.execute(query, params + [limit, offset])
            rows = cursor.fetchall()

            images = []
            for r in rows:
                p = Path(r["file_path"])
                f_size = int(r["file_size"] or 0)
                dur = float(r["duration"] or 0.0)
                images.append({
                    "id": r["id"],
                    "file_path": r["file_path"],
                    "filename": p.name,
                    "is_video": p.suffix.lower() in VIDEO_EXTENSIONS,
                    "scanned_at": r["scanned_at"],
                    "is_favorite": bool(r["is_favorite"]),
                    "width": r["width"] or 0,
                    "height": r["height"] or 0,
                    "file_size": f_size,
                    "formatted_size": format_file_size(f_size),
                    "duration": dur,
                    "formatted_duration": format_duration(dur),
                    "face_count": r["face_count"],
                })

            total_pages = max(1, (total_count + limit - 1) // limit)
            return {
                "images": images,
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
                "sort_by": sort_by,
                "sort_order": sort_order,
            }

    def get_adjacent_image_ids(
        self,
        image_id: int,
        filter_type: str = "all",
        person_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        folder_path: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "date",
        sort_order: str = "desc",
        tag_id: Optional[int] = None,
        similar_to: Optional[int] = None,
        threshold: Optional[float] = None,
        exclude_duplicates: bool = False,
    ) -> Dict[str, Optional[int]]:
        """Get previous and next image IDs for modal navigation."""
        if similar_to is not None:
            active_threshold = threshold
            if active_threshold is None:
                try:
                    active_threshold = float(self.get_setting("similarity_threshold", 0.60))
                except (TypeError, ValueError):
                    active_threshold = 0.60

            # Retrieve ordered list of matching IDs for similar_to query (including main image)
            similar_data = self.get_similar_images(
                image_id=similar_to,
                limit=10000,
                threshold=active_threshold,
                page=1,
                exclude_duplicates=exclude_duplicates,
            )
            ids = [img["id"] for img in similar_data.get("images", [])]

            if image_id not in ids:
                return {"prev_id": None, "next_id": None, "current_index": -1, "total_count": len(ids)}

            idx = ids.index(image_id)
            prev_id = ids[idx - 1] if idx > 0 else None
            next_id = ids[idx + 1] if idx < len(ids) - 1 else None

            return {
                "prev_id": prev_id,
                "next_id": next_id,
                "current_index": idx + 1,
                "total_count": len(ids),
            }

        params: List[Any] = []
        where_clauses: List[str] = []

        if exclude_duplicates:
            where_clauses.append(
                "(i.content_hash = '' OR i.id = (SELECT MIN(mi.id) FROM images mi WHERE mi.content_hash = i.content_hash AND mi.content_hash != ''))"
            )

        if filter_type == "favorites":
            where_clauses.append("i.is_favorite = 1")
        if person_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE person_id = ?)")
            params.append(person_id)
        elif cluster_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE cluster_id = ?)")
            params.append(cluster_id)
        if tag_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM image_tags WHERE tag_id = ?)")
            params.append(tag_id)
        if folder_path:
            norm_folder = str(Path(folder_path).resolve())
            where_clauses.append("(i.file_path LIKE ? OR i.file_path LIKE ?)")
            params.extend([f"{norm_folder}/%", f"{norm_folder}\\%"])
        if search:
            search_like = f"%{search.strip()}%"
            where_clauses.append(
                "(i.file_path LIKE ? OR i.id IN (SELECT f.image_id FROM faces f JOIN people p ON f.person_id = p.id WHERE p.name LIKE ?))"
            )
            params.extend([search_like, search_like])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"
        if sort_by == "name":
            order_sql = f"ORDER BY i.file_path {order_dir}, i.id {order_dir}"
        elif sort_by == "faces":
            order_sql = f"ORDER BY face_count {order_dir}, i.id {order_dir}"
        elif sort_by == "size":
            order_sql = f"ORDER BY i.file_size {order_dir}, i.id {order_dir}"
        else:
            order_sql = f"ORDER BY i.scanned_at {order_dir}, i.id {order_dir}"

        query = f"""
            SELECT i.id, COUNT(DISTINCT COALESCE(f.person_id, f.cluster_id, f.id)) as face_count
            FROM images i
            LEFT JOIN faces f ON i.id = f.image_id
            {where_sql}
            GROUP BY i.id
            {order_sql}
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            ids = [r["id"] for r in rows]

            if image_id not in ids:
                return {"prev_id": None, "next_id": None, "current_index": -1, "total_count": len(ids)}

            idx = ids.index(image_id)
            prev_id = ids[idx - 1] if idx > 0 else None
            next_id = ids[idx + 1] if idx < len(ids) - 1 else None

            return {
                "prev_id": prev_id,
                "next_id": next_id,
                "current_index": idx + 1,
                "total_count": len(ids),
            }

    def get_folders(self, current_folder: Optional[str] = None) -> Dict[str, Any]:
        """Extract hierarchical folder structure and subdirectories with photo counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path FROM images ORDER BY file_path ASC")
            rows = cursor.fetchall()

        if not rows:
            return {
                "current_folder": "",
                "parent_folder": None,
                "breadcrumbs": [],
                "subfolders": [],
                "direct_images_count": 0,
            }

        # Resolve all parent directories of files
        all_paths = [Path(r["file_path"]).resolve() for r in rows]
        # Find common base directory of all media
        try:
            import os
            common_root = Path(os.path.commonpath([str(p.parent) for p in all_paths]))
        except Exception:
            common_root = all_paths[0].parent

        if current_folder:
            target_dir = Path(current_folder).resolve()
            if not target_dir.exists() and (common_root / current_folder).exists():
                target_dir = (common_root / current_folder).resolve()
        else:
            target_dir = common_root

        # Find immediate subdirectories and direct media
        subfolder_counts: Dict[str, Dict[str, Any]] = {}
        direct_images_count = 0

        for r, p in zip(rows, all_paths):
            parent = p.parent
            if parent == target_dir:
                direct_images_count += 1
            elif target_dir in parent.parents or parent.is_relative_to(target_dir):
                # Immediate subfolder relative to target_dir
                rel = parent.relative_to(target_dir)
                top_child_name = rel.parts[0]
                child_full_path = str(target_dir / top_child_name)
                if top_child_name not in subfolder_counts:
                    subfolder_counts[top_child_name] = {
                        "name": top_child_name,
                        "full_path": child_full_path,
                        "rel_path": str(Path(child_full_path).relative_to(common_root)) if child_full_path != str(common_root) else "",
                        "item_count": 0,
                        "cover_image_id": r["id"],
                    }
                subfolder_counts[top_child_name]["item_count"] += 1

        subfolders = sorted(list(subfolder_counts.values()), key=lambda x: x["name"].lower())

        # Build breadcrumbs
        breadcrumbs = [{"name": "Root", "path": str(common_root)}]
        if target_dir != common_root:
            try:
                rel_parts = target_dir.relative_to(common_root).parts
                cur_accum = common_root
                for part in rel_parts:
                    cur_accum = cur_accum / part
                    breadcrumbs.append({"name": part, "path": str(cur_accum)})
            except ValueError:
                breadcrumbs.append({"name": target_dir.name, "path": str(target_dir)})

        parent_folder = None
        if target_dir != common_root and target_dir.parent:
            parent_folder = str(target_dir.parent)

        return {
            "current_folder": str(target_dir),
            "folder_name": target_dir.name if target_dir != common_root else "All Media Folders",
            "parent_folder": parent_folder,
            "breadcrumbs": breadcrumbs,
            "subfolders": subfolders,
            "direct_images_count": direct_images_count,
        }

    def get_people(self, search: Optional[str] = None, cluster_limit: Optional[int] = None, hide_low_quality: bool = False) -> List[Dict[str, Any]]:
        """Retrieve all people and unnamed clusters with face count, photo count, and cover face ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. First fetch explicitly created / named people (appear on top)
            query = """
                SELECT p.id as person_id, p.name, p.cover_face_id, p.created_at,
                       COUNT(DISTINCT f.id) as face_count,
                       COUNT(DISTINCT f.image_id) as photo_count,
                       MIN(f.id) as fallback_face_id
                FROM people p
                LEFT JOIN faces f ON f.person_id = p.id
                GROUP BY p.id
                ORDER BY photo_count DESC, p.name ASC
            """
            cursor.execute(query)
            people = []
            for r in cursor.fetchall():
                people.append({
                    "id": r["person_id"],
                    "type": "person",
                    "is_named": True,
                    "name": r["name"],
                    "cover_face_id": r["cover_face_id"] or r["fallback_face_id"],
                    "face_count": r["face_count"],
                    "photo_count": r["photo_count"],
                })

            # 2. Then include unnamed clusters that have cluster_id >= 0 but no person_id
            cluster_limit_sql = f"LIMIT {int(cluster_limit)}" if cluster_limit else ""
            cluster_quality_filter = ""
            cluster_having = ""
            cluster_params: List[Any] = []
            if hide_low_quality:
                cluster_quality_filter = " AND (f.box_x2 - f.box_x1) >= ? AND (f.box_y2 - f.box_y1) >= ? AND f.confidence >= ?"
                cluster_having = " HAVING COUNT(DISTINCT f.image_id) > 1"
                cluster_params = [LOW_QUALITY_FACE_MIN_SIDE, LOW_QUALITY_FACE_MIN_SIDE, LOW_QUALITY_FACE_MIN_CONF]
            cluster_query = f"""
                SELECT f.cluster_id,
                       COUNT(DISTINCT f.id) as face_count,
                       COUNT(DISTINCT f.image_id) as photo_count,
                       MIN(f.id) as cover_face_id
                FROM faces f
                WHERE f.cluster_id >= 0 AND f.person_id IS NULL
                {cluster_quality_filter}
                GROUP BY f.cluster_id
                {cluster_having}
                ORDER BY photo_count DESC, face_count DESC
                {cluster_limit_sql}
            """
            cursor.execute(cluster_query, cluster_params)
            for r in cursor.fetchall():
                people.append({
                    "id": r["cluster_id"],
                    "type": "cluster",
                    "is_named": False,
                    "name": f"Person #{r['cluster_id']}",
                    "cover_face_id": r["cover_face_id"],
                    "face_count": r["face_count"],
                    "photo_count": r["photo_count"],
                    "cluster_id": r["cluster_id"],
                })

            if search:
                s = search.lower().strip()
                people = [p for p in people if s in p["name"].lower() or (p["type"] == "cluster" and s in str(p["id"]))]

            return people

    def get_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        """Get person profile info by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.id, p.name, p.cover_face_id, p.created_at,
                       COUNT(DISTINCT f.id) as face_count,
                       COUNT(DISTINCT f.image_id) as photo_count,
                       MIN(f.id) as fallback_face_id
                FROM people p
                LEFT JOIN faces f ON f.person_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
            """, (person_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "name": row["name"],
                "cover_face_id": row["cover_face_id"] or row["fallback_face_id"],
                "created_at": row["created_at"],
                "face_count": row["face_count"],
                "photo_count": row["photo_count"],
            }

    def name_person(self, name: str, cluster_id: Optional[int] = None, person_id: Optional[int] = None, face_id: Optional[int] = None) -> int:
        """
        Assign or update a person name:
        - If person_id is given, rename that person (updates across all associated photos and videos).
        - If cluster_id is given, create or associate person with this name and link all faces in cluster across all media.
        - If face_id is given, resolve its cluster and link all matching faces in that cluster across all photos and videos.
        """
        name = name.strip()
        if not name:
            name = "Unnamed"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # If face_id is provided, resolve cluster_id if not given
            if face_id:
                cursor.execute("SELECT cluster_id FROM faces WHERE id = ?", (face_id,))
                fr = cursor.fetchone()
                if fr:
                    if cluster_id is None and fr["cluster_id"] is not None and fr["cluster_id"] >= 0:
                        cluster_id = fr["cluster_id"]

            if person_id:
                # Rename existing person record
                cursor.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
                target_person_id = person_id
            else:
                # Check if a person with this name already exists (case-insensitive)
                cursor.execute("SELECT id FROM people WHERE LOWER(TRIM(name)) = LOWER(?)", (name,))
                existing_p = cursor.fetchone()
                if existing_p:
                    target_person_id = int(existing_p["id"])
                else:
                    cover_face = face_id
                    if not cover_face and cluster_id is not None and cluster_id >= 0:
                        cursor.execute("SELECT id FROM faces WHERE cluster_id = ? LIMIT 1", (cluster_id,))
                        r = cursor.fetchone()
                        if r:
                            cover_face = r["id"]

                    cursor.execute("INSERT INTO people (name, cover_face_id) VALUES (?, ?)", (name, cover_face))
                    target_person_id = int(cursor.lastrowid)

            # Link all faces in cluster across all photos and videos to this person
            if cluster_id is not None and cluster_id >= 0:
                cursor.execute("UPDATE faces SET person_id = ? WHERE cluster_id = ?", (target_person_id, cluster_id))
            if face_id:
                cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (target_person_id, face_id))

            return target_person_id

    def rename_cluster(self, cluster_id: int, name: str) -> int:
        """Create or associate a person name for all faces belonging to a cluster."""
        return self.name_person(name=name, cluster_id=cluster_id)

    def merge_people(self, source_person_id: int, target_person_id: int) -> None:
        """Merge all faces from source_person into target_person, then delete source_person."""
        if source_person_id == target_person_id:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure target person has a cover face if currently None
            cursor.execute("SELECT cover_face_id FROM people WHERE id = ?", (target_person_id,))
            target_row = cursor.fetchone()
            if target_row and not target_row["cover_face_id"]:
                cursor.execute("SELECT cover_face_id FROM people WHERE id = ?", (source_person_id,))
                src_row = cursor.fetchone()
                if src_row and src_row["cover_face_id"]:
                    cursor.execute("UPDATE people SET cover_face_id = ? WHERE id = ?", (src_row["cover_face_id"], target_person_id))

            cursor.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (target_person_id, source_person_id))
            cursor.execute("DELETE FROM people WHERE id = ?", (source_person_id,))

    def merge_person_into_cluster(self, source_person_id: int, target_cluster_id: int) -> None:
        """Merge all faces from source_person into target_cluster, then delete source_person."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE faces SET cluster_id = ?, person_id = NULL WHERE person_id = ?", (target_cluster_id, source_person_id))
            cursor.execute("DELETE FROM people WHERE id = ?", (source_person_id,))

    def merge_cluster_into_person(self, source_cluster_id: int, target_person_id: int) -> None:
        """Merge all faces from source_cluster into target_person."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE faces SET person_id = ? WHERE cluster_id = ?", (target_person_id, source_cluster_id))

    def merge_clusters(self, source_cluster_id: int, target_cluster_id: int) -> None:
        """Merge all faces from source_cluster into target_cluster."""
        if source_cluster_id == target_cluster_id:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE faces SET cluster_id = ?, person_id = NULL WHERE cluster_id = ?", (target_cluster_id, source_cluster_id))

    def move_face(
        self,
        face_id: int,
        target_person_id: Optional[int] = None,
        target_cluster_id: Optional[int] = None,
        new_person_name: Optional[str] = None,
        unlink: bool = False,
    ) -> Optional[int]:
        """
        Move a face to a target person, cluster, new person, or unlink it.
        Returns the resulting person_id if assigned to a person, else None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch current person_id before moving/unlinking
            cursor.execute("SELECT person_id FROM faces WHERE id = ?", (face_id,))
            row = cursor.fetchone()
            current_person_id = row["person_id"] if row else None

            if unlink:
                if current_person_id is not None:
                    cursor.execute(
                        "INSERT OR IGNORE INTO person_exclusions (face_id, person_id) VALUES (?, ?)",
                        (face_id, current_person_id),
                    )
                cursor.execute("UPDATE faces SET person_id = NULL, cluster_id = -1 WHERE id = ?", (face_id,))
                return None

            if new_person_name and new_person_name.strip():
                name = new_person_name.strip()
                cursor.execute("SELECT id FROM people WHERE LOWER(TRIM(name)) = LOWER(?)", (name,))
                existing = cursor.fetchone()
                if existing:
                    target_person_id = int(existing["id"])
                else:
                    cursor.execute("INSERT INTO people (name, cover_face_id) VALUES (?, ?)", (name, face_id))
                    target_person_id = int(cursor.lastrowid)

            if target_person_id is not None:
                # If moving away from a previous person, record exclusion for old person
                if current_person_id is not None and current_person_id != target_person_id:
                    cursor.execute(
                        "INSERT OR IGNORE INTO person_exclusions (face_id, person_id) VALUES (?, ?)",
                        (face_id, current_person_id),
                    )
                # Remove any existing exclusion for target person
                cursor.execute(
                    "DELETE FROM person_exclusions WHERE face_id = ? AND person_id = ?",
                    (face_id, target_person_id),
                )
                cursor.execute("UPDATE faces SET person_id = ? WHERE id = ?", (target_person_id, face_id))
                return target_person_id
            elif target_cluster_id is not None:
                if current_person_id is not None:
                    cursor.execute(
                        "INSERT OR IGNORE INTO person_exclusions (face_id, person_id) VALUES (?, ?)",
                        (face_id, current_person_id),
                    )
                cursor.execute("UPDATE faces SET cluster_id = ?, person_id = NULL WHERE id = ?", (target_cluster_id, face_id))
                return None
            return None

    def get_face(self, face_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve single face detection info."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.image_id, f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                       f.confidence, f.embedding, f.cluster_id, f.person_id,
                       i.file_path, p.name as person_name
                FROM faces f
                JOIN images i ON f.image_id = i.id
                LEFT JOIN people p ON f.person_id = p.id
                WHERE f.id = ?
            """, (face_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return {
                "id": r["id"],
                "image_id": r["image_id"],
                "file_path": r["file_path"],
                "bbox": (int(r["box_x1"]), int(r["box_y1"]), int(r["box_x2"]), int(r["box_y2"])),
                "confidence": float(r["confidence"]),
                "embedding": self.deserialize_embedding(r["embedding"]),
                "cluster_id": int(r["cluster_id"]),
                "person_id": r["person_id"],
                "person_name": r["person_name"],
            }

    def get_all_unclustered_or_all_faces(self) -> List[Dict[str, Any]]:
        """Retrieve all face records and their deserialized embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.image_id, f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                       f.confidence, f.embedding, f.cluster_id, f.person_id, i.file_path
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
                    "person_id": r["person_id"],
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

            cursor.execute("SELECT COUNT(*) as cnt FROM images WHERE is_favorite = 1")
            favorites_count = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM faces")
            faces_detected = cursor.fetchone()["cnt"]

            cursor.execute("SELECT COUNT(*) as cnt FROM people")
            named_people_count = cursor.fetchone()["cnt"]

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

            cursor.execute("SELECT COUNT(*) as cnt FROM faces WHERE cluster_id < 0")
            unclustered_count = cursor.fetchone()["cnt"]
            num_clusters = len(clusters)

            return {
                "images_scanned": images_scanned,
                "favorites_count": favorites_count,
                "faces_detected": faces_detected,
                "named_people_count": named_people_count,
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
        """Clear all records from images, faces, and people."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM faces;")
            conn.execute("DELETE FROM people;")
            conn.execute("DELETE FROM images;")

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored setting value by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (str(key),))
            row = cursor.fetchone()
            if row is not None:
                return row["value"]
            return default

    def set_setting(self, key: str, value: Any) -> None:
        """Save or update a setting value by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(key), str(value))
            )

    def get_all_settings(self) -> Dict[str, str]:
        """Retrieve all stored settings as a dictionary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM settings")
            return {r["key"]: r["value"] for r in cursor.fetchall()}

    def set_settings(self, settings_dict: Dict[str, Any]) -> None:
        """Batch save multiple settings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for k, v in settings_dict.items():
                if v is not None:
                    cursor.execute(
                        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (str(k), str(v))
                    )

    def add_person_exclusion(self, face_id: int, person_id: int) -> None:
        """Record a hard cannot-link exclusion between a face and a person."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO person_exclusions (face_id, person_id) VALUES (?, ?)",
                (int(face_id), int(person_id))
            )

    def remove_person_exclusion(self, face_id: int, person_id: int) -> None:
        """Remove a cannot-link exclusion."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM person_exclusions WHERE face_id = ? AND person_id = ?",
                (int(face_id), int(person_id))
            )

    def get_person_exclusions(self) -> Dict[int, Set[int]]:
        """Retrieve mapping of {face_id: set_of_excluded_person_ids}."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT face_id, person_id FROM person_exclusions")
            exclusions: Dict[int, Set[int]] = {}
            for r in cursor.fetchall():
                f_id = int(r["face_id"])
                p_id = int(r["person_id"])
                if f_id not in exclusions:
                    exclusions[f_id] = set()
                exclusions[f_id].add(p_id)
            return exclusions

    def get_all_person_exemplars(self, max_exemplars: int = 5) -> Dict[int, List[np.ndarray]]:
        """
        Retrieve multi-exemplar embeddings for all named people in the database.
        Returns {person_id: [exemplar_vec1, exemplar_vec2, ...]}.
        """
        from app.recognition import select_k_medoids

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT person_id, embedding
                FROM faces
                WHERE person_id IS NOT NULL
                ORDER BY person_id ASC, id ASC
            """)
            person_embeddings: Dict[int, List[np.ndarray]] = {}
            for r in cursor.fetchall():
                p_id = int(r["person_id"])
                emb = self.deserialize_embedding(r["embedding"])
                if p_id not in person_embeddings:
                    person_embeddings[p_id] = []
                person_embeddings[p_id].append(emb)

            exemplars: Dict[int, List[np.ndarray]] = {}
            for p_id, embs in person_embeddings.items():
                exemplars[p_id] = select_k_medoids(embs, k=max_exemplars)

            return exemplars

    def get_person_exemplars(self, person_id: int, max_exemplars: int = 5) -> List[np.ndarray]:
        """Retrieve multi-exemplar embeddings for a single person."""
        from app.recognition import select_k_medoids

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding
                FROM faces
                WHERE person_id = ?
                ORDER BY id ASC
            """, (int(person_id),))
            embs = [self.deserialize_embedding(r["embedding"]) for r in cursor.fetchall()]
            return select_k_medoids(embs, k=max_exemplars)

    def get_unassigned_faces(self) -> List[Dict[str, Any]]:
        """Retrieve all faces that currently have no person_id assigned."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.image_id, f.box_x1, f.box_y1, f.box_x2, f.box_y2,
                       f.confidence, f.embedding, f.cluster_id, i.file_path
                FROM faces f
                JOIN images i ON f.image_id = i.id
                WHERE f.person_id IS NULL
                ORDER BY f.id ASC
            """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": int(r["id"]),
                    "image_id": int(r["image_id"]),
                    "bbox": (int(r["box_x1"]), int(r["box_y1"]), int(r["box_x2"]), int(r["box_y2"])),
                    "confidence": float(r["confidence"]),
                    "embedding": self.deserialize_embedding(r["embedding"]),
                    "cluster_id": int(r["cluster_id"]),
                    "person_id": None,
                    "file_path": r["file_path"],
                })
            return results

    def assign_faces_to_person(self, face_ids: List[int], person_id: int) -> None:
        """Batch assign a list of face IDs to a person."""
        if not face_ids:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in face_ids)
            cursor.execute(
                f"UPDATE faces SET person_id = ? WHERE id IN ({placeholders})",
                [int(person_id)] + [int(fid) for fid in face_ids]
            )
            # Clear any exclusions for this person
            cursor.execute(
                f"DELETE FROM person_exclusions WHERE person_id = ? AND face_id IN ({placeholders})",
                [int(person_id)] + [int(fid) for fid in face_ids]
            )
    def update_folder_paths(self, old_dir: str, new_dir: str) -> List[str]:
        """Rewrite file_path for every image under old_dir to sit under new_dir after a folder rename.

        Returns the list of old file paths that were updated so callers can invalidate cache.
        """
        old_prefix = str(Path(old_dir).resolve())
        new_prefix = str(Path(new_dir).resolve())
        escape = old_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, file_path FROM images WHERE file_path = ? OR file_path LIKE ? ESCAPE '\\'",
                (old_prefix, escape),
            )
            rows = cursor.fetchall()
            old_paths = [r["file_path"] for r in rows]
            for r in rows:
                rel = r["file_path"][len(old_prefix):]
                cursor.execute(
                    "UPDATE images SET file_path = ? WHERE id = ?",
                    (new_prefix + rel, r["id"]),
                )
        return old_paths

    def update_image_path(self, image_id: int, new_path: str) -> None:
        """Update the file_path of an existing image record on disk move or rename."""
        resolved = str(Path(new_path).resolve())
        p = Path(resolved)
        mtime = p.stat().st_mtime if p.exists() else 0.0
        file_size = p.stat().st_size if p.exists() else 0
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE images SET file_path = ?, mtime = ?, file_size = ? WHERE id = ?",
                (resolved, mtime, file_size, int(image_id)),
            )

    def delete_image_records(self, image_ids: List[int]) -> int:
        """Delete specific image records by IDs (cascades to faces & exclusions)."""
        if not image_ids:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in image_ids)
            cursor.execute(f"DELETE FROM images WHERE id IN ({placeholders})", [int(i) for i in image_ids])
            return cursor.rowcount

    def clone_image_record(self, source_image_id: int, new_path: str) -> Optional[int]:
        """Clone an image and all its face detections/embeddings for a copied file."""
        resolved = str(Path(new_path).resolve())
        p = Path(resolved)
        mtime = p.stat().st_mtime if p.exists() else 0.0
        file_size = p.stat().st_size if p.exists() else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Fetch source image metadata
            cursor.execute("SELECT * FROM images WHERE id = ?", (int(source_image_id),))
            src_img = cursor.fetchone()
            if not src_img:
                return None

            cursor.execute(
                """
                INSERT INTO images (file_path, is_favorite, width, height, file_size, mtime, content_hash, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved,
                    src_img["is_favorite"],
                    src_img["width"],
                    src_img["height"],
                    file_size or src_img["file_size"],
                    mtime or src_img["mtime"],
                    src_img["content_hash"],
                    src_img["duration"],
                ),
            )
            new_image_id = cursor.lastrowid

            # Clone all face detections for this image
            cursor.execute("SELECT * FROM faces WHERE image_id = ?", (int(source_image_id),))
            src_faces = cursor.fetchall()
            for sf in src_faces:
                cursor.execute(
                    """
                    INSERT INTO faces (image_id, box_x1, box_y1, box_x2, box_y2, confidence, embedding, cluster_id, person_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_image_id,
                        sf["box_x1"],
                        sf["box_y1"],
                        sf["box_x2"],
                        sf["box_y2"],
                        sf["confidence"],
                        sf["embedding"],
                        sf["cluster_id"],
                        sf["person_id"],
                    ),
                )

            # Clone all tags for this image
            cursor.execute("SELECT tag_id FROM image_tags WHERE image_id = ?", (int(source_image_id),))
            for it in cursor.fetchall():
                cursor.execute(
                    "INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)",
                    (new_image_id, int(it["tag_id"])),
                )
            return new_image_id

    def get_all_folder_paths(self) -> List[str]:
        """Return a sorted list of all unique folder paths containing indexed images."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM images")
            rows = cursor.fetchall()
            folders = set()
            for r in rows:
                if r["file_path"]:
                    folders.add(str(Path(r["file_path"]).parent.resolve()))
            
            # Also add root scan dir from settings if configured
            input_dir = self.get_setting("input_dir")
            if input_dir and Path(input_dir).exists():
                folders.add(str(Path(input_dir).resolve()))
            return sorted(list(folders))

    def batch_toggle_favorites(self, image_ids: List[int], is_favorite: bool) -> int:
        """Batch set favorite status for multiple images."""
        if not image_ids:
            return 0
        val = 1 if is_favorite else 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in image_ids)
            cursor.execute(
                f"UPDATE images SET is_favorite = ? WHERE id IN ({placeholders})",
                [val] + [int(i) for i in image_ids],
            )
            return cursor.rowcount

    def add_tag(self, name: str) -> Optional[int]:
        """Create a tag if missing, else return existing tag id. Returns None for blank names."""
        name = (name or "").strip()
        if not name:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM tags WHERE LOWER(name) = LOWER(?)", (name,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])
            cursor.execute("INSERT INTO tags (name) VALUES (?)", (name,))
            return int(cursor.lastrowid)

    def get_tag(self, tag_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single tag by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.id, t.name, COUNT(it.image_id) as photo_count
                FROM tags t
                LEFT JOIN image_tags it ON it.tag_id = t.id
                WHERE t.id = ?
                GROUP BY t.id
            """, (int(tag_id),))
            row = cursor.fetchone()
            if not row:
                return None
            return {"id": row["id"], "name": row["name"], "photo_count": row["photo_count"]}

    def get_all_tags(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all tags with photo counts, sorted by name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT t.id, t.name, COUNT(it.image_id) as photo_count
                FROM tags t
                LEFT JOIN image_tags it ON it.tag_id = t.id
            """
            params: List[Any] = []
            if search and search.strip():
                query += " WHERE t.name LIKE ?"
                params.append(f"%{search.strip()}%")
            query += " GROUP BY t.id ORDER BY t.name COLLATE NOCASE ASC"
            cursor.execute(query, params)
            return [{"id": r["id"], "name": r["name"], "photo_count": r["photo_count"]} for r in cursor.fetchall()]

    def get_image_tags(self, image_id: int) -> List[Dict[str, Any]]:
        """Retrieve tags assigned to an image."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.id, t.name
                FROM tags t
                JOIN image_tags it ON it.tag_id = t.id
                WHERE it.image_id = ?
                ORDER BY t.name COLLATE NOCASE ASC
            """, (int(image_id),))
            return [{"id": r["id"], "name": r["name"]} for r in cursor.fetchall()]

    def add_tags_to_image(self, image_id: int, tag_names: List[str]) -> int:
        """Assign a list of tags to an image. Existing tags are reused. Returns count added.

        When the image has a content fingerprint, the tags are also applied to every
        duplicate of that image (any file sharing the same content hash).
        """
        tag_ids = [tid for tid in (self.add_tag(name) for name in (tag_names or [])) if tid is not None]
        if not tag_ids:
            return 0
        added = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Resolve the source image's content hash and propagate to all duplicates
            cursor.execute("SELECT content_hash FROM images WHERE id = ?", (int(image_id),))
            row = cursor.fetchone()
            if row and row["content_hash"]:
                cursor.execute(
                    "SELECT id FROM images WHERE content_hash = ?",
                    (row["content_hash"],),
                )
                target_ids = [int(r["id"]) for r in cursor.fetchall()]
            else:
                target_ids = [int(image_id)]

            for target_id in target_ids:
                for tag_id in tag_ids:
                    cursor.execute(
                        "INSERT OR IGNORE INTO image_tags (image_id, tag_id) VALUES (?, ?)",
                        (int(target_id), int(tag_id)),
                    )
                    if cursor.rowcount:
                        added += 1
        return added

    def remove_tag_from_image(self, image_id: int, tag_id: int) -> bool:
        """Remove a tag from an image. Returns True if a link was removed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM image_tags WHERE image_id = ? AND tag_id = ?",
                (int(image_id), int(tag_id)),
            )
            return cursor.rowcount > 0

    def save_media_embedding(self, image_id: int, embedding: np.ndarray, model: str = "clip-vit-b32") -> None:
        """Save or update whole-media visual embedding."""
        blob = self.serialize_embedding(embedding)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO media_embeddings (image_id, embedding, model)
                VALUES (?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    model = excluded.model,
                    created_at = CURRENT_TIMESTAMP
            """, (int(image_id), blob, model))

    def get_media_embedding(self, image_id: int) -> Optional[np.ndarray]:
        """Retrieve whole-media visual embedding for an image."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM media_embeddings WHERE image_id = ?", (int(image_id),))
            row = cursor.fetchone()
            if row and row["embedding"]:
                return self.deserialize_embedding(row["embedding"])
            return None

    def get_all_media_embeddings(self) -> Dict[int, np.ndarray]:
        """Retrieve all media embeddings as a dictionary of image_id -> vector."""
        result: Dict[int, np.ndarray] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT image_id, embedding FROM media_embeddings")
            for row in cursor.fetchall():
                result[row["image_id"]] = self.deserialize_embedding(row["embedding"])
        return result

    def get_media_embedding_stats(self) -> Dict[str, int]:
        """Return count of total images vs indexed media embeddings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM images")
            total = cursor.fetchone()["total"]
            cursor.execute("SELECT COUNT(*) as embedded FROM media_embeddings")
            embedded = cursor.fetchone()["embedded"]
            return {"total_images": total, "embedded_images": embedded, "missing": max(0, total - embedded)}

    def get_unembedded_image_ids(self) -> List[int]:
        """Retrieve all image IDs that lack a media embedding."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT i.id FROM images i
                LEFT JOIN media_embeddings me ON me.image_id = i.id
                WHERE me.image_id IS NULL
                ORDER BY i.id ASC
            """)
            return [r["id"] for r in cursor.fetchall()]

    def get_similar_images(
        self,
        image_id: int,
        limit: int = 48,
        threshold: float = 0.60,
        page: int = 1,
        exclude_duplicates: bool = False,
    ) -> Dict[str, Any]:
        """
        Find and return images visually and semantically similar to image_id,
        sorted by cosine similarity descending.
        """
        target_img = self.get_image(image_id)
        if not target_img:
            return {
                "images": [],
                "total": 0,
                "page": page,
                "limit": limit,
                "total_pages": 0,
                "target_image": None,
            }

        target_emb = self.get_media_embedding(image_id)
        if target_emb is None:
            t_copy = dict(target_img)
            t_copy["is_target"] = True
            t_copy["similarity_score"] = 1.0
            t_copy["similarity_pct"] = 100
            t_copy["face_count"] = t_copy.get("face_count", len(t_copy.get("faces", [])))
            return {
                "images": [t_copy],
                "total": 1,
                "page": page,
                "limit": limit,
                "total_pages": 1,
                "target_image": target_img,
            }

        all_embs = self.get_all_media_embeddings()
        all_embs.pop(int(image_id), None)
        if not all_embs:
            t_copy = dict(target_img)
            t_copy["is_target"] = True
            t_copy["similarity_score"] = 1.0
            t_copy["similarity_pct"] = 100
            t_copy["face_count"] = t_copy.get("face_count", len(t_copy.get("faces", [])))
            return {
                "images": [t_copy],
                "total": 1,
                "page": page,
                "limit": limit,
                "total_pages": 1,
                "target_image": target_img,
            }

        candidate_ids = list(all_embs.keys())
        matrix = np.vstack([all_embs[cid] for cid in candidate_ids])

        # Handle CLIP embedding anisotropy (shared bias vector across all media).
        # When dataset is large enough and a non-trivial mean vector exists (>0.30 norm),
        # subtract the global mean vector before computing cosine similarity to eliminate
        # false positive baseline matches (e.g. video multi-frame averaging collapse).
        if len(matrix) >= 8:
            mean_vec = np.mean(matrix, axis=0, keepdims=True)
            if np.linalg.norm(mean_vec) > 0.30:
                target_centered = target_emb - mean_vec.ravel()
                target_norm = target_centered / (np.linalg.norm(target_centered) + 1e-9)

                matrix_centered = matrix - mean_vec
                matrix_norms = np.linalg.norm(matrix_centered, axis=1, keepdims=True) + 1e-9
                norm_matrix = matrix_centered / matrix_norms
            else:
                target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
                matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
                norm_matrix = matrix / matrix_norms
        else:
            target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
            matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            norm_matrix = matrix / matrix_norms

        # Compute cosine similarity
        sims = np.dot(norm_matrix, target_norm).ravel()

        # Filter by threshold and sort descending
        valid_indices = np.where(sims >= threshold)[0]
        sorted_valid = valid_indices[np.argsort(-sims[valid_indices])]

        # Include main/target image at index 0, followed by similar candidates
        ranked_matches: List[Tuple[int, float]] = [
            (int(image_id), 1.0)
        ] + [
            (candidate_ids[idx], float(sims[idx])) for idx in sorted_valid
        ]

        if exclude_duplicates:
            # Filter out files with identical content_hash (keep earliest, preserving main image)
            seen_hashes = set()
            filtered_ranked = []
            for cid, score in ranked_matches:
                img_data = target_img if cid == int(image_id) else self.get_image(cid)
                if img_data:
                    chash = img_data.get("content_hash")
                    if chash:
                        if chash in seen_hashes:
                            continue
                        seen_hashes.add(chash)
                    filtered_ranked.append((cid, score))
            ranked_matches = filtered_ranked

        total = len(ranked_matches)
        total_pages = max(1, (total + limit - 1) // limit) if total > 0 else 0
        offset = max(0, (page - 1) * limit)
        page_matches = ranked_matches[offset : offset + limit]

        images_list = []
        for cid, score in page_matches:
            if cid == int(image_id):
                img = dict(target_img)
                img["is_target"] = True
            else:
                img = self.get_image(cid)
                if img:
                    img["is_target"] = False

            if img:
                img["similarity_score"] = score
                img["similarity_pct"] = int(round(score * 100))
                img["face_count"] = img.get("face_count", len(img.get("faces", [])))
                images_list.append(img)

        return {
            "images": images_list,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "target_image": target_img,
        }

    def suggest_tags_for_image(
        self,
        image_id: int,
        top_k: int = 15,
        min_score: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """
        Suggest tags for an image using k-NN tag transfer from similar media.
        Returns ranked list of candidate tags with confidence scores.
        """
        target_emb = self.get_media_embedding(image_id)
        if target_emb is None:
            return []

        all_embs = self.get_all_media_embeddings()
        all_embs.pop(int(image_id), None)
        if not all_embs:
            return []

        # Get existing tags on this image to exclude them from suggestions
        existing_tags = {t["id"] for t in self.get_image_tags(image_id)}

        candidate_ids = list(all_embs.keys())
        matrix = np.vstack([all_embs[cid] for cid in candidate_ids])

        if len(matrix) >= 8:
            mean_vec = np.mean(matrix, axis=0, keepdims=True)
            if np.linalg.norm(mean_vec) > 0.30:
                target_centered = target_emb - mean_vec.ravel()
                target_norm = target_centered / (np.linalg.norm(target_centered) + 1e-9)

                matrix_centered = matrix - mean_vec
                matrix_norms = np.linalg.norm(matrix_centered, axis=1, keepdims=True) + 1e-9
                norm_matrix = matrix_centered / matrix_norms
            else:
                target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
                matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
                norm_matrix = matrix / matrix_norms
        else:
            target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-9)
            matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            norm_matrix = matrix / matrix_norms

        sims = np.dot(norm_matrix, target_norm).ravel()

        # Take top-K most similar items with similarity >= 0.50
        top_indices = np.argsort(-sims)[:top_k]
        top_indices = [idx for idx in top_indices if sims[idx] >= 0.50]
        if not top_indices:
            return []

        neighbor_ids = [candidate_ids[idx] for idx in top_indices]
        neighbor_sims = {candidate_ids[idx]: float(sims[idx]) for idx in top_indices}

        # Query tags for these neighbors
        tag_scores: Dict[int, Dict[str, Any]] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in neighbor_ids)
            cursor.execute(f"""
                SELECT it.image_id, t.id as tag_id, t.name as tag_name
                FROM image_tags it
                JOIN tags t ON t.id = it.tag_id
                WHERE it.image_id IN ({placeholders})
            """, neighbor_ids)
            rows = cursor.fetchall()

        total_sim_weight = sum(neighbor_sims.values()) + 1e-9
        for row in rows:
            tid = row["tag_id"]
            if tid in existing_tags:
                continue
            tname = row["tag_name"]
            sim = neighbor_sims.get(row["image_id"], 0.0)

            if tid not in tag_scores:
                tag_scores[tid] = {
                    "id": tid,
                    "name": tname,
                    "weighted_sim": 0.0,
                    "count": 0,
                    "max_sim": 0.0,
                }
            tag_scores[tid]["weighted_sim"] += sim
            tag_scores[tid]["count"] += 1
            if sim > tag_scores[tid]["max_sim"]:
                tag_scores[tid]["max_sim"] = sim

        suggestions = []
        for tid, data in tag_scores.items():
            # Normalized score combines frequency in top-K and max similarity
            freq_score = data["weighted_sim"] / total_sim_weight
            confidence = 0.6 * data["max_sim"] + 0.4 * min(1.0, freq_score * 2.5)
            if confidence >= min_score:
                suggestions.append({
                    "id": tid,
                    "name": data["name"],
                    "score": round(confidence, 2),
                    "confidence_pct": int(round(confidence * 100)),
                    "matched_count": data["count"],
                })

        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions

