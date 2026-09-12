"""
SQLite database layer for storing images, face detections, embeddings, clusters, and named people.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}


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
                    height INTEGER DEFAULT 0
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

            cursor.execute("PRAGMA table_info(faces)")
            face_cols = {col["name"] for col in cursor.fetchall()}
            if "person_id" not in face_cols:
                cursor.execute("ALTER TABLE faces ADD COLUMN person_id INTEGER DEFAULT NULL")

            # 3. Create indices after verifying all columns exist
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(is_favorite);
                CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path);
                CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
                CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
                CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
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
    ) -> int:
        """Insert or retrieve an image ID given its file path and metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO images (file_path, width, height, file_size, mtime, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_path, int(width), int(height), int(file_size), float(mtime), content_hash)
            )
            if width > 0 and height > 0:
                cursor.execute(
                    """
                    UPDATE images 
                    SET width = ?, height = ?, file_size = COALESCE(NULLIF(?, 0), file_size),
                        mtime = COALESCE(NULLIF(?, 0.0), mtime), content_hash = COALESCE(NULLIF(?, ''), content_hash)
                    WHERE file_path = ?
                    """,
                    (int(width), int(height), int(file_size), float(mtime), content_hash, file_path)
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
                "SELECT id, file_path, file_size, mtime, content_hash, width, height FROM images WHERE file_path = ?",
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
            cursor.execute("SELECT id, file_path, file_size, mtime, content_hash, width, height FROM images")
            rows = cursor.fetchall()
            return {str(row["file_path"]): dict(row) for row in rows}


    def update_image_meta(
        self,
        image_id: int,
        file_size: int,
        mtime: float,
        content_hash: str,
        width: int = 0,
        height: int = 0,
    ) -> None:
        """Update file stat and hash metadata for an image record."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if width > 0 and height > 0:
                cursor.execute(
                    """
                    UPDATE images 
                    SET file_size = ?, mtime = ?, content_hash = ?, width = ?, height = ?
                    WHERE id = ?
                    """,
                    (int(file_size), float(mtime), content_hash, int(width), int(height), int(image_id))
                )
            else:
                cursor.execute(
                    """
                    UPDATE images 
                    SET file_size = ?, mtime = ?, content_hash = ?
                    WHERE id = ?
                    """,
                    (int(file_size), float(mtime), content_hash, int(image_id))
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

            p = Path(img["file_path"])
            img["filename"] = p.name
            img["is_video"] = p.suffix.lower() in VIDEO_EXTENSIONS
            # Priority: Named faces first, then cluster faces, then confidence
            sorted_faces = sorted(
                seen_people.values(),
                key=lambda f: (0 if f["person_id"] is not None else (1 if f["cluster_id"] >= 0 else 2), -f["confidence"])
            )
            img["faces"] = sorted_faces
            return img

    def get_images(
        self,
        filter_type: str = "all",  # 'all', 'favorites', 'person', 'unclustered'
        person_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        folder_path: Optional[str] = None,
        search: Optional[str] = None,
        sort_by: str = "date",  # 'date', 'name', 'faces'
        sort_order: str = "desc",  # 'desc', 'asc'
        page: int = 1,
        limit: int = 60,
    ) -> Dict[str, Any]:
        """Query images with filtering, search, sorting, folder filtering, and pagination."""
        offset = max(0, (page - 1) * limit)
        params: List[Any] = []
        where_clauses: List[str] = []

        if filter_type == "favorites":
            where_clauses.append("i.is_favorite = 1")

        if person_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE person_id = ?)")
            params.append(person_id)
        elif cluster_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE cluster_id = ?)")
            params.append(cluster_id)

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
        else:  # default date
            order_sql = f"ORDER BY i.scanned_at {order_dir}, i.id {order_dir}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            count_query = f"SELECT COUNT(DISTINCT i.id) as total FROM images i {where_sql}"
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()["total"]

            query = f"""
                SELECT i.id, i.file_path, i.scanned_at, i.is_favorite, i.width, i.height,
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
                images.append({
                    "id": r["id"],
                    "file_path": r["file_path"],
                    "filename": p.name,
                    "is_video": p.suffix.lower() in VIDEO_EXTENSIONS,
                    "scanned_at": r["scanned_at"],
                    "is_favorite": bool(r["is_favorite"]),
                    "width": r["width"] or 0,
                    "height": r["height"] or 0,
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
    ) -> Dict[str, Optional[int]]:
        """Get previous and next image IDs for modal navigation."""
        params: List[Any] = []
        where_clauses: List[str] = []

        if filter_type == "favorites":
            where_clauses.append("i.is_favorite = 1")
        if person_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE person_id = ?)")
            params.append(person_id)
        elif cluster_id is not None:
            where_clauses.append("i.id IN (SELECT image_id FROM faces WHERE cluster_id = ?)")
            params.append(cluster_id)
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

    def get_people(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
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
            cluster_query = """
                SELECT f.cluster_id,
                       COUNT(DISTINCT f.id) as face_count,
                       COUNT(DISTINCT f.image_id) as photo_count,
                       MIN(f.id) as cover_face_id
                FROM faces f
                WHERE f.cluster_id >= 0 AND f.person_id IS NULL
                GROUP BY f.cluster_id
                ORDER BY photo_count DESC, face_count DESC
            """
            cursor.execute(cluster_query)
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
                people = [p for p in people if s in p["name"].lower()]

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

            # If face_id is provided, resolve cluster_id and existing person_id
            if face_id:
                cursor.execute("SELECT cluster_id, person_id FROM faces WHERE id = ?", (face_id,))
                fr = cursor.fetchone()
                if fr:
                    if cluster_id is None and fr["cluster_id"] is not None and fr["cluster_id"] >= 0:
                        cluster_id = fr["cluster_id"]
                    if person_id is None and fr["person_id"] is not None:
                        person_id = fr["person_id"]

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
            cursor.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (target_person_id, source_person_id))
            cursor.execute("DELETE FROM people WHERE id = ?", (source_person_id,))

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
