"""
FastAPI web server for the Kimera Media Gallery.
Provides HTMX-driven HTML views, REST APIs, and cached thumbnail/crop streaming.
"""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import Body, FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.cache import ThumbnailCache
from app.db import Database
from app.recognition import MultiExemplarMatcher
from app.scanner import compute_quick_hash, scan_media_paths

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"


class ScanManager:
    """Manages background scanning & progress reporting."""

    def __init__(self, db_path: str = "face_clusters.db", cache_dir: Optional[str] = None):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.is_running = False
        self.status = "idle"  # idle, running, completed, error
        self.progress_message = "Ready to scan"
        self.percent = 0
        self.current_count = 0
        self.total_count = 0
        self.logs: List[str] = []
        self.stats: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def start_scan(
        self,
        input_dir: str,
        conf_threshold: float = 0.60,
        eps: float = 0.43,
        min_samples: int = 1,
        algorithm: str = "agglomerative",
        min_interval_sec: float = 60.0,
        match_threshold: Optional[float] = None,
        include_cluster_references: Optional[bool] = None,
        intra_video_merge_threshold: Optional[float] = None,
    ) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self.status = "running"
            self.progress_message = f"Scanning directory: {input_dir}..."
            self.percent = 0
            self.current_count = 0
            self.total_count = 0
            self.logs = [f"Starting scan on directory: {input_dir}"]
            self.error = None
            self.stats = {}

        thread = threading.Thread(
            target=self._run_scan_thread,
            args=(input_dir, conf_threshold, eps, min_samples, algorithm, min_interval_sec, match_threshold, include_cluster_references, intra_video_merge_threshold),
            daemon=True,
        )
        thread.start()
        return True

    def _on_progress(self, msg: str, percent: int = 0, current: int = 0, total: int = 0) -> None:
        with self._lock:
            self.progress_message = msg
            self.percent = max(0, min(100, percent))
            self.current_count = current
            self.total_count = total
            if msg and (not self.logs or self.logs[-1] != msg):
                self.logs.append(msg)
                if len(self.logs) > 60:
                    self.logs = self.logs[-60:]

    def _run_scan_thread(
        self,
        input_dir: str,
        conf_threshold: float,
        eps: float,
        min_samples: int,
        algorithm: str,
        min_interval_sec: float = 30.0,
        match_threshold: Optional[float] = None,
        include_cluster_references: Optional[bool] = None,
        intra_video_merge_threshold: Optional[float] = None,
    ) -> None:
        from app.pipeline import run_pipeline
        try:
            stats = run_pipeline(
                input_dir=input_dir,
                db_path=self.db_path,
                cache_dir=self.cache_dir,
                conf_threshold=conf_threshold,
                eps=eps,
                min_samples=min_samples,
                clustering_algorithm=algorithm,
                min_interval_sec=min_interval_sec,
                match_threshold=match_threshold,
                include_cluster_references=include_cluster_references,
                intra_video_merge_threshold=intra_video_merge_threshold,
                progress_callback=self._on_progress,
            )
            with self._lock:
                self.is_running = False
                self.status = "completed"
                self.percent = 100
                self.progress_message = (
                    f"Scan completed! {stats.get('images_scanned', 0)} media scanned, "
                    f"{stats.get('faces_detected', 0)} faces in {stats.get('num_clusters', 0)} clusters."
                )
                self.stats = stats
                self.logs.append("Scan pipeline finished successfully.")
        except Exception as e:
            with self._lock:
                self.is_running = False
                self.status = "error"
                self.error = str(e)
                self.progress_message = f"Scan error: {e}"
                self.logs.append(f"Failed with exception: {e}")


class EmbeddingManager:
    """Manages background generation of visual embeddings for unindexed media."""

    def __init__(self, db_path: str = "face_clusters.db"):
        self.db_path = db_path
        self.is_running = False
        self.status = "idle"  # idle, running, completed, error
        self.progress_message = "Ready"
        self.percent = 0
        self.current_count = 0
        self.total_count = 0
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def start_indexing(self) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self.status = "running"
            self.progress_message = "Starting visual embedding indexing..."
            self.percent = 0
            self.error = None

        thread = threading.Thread(target=self._run_thread, daemon=True)
        thread.start()
        return True

    def _on_progress(self, msg: str, percent: int = 0, current: int = 0, total: int = 0) -> None:
        with self._lock:
            self.progress_message = msg
            self.percent = max(0, min(100, percent))
            self.current_count = current
            self.total_count = total

    def _run_thread(self) -> None:
        from app.pipeline import index_missing_media_embeddings
        try:
            db = Database(self.db_path)
            indexed = index_missing_media_embeddings(db, progress_callback=self._on_progress)
            with self._lock:
                self.is_running = False
                self.status = "completed"
                self.percent = 100
                self.progress_message = f"Indexed {indexed} visual embeddings successfully."
        except Exception as e:
            with self._lock:
                self.is_running = False
                self.status = "error"
                self.error = str(e)
                self.progress_message = f"Embedding indexing error: {e}"


class ThumbnailManager:
    """Regenerates cached thumbnails for indexed media that are missing one. Only missing files are processed."""

    def __init__(self, db_path: str = "face_clusters.db", cache_dir: Optional[str] = None):
        self.db_path = db_path
        self.cache_dir = cache_dir
        self.is_running = False
        self.status = "idle"  # idle, running, completed, error
        self.progress_message = "Ready"
        self.percent = 0
        self.current_count = 0
        self.total_count = 0
        self.error: Optional[str] = None
        self.stats: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def start_generation(self) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self.status = "running"
            self.progress_message = "Scanning library for missing thumbnails..."
            self.percent = 0
            self.current_count = 0
            self.total_count = 0
            self.error = None
            self.stats = {}

        thread = threading.Thread(target=self._run_thread, daemon=True)
        thread.start()
        return True

    def _on_progress(self, msg: str, percent: int = 0, current: int = 0, total: int = 0) -> None:
        with self._lock:
            self.progress_message = msg
            self.percent = max(0, min(100, percent))
            self.current_count = current
            self.total_count = total

    def _run_thread(self) -> None:
        try:
            db = Database(self.db_path)
            cache = ThumbnailCache(self.cache_dir)
            all_paths = db.get_all_media_paths()
            total = len(all_paths)
            self._on_progress("Checking cached thumbnails...", 0, 0, total)

            missing = []
            for i, p in enumerate(all_paths):
                if Path(p).is_file() and not cache.has_thumbnail(p, 480):
                    missing.append(p)
                if (i + 1) % 100 == 0 or i + 1 == total:
                    self._on_progress(
                        f"Checked {i + 1}/{total} media files for missing thumbnails...",
                        int((i + 1) / max(1, total) * 100), i + 1, total,
                    )

            missing_total = len(missing)
            if missing_total == 0:
                with self._lock:
                    self.is_running = False
                    self.status = "completed"
                    self.percent = 100
                    self.stats = {"checked": total, "missing": 0, "generated": 0, "failed": 0}
                    self.progress_message = "No missing thumbnails found. All cached media already has one."
                return

            self._on_progress(f"Found {missing_total} missing thumbnails. Generating...", 0, 0, missing_total)

            generated = 0
            failed = 0
            for i, p in enumerate(missing):
                if not Path(p).is_file():
                    failed += 1
                elif cache.has_thumbnail(p, 480):
                    pass  # cached by another request meanwhile
                elif cache.get_thumbnail(p, 480) is not None:
                    generated += 1
                else:
                    failed += 1
                if (i + 1) % 10 == 0 or i + 1 == missing_total:
                    self._on_progress(
                        f"Generated {generated} of {missing_total} missing thumbnails...",
                        int((i + 1) / max(1, missing_total) * 100), i + 1, missing_total,
                    )

            with self._lock:
                self.is_running = False
                self.status = "completed"
                self.percent = 100
                self.stats = {
                    "checked": total,
                    "missing": missing_total,
                    "generated": generated,
                    "failed": failed,
                }
                self.progress_message = (
                    f"Thumbnail regeneration complete: {generated} generated, "
                    f"{failed} failed, {missing_total - generated - failed} already cached."
                )
        except Exception as e:
            with self._lock:
                self.is_running = False
                self.status = "error"
                self.error = str(e)
                self.progress_message = f"Thumbnail regeneration error: {e}"


class DeleteFilesRequest(BaseModel):
    image_ids: List[int]


class RenameFileRequest(BaseModel):
    image_id: int
    new_name: str


class BatchRenameRequest(BaseModel):
    image_ids: List[int]
    mode: str = "prefix"  # prefix, suffix, replace, pattern
    prefix: Optional[str] = ""
    suffix: Optional[str] = ""
    find_text: Optional[str] = ""
    replace_text: Optional[str] = ""
    pattern: Optional[str] = ""
    start_index: int = 1


class MoveFilesRequest(BaseModel):
    image_ids: List[int]
    destination_folder: str


class CopyFilesRequest(BaseModel):
    image_ids: List[int]
    destination_folder: Optional[str] = None


class BatchFavoriteRequest(BaseModel):
    image_ids: List[int]
    is_favorite: bool = True


def _parse_image_ids(raw_ids: Any) -> List[int]:
    """Extract list of integer image IDs from list, string, or int input."""
    if isinstance(raw_ids, list):
        out = []
        for x in raw_ids:
            if isinstance(x, int):
                out.append(x)
            elif str(x).isdigit():
                out.append(int(x))
        return out
    if isinstance(raw_ids, str):
        return [int(x.strip()) for x in raw_ids.split(",") if x.strip().isdigit()]
    if isinstance(raw_ids, int):
        return [raw_ids]
    return []


def _get_unique_destination_path(target_dir: Path, filename: str) -> Path:
    """Ensure unique destination path by appending _1, _2, etc. if candidate exists."""
    dest = target_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def create_app(db_path: str = "face_clusters.db", cache_dir: Optional[str] = None) -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    app = FastAPI(title="Kimera Media Gallery")

    resolved_cache_dir = str(Path(cache_dir).resolve()) if cache_dir else str(Path.cwd() / ".cache")
    db = Database(db_path)
    cache = ThumbnailCache(resolved_cache_dir)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    scan_mgr = ScanManager(db_path, cache_dir=resolved_cache_dir)
    embedding_mgr = EmbeddingManager(db_path)
    thumbnail_mgr = ThumbnailManager(db_path, cache_dir=resolved_cache_dir)

    def _media_source_root() -> Optional[Path]:
        input_dir = db.get_setting("input_dir")
        if input_dir and Path(input_dir).exists():
            return Path(input_dir).resolve()
        return None

    def _within_source_root(path: Path) -> bool:
        root = _media_source_root()
        if root is None:
            return True
        try:
            return path.resolve().is_relative_to(root)
        except ValueError:
            return False

    # On server start/restart: trigger background download of CLIP vision model if missing
    from app.models import trigger_clip_download_async
    trigger_clip_download_async()

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def gallery_view(
        request: Request,
        filter: str = Query("all", alias="filter"),
        search: Optional[str] = Query(None),
        person_id: Optional[int] = Query(None),
        cluster_id: Optional[int] = Query(None),
        tag_id: Optional[int] = Query(None),
        folder_path: Optional[str] = Query(None),
        similar_to: Optional[int] = Query(None),
        threshold: Optional[float] = Query(None),
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
        page: int = Query(1, ge=1),
        infinite: int = Query(0),
    ):
        exclude_duplicates = db.get_setting("exclude_duplicates") == "1"

        similar_to_image = None
        if similar_to is not None:
            active_threshold = threshold
            if active_threshold is None:
                try:
                    active_threshold = float(db.get_setting("similarity_threshold", 0.60))
                except (TypeError, ValueError):
                    active_threshold = 0.60

            data = db.get_similar_images(
                image_id=similar_to,
                limit=48,
                threshold=active_threshold,
                page=page,
                exclude_duplicates=exclude_duplicates,
            )
            similar_to_image = data.get("target_image")
            data["has_next"] = data["page"] < data["total_pages"]
            data["has_prev"] = data["page"] > 1
        else:
            data = db.get_images(
                filter_type=filter,
                person_id=person_id,
                cluster_id=cluster_id,
                folder_path=folder_path,
                search=search,
                sort_by=sort_by,
                sort_order=sort_order,
                page=page,
                limit=48,
                tag_id=tag_id,
                exclude_duplicates=exclude_duplicates,
            )

        active_tab = "favorites" if filter == "favorites" else "photos"
        all_tags = db.get_all_tags()
        active_tag = db.get_tag(tag_id) if tag_id is not None else None
        
        # If cluster_id is specified, fetch cluster info for top rename banner
        cluster_info = None
        if cluster_id is not None:
            cluster_info = {
                "cluster_id": cluster_id,
                "name": f"Person #{cluster_id}",
                "cover_face_id": None,
            }
            if data["images"]:
                # Try to get a cover face
                first_img = db.get_image(data["images"][0]["id"])
                if first_img and first_img.get("faces"):
                    for fc in first_img["faces"]:
                        if fc.get("cluster_id") == cluster_id:
                            cluster_info["cover_face_id"] = fc["face_id"]
                            break

        # Check if HTMX infinite scroll request
        if request.headers.get("HX-Request") and infinite == 1:
            return templates.TemplateResponse(
                request=request,
                name="partials/photo_batch.html",
                context={
                    "images": data["images"],
                    "page": data["page"],
                    "has_next": data["has_next"],
                    "filter_type": filter,
                    "search": search,
                    "person_id": person_id,
                    "cluster_id": cluster_id,
                    "folder_path": folder_path,
                    "similar_to": similar_to,
                    "similar_to_image": similar_to_image,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "tag_id": tag_id,
                    "target_url": "/",
                },
            )

        return templates.TemplateResponse(
            request=request,
            name="gallery.html",
            context={
                "images": data["images"],
                "total": data["total"],
                "page": data["page"],
                "total_pages": data["total_pages"],
                "has_next": data["has_next"],
                "has_prev": data["has_prev"],
                "filter_type": filter,
                "search": search,
                "person_id": person_id,
                "cluster_id": cluster_id,
                "cluster_info": cluster_info,
                "folder_path": folder_path,
                "similar_to": similar_to,
                "similar_to_image": similar_to_image,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "active_page": active_tab,
                "active_tag": active_tag,
                "all_tags": all_tags,
            },
        )

    @app.get("/folders", response_class=HTMLResponse)
    async def folders_view(
        request: Request,
        path: Optional[str] = Query(None),
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
        page: int = Query(1, ge=1),
        infinite: int = Query(0),
    ):
        folders_data = db.get_folders(current_folder=path)
        cur_folder = folders_data["current_folder"]
        
        # Load images directly in current folder
        exclude_duplicates = db.get_setting("exclude_duplicates") == "1"
        images_data = db.get_images(
            folder_path=cur_folder,
            folder_direct_only=True,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            limit=48,
            exclude_duplicates=exclude_duplicates,
        )

        if request.headers.get("HX-Request") and infinite == 1:
            from urllib.parse import quote_plus
            return templates.TemplateResponse(
                request=request,
                name="partials/photo_batch.html",
                context={
                    "images": images_data["images"],
                    "page": images_data["page"],
                    "has_next": images_data["has_next"],
                    "folder_path": cur_folder,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "target_url": f"/folders?path={quote_plus(cur_folder)}",
                },
            )

        return templates.TemplateResponse(
            request=request,
            name="folders.html",
            context={
                "folder_info": folders_data,
                "images": images_data["images"],
                "total": images_data["total"],
                "page": images_data["page"],
                "total_pages": images_data["total_pages"],
                "has_next": images_data["has_next"],
                "has_prev": images_data["has_prev"],
                "sort_by": sort_by,
                "sort_order": sort_order,
                "active_page": "folders",
            },
        )

    @app.get("/people", response_class=HTMLResponse)
    async def people_view(request: Request, search: Optional[str] = Query(None)):
        saved_settings = db.get_all_settings()
        hide_low_quality = saved_settings.get("hide_low_quality_faces") == "1"
        people = db.get_people(search=search, hide_low_quality=hide_low_quality)
        return templates.TemplateResponse(
            request=request,
            name="people.html",
            context={
                "people": people,
                "search": search,
                "hide_low_quality": hide_low_quality,
                "active_page": "people",
            },
        )

    @app.get("/person/{person_id}", response_class=HTMLResponse)
    async def person_detail_view(
        request: Request,
        person_id: int,
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
        page: int = Query(1, ge=1),
        infinite: int = Query(0),
    ):
        person = db.get_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")

        exclude_duplicates = db.get_setting("exclude_duplicates") == "1"
        data = db.get_images(
            filter_type="person",
            person_id=person_id,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            limit=48,
            exclude_duplicates=exclude_duplicates,
        )

        if request.headers.get("HX-Request") and infinite == 1:
            return templates.TemplateResponse(
                request=request,
                name="partials/photo_batch.html",
                context={
                    "images": data["images"],
                    "page": data["page"],
                    "has_next": data["has_next"],
                    "person_id": person_id,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "target_url": f"/person/{person_id}",
                },
            )

        return templates.TemplateResponse(
            request=request,
            name="person_detail.html",
            context={
                "person": person,
                "images": data["images"],
                "total": data["total"],
                "page": data["page"],
                "total_pages": data["total_pages"],
                "has_next": data["has_next"],
                "has_prev": data["has_prev"],
                "sort_by": sort_by,
                "sort_order": sort_order,
                "active_page": "people",
            },
        )

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_view(request: Request):
        stats = db.get_summary_stats()
        saved_settings = db.get_all_settings()
        default_dir = saved_settings.get("input_dir") or (str(Path("pictures").resolve()) if Path("pictures").exists() else str(Path.cwd()))
        cache_stats = cache.get_cache_stats()
        embedding_stats = db.get_media_embedding_stats()
        try:
            missing_thumb_count = cache.count_missing_thumbnails(db.get_all_media_paths(), max_dim=480)
        except Exception:
            missing_thumb_count = 0
        return templates.TemplateResponse(
            request=request,
            name="settings.html",
            context={
                "stats": stats,
                "default_dir": default_dir,
                "settings": saved_settings,
                "cache_dir": str(cache.cache_dir),
                "cache_stats": cache_stats,
                "missing_thumb_count": missing_thumb_count,
                "scan_mgr": scan_mgr,
                "embedding_mgr": embedding_mgr,
                "embedding_stats": embedding_stats,
                "active_page": "settings",
            },
        )

    @app.post("/api/embeddings/generate-all")
    async def generate_all_embeddings():
        if embedding_mgr.is_running:
            return {"status": "already_running", "message": embedding_mgr.progress_message}
        started = embedding_mgr.start_indexing()
        return {"status": "started" if started else "failed"}

    @app.get("/api/embeddings/status")
    async def get_embeddings_status():
        stats = db.get_media_embedding_stats()
        return {
            "is_running": embedding_mgr.is_running,
            "status": embedding_mgr.status,
            "percent": embedding_mgr.percent,
            "current": embedding_mgr.current_count,
            "total": embedding_mgr.total_count,
            "message": embedding_mgr.progress_message,
            "error": embedding_mgr.error,
            "stats": stats,
        }

    @app.post("/api/thumbnails/generate-missing")
    async def generate_missing_thumbnails():
        if thumbnail_mgr.is_running:
            return {"status": "already_running", "message": thumbnail_mgr.progress_message}
        started = thumbnail_mgr.start_generation()
        return {"status": "started" if started else "failed"}

    @app.get("/api/thumbnails/status")
    async def get_thumbnails_status():
        stats = thumbnail_mgr.stats if thumbnail_mgr.status in ("completed", "error") else None
        return {
            "is_running": thumbnail_mgr.is_running,
            "status": thumbnail_mgr.status,
            "percent": thumbnail_mgr.percent,
            "current": thumbnail_mgr.current_count,
            "total": thumbnail_mgr.total_count,
            "message": thumbnail_mgr.progress_message,
            "error": thumbnail_mgr.error,
            "stats": stats,
        }

    @app.get("/api/photos/{image_id}/suggested-tags")
    async def get_photo_suggested_tags(image_id: int):
        try:
            return {"suggested_tags": db.suggest_tags_for_image(image_id, top_k=8, min_score=0.50)}
        except Exception as e:
            return {"suggested_tags": [], "error": str(e)}

    @app.post("/api/settings/save", response_class=HTMLResponse)
    async def save_settings_endpoint(
        request: Request,
        input_dir: str = Form(...),
        conf_threshold: float = Form(0.60),
        eps: float = Form(0.43),
        min_samples: int = Form(1),
        algorithm: str = Form("agglomerative"),
        min_interval_sec: float = Form(60.0),
        recognition_match_threshold: float = Form(0.42),
        similarity_threshold: float = Form(0.60),
        hide_low_quality_faces: str = Form(""),
        include_cluster_references: str = Form(""),
        intra_video_merge_threshold: float = Form(0.30),
    ):
        db.set_settings({
            "input_dir": input_dir,
            "conf_threshold": str(conf_threshold),
            "eps": str(eps),
            "min_samples": str(min_samples),
            "algorithm": algorithm,
            "min_interval_sec": str(min_interval_sec),
            "recognition_match_threshold": str(recognition_match_threshold),
            "similarity_threshold": str(similarity_threshold),
            "hide_low_quality_faces": "1" if str(hide_low_quality_faces).lower() in ("1", "true", "on", "yes") else "",
            "include_cluster_references": "1" if str(include_cluster_references).lower() in ("1", "true", "on", "yes") else "",
            "intra_video_merge_threshold": str(intra_video_merge_threshold),
        })
        return HTMLResponse("""
            <div class="alert alert-success" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.875rem;">
                ✓ Settings saved successfully!
            </div>
        """)

    @app.post("/api/settings/cache", response_class=HTMLResponse)
    async def update_cache_settings(request: Request, cache_dir: str = Form(...)):
        new_path = Path(cache_dir).resolve()
        cache.set_cache_dir(new_path)
        scan_mgr.cache_dir = str(new_path)
        cache_stats = cache.get_cache_stats()
        return HTMLResponse(f"""
            <div class="alert alert-success" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ✓ Cache directory updated to: <code>{new_path}</code> ({cache_stats['thumbnail_count']} thumbnails, {cache_stats['face_count']} faces, {cache_stats['total_size_mb']} MB)
            </div>
        """)

    @app.post("/api/settings/cache/clear", response_class=HTMLResponse)
    async def clear_cache_endpoint(request: Request):
        cache.clear_cache()
        cache_stats = cache.get_cache_stats()
        return HTMLResponse(f"""
            <div class="alert alert-success" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ✓ Cache cleared! {cache_stats['thumbnail_count']} thumbnails, {cache_stats['total_size_mb']} MB remaining.
            </div>
        """)

    @app.post("/api/settings/exclude-duplicates", response_class=HTMLResponse)
    async def set_exclude_duplicates(request: Request, exclude_duplicates: str = Form("")):
        enabled = str(exclude_duplicates).lower() in ("1", "true", "on", "yes")
        db.set_setting("exclude_duplicates", "1" if enabled else "")
        checked = "checked" if enabled else ""
        return HTMLResponse(f"""
        <div id="exclude-duplicates-control">
          <div class="alert alert-success" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 0.5rem 0.85rem; border-radius: 8px; margin-bottom: 0.85rem; font-size: 0.8125rem;">
            ✓ Exclude duplicates {"enabled" if enabled else "disabled"}. Open the gallery to apply.
          </div>
          <form hx-post="/api/settings/exclude-duplicates" hx-target="#exclude-duplicates-control" hx-swap="outerHTML">
            <label class="form-label" style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
              <input type="checkbox" name="exclude_duplicates" value="1" onchange="this.form.requestSubmit()" {checked} style="width: 16px; height: 16px;">
              Exclude duplicates from gallery
            </label>
            <span class="form-help">Show only one file per duplicate group in the media list, even when duplicates sit in different folders.</span>
          </form>
        </div>
        """)

    @app.post("/api/duplicates/scan", response_class=HTMLResponse)
    async def scan_duplicates_endpoint(request: Request):
        # 1. Collect indexed records, refreshing fingerprints only when size/mtime changed
        entries: List[Dict[str, Any]] = []
        recomputed = 0
        for rec in db.get_hash_records():
            p = Path(rec["file_path"])
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            cached = rec.get("content_hash") or ""
            if cached and int(rec.get("file_size") or 0) == st.st_size and abs(float(rec.get("mtime") or 0.0) - st.st_mtime) < 0.001:
                h = cached
            else:
                h = compute_quick_hash(p)
                db.update_image_meta(int(rec["id"]), file_size=st.st_size, mtime=st.st_mtime, content_hash=h)
                recomputed += 1
            entries.append({
                "id": int(rec["id"]),
                "file_path": rec["file_path"],
                "filename": p.name,
                "file_size": st.st_size,
                "content_hash": h,
            })

        # 2. Walk the filesystem for media files that are not yet indexed
        existing_paths = {e["file_path"] for e in entries}
        new_found = 0
        input_dir = db.get_setting("input_dir") or (str(Path("pictures").resolve()) if Path("pictures").exists() else str(Path.cwd()))
        if Path(input_dir).exists():
            for p in scan_media_paths(input_dir):
                rp = str(p.resolve())
                if rp in existing_paths:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                entries.append({
                    "id": None,
                    "file_path": rp,
                    "filename": p.name,
                    "file_size": st.st_size,
                    "content_hash": compute_quick_hash(p),
                })
                new_found += 1

        # 3. Group by content fingerprint
        by_hash: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            if e["content_hash"]:
                by_hash.setdefault(e["content_hash"], []).append(e)

        raw_groups = [g for g in by_hash.values() if len(g) >= 2]
        raw_groups.sort(key=lambda g: (-len(g), g[0]["file_path"].lower()))

        groups = []
        total_duplicates = 0
        for g in raw_groups:
            g_sorted = sorted(g, key=lambda e: (e["id"] is None, e["id"] or 0))
            keep, dups = g_sorted[0], g_sorted[1:]
            total_duplicates += len(dups)
            for item in g_sorted:
                item["has_image_id"] = item["id"] is not None
            groups.append({
                "content_hash": g[0]["content_hash"],
                "size": len(g),
                "keep": keep,
                "duplicates": dups,
            })

        results = {
            "total_files": len(entries),
            "total_indexed": len([e for e in entries if e["id"] is not None]),
            "total_groups": len(groups),
            "total_duplicates": total_duplicates,
            "new_found": new_found,
            "groups": groups,
        }
        return templates.TemplateResponse(
            request=request,
            name="partials/duplicates_results.html",
            context={
                "duplicates": results,
                "recomputed": recomputed,
                "exclude_duplicates": db.get_setting("exclude_duplicates") == "1",
            },
        )

    @app.post("/api/scan/trigger", response_class=HTMLResponse)
    async def trigger_scan(
        request: Request,
        input_dir: str = Form(...),
        conf_threshold: float = Form(0.60),
        eps: float = Form(0.43),
        min_samples: int = Form(1),
        algorithm: str = Form("agglomerative"),
        min_interval_sec: float = Form(60.0),
        recognition_match_threshold: float = Form(0.42),
        include_cluster_references: str = Form(""),
        intra_video_merge_threshold: float = Form(0.30),
    ):
        if not Path(input_dir).exists():
            return HTMLResponse(
                f"""<div class="alert alert-error" style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; padding: 0.75rem 1rem; border-radius: 8px;">
                    Directory not found: {input_dir}
                </div>"""
            )

        # Save settings to DB
        db.set_settings({
            "input_dir": input_dir,
            "conf_threshold": str(conf_threshold),
            "eps": str(eps),
            "min_samples": str(min_samples),
            "algorithm": algorithm,
            "min_interval_sec": str(min_interval_sec),
            "recognition_match_threshold": str(recognition_match_threshold),
            "include_cluster_references": "1" if str(include_cluster_references).lower() in ("1", "true", "on", "yes") else "",
            "intra_video_merge_threshold": str(intra_video_merge_threshold),
        })

        scan_mgr.start_scan(
            input_dir=input_dir,
            conf_threshold=conf_threshold,
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            min_interval_sec=min_interval_sec,
            match_threshold=recognition_match_threshold,
            include_cluster_references=str(include_cluster_references).lower() in ("1", "true", "on", "yes"),
            intra_video_merge_threshold=intra_video_merge_threshold,
        )

        return templates.TemplateResponse(
            request=request,
            name="partials/scan_status.html",
            context={"scan_mgr": scan_mgr},
        )

    @app.get("/api/scan/status", response_class=HTMLResponse)
    async def get_scan_status(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="partials/scan_status.html",
            context={"scan_mgr": scan_mgr},
        )

    # API endpoints
    @app.get("/api/photos/{image_id}/thumbnail")
    async def get_photo_thumbnail(image_id: int):
        img_meta = db.get_image(image_id)
        if not img_meta or not Path(img_meta["file_path"]).exists():
            raise HTTPException(status_code=404, detail="Image not found")

        thumb = cache.get_thumbnail(img_meta["file_path"], max_dim=480)
        if not thumb or not thumb.exists():
            if cache.is_video(img_meta["file_path"]):
                raise HTTPException(status_code=404, detail="Thumbnail unavailable")
            return FileResponse(img_meta["file_path"])

        return FileResponse(
            thumb,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.get("/api/photos/{image_id}/raw")
    async def get_photo_raw(image_id: int):
        img_meta = db.get_image(image_id)
        if not img_meta or not Path(img_meta["file_path"]).exists():
            raise HTTPException(status_code=404, detail="Image or video not found")
        
        mime_type, _ = mimetypes.guess_type(img_meta["file_path"])
        return FileResponse(
            img_meta["file_path"],
            media_type=mime_type or "application/octet-stream",
            headers={"Accept-Ranges": "bytes"}
        )

    @app.get("/api/photos/{image_id}/stream")
    async def get_photo_stream(image_id: int):
        img_meta = db.get_image(image_id)
        if not img_meta or not Path(img_meta["file_path"]).exists():
            raise HTTPException(status_code=404, detail="Image or video not found")

        # If not a video, fallback to raw response
        if not cache.is_video(img_meta["file_path"]):
            mime_type, _ = mimetypes.guess_type(img_meta["file_path"])
            return FileResponse(
                img_meta["file_path"],
                media_type=mime_type or "application/octet-stream",
                headers={"Accept-Ranges": "bytes"}
            )

        transcoded = cache.get_transcoded_video(img_meta["file_path"])
        if not transcoded or not transcoded.exists():
            # Fallback to direct raw file if transcoding is unavailable or fails
            mime_type, _ = mimetypes.guess_type(img_meta["file_path"])
            return FileResponse(
                img_meta["file_path"],
                media_type=mime_type or "video/mp4",
                headers={"Accept-Ranges": "bytes"}
            )

        return FileResponse(
            transcoded,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=86400, immutable"
            }
        )

    def _render_photo_modal_response(
        request: Request,
        image_id: int,
        filter_type: str = "all",
        search: Optional[str] = None,
        person_id: Optional[int] = None,
        cluster_id: Optional[int] = None,
        folder_path: Optional[str] = None,
        folder_direct_only: bool = False,
        similar_to: Optional[int] = None,
        threshold: Optional[float] = None,
        sort_by: str = "date",
        sort_order: str = "desc",
        tag_id: Optional[int] = None,
    ) -> HTMLResponse:
        img_meta = db.get_image(image_id)
        if not img_meta:
            raise HTTPException(status_code=404, detail="Image not found")
        img_p = Path(img_meta["file_path"])
        img_meta["filename"] = img_p.name
        img_meta["parent_folder"] = str(img_p.parent)

        adj = db.get_adjacent_image_ids(
            image_id=image_id,
            filter_type=filter_type,
            person_id=person_id,
            cluster_id=cluster_id,
            folder_path=folder_path,
            folder_direct_only=folder_direct_only,
            similar_to=similar_to,
            threshold=threshold,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            tag_id=tag_id,
            exclude_duplicates=db.get_setting("exclude_duplicates") == "1",
        )

        try:
            suggested_tags = db.suggest_tags_for_image(image_id, top_k=6, min_score=0.50)
        except Exception:
            suggested_tags = []

        return templates.TemplateResponse(
            request=request,
            name="partials/photo_modal.html",
            context={
                "image": img_meta,
                "prev_id": adj["prev_id"],
                "next_id": adj["next_id"],
                "current_index": adj["current_index"],
                "total_count": adj["total_count"],
                "filter_type": filter_type,
                "search": search,
                "person_id": person_id,
                "cluster_id": cluster_id,
                "folder_path": folder_path,
                "similar_to": similar_to,
                "threshold": threshold,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "tag_id": tag_id,
                "suggested_tags": suggested_tags,
                "modal_ctx": {
                    "filter_type": filter_type,
                    "search": search,
                    "person_id": person_id,
                    "cluster_id": cluster_id,
                    "folder_path": folder_path,
                    "similar_to": similar_to,
                    "threshold": threshold,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                    "ctx_tag_id": tag_id,
                },
            },
        )

    @app.get("/api/photos/{image_id}/modal", response_class=HTMLResponse)
    async def get_photo_modal(
        request: Request,
        image_id: int,
        filter: str = Query("all", alias="filter"),
        search: Optional[str] = Query(None),
        person_id: Optional[int] = Query(None),
        cluster_id: Optional[int] = Query(None),
        folder_path: Optional[str] = Query(None),
        path: Optional[str] = Query(None),
        direct_only: Optional[int] = Query(None),
        similar_to: Optional[int] = Query(None),
        threshold: Optional[float] = Query(None),
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
        tag_id: Optional[int] = Query(None),
    ):
        effective_folder = folder_path or path
        is_direct = bool(direct_only == 1 or path or ("/folders" in request.headers.get("referer", "")))
        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
            filter_type=filter,
            search=search,
            person_id=person_id,
            cluster_id=cluster_id,
            folder_path=effective_folder,
            folder_direct_only=is_direct if effective_folder else False,
            similar_to=similar_to,
            threshold=threshold,
            sort_by=sort_by,
            sort_order=sort_order,
            tag_id=tag_id,
        )

    @app.post("/api/photos/{image_id}/favorite", response_class=HTMLResponse)
    async def toggle_favorite_photo(request: Request, image_id: int):
        db.toggle_favorite(image_id)
        img_meta = db.get_image(image_id)
        if not img_meta:
            raise HTTPException(status_code=404, detail="Image not found")
        img_meta["filename"] = Path(img_meta["file_path"]).name
        img_meta["face_count"] = len(img_meta.get("faces", []))

        # Check if requested from modal or grid
        if "modal" in request.headers.get("HX-Target", ""):
            fav_active = "active" if img_meta["is_favorite"] else ""
            fill_color = "currentColor" if img_meta["is_favorite"] else "none"
            return HTMLResponse(f"""
                <button 
                  class="fav-btn {fav_active}"
                  hx-post="/api/photos/{image_id}/favorite"
                  hx-target="#modal-fav-btn"
                  hx-swap="outerHTML"
                  id="modal-fav-btn"
                  title="Toggle favorite"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="{fill_color}" stroke="currentColor" stroke-width="2">
                    <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
                  </svg>
                </button>
            """)

        return templates.TemplateResponse(
            request=request,
            name="partials/photo_card.html",
            context={"image": img_meta},
        )

    @app.get("/api/tags/list")
    async def get_tags_list(search: Optional[str] = Query(None)):
        return JSONResponse({"tags": db.get_all_tags(search=search)})

    @app.get("/api/tags/picker", response_class=HTMLResponse)
    async def get_tag_picker(
        request: Request,
        image_id: int = Query(...),
        filter_type: str = Query("all"),
        search: Optional[str] = Query(None),
        person_id: Optional[int] = Query(None),
        cluster_id: Optional[int] = Query(None),
        folder_path: Optional[str] = Query(None),
        sort_by: str = Query("date"),
        sort_order: str = Query("desc"),
        ctx_tag_id: Optional[int] = Query(None),
    ):
        img = db.get_image(image_id)
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        all_tags = db.get_all_tags()
        img_tags = {t["id"] for t in img.get("tags", [])}
        return templates.TemplateResponse(
            request=request,
            name="partials/tag_picker_modal.html",
            context={
                "image_id": image_id,
                "all_tags": all_tags,
                "img_tags": img_tags,
                "modal_ctx": {
                    k: v
                    for k, v in {
                        "filter_type": filter_type,
                        "search": search,
                        "person_id": person_id,
                        "cluster_id": cluster_id,
                        "folder_path": folder_path,
                        "sort_by": sort_by,
                        "sort_order": sort_order,
                        "ctx_tag_id": ctx_tag_id,
                    }.items()
                    if v is not None
                },
            },
        )

    @app.post("/api/photos/{image_id}/tags", response_class=HTMLResponse)
    async def add_tags_to_photo(
        request: Request,
        image_id: int,
        tags: str = Form(""),
        filter_type: str = Form("all"),
        search: Optional[str] = Form(None),
        person_id: Optional[int] = Form(None),
        cluster_id: Optional[int] = Form(None),
        folder_path: Optional[str] = Form(None),
        similar_to: Optional[int] = Form(None),
        threshold: Optional[float] = Form(None),
        sort_by: str = Form("date"),
        sort_order: str = Form("desc"),
        ctx_tag_id: Optional[int] = Form(None),
    ):
        tag_names = [t.strip() for t in tags.split(",") if t.strip()]
        db.add_tags_to_image(image_id, tag_names)
        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
            filter_type=filter_type,
            search=search,
            person_id=person_id,
            cluster_id=cluster_id,
            folder_path=folder_path,
            similar_to=similar_to,
            threshold=threshold,
            sort_by=sort_by,
            sort_order=sort_order,
            tag_id=ctx_tag_id,
        )

    @app.post("/api/photos/{image_id}/tags/{tag_id}/remove", response_class=HTMLResponse)
    async def remove_tag_from_photo(
        request: Request,
        image_id: int,
        tag_id: int,
        filter_type: str = Form("all"),
        search: Optional[str] = Form(None),
        person_id: Optional[int] = Form(None),
        cluster_id: Optional[int] = Form(None),
        folder_path: Optional[str] = Form(None),
        similar_to: Optional[int] = Form(None),
        threshold: Optional[float] = Form(None),
        sort_by: str = Form("date"),
        sort_order: str = Form("desc"),
        ctx_tag_id: Optional[int] = Form(None),
    ):
        db.remove_tag_from_image(image_id, tag_id)
        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
            filter_type=filter_type,
            search=search,
            person_id=person_id,
            cluster_id=cluster_id,
            folder_path=folder_path,
            similar_to=similar_to,
            threshold=threshold,
            sort_by=sort_by,
            sort_order=sort_order,
            tag_id=ctx_tag_id,
        )

    @app.get("/api/faces/{face_id}/crop")
    async def get_face_avatar(face_id: int):
        face = db.get_face(face_id)
        if not face or not Path(face["file_path"]).exists():
            raise HTTPException(status_code=404, detail="Face record not found")

        crop = cache.get_face_crop(face_id, face["file_path"], face["bbox"], size=200)
        if not crop or not crop.exists():
            raise HTTPException(status_code=500, detail="Failed to crop face")

        return FileResponse(
            crop,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.post("/api/faces/{face_id}/name", response_class=HTMLResponse)
    async def assign_face_name(
        request: Request,
        face_id: int,
        name: str = Form(...),
        image_id: int = Form(...),
    ):
        face = db.get_face(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")

        db.name_person(name=name, cluster_id=face["cluster_id"], face_id=face_id)

        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
        )

    @app.post("/api/clusters/{cluster_id}/name", response_class=HTMLResponse)
    async def assign_cluster_name(
        request: Request,
        cluster_id: int,
        name: str = Form(...),
    ):
        person_id = db.rename_cluster(cluster_id=cluster_id, name=name)
        return HTMLResponse(f"""
            <div id="cluster-name-section" style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #a5b4fc;">{name}</span>
                <a href="/person/{person_id}" class="btn btn-primary" style="padding: 0.3rem 0.75rem; font-size: 0.8125rem;">View Person Profile →</a>
            </div>
        """)

    @app.post("/api/people/{person_id}/name", response_class=HTMLResponse)
    async def rename_person(person_id: int, name: str = Form(...)):
        db.name_person(name=name, person_id=person_id)
        return HTMLResponse(f"""
          <div id="person-name-section">
            <form hx-post="/api/people/{person_id}/name" hx-target="#person-name-section" hx-swap="outerHTML" style="display: flex; align-items: center; gap: 0.5rem; max-width: 400px;">
              <input type="text" name="name" value="{name}" class="inline-edit-input" style="font-size: 1.25rem; font-weight: 700; padding: 0.4rem 0.75rem;">
              <button type="submit" class="btn btn-primary" style="white-space: nowrap;">Saved ✓</button>
            </form>
          </div>
        """)

    @app.get("/api/targets/picker", response_class=HTMLResponse)
    async def get_targets_picker(
        request: Request,
        mode: str = Query("move_face"),
        source_id: int = Query(...),
        image_id: Optional[int] = Query(None),
        source_name: Optional[str] = Query(""),
    ):
        all_targets = db.get_people(cluster_limit=50)
        if mode == "merge_person":
            targets = [t for t in all_targets if not (t["type"] == "person" and t["id"] == source_id)]
        elif mode == "merge_cluster":
            targets = [t for t in all_targets if not (t["type"] == "cluster" and t["id"] == source_id)]
        else:
            targets = all_targets

        return templates.TemplateResponse(
            request=request,
            name="partials/target_picker_modal.html",
            context={
                "action_mode": mode,
                "source_id": source_id,
                "image_id": image_id,
                "source_name": source_name,
                "targets": targets,
            },
        )

    @app.get("/api/targets/items", response_class=HTMLResponse)
    async def get_targets_picker_items(
        request: Request,
        mode: str = Query("move_face"),
        source_id: int = Query(...),
        image_id: Optional[int] = Query(None),
        search: Optional[str] = Query(None),
    ):
        search_term = (search or "").strip()
        limit = 100 if search_term else 50
        all_targets = db.get_people(search=search_term, cluster_limit=limit)
        if mode == "merge_person":
            targets = [t for t in all_targets if not (t["type"] == "person" and t["id"] == source_id)]
        elif mode == "merge_cluster":
            targets = [t for t in all_targets if not (t["type"] == "cluster" and t["id"] == source_id)]
        else:
            targets = all_targets

        return templates.TemplateResponse(
            request=request,
            name="partials/target_picker_items.html",
            context={
                "action_mode": mode,
                "source_id": source_id,
                "image_id": image_id,
                "targets": targets,
            },
        )

    @app.post("/api/faces/{face_id}/move", response_class=HTMLResponse)
    async def move_face_endpoint(
        request: Request,
        face_id: int,
        image_id: int = Form(...),
        target_type: str = Form(...),
        target_id: Optional[int] = Form(None),
        new_name: Optional[str] = Form(None),
    ):
        if target_type == "person":
            db.move_face(face_id, target_person_id=target_id)
        elif target_type == "cluster":
            db.move_face(face_id, target_cluster_id=target_id)
        elif target_type == "new" and new_name:
            db.move_face(face_id, new_person_name=new_name)
        elif target_type == "unlink":
            db.move_face(face_id, unlink=True)

        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
        )

    @app.post("/api/faces/{face_id}/unlink", response_class=HTMLResponse)
    async def unlink_face_endpoint(
        request: Request,
        face_id: int,
        image_id: int = Form(...),
    ):
        db.move_face(face_id, unlink=True)
        return _render_photo_modal_response(
            request=request,
            image_id=image_id,
        )

    @app.post("/api/people/{person_id}/merge")
    async def merge_person_endpoint(
        person_id: int,
        target_person_id: Optional[int] = Form(None),
        target_type: Optional[str] = Form("person"),
        target_id: Optional[int] = Form(None),
    ):
        tgt_id = target_person_id if target_person_id is not None else target_id
        if tgt_id is None:
            return Response(status_code=400, content="Missing target ID")
        if target_type == "cluster":
            db.merge_person_into_cluster(source_person_id=person_id, target_cluster_id=tgt_id)
            return Response(status_code=200, headers={"HX-Redirect": f"/?cluster_id={tgt_id}"})
        else:
            db.merge_people(source_person_id=person_id, target_person_id=tgt_id)
            return Response(status_code=200, headers={"HX-Redirect": f"/person/{tgt_id}"})

    @app.post("/api/clusters/{cluster_id}/merge")
    async def merge_cluster_endpoint(
        cluster_id: int,
        target_type: str = Form(...),
        target_id: int = Form(...),
    ):
        if target_type == "person":
            db.merge_cluster_into_person(source_cluster_id=cluster_id, target_person_id=target_id)
            return Response(status_code=200, headers={"HX-Redirect": f"/person/{target_id}"})
        else:
            db.merge_clusters(source_cluster_id=cluster_id, target_cluster_id=target_id)
            return Response(status_code=200, headers={"HX-Redirect": f"/?cluster_id={target_id}"})

    @app.post("/api/people/{person_id}/autotag", response_class=HTMLResponse)
    async def autotag_person_endpoint(person_id: int):
        p = db.get_person(person_id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")

        exemplars = db.get_person_exemplars(person_id, max_exemplars=5)
        if not exemplars:
            return HTMLResponse(f"""
                <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    No reference face embeddings found for {p['name']}. Assign at least one face first.
                </div>
            """)

        unassigned_faces = db.get_unassigned_faces()
        if not unassigned_faces:
            return HTMLResponse("""
                <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    All faces in library are already assigned to people.
                </div>
            """)

        threshold = float(db.get_setting("recognition_match_threshold", 0.42))
        exclusions = db.get_person_exclusions()
        matcher = MultiExemplarMatcher({person_id: exemplars})
        matched_results = matcher.match_faces_batch(unassigned_faces, exclusions=exclusions, threshold=threshold)

        matched_face_ids = list(matched_results.keys())
        if matched_face_ids:
            db.assign_faces_to_person(matched_face_ids, person_id)
            return Response(status_code=200, headers={"HX-Refresh": "true"})

        return HTMLResponse(f"""
            <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                Scanned {len(unassigned_faces)} unassigned faces: 0 matched {p['name']} (threshold: {threshold:.2f}).
            </div>
        """)

    @app.get("/api/people/all/candidates")
    async def get_all_people_candidates_endpoint(
        limit: int = Query(10, ge=1, le=50),
        min_dist: Optional[float] = Query(None),
        max_dist: Optional[float] = Query(None),
    ):
        all_exemplars = db.get_all_person_exemplars(max_exemplars=5)
        unassigned_faces = db.get_unassigned_faces()
        total_unassigned = len(unassigned_faces)

        if not all_exemplars:
            return JSONResponse({
                "candidates": [],
                "total_unassigned": total_unassigned,
                "message": "No named people found in library. Name at least one person first.",
            })

        if not unassigned_faces:
            return JSONResponse({
                "candidates": [],
                "total_unassigned": 0,
                "message": "All faces in library are already assigned to people.",
            })

        rec_thresh = float(db.get_setting("recognition_match_threshold", 0.42))
        _min_dist = min_dist if min_dist is not None else max(0.20, rec_thresh - 0.05)
        _max_dist = max_dist if max_dist is not None else 0.60

        exclusions = db.get_person_exclusions()
        matcher = MultiExemplarMatcher(all_exemplars)
        cands = matcher.find_uncertain_candidates_all(
            faces=unassigned_faces,
            exclusions=exclusions,
            min_dist=_min_dist,
            max_dist=_max_dist,
            limit=limit,
        )

        all_people = {p["id"]: p for p in db.get_people()}

        formatted_cands = []
        for c in cands:
            p_id = c["target_person_id"]
            p_info = all_people.get(p_id, {})
            formatted_cands.append({
                "face_id": c["id"],
                "image_id": c["image_id"],
                "person_id": p_id,
                "person_name": p_info.get("name", f"Person #{p_id}"),
                "person_cover_face_id": p_info.get("cover_face_id"),
                "distance": c["distance"],
                "similarity_pct": c["similarity_pct"],
                "file_path": c["file_path"],
                "filename": Path(c["file_path"]).name if c.get("file_path") else "",
                "bbox": c["bbox"],
            })

        return JSONResponse({
            "candidates": formatted_cands,
            "total_unassigned": total_unassigned,
            "message": "No uncertain matches found in library." if not formatted_cands else "",
        })

    @app.get("/api/people/{person_id}/candidates")
    async def get_person_candidates_endpoint(
        person_id: int,
        limit: int = Query(10, ge=1, le=50),
        min_dist: Optional[float] = Query(None),
        max_dist: Optional[float] = Query(None),
    ):
        p = db.get_person(person_id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")

        exemplars = db.get_person_exemplars(person_id, max_exemplars=5)
        if not exemplars:
            return JSONResponse({
                "person": {"id": p["id"], "name": p["name"], "cover_face_id": p.get("cover_face_id")},
                "candidates": [],
                "message": f"No reference face embeddings found for {p['name']}. Assign at least one face first.",
            })

        unassigned_faces = db.get_unassigned_faces()
        if not unassigned_faces:
            return JSONResponse({
                "person": {"id": p["id"], "name": p["name"], "cover_face_id": p.get("cover_face_id")},
                "candidates": [],
                "message": "No unassigned faces left in library.",
            })

        rec_thresh = float(db.get_setting("recognition_match_threshold", 0.42))
        _min_dist = min_dist if min_dist is not None else max(0.20, rec_thresh - 0.05)
        _max_dist = max_dist if max_dist is not None else 0.60

        exclusions = db.get_person_exclusions()
        matcher = MultiExemplarMatcher({person_id: exemplars})
        cands = matcher.find_uncertain_candidates(
            person_id=person_id,
            faces=unassigned_faces,
            exclusions=exclusions,
            min_dist=_min_dist,
            max_dist=_max_dist,
            limit=limit,
        )

        formatted_cands = []
        for c in cands:
            formatted_cands.append({
                "face_id": c["id"],
                "image_id": c["image_id"],
                "distance": c["distance"],
                "similarity_pct": c["similarity_pct"],
                "file_path": c["file_path"],
                "filename": Path(c["file_path"]).name if c.get("file_path") else "",
                "bbox": c["bbox"],
            })

        return JSONResponse({
            "person": {
                "id": p["id"],
                "name": p["name"],
                "cover_face_id": p.get("cover_face_id"),
            },
            "candidates": formatted_cands,
            "total_unassigned": len(unassigned_faces),
        })

    @app.post("/api/people/{person_id}/confirm-match")
    async def confirm_person_match_endpoint(
        person_id: int,
        request: Request,
    ):
        p = db.get_person(person_id)
        if not p:
            raise HTTPException(status_code=404, detail="Person not found")

        body = await request.json()
        face_id = body.get("face_id")
        matched = body.get("matched", False)

        if not face_id:
            raise HTTPException(status_code=400, detail="face_id is required")

        if matched:
            db.assign_faces_to_person([int(face_id)], person_id)
            action = "assigned"
        else:
            db.add_person_exclusion(int(face_id), person_id)
            action = "excluded"

        return JSONResponse({"status": "ok", "action": action, "face_id": face_id, "person_id": person_id})

    @app.post("/api/people/autotag-all", response_class=HTMLResponse)
    async def autotag_all_people_endpoint():
        all_exemplars = db.get_all_person_exemplars(max_exemplars=5)
        if not all_exemplars:
            return HTMLResponse("""
                <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    No named people found in library. Name at least one person first.
                </div>
            """)

        unassigned_faces = db.get_unassigned_faces()
        if not unassigned_faces:
            return HTMLResponse("""
                <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    All faces in library are already assigned to people!
                </div>
            """)

        threshold = float(db.get_setting("recognition_match_threshold", 0.42))
        exclusions = db.get_person_exclusions()
        matcher = MultiExemplarMatcher(all_exemplars)
        matched_results = matcher.match_faces_batch(unassigned_faces, exclusions=exclusions, threshold=threshold)

        if not matched_results:
            return HTMLResponse(f"""
                <div class="alert alert-info" style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; color: #93c5fd; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    Scanned {len(unassigned_faces)} unassigned faces: 0 matches found for named people (threshold: {threshold:.2f}).
                </div>
            """)

        person_to_faces: Dict[int, List[int]] = {}
        for f_id, (p_id, _dist) in matched_results.items():
            if p_id not in person_to_faces:
                person_to_faces[p_id] = []
            person_to_faces[p_id].append(f_id)

        for p_id, f_ids in person_to_faces.items():
            db.assign_faces_to_person(f_ids, p_id)

        total_matched = len(matched_results)
        num_people_matched = len(person_to_faces)
        return HTMLResponse(f"""
            <div class="alert alert-success" style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                ✓ Auto-tagged {total_matched} faces across {num_people_matched} named people!
            </div>
        """)

    @app.post("/api/people/merge")
    async def merge_people(source_id: int = Form(...), target_id: int = Form(...)):
        db.merge_people(source_person_id=source_id, target_person_id=target_id)
        return {"status": "success", "merged_into": target_id}

    # ==================== FILE MANAGEMENT ENDPOINTS ====================

    async def _extract_request_payload(req: Request) -> Dict[str, Any]:
        """Extract parameters from either application/json or multipart/form-data / x-www-form-urlencoded."""
        c_type = req.headers.get("content-type", "")
        if "application/json" in c_type:
            try:
                return await req.json()
            except Exception:
                return {}
        try:
            form = await req.form()
            return dict(form)
        except Exception:
            return {}

    @app.post("/api/files/delete")
    async def delete_files_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        raw_ids = _parse_image_ids(payload.get("image_ids") or payload.get("image_id"))

        if not raw_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        deleted_count = 0
        for img_id in raw_ids:
            img = db.get_image(img_id)
            if img and img.get("file_path"):
                fpath = Path(img["file_path"])
                if fpath.exists() and fpath.is_file():
                    try:
                        fpath.unlink(missing_ok=True)
                    except Exception:
                        pass
                cache.invalidate_media_cache(img["file_path"])
                deleted_count += 1

        db.delete_image_records(raw_ids)
        return JSONResponse({"status": "success", "deleted_count": deleted_count, "deleted_ids": raw_ids})

    @app.post("/api/files/rename")
    async def rename_file_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        img_id = payload.get("image_id")
        new_name = payload.get("new_name")

        if img_id is None or not new_name:
            raise HTTPException(status_code=400, detail="image_id and new_name are required")

        try:
            img_id = int(img_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid image_id")

        new_name = str(new_name).strip()
        if not new_name or "/" in new_name or "\\" in new_name or ".." in new_name:
            raise HTTPException(status_code=400, detail="Invalid filename")

        img = db.get_image(img_id)
        if not img or not img.get("file_path"):
            raise HTTPException(status_code=404, detail="Image not found")

        old_path = Path(img["file_path"])
        if not old_path.exists():
            raise HTTPException(status_code=404, detail="File does not exist on disk")

        # If user didn't specify extension, keep existing extension
        if not Path(new_name).suffix and old_path.suffix:
            new_name = f"{new_name}{old_path.suffix}"

        dest_path = old_path.parent / new_name
        if dest_path.resolve() == old_path.resolve():
            return JSONResponse({"status": "success", "image_id": img_id, "new_name": new_name, "new_path": str(dest_path)})

        if dest_path.exists():
            raise HTTPException(status_code=400, detail=f"A file named '{new_name}' already exists in this folder")

        try:
            old_path.rename(dest_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to rename file: {e}")

        cache.invalidate_media_cache(old_path)
        db.update_image_path(img_id, str(dest_path))

        return JSONResponse({
            "status": "success",
            "image_id": img_id,
            "new_name": new_name,
            "new_path": str(dest_path)
        })

    @app.post("/api/files/batch-rename")
    async def batch_rename_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        raw_ids = _parse_image_ids(payload.get("image_ids"))
        mode = str(payload.get("mode", "prefix"))
        prefix = str(payload.get("prefix", ""))
        suffix = str(payload.get("suffix", ""))
        find_text = str(payload.get("find_text", ""))
        replace_text = str(payload.get("replace_text", ""))
        pattern = str(payload.get("pattern", ""))
        try:
            start_index = int(payload.get("start_index", 1))
        except (ValueError, TypeError):
            start_index = 1

        if not raw_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        renamed_count = 0
        for idx, img_id in enumerate(raw_ids):
            img = db.get_image(img_id)
            if not img or not img.get("file_path"):
                continue
            old_path = Path(img["file_path"])
            if not old_path.exists():
                continue

            stem = old_path.stem
            ext = old_path.suffix

            if mode == "prefix":
                new_filename = f"{prefix}{stem}{ext}"
            elif mode == "suffix":
                new_filename = f"{stem}{suffix}{ext}"
            elif mode == "replace":
                if find_text:
                    new_filename = f"{stem.replace(find_text, replace_text)}{ext}"
                else:
                    new_filename = old_path.name
            elif mode == "pattern":
                seq_num = start_index + idx
                if "{n}" in pattern:
                    p_name = pattern.replace("{n}", f"{seq_num:02d}")
                elif "{n:03d}" in pattern:
                    p_name = pattern.replace("{n:03d}", f"{seq_num:03d}")
                elif "{n:04d}" in pattern:
                    p_name = pattern.replace("{n:04d}", f"{seq_num:04d}")
                elif "{name}" in pattern:
                    p_name = pattern.replace("{name}", stem)
                else:
                    p_name = f"{pattern}_{seq_num}" if pattern else f"{stem}_{seq_num}"
                new_filename = f"{p_name}{ext}" if not Path(p_name).suffix else p_name
            else:
                new_filename = old_path.name

            new_filename = new_filename.strip()
            if not new_filename or "/" in new_filename or "\\" in new_filename or ".." in new_filename:
                continue

            dest_path = _get_unique_destination_path(old_path.parent, new_filename)
            if dest_path.resolve() == old_path.resolve():
                continue

            try:
                old_path.rename(dest_path)
                cache.invalidate_media_cache(old_path)
                db.update_image_path(img_id, str(dest_path))
                renamed_count += 1
            except Exception:
                pass

        return JSONResponse({"status": "success", "renamed_count": renamed_count})

    @app.post("/api/files/move")
    async def move_files_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        raw_ids = _parse_image_ids(payload.get("image_ids"))
        destination_folder = str(payload.get("destination_folder", "")).strip()

        if not raw_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")
        if not destination_folder:
            raise HTTPException(status_code=400, detail="destination_folder is required")

        dest_dir = Path(destination_folder).resolve()
        if not _within_source_root(dest_dir):
            raise HTTPException(status_code=400, detail="Destination must be within the media source directory")
        dest_dir.mkdir(parents=True, exist_ok=True)

        moved_count = 0
        for img_id in raw_ids:
            img = db.get_image(img_id)
            if not img or not img.get("file_path"):
                continue
            old_path = Path(img["file_path"])
            if not old_path.exists():
                continue

            target_path = _get_unique_destination_path(dest_dir, old_path.name)
            if target_path.resolve() == old_path.resolve():
                continue

            try:
                shutil.move(str(old_path), str(target_path))
                cache.invalidate_media_cache(old_path)
                db.update_image_path(img_id, str(target_path))
                moved_count += 1
            except Exception:
                pass

        return JSONResponse({
            "status": "success",
            "moved_count": moved_count,
            "destination": str(dest_dir)
        })

    @app.post("/api/files/copy")
    async def copy_files_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        raw_ids = _parse_image_ids(payload.get("image_ids"))
        destination_folder = payload.get("destination_folder")
        if destination_folder:
            destination_folder = str(destination_folder).strip()

        if not raw_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        copied_count = 0
        for img_id in raw_ids:
            img = db.get_image(img_id)
            if not img or not img.get("file_path"):
                continue
            src_path = Path(img["file_path"])
            if not src_path.exists():
                continue

            if destination_folder:
                dest_dir = Path(destination_folder).resolve()
                if not _within_source_root(dest_dir):
                    raise HTTPException(status_code=400, detail="Destination must be within the media source directory")
                dest_dir.mkdir(parents=True, exist_ok=True)
                candidate_name = src_path.name
            else:
                dest_dir = src_path.parent
                candidate_name = f"{src_path.stem}_copy{src_path.suffix}"

            target_path = _get_unique_destination_path(dest_dir, candidate_name)
            try:
                shutil.copy2(str(src_path), str(target_path))
                db.clone_image_record(img_id, str(target_path))
                copied_count += 1
            except Exception:
                pass

        return JSONResponse({"status": "success", "copied_count": copied_count})

    @app.post("/api/files/batch-favorite")
    async def batch_favorite_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        raw_ids = _parse_image_ids(payload.get("image_ids"))
        raw_fav = payload.get("is_favorite", True)
        if isinstance(raw_fav, str):
            is_favorite = raw_fav.lower() in ("true", "1", "yes")
        else:
            is_favorite = bool(raw_fav)

        if not raw_ids:
            raise HTTPException(status_code=400, detail="No image IDs provided")

        count = db.batch_toggle_favorites(raw_ids, is_favorite)
        return JSONResponse({"status": "success", "count": count, "is_favorite": is_favorite})

    @app.post("/api/folders/rename")
    async def rename_folder_endpoint(request: Request):
        payload = await _extract_request_payload(request)
        folder_path = payload.get("folder_path")
        new_name = payload.get("new_name")

        if not folder_path or not new_name:
            raise HTTPException(status_code=400, detail="folder_path and new_name are required")

        new_name = str(new_name).strip()
        if not new_name or "/" in new_name or "\\" in new_name or ".." in new_name:
            raise HTTPException(status_code=400, detail="Invalid folder name")

        old_dir = Path(folder_path).resolve()
        if not old_dir.exists() or not old_dir.is_dir():
            raise HTTPException(status_code=404, detail="Folder does not exist on disk")

        dest_dir = old_dir.parent / new_name
        if dest_dir.resolve() == old_dir:
            return JSONResponse({"status": "success", "renamed_count": 0, "new_path": str(dest_dir.resolve()), "new_parent": str(dest_dir.parent.resolve())})

        if dest_dir.exists():
            raise HTTPException(status_code=400, detail=f"A folder named '{new_name}' already exists here")

        try:
            old_dir.rename(dest_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to rename folder: {e}")

        old_paths = db.update_folder_paths(old_dir, dest_dir)
        for old_p in old_paths:
            cache.invalidate_media_cache(old_p)

        return JSONResponse({
            "status": "success",
            "renamed_count": len(old_paths),
            "new_path": str(dest_dir.resolve()),
            "new_parent": str(dest_dir.parent.resolve()),
        })

    @app.get("/api/folders/list")
    async def get_folders_list():
        folders = db.get_all_folder_paths()
        return JSONResponse({"folders": folders})

    @app.get("/api/folders/picker-modal", response_class=HTMLResponse)
    async def get_folder_picker_modal(
        request: Request,
        action: str = Query("move"),
        image_ids: str = Query(""),
        current_folder: Optional[str] = Query(None),
    ):
        folders = db.get_all_folder_paths()
        source_root = _media_source_root()
        if source_root is not None:
            folders = [f for f in folders if Path(f).is_relative_to(source_root)]
        return templates.TemplateResponse(
            request=request,
            name="partials/folder_picker_modal.html",
            context={
                "action": action,
                "image_ids": image_ids,
                "folders": folders,
                "source_root": str(source_root) if source_root else None,
                "current_folder": current_folder,
            },
        )

    @app.get("/api/files/batch-rename-modal", response_class=HTMLResponse)
    async def get_batch_rename_modal(
        request: Request,
        image_ids: str = Query(""),
    ):
        return templates.TemplateResponse(
            request=request,
            name="partials/batch_rename_modal.html",
            context={"image_ids": image_ids},
        )

    return app

