"""
FastAPI web server for the Kimera Media Gallery.
Provides HTMX-driven HTML views, REST APIs, and cached thumbnail/crop streaming.
"""

from __future__ import annotations

import mimetypes
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.cache import ThumbnailCache
from app.db import Database

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
        conf_threshold: float = 0.5,
        eps: float = 0.65,
        min_samples: int = 1,
        algorithm: str = "dbscan",
        min_interval_sec: float = 30.0,
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
            args=(input_dir, conf_threshold, eps, min_samples, algorithm, min_interval_sec),
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


def create_app(db_path: str = "face_clusters.db", cache_dir: Optional[str] = None) -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    app = FastAPI(title="Kimera Media Gallery")

    resolved_cache_dir = str(Path(cache_dir).resolve()) if cache_dir else str(Path.cwd() / ".cache")
    db = Database(db_path)
    cache = ThumbnailCache(resolved_cache_dir)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    scan_mgr = ScanManager(db_path, cache_dir=resolved_cache_dir)

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def gallery_view(
        request: Request,
        filter: str = Query("all", alias="filter"),
        search: Optional[str] = Query(None),
        person_id: Optional[int] = Query(None),
        cluster_id: Optional[int] = Query(None),
        folder_path: Optional[str] = Query(None),
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
        page: int = Query(1, ge=1),
        infinite: int = Query(0),
    ):
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
        )
        active_tab = "favorites" if filter == "favorites" else "photos"
        
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
                    "sort_by": sort_by,
                    "sort_order": sort_order,
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
                "sort_by": sort_by,
                "sort_order": sort_order,
                "active_page": active_tab,
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
        
        # Load images directly in current folder (or under it)
        images_data = db.get_images(
            folder_path=cur_folder,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            limit=48,
        )

        if request.headers.get("HX-Request") and infinite == 1:
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
                    "target_url": f"/folders?path={cur_folder}",
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
        people = db.get_people(search=search)
        return templates.TemplateResponse(
            request=request,
            name="people.html",
            context={
                "people": people,
                "search": search,
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

        data = db.get_images(
            filter_type="person",
            person_id=person_id,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            limit=48,
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
        # Default scan folder if pictures directory exists
        default_dir = str(Path("pictures").resolve()) if Path("pictures").exists() else str(Path.cwd())
        cache_stats = cache.get_cache_stats()
        return templates.TemplateResponse(
            request=request,
            name="settings.html",
            context={
                "stats": stats,
                "default_dir": default_dir,
                "cache_dir": str(cache.cache_dir),
                "cache_stats": cache_stats,
                "scan_mgr": scan_mgr,
                "active_page": "settings",
            },
        )

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

    @app.post("/api/scan/trigger", response_class=HTMLResponse)
    async def trigger_scan(
        request: Request,
        input_dir: str = Form(...),
        conf_threshold: float = Form(0.5),
        eps: float = Form(0.65),
        min_samples: int = Form(1),
        algorithm: str = Form("dbscan"),
        min_interval_sec: float = Form(30.0),
    ):
        if not Path(input_dir).exists():
            return HTMLResponse(
                f"""<div class="alert alert-error" style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; padding: 0.75rem 1rem; border-radius: 8px;">
                    Directory not found: {input_dir}
                </div>"""
            )

        scan_mgr.start_scan(
            input_dir=input_dir,
            conf_threshold=conf_threshold,
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            min_interval_sec=min_interval_sec,
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

    @app.get("/api/photos/{image_id}/modal", response_class=HTMLResponse)
    async def get_photo_modal(
        request: Request,
        image_id: int,
        filter: str = Query("all", alias="filter"),
        search: Optional[str] = Query(None),
        person_id: Optional[int] = Query(None),
        cluster_id: Optional[int] = Query(None),
        folder_path: Optional[str] = Query(None),
        sort_by: str = Query("date", alias="sort"),
        sort_order: str = Query("desc", alias="order"),
    ):
        img_meta = db.get_image(image_id)
        if not img_meta:
            raise HTTPException(status_code=404, detail="Image not found")
        img_meta["filename"] = Path(img_meta["file_path"]).name

        # Calculate adjacent IDs for modal arrow/swipe navigation
        adj = db.get_adjacent_image_ids(
            image_id=image_id,
            filter_type=filter,
            person_id=person_id,
            cluster_id=cluster_id,
            folder_path=folder_path,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return templates.TemplateResponse(
            request=request,
            name="partials/photo_modal.html",
            context={
                "image": img_meta,
                "prev_id": adj["prev_id"],
                "next_id": adj["next_id"],
                "current_index": adj["current_index"],
                "total_count": adj["total_count"],
                "filter_type": filter,
                "search": search,
                "person_id": person_id,
                "cluster_id": cluster_id,
                "folder_path": folder_path,
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
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

        # Return updated modal
        img_meta = db.get_image(image_id)
        if not img_meta:
            raise HTTPException(status_code=404, detail="Image not found")
        img_meta["filename"] = Path(img_meta["file_path"]).name
        return templates.TemplateResponse(
            request=request,
            name="partials/photo_modal.html",
            context={"image": img_meta},
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

    @app.post("/api/people/merge")
    async def merge_people(source_id: int = Form(...), target_id: int = Form(...)):
        db.merge_people(source_person_id=source_id, target_person_id=target_id)
        return {"status": "success", "merged_into": target_id}

    return app

