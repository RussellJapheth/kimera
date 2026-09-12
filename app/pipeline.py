"""
Pipeline orchestrator connecting scanning, detection, embedding, clustering, and SQLite storage.
"""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
from tqdm import tqdm

from app.clustering import cluster_embeddings
from app.db import Database
from app.models import FaceDetector, FaceEmbedder
from app.scanner import (
    compute_quick_hash,
    extract_video_keyframes,
    is_video_file,
    load_image_rgb,
    scan_media_paths,
)


class FileScanTask(NamedTuple):
    path: Path
    path_str: str
    needs_scan: bool
    image_id: Optional[int]
    file_size: int
    mtime: float
    content_hash: str
    is_video: bool


def _prefilter_file(
    path: Path,
    meta: Optional[Dict[str, Any]],
) -> Tuple[FileScanTask, Optional[Tuple[int, int, float, str]]]:
    """
    Worker function to check file stats and content hashes in parallel.
    Returns (FileScanTask, touch_update_tuple)
    """
    path_str = str(path.resolve())
    is_video = is_video_file(path)
    try:
        stat = path.stat()
        cur_size = stat.st_size
        cur_mtime = stat.st_mtime
    except Exception:
        return (
            FileScanTask(
                path=path,
                path_str=path_str,
                needs_scan=False,
                image_id=None,
                file_size=0,
                mtime=0.0,
                content_hash="",
                is_video=is_video,
            ),
            None,
        )

    if meta is None:
        # New file
        cur_hash = compute_quick_hash(path)
        return (
            FileScanTask(
                path=path,
                path_str=path_str,
                needs_scan=True,
                image_id=None,
                file_size=cur_size,
                mtime=cur_mtime,
                content_hash=cur_hash,
                is_video=is_video,
            ),
            None,
        )

    image_id = meta["id"]
    # 1. Quick check: mtime and size match
    if meta.get("file_size") == cur_size and abs(float(meta.get("mtime") or 0) - cur_mtime) < 0.001:
        return (
            FileScanTask(
                path=path,
                path_str=path_str,
                needs_scan=False,
                image_id=image_id,
                file_size=cur_size,
                mtime=cur_mtime,
                content_hash=meta.get("content_hash") or "",
                is_video=is_video,
            ),
            None,
        )

    # 2. Mtime or size changed: check 64KB quick hash
    cur_hash = compute_quick_hash(path)
    if cur_hash and cur_hash == meta.get("content_hash"):
        # Content did not change, just touched mtime
        return (
            FileScanTask(
                path=path,
                path_str=path_str,
                needs_scan=False,
                image_id=image_id,
                file_size=cur_size,
                mtime=cur_mtime,
                content_hash=cur_hash,
                is_video=is_video,
            ),
            (image_id, cur_size, cur_mtime, cur_hash),
        )

    # Content genuinely modified
    return (
        FileScanTask(
            path=path,
            path_str=path_str,
            needs_scan=True,
            image_id=image_id,
            file_size=cur_size,
            mtime=cur_mtime,
            content_hash=cur_hash,
            is_video=is_video,
        ),
        None,
    )


def run_pipeline(
    input_dir: Path | str,
    db_path: Path | str = "face_clusters.db",
    cache_dir: Optional[Path | str] = None,
    conf_threshold: float = 0.5,
    eps: float = 0.65,
    min_samples: int = 1,
    clustering_algorithm: str = "dbscan",
    export_dir: Optional[Path | str] = None,
    scene_threshold: float = 0.35,
    min_interval_sec: float = 30.0,
    max_interval_sec: float = 90.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run high-performance incremental face scanning, detection, embedding, and clustering.
    Uses multi-threaded parallel pre-filtering and background image decode prefetching.
    """
    db = Database(db_path)
    media_paths = scan_media_paths(input_dir)

    if not media_paths:
        return {
            "images_scanned": 0,
            "faces_detected": 0,
            "num_clusters": 0,
            "clusters": [],
            "unclustered_faces": 0,
        }

    from app.cache import ThumbnailCache

    cache = ThumbnailCache(cache_dir)
    detector: Optional[FaceDetector] = None
    embedder: Optional[FaceEmbedder] = None

    def notify_progress(msg: str, percent: int, current: int, total: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(msg, percent, current, total)
        except TypeError:
            try:
                progress_callback(msg)
            except Exception:
                pass

    def get_models() -> tuple[FaceDetector, FaceEmbedder]:
        nonlocal detector, embedder
        if detector is None:
            notify_progress("Loading face detection model (InsightFace / SCRFD)...", 3, 0, len(media_paths))
            detector = FaceDetector(conf_threshold=conf_threshold, progress_callback=notify_progress)
        if embedder is None:
            notify_progress("Loading face recognition embedding model (ArcFace)...", 4, 0, len(media_paths))
            embedder = FaceEmbedder(progress_callback=notify_progress)
        return detector, embedder

    total_media = len(media_paths)

    # 1. Clean up records for deleted files
    existing_paths = {str(p.resolve()) for p in media_paths}
    db.delete_missing_images(existing_paths)

    # 2. Parallel pre-filtering and metadata checking
    notify_progress(f"Checking library status for {total_media} files...", 2, 0, total_media)
    meta_map = db.get_all_images_file_meta_map()

    tasks: List[FileScanTask] = []
    touch_updates: List[Tuple[int, int, float, str]] = []

    # Check file stats & hashes concurrently across CPU threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(4, total_media))) as executor:
        future_to_path = [
            executor.submit(_prefilter_file, p, meta_map.get(str(p.resolve())))
            for p in media_paths
        ]
        for fut in concurrent.futures.as_completed(future_to_path):
            task, touch_up = fut.result()
            tasks.append(task)
            if touch_up:
                touch_updates.append(touch_up)

    # Apply touches to database
    for img_id, sz, mt, hsh in touch_updates:
        db.update_image_meta(img_id, file_size=sz, mtime=mt, content_hash=hsh)

    # Invalidate modified items
    items_to_scan = [t for t in tasks if t.needs_scan]
    for task in items_to_scan:
        if task.image_id is not None:
            cache.invalidate_thumbnail(task.path_str)
            db.clear_faces_for_image(task.image_id)

    total_to_scan = len(items_to_scan)
    if total_to_scan > 0:
        notify_progress(f"Found {total_to_scan} new/modified media items. Initializing AI models...", 4, 0, total_to_scan)
        get_models()
    else:
        notify_progress(f"All {total_media} files up-to-date. Checking clusters...", 90, total_media, total_media)

    # 3. Process new/modified files with worker prefetch for images
    if total_to_scan > 0:
        notify_progress(f"Scanning {total_to_scan} media items...", 5, 0, total_to_scan)
        det, emb = get_models()

        # Prefetch pipeline for images
        def image_loader(task_list: List[FileScanTask], out_q: queue.Queue, stop_ev: threading.Event) -> None:
            for item in task_list:
                if stop_ev.is_set():
                    break
                if item.is_video:
                    out_q.put((item, None))
                else:
                    arr = load_image_rgb(item.path)
                    out_q.put((item, arr))
            out_q.put(None)  # Sentinel

        prefetch_q: queue.Queue = queue.Queue(maxsize=16)
        stop_event = threading.Event()
        loader_thread = threading.Thread(
            target=image_loader,
            args=(items_to_scan, prefetch_q, stop_event),
            daemon=True,
        )
        loader_thread.start()

        idx = 0
        try:
            with tqdm(total=total_to_scan, desc="Scanning media library") as pbar:
                while True:
                    queue_item = prefetch_q.get()
                    if queue_item is None:
                        break

                    task, img_rgb = queue_item
                    idx += 1
                    pct = int(5 + (idx / max(1, total_to_scan)) * 85)
                    notify_progress(f"Processing ({idx}/{total_to_scan}): {task.path.name}", pct, idx, total_to_scan)

                    if task.is_video:
                        video_image_id = task.image_id
                        vid_w, vid_h = 0, 0
                        for kf in extract_video_keyframes(
                            task.path,
                            scene_threshold=scene_threshold,
                            min_interval_sec=min_interval_sec,
                            max_interval_sec=max_interval_sec,
                        ):
                            if video_image_id is None:
                                vid_h, vid_w, _ = kf.frame_rgb.shape
                                video_image_id = db.insert_image(
                                    task.path_str,
                                    width=vid_w,
                                    height=vid_h,
                                    file_size=task.file_size,
                                    mtime=task.mtime,
                                    content_hash=task.content_hash,
                                )

                            detected_faces = det.detect(kf.frame_rgb)
                            for face in detected_faces:
                                vec = emb.extract_embedding(kf.frame_rgb, face.raw_face)
                                face_id = db.insert_face(
                                    image_id=video_image_id,
                                    bbox=face.bbox,
                                    confidence=face.confidence,
                                    embedding=vec,
                                    cluster_id=-1,
                                )
                                cache.save_face_crop_from_array(face_id, kf.frame_rgb, face.bbox)

                        if video_image_id is None:
                            db.insert_image(
                                task.path_str,
                                width=0,
                                height=0,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                            )
                        else:
                            db.update_image_meta(
                                video_image_id,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                                width=vid_w,
                                height=vid_h,
                            )
                    else:
                        # Image file
                        if img_rgb is None:
                            pbar.update(1)
                            continue

                        h, w, _ = img_rgb.shape
                        image_id = task.image_id
                        if image_id is None:
                            image_id = db.insert_image(
                                task.path_str,
                                width=w,
                                height=h,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                            )
                        else:
                            db.update_image_meta(
                                image_id,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                                width=w,
                                height=h,
                            )

                        detected_faces = det.detect(img_rgb)
                        for face in detected_faces:
                            vec = emb.extract_embedding(img_rgb, face.raw_face)
                            db.insert_face(
                                image_id=image_id,
                                bbox=face.bbox,
                                confidence=face.confidence,
                                embedding=vec,
                                cluster_id=-1,
                            )

                    pbar.update(1)
        finally:
            stop_event.set()

    # 4. Perform clustering across all faces in library
    notify_progress("Clustering face embeddings into people...", 95, total_media, total_media)
    all_faces = db.get_all_unclustered_or_all_faces()
    if all_faces:
        all_face_ids = [f["id"] for f in all_faces]
        all_embeddings = [f["embedding"] for f in all_faces]
        cluster_labels = cluster_embeddings(
            embeddings=all_embeddings,
            eps=eps,
            min_samples=min_samples,
            algorithm=clustering_algorithm,
        )
        db.update_face_clusters(all_face_ids, cluster_labels)

        # Export cutouts if requested
        if export_dir:
            import cv2
            out_root = Path(export_dir)
            import shutil
            if out_root.exists():
                shutil.rmtree(out_root)
            out_root.mkdir(parents=True, exist_ok=True)

            unique_clusters = sorted([c for c in set(cluster_labels) if c >= 0])
            cluster_name_map = {c: f"person_{i + 1}" for i, c in enumerate(unique_clusters)}
            cluster_name_map[-1] = "unclustered"

            for face_rec, cluster_id in zip(all_faces, cluster_labels):
                person_folder = cluster_name_map.get(cluster_id, f"person_{cluster_id + 1}")
                dest_dir = out_root / person_folder
                dest_dir.mkdir(parents=True, exist_ok=True)

                img_rgb = load_image_rgb(face_rec["file_path"])
                if img_rgb is not None:
                    h, w, _ = img_rgb.shape
                    x1, y1, x2, y2 = face_rec["bbox"]
                    bw, bh = x2 - x1, y2 - y1
                    mx, my = int(bw * 0.15), int(bh * 0.15)
                    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                    cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

                    crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    out_name = f"{Path(face_rec['file_path']).stem}_face{face_rec['id']}.jpg"
                    cv2.imwrite(str(dest_dir / out_name), crop_bgr)

    return db.get_summary_stats()
