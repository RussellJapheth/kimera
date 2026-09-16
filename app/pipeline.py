# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Pipeline orchestrator connecting scanning, detection, embedding, clustering, and SQLite storage.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from app.clustering import cluster_embeddings
from app.db import Database
from app.models import FaceDetector, FaceEmbedder, get_media_embedder
from app.recognition import MultiExemplarMatcher, group_intra_video
from app.scanner import (
    compute_quick_hash,
    extract_video_keyframes,
    get_video_duration,
    is_video_file,
    load_image_rgb,
    scan_media_paths,
)

logger = logging.getLogger(__name__)


class FileScanTask(NamedTuple):
    """Work item describing one file and whether it needs (re)scanning."""

    path: Path
    path_str: str
    needs_scan: bool
    image_id: int | None
    file_size: int
    mtime: float
    content_hash: str
    is_video: bool


def _prefilter_file(
    path: Path,
    meta: dict[str, Any] | None,
) -> tuple[FileScanTask, tuple[int, int, float, str] | None]:
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
    except OSError:
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


def run_intra_video_merge(db: Database, threshold: float = 0.30) -> None:
    """Merge near-identical faces found within the same video so angle/lighting
    variation does not split one person into multiple identities.

    Resolution rules per linked component:
    - Single named person: promote the whole component to that person.
    - Multiple named people: ambiguous, skip.
    - Existing clusters only: merge into the most common cluster.
    - Noise only: group into a fresh cluster.
    """
    video_faces = db.get_video_faces_for_merge()
    if not video_faces:
        return

    components = group_intra_video(video_faces, threshold=threshold)
    if not components:
        return

    all_cids: set[int] = set()
    for comp in components:
        all_cids.update(comp["clusters"].keys())
    member_map = db.get_face_ids_by_clusters(sorted(all_cids)) if all_cids else {}

    next_cluster_id = db.get_max_cluster_id()
    for comp in components:
        if len(comp["persons"]) == 1:
            person_id = max(comp["persons"].items(), key=lambda kv: (kv[1], -kv[0]))[0]
            merge_ids = list(comp["noise_ids"])
            for cid in comp["clusters"]:
                merge_ids.extend(member_map.get(cid, []))
            db.assign_faces_to_person(merge_ids, person_id)
        elif len(comp["persons"]) > 1:
            continue
        elif comp["clusters"]:
            target_cid = max(comp["clusters"].items(), key=lambda kv: (kv[1], kv[0]))[0]
            merge_ids = list(comp["noise_ids"])
            for cid in comp["clusters"]:
                if cid != target_cid:
                    merge_ids.extend(member_map.get(cid, []))
            db.assign_faces_to_cluster(merge_ids, target_cid)
        else:
            next_cluster_id += 1
            db.assign_faces_to_cluster(comp["noise_ids"], next_cluster_id)


def run_pipeline(
    input_dir: Path | str,
    db_path: Path | str = "face_clusters.db",
    cache_dir: Path | str | None = None,
    conf_threshold: float = 0.60,
    eps: float = 0.43,
    min_samples: int = 1,
    clustering_algorithm: str = "agglomerative",
    export_dir: Path | str | None = None,
    scene_threshold: float = 0.35,
    min_interval_sec: float = 60.0,
    max_interval_sec: float = 90.0,
    match_threshold: float | None = None,
    include_cluster_references: bool | None = None,
    intra_video_merge_threshold: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Run high-performance incremental face scanning, detection, embedding, and clustering.
    Uses multi-threaded parallel pre-filtering and background image decode prefetching.
    Includes multi-exemplar supervised matching for named people followed by unsupervised clustering.
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
    detector: FaceDetector | None = None
    embedder: FaceEmbedder | None = None

    def notify_progress(msg: str, percent: int, current: int, total: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(msg, percent, current, total)
        except TypeError:
            with contextlib.suppress(Exception):
                progress_callback(msg)

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

    tasks: list[FileScanTask] = []
    touch_updates: list[tuple[int, int, float, str]] = []

    # Check file stats & hashes concurrently across CPU threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(4, total_media))) as executor:
        future_to_path = [executor.submit(_prefilter_file, p, meta_map.get(str(p.resolve()))) for p in media_paths]
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
        notify_progress(
            f"Found {total_to_scan} new/modified media items. Initializing AI models...", 4, 0, total_to_scan
        )
        get_models()
    else:
        notify_progress(f"All {total_media} files up-to-date. Checking clusters...", 90, total_media, total_media)

    # 3. Process new/modified files with worker prefetch for images
    if total_to_scan > 0:
        notify_progress(f"Scanning {total_to_scan} media items...", 5, 0, total_to_scan)
        det, emb = get_models()

        # Prefetch pipeline for images
        def image_loader(task_list: list[FileScanTask], out_q: queue.Queue, stop_ev: threading.Event) -> None:
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
                        vid_dur = get_video_duration(task.path)
                        vid_frames: list[np.ndarray] = []
                        for kf in extract_video_keyframes(
                            task.path,
                            scene_threshold=scene_threshold,
                            min_interval_sec=min_interval_sec,
                            max_interval_sec=max_interval_sec,
                        ):
                            if len(vid_frames) < 3:
                                vid_frames.append(kf.frame_rgb)

                            if video_image_id is None:
                                vid_h, vid_w, _ = kf.frame_rgb.shape
                                video_image_id = db.insert_image(
                                    task.path_str,
                                    width=vid_w,
                                    height=vid_h,
                                    file_size=task.file_size,
                                    mtime=task.mtime,
                                    content_hash=task.content_hash,
                                    duration=vid_dur,
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
                            video_image_id = db.insert_image(
                                task.path_str,
                                width=0,
                                height=0,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                                duration=vid_dur,
                            )
                        else:
                            db.update_image_meta(
                                video_image_id,
                                file_size=task.file_size,
                                mtime=task.mtime,
                                content_hash=task.content_hash,
                                width=vid_w,
                                height=vid_h,
                                duration=vid_dur,
                            )

                        # Visual embedding is best-effort; a failure must not abort the scan.
                        try:
                            v_embedder = get_media_embedder(auto_download=False)
                            if v_embedder is not None and video_image_id is not None and vid_frames:
                                v_emb = v_embedder.embed_video_frames(vid_frames)
                                if v_emb is not None:
                                    db.save_media_embedding(video_image_id, v_emb)
                        except Exception:
                            logger.debug("Visual embedding failed for video %s", task.path, exc_info=True)
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

                        # Visual embedding is best-effort; a failure must not abort the scan.
                        try:
                            v_embedder = get_media_embedder(auto_download=False)
                            if v_embedder is not None and image_id is not None:
                                v_emb = v_embedder.embed_image(img_rgb)
                                db.save_media_embedding(image_id, v_emb)
                        except Exception:
                            logger.debug("Visual embedding failed for image %s", task.path, exc_info=True)

                    pbar.update(1)
        finally:
            stop_event.set()

    # 4. Perform Two-Stage Identification & Clustering
    # Stage 4A: Supervised matching against all verified person exemplars (already-named & newly assigned)
    notify_progress("Matching unassigned faces against named people...", 93, total_media, total_media)
    effective_match_threshold = (
        float(match_threshold)
        if match_threshold is not None
        else float(db.get_setting("recognition_match_threshold", eps))
    )
    effective_include_cluster_refs = (
        include_cluster_references
        if include_cluster_references is not None
        else db.get_setting("include_cluster_references", "1") == "1"
    )
    effective_intra_video_threshold = (
        float(intra_video_merge_threshold)
        if intra_video_merge_threshold is not None
        else float(db.get_setting("intra_video_merge_threshold", 0.30))
    )

    person_exemplars = db.get_all_person_exemplars(max_exemplars=5)
    unassigned_faces = db.get_unassigned_faces()

    if person_exemplars and unassigned_faces:
        matcher = MultiExemplarMatcher(person_exemplars)
        exclusions = db.get_person_exclusions()
        matched_results = matcher.match_faces_batch(
            unassigned_faces,
            exclusions=exclusions,
            threshold=effective_match_threshold,
        )
        if matched_results:
            # Group by person_id for batch DB update
            person_to_faces: dict[int, list[int]] = {}
            for f_id, (p_id, _dist) in matched_results.items():
                if p_id not in person_to_faces:
                    person_to_faces[p_id] = []
                person_to_faces[p_id].append(f_id)

            for p_id, f_ids in person_to_faces.items():
                db.assign_faces_to_person(f_ids, p_id)

    # Stage 4A2: Match orphan faces (unclustered) against existing unnamed clusters,
    # so known-but-unnamed faces act as references too. Matches merge into the cluster.
    if effective_include_cluster_refs:
        notify_progress("Matching orphan faces against existing clusters...", 94, total_media, total_media)
        cluster_exemplars = db.get_all_cluster_exemplars(max_exemplars=5)
        orphan_faces = db.get_orphan_faces()
        if cluster_exemplars and orphan_faces:
            cluster_matcher = MultiExemplarMatcher(cluster_exemplars)
            cluster_matches = cluster_matcher.match_faces_batch(
                orphan_faces,
                threshold=effective_match_threshold,
            )
            cluster_to_faces: dict[int, list[int]] = {}
            for f_id, (c_id, _dist) in cluster_matches.items():
                if c_id not in cluster_to_faces:
                    cluster_to_faces[c_id] = []
                cluster_to_faces[c_id].append(f_id)

            for c_id, f_ids in cluster_to_faces.items():
                db.assign_faces_to_cluster(f_ids, c_id)

    # Stage 4B: Unsupervised clustering on remaining unassigned faces
    notify_progress("Clustering remaining unassigned faces...", 96, total_media, total_media)
    remaining_unassigned = db.get_unassigned_faces()
    if remaining_unassigned:
        all_unassigned_ids = [f["id"] for f in remaining_unassigned]
        all_unassigned_embeddings = [f["embedding"] for f in remaining_unassigned]
        cluster_labels = cluster_embeddings(
            embeddings=all_unassigned_embeddings,
            eps=eps,
            min_samples=min_samples,
            algorithm=clustering_algorithm,
        )
        db.update_face_clusters(all_unassigned_ids, cluster_labels)

    # Stage 4C: Intra-video merge. Group near-identical faces from the same video
    # so angle/lighting variation does not split one person into multiple identities.
    notify_progress("Merging similar faces within videos...", 97, total_media, total_media)
    run_intra_video_merge(db, threshold=effective_intra_video_threshold)

    # Export cutouts if requested
    if export_dir:
        import cv2

        out_root = Path(export_dir)
        import shutil

        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        all_faces = db.get_all_unclustered_or_all_faces()
        for face_rec in all_faces:
            p_id = face_rec["person_id"]
            c_id = face_rec["cluster_id"]
            if p_id is not None:
                p_info = db.get_person(p_id)
                folder_name = f"person_{p_info['name']}" if p_info else f"person_id_{p_id}"
            elif c_id >= 0:
                folder_name = f"cluster_{c_id}"
            else:
                folder_name = "unclustered"

            dest_dir = out_root / folder_name
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


def index_missing_media_embeddings(
    db: Database,
    progress_callback: Callable | None = None,
) -> int:
    """
    Compute and save visual embeddings for all media files in the library
    that currently lack an entry in media_embeddings.
    Returns the number of embeddings successfully generated.
    """
    missing_ids = db.get_unembedded_image_ids()
    if not missing_ids:
        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback("All media embeddings up to date.", 100, 0, 0)
        return 0

    embedder = get_media_embedder(auto_download=True)
    if embedder is None:
        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback("CLIP vision model unavailable.", 0, 0, len(missing_ids))
        return 0

    total = len(missing_ids)
    indexed = 0
    for idx, img_id in enumerate(missing_ids):
        pct = int(((idx + 1) / total) * 100)
        img_info = db.get_image(img_id)
        if not img_info:
            continue

        file_path = Path(img_info["file_path"])
        if not file_path.exists():
            continue

        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback(
                    f"Indexing visual embedding ({idx + 1}/{total}): {file_path.name}",
                    pct,
                    idx + 1,
                    total,
                )

        try:
            if img_info.get("is_video"):
                frames = []
                for kf in extract_video_keyframes(file_path, min_interval_sec=5.0, max_interval_sec=30.0):
                    frames.append(kf.frame_rgb)
                    if len(frames) >= 3:
                        break
                if frames:
                    emb = embedder.embed_video_frames(frames)
                    if emb is not None:
                        db.save_media_embedding(img_id, emb)
                        indexed += 1
            else:
                img_rgb = load_image_rgb(file_path)
                if img_rgb is not None:
                    emb = embedder.embed_image(img_rgb)
                    db.save_media_embedding(img_id, emb)
                    indexed += 1
        # One bad file must not abort the batch.
        except Exception:
            logger.warning("Error embedding media %s", img_id, exc_info=True)

    if progress_callback:
        with contextlib.suppress(Exception):
            progress_callback(f"Successfully generated {indexed} visual embeddings.", 100, total, total)

    return indexed
