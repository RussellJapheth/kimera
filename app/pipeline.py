"""
Pipeline orchestrator connecting scanning, detection, embedding, clustering, and SQLite storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from tqdm import tqdm

from app.clustering import cluster_embeddings
from app.db import Database
from app.models import FaceDetector, FaceEmbedder
from app.scanner import (
    extract_video_keyframes,
    is_video_file,
    load_image_rgb,
    scan_media_paths,
)


def run_pipeline(
    input_dir: Path | str,
    db_path: Path | str = "face_clusters.db",
    conf_threshold: float = 0.5,
    eps: float = 0.65,
    min_samples: int = 1,
    clustering_algorithm: str = "dbscan",
    export_dir: Optional[Path | str] = None,
    scene_threshold: float = 0.35,
    min_interval_sec: float = 0.5,
    max_interval_sec: float = 3.0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete face scanning, detection, embedding, and clustering pipeline for images and videos.

    Returns:
        Summary dictionary containing images scanned, faces detected, clusters, and breakdown.
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

    # Initialize models
    detector = FaceDetector(conf_threshold=conf_threshold)
    embedder = FaceEmbedder()

    all_face_ids: List[int] = []
    all_embeddings: List[np.ndarray] = []
    all_face_records: List[Dict[str, Any]] = []

    # Process media files with progress bar
    for path in tqdm(media_paths, desc="Scanning & detecting faces"):
        if is_video_file(path):
            # Process video keyframes
            for kf in extract_video_keyframes(
                path,
                scene_threshold=scene_threshold,
                min_interval_sec=min_interval_sec,
                max_interval_sec=max_interval_sec,
            ):
                source_id = f"{path}#t={kf.timestamp_sec:.2f}s"
                image_id = db.insert_image(source_id)
                detected_faces = detector.detect(kf.frame_rgb)

                for idx, face in enumerate(detected_faces):
                    emb = embedder.extract_embedding(kf.frame_rgb, face.raw_face)
                    face_id = db.insert_face(
                        image_id=image_id,
                        bbox=face.bbox,
                        confidence=face.confidence,
                        embedding=emb,
                        cluster_id=-1,
                    )
                    all_face_ids.append(face_id)
                    all_embeddings.append(emb)
                    all_face_records.append({
                        "face_id": face_id,
                        "path": path,
                        "face_idx": idx,
                        "label_suffix": f"t{kf.timestamp_sec:.2f}s",
                        "bbox": face.bbox,
                        "img_rgb": kf.frame_rgb,
                    })
        else:
            # Process standard image
            img_rgb = load_image_rgb(path)
            if img_rgb is None:
                continue

            image_id = db.insert_image(str(path))
            detected_faces = detector.detect(img_rgb)

            for idx, face in enumerate(detected_faces):
                emb = embedder.extract_embedding(img_rgb, face.raw_face)
                face_id = db.insert_face(
                    image_id=image_id,
                    bbox=face.bbox,
                    confidence=face.confidence,
                    embedding=emb,
                    cluster_id=-1,
                )
                all_face_ids.append(face_id)
                all_embeddings.append(emb)
                all_face_records.append({
                    "face_id": face_id,
                    "path": path,
                    "face_idx": idx,
                    "label_suffix": "",
                    "bbox": face.bbox,
                    "img_rgb": img_rgb,
                })

    # Perform clustering if any faces detected
    if all_face_ids and all_embeddings:
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

            for record, cluster_id in zip(all_face_records, cluster_labels):
                person_folder = cluster_name_map.get(cluster_id, f"person_{cluster_id + 1}")
                dest_dir = out_root / person_folder
                dest_dir.mkdir(parents=True, exist_ok=True)

                img_rgb = record["img_rgb"]
                h, w, _ = img_rgb.shape
                x1, y1, x2, y2 = record["bbox"]
                bw, bh = x2 - x1, y2 - y1
                mx, my = int(bw * 0.15), int(bh * 0.15)
                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)

                crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

                suffix = f"_{record['label_suffix']}" if record.get("label_suffix") else ""
                out_name = f"{record['path'].stem}{suffix}_face{record['face_idx']}.jpg"
                cv2.imwrite(str(dest_dir / out_name), crop_bgr)

    return db.get_summary_stats()
