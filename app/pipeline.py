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
from app.scanner import load_image_rgb, scan_image_paths


def run_pipeline(
    input_dir: Path | str,
    db_path: Path | str = "face_clusters.db",
    conf_threshold: float = 0.5,
    eps: float = 0.65,
    min_samples: int = 1,
    clustering_algorithm: str = "dbscan",
    export_dir: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Run the complete face scanning, detection, embedding, and clustering pipeline.

    Returns:
        Summary dictionary containing images scanned, faces detected, clusters, and breakdown.
    """
    db = Database(db_path)
    image_paths = scan_image_paths(input_dir)

    if not image_paths:
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

    # Process images with progress bar
    for path in tqdm(image_paths, desc="Scanning & detecting faces"):
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

                out_name = f"{record['path'].stem}_face{record['face_idx']}.jpg"
                cv2.imwrite(str(dest_dir / out_name), crop_bgr)

    return db.get_summary_stats()
