"""
Face detection and embedding models using InsightFace (SCRFD + ArcFace) with fallback to OpenCV Zoo (YuNet + SFace).
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np

# Lightweight ONNX models
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "kimera" / "models"


class DetectedFace(NamedTuple):
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    raw_face: Any  # raw face object or keypoints/landmarks


def download_file(url: str, dest_path: Path) -> None:
    """Download a model file with temporary file safety."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    print(f"Downloading model from {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, temp_path)
        temp_path.rename(dest_path)
        print(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e


def ensure_buffalo_s(model_dir: Optional[Path] = None) -> Path:
    """Ensure buffalo_s models are present and return root directory."""
    root_dir = model_dir or (Path.home() / ".insightface")
    buffalo_dir = root_dir / "models" / "buffalo_s"
    if not (buffalo_dir / "w600k_mbf.onnx").exists() or not (buffalo_dir / "det_500m.onnx").exists():
        buffalo_dir.mkdir(parents=True, exist_ok=True)
        zip_path = buffalo_dir / "buffalo_s.zip"
        download_file(BUFFALO_S_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(buffalo_dir)
    return root_dir


class FaceDetector:
    """
    Face detector using InsightFace SCRFD or YuNet ONNX.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        engine: str = "auto",  # 'auto', 'insightface', 'yunet'
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.engine = engine
        self._app = None
        self._detector = None

        if self.engine in ("auto", "insightface"):
            try:
                from insightface.app import FaceAnalysis
                root_dir = ensure_buffalo_s()
                self._app = FaceAnalysis(name="buffalo_s", root=str(root_dir), providers=["CPUExecutionProvider"])
                self._app.prepare(ctx_id=0, det_size=(640, 640))
                self.engine = "insightface"
            except Exception as e:
                if self.engine == "insightface":
                    raise e
                self.engine = "yunet"

        if self.engine == "yunet":
            if model_path is None:
                model_path = DEFAULT_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
                if not Path(model_path).exists():
                    download_file(YUNET_MODEL_URL, Path(model_path))

            self.model_path = str(model_path)
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )

    def detect(self, image_rgb: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in an RGB image.
        Returns a list of DetectedFace objects.
        """
        h, w, _ = image_rgb.shape
        if h == 0 or w == 0:
            return []

        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if self.engine == "insightface" and self._app is not None:
            faces = self._app.get(img_bgr)
            results: List[DetectedFace] = []
            for face in faces:
                score = float(face.det_score)
                if score < self.conf_threshold:
                    continue
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                results.append(DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    raw_face=face,
                ))
            return results

        # Fallback YuNet
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img_bgr)

        results: List[DetectedFace] = []
        if faces is None or len(faces) == 0:
            return results

        for face in faces:
            score = float(face[-1])
            if score < self.conf_threshold:
                continue

            x, y, bw, bh = face[0:4]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(w, int(round(x + bw)))
            y2 = min(h, int(round(y + bh)))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(DetectedFace(
                bbox=(x1, y1, x2, y2),
                confidence=score,
                raw_face=face,
            ))

        return results


class FaceEmbedder:
    """
    Face embedding generator using InsightFace ArcFace (w600k_mbf 512D) or SFace ONNX (128D).
    Generates L2-normalized feature vectors.
    """

    def __init__(self, model_path: Optional[Path | str] = None, engine: str = "auto"):
        self.engine = engine
        self._recognizer = None

        if self.engine in ("auto", "insightface"):
            self.engine = "insightface"
        else:
            self.engine = "sface"

        if self.engine == "sface":
            if model_path is None:
                model_path = DEFAULT_MODEL_DIR / "face_recognition_sface_2021dec.onnx"
                if not Path(model_path).exists():
                    download_file(SFACE_MODEL_URL, Path(model_path))

            self.model_path = str(model_path)
            self._recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

    def extract_embedding(self, image_rgb: np.ndarray, raw_face: Any) -> np.ndarray:
        """
        Align and extract a normalized embedding vector for a detected face.
        """
        # If raw_face is InsightFace Face object and has embedding
        if hasattr(raw_face, "embedding") and raw_face.embedding is not None:
            feature = raw_face.embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            return feature

        if self._recognizer is None:
            if not hasattr(self, "model_path") or self.model_path is None:
                self.model_path = str(DEFAULT_MODEL_DIR / "face_recognition_sface_2021dec.onnx")
                if not Path(self.model_path).exists():
                    download_file(SFACE_MODEL_URL, Path(self.model_path))
            self._recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        aligned_face = self._recognizer.alignCrop(img_bgr, raw_face)
        feature = self._recognizer.feature(aligned_face).flatten().astype(np.float32)

        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm
        return feature
