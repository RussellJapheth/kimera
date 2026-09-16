# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Face detection and embedding models using InsightFace (SCRFD + ArcFace) with fallback to OpenCV Zoo (YuNet + SFace).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Model URLs
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_S_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
CLIP_VISION_MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx"
)

DEFAULT_MODEL_DIR = Path.cwd() / ".cache" / "models"


class DetectedFace(NamedTuple):
    """A single detected face: bounding box, confidence, and the raw model output."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    raw_face: Any  # raw face object or keypoints/landmarks


def _download_stream(
    url: str,
    temp_path: Path,
    total_bytes: int,
    description: str,
    progress_callback: Callable | None = None,
) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Kimera-FaceCluster/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        chunk_size = 512 * 1024  # 512KB chunks
        downloaded = 0
        with (
            tqdm(
                total=total_bytes if total_bytes > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=description,
                leave=True,
            ) as pbar,
            open(temp_path, "wb") as f,
        ):
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pbar.update(len(chunk))
                if progress_callback and total_bytes > 0:
                    pct = int((downloaded / total_bytes) * 100)
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_bytes / (1024 * 1024)
                    try:
                        progress_callback(
                            f"{description}: {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)",
                            pct,
                            downloaded,
                            total_bytes,
                        )
                    except TypeError:
                        with contextlib.suppress(Exception):
                            progress_callback(f"{description}: {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)")


def _download_parallel(
    url: str,
    temp_path: Path,
    total_bytes: int,
    description: str,
    num_threads: int = 8,
    progress_callback: Callable | None = None,
) -> bool:
    """Download large files with parallel HTTP Range connections."""
    try:
        with open(temp_path, "wb") as f:
            f.truncate(total_bytes)

        chunk_size = total_bytes // num_threads
        ranges = []
        for i in range(num_threads):
            s = i * chunk_size
            e = total_bytes - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
            ranges.append((s, e))

        lock = threading.Lock()
        downloaded = [0]
        stop_event = threading.Event()

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"{description} (fast multi-connection)",
            leave=True,
        ) as pbar:

            def worker(start: int, end: int) -> None:
                if stop_event.is_set():
                    return
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Kimera-FaceCluster/1.0)",
                        "Range": f"bytes={start}-{end}",
                    },
                )
                with urllib.request.urlopen(req, timeout=45) as resp, open(temp_path, "r+b") as f_worker:
                    curr = start
                    while curr <= end and not stop_event.is_set():
                        to_read = min(256 * 1024, end - curr + 1)
                        chunk = resp.read(to_read)
                        if not chunk:
                            break
                        f_worker.seek(curr)
                        f_worker.write(chunk)
                        curr += len(chunk)
                        with lock:
                            downloaded[0] += len(chunk)
                            pbar.update(len(chunk))
                            if progress_callback:
                                pct = int((downloaded[0] / total_bytes) * 100)
                                mb_done = downloaded[0] / (1024 * 1024)
                                mb_total = total_bytes / (1024 * 1024)
                                try:
                                    progress_callback(
                                        f"{description}: {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)",
                                        pct,
                                        downloaded[0],
                                        total_bytes,
                                    )
                                except TypeError:
                                    with contextlib.suppress(Exception):
                                        progress_callback(f"{description}: {mb_done:.1f}MB / {mb_total:.1f}MB ({pct}%)")

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, s, e) for s, e in ranges]
                concurrent.futures.wait(futures)
                for fut in futures:
                    fut.result()

        return downloaded[0] >= total_bytes
    except Exception:
        return False


def download_file(
    url: str,
    dest_path: Path,
    desc: str | None = None,
    progress_callback: Callable | None = None,
) -> None:
    """Download a model file with high-speed parallel range downloads and progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    description = desc or f"Downloading {dest_path.name}"

    try:
        # Check file size & range capability
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Kimera-FaceCluster/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size_header = response.info().get("Content-Length")
            accept_ranges = response.info().get("Accept-Ranges", "")
            total_bytes = int(total_size_header) if total_size_header and total_size_header.isdigit() else 0

        success = False
        if total_bytes > 5 * 1024 * 1024 and ("bytes" in accept_ranges.lower() or "github" in url):
            success = _download_parallel(
                url=url,
                temp_path=temp_path,
                total_bytes=total_bytes,
                description=description,
                num_threads=8,
                progress_callback=progress_callback,
            )

        if not success:
            _download_stream(
                url=url,
                temp_path=temp_path,
                total_bytes=total_bytes,
                description=description,
                progress_callback=progress_callback,
            )

        temp_path.rename(dest_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e


def ensure_buffalo_model(
    model_name: str = "buffalo_l",
    model_dir: Path | None = None,
    progress_callback: Callable | None = None,
) -> Path:
    """Ensure buffalo models are present and return root directory."""
    root_dir = model_dir or (Path.cwd() / ".cache")
    buffalo_dir = root_dir / "models" / model_name

    if model_name == "buffalo_l":
        required = ["w600k_r50.onnx", "det_10g.onnx"]
        url = BUFFALO_L_URL
    else:
        required = ["w600k_mbf.onnx", "det_500m.onnx"]
        url = BUFFALO_S_URL

    if not all((buffalo_dir / f).exists() for f in required):
        buffalo_dir.mkdir(parents=True, exist_ok=True)
        zip_path = buffalo_dir / f"{model_name}.zip"
        if not zip_path.exists():
            if progress_callback:
                with contextlib.suppress(Exception):
                    progress_callback(f"Downloading InsightFace {model_name} model pack (~300MB)...", 0, 0, 100)
            download_file(
                url,
                zip_path,
                desc=f"Downloading {model_name}.zip",
                progress_callback=progress_callback,
            )

        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback(f"Extracting {model_name}.zip models...", 98, 98, 100)
        logger.info("Extracting %s to %s...", zip_path.name, buffalo_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(buffalo_dir)
        logger.info("InsightFace %s model ready.", model_name)
    return root_dir


def ensure_buffalo_s(
    model_dir: Path | None = None,
    progress_callback: Callable | None = None,
) -> Path:
    """Backward compatibility helper."""
    return ensure_buffalo_model("buffalo_s", model_dir, progress_callback=progress_callback)


class FaceDetector:
    """
    Face detector using InsightFace SCRFD (buffalo_l / buffalo_s) or YuNet ONNX.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        conf_threshold: float = 0.60,
        nms_threshold: float = 0.3,
        engine: str = "auto",  # 'auto', 'insightface', 'yunet'
        model_pack: str = "buffalo_l",
        progress_callback: Callable | None = None,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.engine = engine
        self.model_pack = model_pack
        self._app = None
        self._detector = None

        if self.engine in ("auto", "insightface"):
            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis

                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ctx_id = 0
                else:
                    providers = ["CPUExecutionProvider"]
                    ctx_id = -1

                root_dir = ensure_buffalo_model(self.model_pack, progress_callback=progress_callback)
                self._app = FaceAnalysis(name=self.model_pack, root=str(root_dir), providers=providers)
                self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self.engine = "insightface"
            except Exception:
                if self.engine == "insightface":
                    raise
                self.engine = "yunet"

        if self.engine == "yunet":
            if model_path is None:
                model_path = DEFAULT_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
                if not Path(model_path).exists():
                    download_file(
                        YUNET_MODEL_URL,
                        Path(model_path),
                        desc="Downloading YuNet ONNX model",
                        progress_callback=progress_callback,
                    )

            self.model_path = str(model_path)
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                score_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
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
            results: list[DetectedFace] = []
            for face in faces:
                score = float(face.det_score)
                if score < self.conf_threshold:
                    continue
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                results.append(
                    DetectedFace(
                        bbox=(x1, y1, x2, y2),
                        confidence=score,
                        raw_face=face,
                    )
                )
            return results

        # Fallback YuNet
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(img_bgr)

        results: list[DetectedFace] = []
        if faces is None or len(faces) == 0:
            return results

        for face in faces:
            score = float(face[-1])
            if score < self.conf_threshold:
                continue

            x, y, bw, bh = face[0:4]
            x1 = max(0, round(x))
            y1 = max(0, round(y))
            x2 = min(w, round(x + bw))
            y2 = min(h, round(y + bh))

            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    raw_face=face,
                )
            )

        return results


class FaceEmbedder:
    """
    Face embedding generator using InsightFace ArcFace (w600k_mbf 512D) or SFace ONNX (128D).
    Generates L2-normalized feature vectors.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        engine: str = "auto",
        progress_callback: Callable | None = None,
    ):
        self.engine = engine
        self._recognizer = None
        self.progress_callback = progress_callback

        if self.engine in ("auto", "insightface"):
            self.engine = "insightface"
        else:
            self.engine = "sface"

        if self.engine == "sface":
            if model_path is None:
                model_path = DEFAULT_MODEL_DIR / "face_recognition_sface_2021dec.onnx"
                if not Path(model_path).exists():
                    download_file(
                        SFACE_MODEL_URL,
                        Path(model_path),
                        desc="Downloading SFace ONNX model",
                        progress_callback=self.progress_callback,
                    )

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
                    download_file(
                        SFACE_MODEL_URL,
                        Path(self.model_path),
                        desc="Downloading SFace ONNX model",
                        progress_callback=self.progress_callback,
                    )
            self._recognizer = cv2.FaceRecognizerSF.create(self.model_path, "")

        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        aligned_face = self._recognizer.alignCrop(img_bgr, raw_face)
        feature = self._recognizer.feature(aligned_face).flatten().astype(np.float32)

        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature = feature / norm
        return feature


def ensure_clip_model(
    model_dir: Path | None = None,
    progress_callback: Callable | None = None,
) -> Path:
    """Ensure CLIP quantized ONNX vision model is present and return file path."""
    root_dir = model_dir or (Path.cwd() / ".cache")
    clip_dir = root_dir / "models" / "clip"
    dest_path = clip_dir / "vision_model_quantized.onnx"
    if not dest_path.exists():
        clip_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback:
            with contextlib.suppress(Exception):
                progress_callback("Downloading CLIP Vision ONNX model (~89MB)...", 0, 0, 100)
        download_file(
            CLIP_VISION_MODEL_URL,
            dest_path,
            desc="Downloading CLIP Vision Model",
            progress_callback=progress_callback,
        )
    return dest_path


class MediaEmbedder:
    """
    Fast CLIP ViT-B/32 ONNX visual embedder for images and videos.
    Runs directly in ONNX Runtime (CPU/CUDA) in ~15-30ms.
    Produces normalized 512-dimensional semantic embeddings.
    """

    def __init__(self, model_path: Path | None = None):
        if model_path is None:
            root_dir = Path.cwd() / ".cache"
            model_path = root_dir / "models" / "clip" / "vision_model_quantized.onnx"

        if not model_path.exists():
            model_path = ensure_clip_model()

        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        """Resize to 224x224, scale [0, 1], apply CLIP normalization."""
        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self.mean) / self.std
        chw = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Compute 512-dim L2-normalized visual embedding for an image."""
        tensor = self.preprocess(image_rgb)
        out = self.session.run([self.output_name], {self.input_name: tensor})[0]
        vec = out[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.astype(np.float32)

    def embed_video_frames(self, frames_rgb: list[np.ndarray]) -> np.ndarray | None:
        """Embed multiple video keyframes, average them, and L2-normalize."""
        if not frames_rgb:
            return None
        embs = [self.embed_image(f) for f in frames_rgb if f is not None and f.size > 0]
        if not embs:
            return None
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-9:
            avg = avg / norm
        return avg.astype(np.float32)


_media_embedder_lock = threading.Lock()
_media_embedder: MediaEmbedder | None = None
_clip_downloading: bool = False


def get_media_embedder(auto_download: bool = False) -> MediaEmbedder | None:
    """Thread-safe accessor for the MediaEmbedder instance."""
    global _media_embedder
    with _media_embedder_lock:
        if _media_embedder is not None:
            return _media_embedder
        clip_path = DEFAULT_MODEL_DIR / "clip" / "vision_model_quantized.onnx"
        if not clip_path.exists():
            if not auto_download:
                return None
            ensure_clip_model()
        try:
            _media_embedder = MediaEmbedder(clip_path)
        except Exception as e:
            logger.warning("Failed to load MediaEmbedder: %s", e)
            return None
        else:
            return _media_embedder


def trigger_clip_download_async(progress_callback: Callable | None = None) -> None:
    """Start background download of CLIP vision model if not already present."""
    clip_path = DEFAULT_MODEL_DIR / "clip" / "vision_model_quantized.onnx"
    if clip_path.exists() or _clip_downloading:
        return

    def _bg() -> None:
        global _clip_downloading
        _clip_downloading = True
        try:
            ensure_clip_model(progress_callback=progress_callback)
            # Pre-warm embedder once downloaded
            get_media_embedder(auto_download=False)
        except Exception as e:
            logger.warning("Background CLIP model download notice: %s", e)
        finally:
            _clip_downloading = False

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
