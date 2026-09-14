"""
High-performance thumbnail and face crop cache engine.
Generates WebP thumbnails for images and videos, plus square face avatar crops with EXIF orientation handling.
Uses 2-tier directory sharding to support massive libraries without filesystem performance degradation.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_CACHE_DIR = Path.cwd() / ".cache"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}


class ThumbnailCache:
    """Manages disk-cached image/video thumbnails, face avatar crops, and web-transcoded videos with 2-tier directory sharding."""

    def __init__(self, cache_dir: Optional[Path | str] = None):
        self.set_cache_dir(cache_dir or DEFAULT_CACHE_DIR)

    def set_cache_dir(self, cache_dir: Path | str) -> None:
        """Update active cache directory on disk and ensure subfolders exist."""
        self.cache_dir = Path(cache_dir).resolve()
        self.thumbs_dir = self.cache_dir / "thumbnails"
        self.faces_dir = self.cache_dir / "faces"
        self.transcoded_dir = self.cache_dir / "transcoded"
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.transcoded_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _shard_dir(base_dir: Path, key: str) -> Path:
        """Return 2-tier sharded directory path (e.g., base/ab/cd/)."""
        norm_key = key.lower() if len(key) >= 4 else key.zfill(4).lower()
        return base_dir / norm_key[:2] / norm_key[2:4]

    def get_cache_stats(self) -> dict:
        """Calculate number of cached items and disk footprint in bytes across sharded directories."""
        thumb_files = [f for f in self.thumbs_dir.rglob("*.webp") if f.is_file()] if self.thumbs_dir.exists() else []
        face_files = [f for f in self.faces_dir.rglob("*.webp") if f.is_file()] if self.faces_dir.exists() else []
        transcoded_files = [f for f in self.transcoded_dir.rglob("*.mp4") if f.is_file()] if self.transcoded_dir.exists() else []
        total_size = sum(f.stat().st_size for f in thumb_files + face_files + transcoded_files)
        return {
            "cache_dir": str(self.cache_dir),
            "thumbnail_count": len(thumb_files),
            "face_count": len(face_files),
            "transcoded_count": len(transcoded_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    def clear_cache(self) -> None:
        """Remove all cached thumbnails, face crops, and transcoded videos."""
        for d in (self.thumbs_dir, self.faces_dir, self.transcoded_dir):
            if d.exists():
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
                d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _path_hash(path_str: str) -> str:
        return hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def is_video(path: Path | str) -> bool:
        return Path(path).suffix.lower() in VIDEO_EXTENSIONS

    def get_transcoded_video(self, media_path: str | Path) -> Optional[Path]:
        """
        Generate or retrieve a browser-compatible H.264+AAC MP4 web stream version of a video.
        Preserves original resolution (scaled only if needed to even dimensions for H.264).
        """
        src = Path(media_path)
        if not src.is_file() or not self.is_video(src):
            return None

        file_hash = self._path_hash(str(src.resolve()))
        shard_folder = self._shard_dir(self.transcoded_dir, file_hash)
        transcoded_path = shard_folder / f"{file_hash}.mp4"
        legacy_path = self.transcoded_dir / f"{file_hash}.mp4"

        # Check sharded location then legacy location
        if transcoded_path.exists() and transcoded_path.stat().st_size > 0:
            return transcoded_path
        if legacy_path.exists() and legacy_path.stat().st_size > 0:
            return legacy_path

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return None

        shard_folder.mkdir(parents=True, exist_ok=True)
        part_path = shard_folder / f"{file_hash}.part.mp4"
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(src.resolve()),
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(part_path)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            if proc.returncode == 0 and part_path.exists() and part_path.stat().st_size > 0:
                part_path.replace(transcoded_path)
                return transcoded_path
            else:
                part_path.unlink(missing_ok=True)
                return None
        except Exception:
            part_path.unlink(missing_ok=True)
            return None

    def _extract_video_frame(self, src: Path) -> Optional[np.ndarray]:
        """Extract a representative RGB frame via OpenCV, falling back to ffmpeg for codecs OpenCV cannot decode (e.g. AV1)."""
        try:
            cap = cv2.VideoCapture(str(src))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 0.5))
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_bgr = cap.read()
                cap.release()
                if ret and frame_bgr is not None:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return None
        try:
            proc = subprocess.run(
                [
                    ffmpeg_bin, "-y", "-ss", "0.5", "-i", str(src.resolve()),
                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout:
                return None
            with Image.open(io.BytesIO(proc.stdout)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                return np.array(img)
        except Exception:
            return None

    def _thumbnail_locations(self, src: Path, max_dim: int) -> Tuple[Path, Path, Path]:
        """Return (shard_folder, sharded_thumb_path, legacy_thumb_path) for a media file."""
        file_hash = self._path_hash(str(src.resolve()))
        shard_folder = self._shard_dir(self.thumbs_dir, file_hash)
        return shard_folder, shard_folder / f"{file_hash}_{max_dim}.webp", self.thumbs_dir / f"{file_hash}_{max_dim}.webp"

    def has_thumbnail(self, media_path: str | Path, max_dim: int = 480) -> bool:
        """Return True if a cached WebP thumbnail exists for the media file."""
        src = Path(media_path)
        if not src.is_file():
            return False
        _, thumb_path, legacy_path = self._thumbnail_locations(src, max_dim)
        return thumb_path.is_file() or legacy_path.is_file()

    def count_missing_thumbnails(self, media_paths: List[Path | str], max_dim: int = 480) -> int:
        """Count media files that exist on disk but lack a cached thumbnail."""
        return sum(1 for p in media_paths if Path(p).is_file() and not self.has_thumbnail(p, max_dim))

    def get_thumbnail(self, media_path: str | Path, max_dim: int = 480) -> Optional[Path]:
        """
        Generate or retrieve a cached WebP thumbnail for an image or video file.
        Returns Path to the cached .webp file.
        """
        src = Path(media_path)
        if not src.is_file():
            return None

        shard_folder, thumb_path, legacy_path = self._thumbnail_locations(src, max_dim)

        if thumb_path.is_file():
            return thumb_path
        if legacy_path.is_file():
            return legacy_path

        # If video file: extract representative frame with OpenCV, fall back to ffmpeg (e.g. AV1)
        if self.is_video(src):
            try:
                frame_rgb = self._extract_video_frame(src)
                if frame_rgb is None:
                    return None

                img = Image.fromarray(frame_rgb)
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                shard_folder.mkdir(parents=True, exist_ok=True)
                img.save(thumb_path, format="WEBP", quality=82, method=4)
                return thumb_path
            except Exception:
                return None

        # Standard image
        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                shard_folder.mkdir(parents=True, exist_ok=True)
                img.save(thumb_path, format="WEBP", quality=82, method=4)
                return thumb_path
        except Exception:
            return None

    def invalidate_thumbnail(self, media_path: str | Path) -> None:
        """Remove cached thumbnails for a modified or removed media file."""
        src = Path(media_path)
        file_hash = self._path_hash(str(src.resolve()))
        shard_folder = self._shard_dir(self.thumbs_dir, file_hash)
        
        # Check sharded folder
        if shard_folder.exists():
            for thumb in shard_folder.glob(f"{file_hash}_*.webp"):
                try:
                    thumb.unlink(missing_ok=True)
                except Exception:
                    pass
        # Check legacy flat folder
        for thumb in self.thumbs_dir.glob(f"{file_hash}_*.webp"):
            try:
                thumb.unlink(missing_ok=True)
            except Exception:
                pass

    def invalidate_media_cache(self, media_path: str | Path) -> None:
        """Invalidate all thumbnails and transcoded video streams for a media file."""
        self.invalidate_thumbnail(media_path)
        src = Path(media_path)
        file_hash = self._path_hash(str(src.resolve()))
        # Check sharded transcoded dir
        shard_folder = self._shard_dir(self.transcoded_dir, file_hash)
        if shard_folder.exists():
            for vid in shard_folder.glob(f"{file_hash}*.mp4"):
                try:
                    vid.unlink(missing_ok=True)
                except Exception:
                    pass
        # Check legacy flat transcoded dir
        for vid in self.transcoded_dir.glob(f"{file_hash}*.mp4"):
            try:
                vid.unlink(missing_ok=True)
            except Exception:
                pass

    def _face_shard_dir(self, face_id: int) -> Path:
        """Return 2-tier sharded directory for face crops."""
        face_key = f"{face_id:08d}"
        return self._shard_dir(self.faces_dir, face_key)

    def save_face_crop_from_array(
        self,
        face_id: int,
        img_rgb: np.ndarray,
        bbox: Tuple[int, int, int, int],
        size: int = 200,
        margin: float = 0.35,
    ) -> Optional[Path]:
        """Directly crop and cache face from an in-memory RGB array (e.g. video frame)."""
        shard_folder = self._face_shard_dir(face_id)
        crop_path = shard_folder / f"face_{face_id}_{size}.webp"
        legacy_path = self.faces_dir / f"face_{face_id}_{size}.webp"

        if crop_path.exists():
            return crop_path
        if legacy_path.exists():
            return legacy_path

        try:
            h, w, _ = img_rgb.shape
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1

            pad_w = int(bw * margin)
            pad_h = int(bh * margin)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            side = max(bw + 2 * pad_w, bh + 2 * pad_h) // 2

            crop_x1 = max(0, cx - side)
            crop_y1 = max(0, cy - side)
            crop_x2 = min(w, cx + side)
            crop_y2 = min(h, cy + side)

            cropped = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            img = Image.fromarray(cropped)
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            shard_folder.mkdir(parents=True, exist_ok=True)
            img.save(crop_path, format="WEBP", quality=85, method=4)
            return crop_path
        except Exception:
            return None

    def get_face_crop(
        self,
        face_id: int,
        image_path: str | Path,
        bbox: Tuple[int, int, int, int],
        size: int = 200,
        margin: float = 0.35,
    ) -> Optional[Path]:
        """
        Generate or retrieve a square face avatar crop with padding margin.
        """
        shard_folder = self._face_shard_dir(face_id)
        crop_path = shard_folder / f"face_{face_id}_{size}.webp"
        legacy_path = self.faces_dir / f"face_{face_id}_{size}.webp"

        if crop_path.exists():
            return crop_path
        if legacy_path.exists():
            return legacy_path

        src = Path(image_path)
        if not src.is_file():
            return None

        if self.is_video(src):
            try:
                frame_rgb = self._extract_video_frame(src)
                if frame_rgb is None:
                    return None
                return self.save_face_crop_from_array(face_id, frame_rgb, bbox, size=size, margin=margin)
            except Exception:
                return None

        try:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                w, h = img.size
                x1, y1, x2, y2 = bbox
                bw = x2 - x1
                bh = y2 - y1

                pad_w = int(bw * margin)
                pad_h = int(bh * margin)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                side = max(bw + 2 * pad_w, bh + 2 * pad_h) // 2

                crop_x1 = max(0, cx - side)
                crop_y1 = max(0, cy - side)
                crop_x2 = min(w, cx + side)
                crop_y2 = min(h, cy + side)

                cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                cropped = cropped.resize((size, size), Image.Resampling.LANCZOS)
                shard_folder.mkdir(parents=True, exist_ok=True)
                cropped.save(crop_path, format="WEBP", quality=85, method=4)
                return crop_path
        except Exception:
            return None

