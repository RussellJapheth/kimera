# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Media scanning, loading, and video keyframe extraction utilities.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Generator
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS


def compute_quick_hash(path: Path | str, chunk_size: int = 65536) -> str:
    """
    Compute fast content fingerprint by hashing first 64KB + last 64KB + total file size.
    Prevents reading entire gigabyte files when validating modifications.
    """
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        size = p.stat().st_size
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            # Read first chunk
            first_chunk = f.read(chunk_size)
            hasher.update(first_chunk)
            # If larger than chunk size, seek and read last chunk
            if size > chunk_size:
                f.seek(max(0, size - chunk_size))
                last_chunk = f.read(chunk_size)
                hasher.update(last_chunk)
        hasher.update(str(size).encode("utf-8"))
        return hasher.hexdigest()[:24]
    except Exception:
        return ""


class VideoKeyframe(NamedTuple):
    """A sampled video frame with its position and scene-cut marker."""

    frame_idx: int
    timestamp_sec: float
    frame_rgb: np.ndarray
    is_scene_cut: bool


def is_image_file(path: Path | str) -> bool:
    """Check if a file has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_video_file(path: Path | str) -> bool:
    """Check if a file has a supported video extension."""
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def get_video_duration(video_path: Path | str) -> float:
    """
    Quickly retrieve video duration in seconds via OpenCV VideoCapture metadata.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
    except Exception:
        return 0.0
    else:
        if fps > 0 and frame_count > 0:
            return float(frame_count / fps)
        return 0.0


def is_media_file(path: Path | str) -> bool:
    """Check if a file has a supported image or video extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def scan_media_paths(directory_or_file: Path | str) -> list[Path]:
    """
    Recursively scan a directory or return a single media file.
    Supports images and videos.
    """
    path = Path(directory_or_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {directory_or_file}")

    if path.is_file():
        if is_media_file(path):
            return [path]
        raise ValueError(f"File format not supported: {path.name}")

    media_paths: list[Path] = []
    for current_root, _, files in os.walk(path):
        for file in sorted(files):
            file_path = Path(current_root) / file
            if is_media_file(file_path):
                media_paths.append(file_path)
    return media_paths


# Backward compatibility alias
scan_image_paths = scan_media_paths


def load_image_rgb(image_path: Path | str) -> np.ndarray | None:
    """
    Load an image from disk as an RGB numpy array (H, W, 3),
    correcting for EXIF orientation if present.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")
            return np.array(image, dtype=np.uint8)
    except Exception:
        return None


def extract_video_keyframes(
    video_path: Path | str,
    scene_threshold: float = 0.35,
    min_interval_sec: float = 60.0,
    max_interval_sec: float = 90.0,
) -> Generator[VideoKeyframe, None, None]:
    """
    Efficiently extract keyframes from a video file using adaptive scene-cut detection
    and maximum interval fallback sampling.

    Parameters:
        video_path: Path to video file.
        scene_threshold: Difference threshold (0.0 to 1.0) for detecting scene cuts.
        min_interval_sec: Minimum seconds between keyframes to prevent duplicate bursts.
        max_interval_sec: Maximum seconds before forcing a keyframe sample.

    Yields:
        VideoKeyframe(frame_idx, timestamp_sec, frame_rgb, is_scene_cut)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0

    min_frame_interval = max(1, int(fps * min_interval_sec))
    max_frame_interval = max(1, int(fps * max_interval_sec))

    prev_hist: np.ndarray | None = None
    last_keyframe_idx = -max_frame_interval
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            frames_since_last = frame_idx - last_keyframe_idx

            # First frame is always a keyframe
            if frame_idx == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Compute normalized HSV histogram for scene cut detection
                small_bgr = cv2.resize(frame_bgr, (160, 90))
                hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                prev_hist = hist
                last_keyframe_idx = 0
                yield VideoKeyframe(
                    frame_idx=0,
                    timestamp_sec=0.0,
                    frame_rgb=frame_rgb,
                    is_scene_cut=True,
                )
                frame_idx += 1
                continue

            # Compute thumbnail histogram
            small_bgr = cv2.resize(frame_bgr, (160, 90))
            hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) if prev_hist is not None else 1.0
            diff = max(0.0, 1.0 - correlation)

            is_scene_cut = diff >= scene_threshold
            should_sample = False

            if (is_scene_cut and frames_since_last >= min_frame_interval) or frames_since_last >= max_frame_interval:
                should_sample = True

            if should_sample:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                prev_hist = hist
                last_keyframe_idx = frame_idx
                yield VideoKeyframe(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    frame_rgb=frame_rgb,
                    is_scene_cut=is_scene_cut,
                )

            frame_idx += 1
    finally:
        cap.release()
