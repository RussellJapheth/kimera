"""
Image scanning and loading utilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def is_image_file(path: Path | str) -> bool:
    """Check if a file has a supported image extension."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def scan_image_paths(directory: Path | str) -> List[Path]:
    """Recursively scan a directory for supported image files."""
    root = Path(directory).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    image_paths: List[Path] = []
    for current_root, _, files in os.walk(root):
        for file in sorted(files):
            file_path = Path(current_root) / file
            if is_image_file(file_path):
                image_paths.append(file_path)
    return image_paths


def load_image_rgb(image_path: Path | str) -> Optional[np.ndarray]:
    """
    Load an image from disk as an RGB numpy array (H, W, 3),
    correcting for EXIF orientation if present.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception:
        return None
