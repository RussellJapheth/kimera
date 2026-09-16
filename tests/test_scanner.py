# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Tests for directory, image, and video keyframe scanning logic.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.scanner import (
    extract_video_keyframes,
    is_image_file,
    is_media_file,
    is_video_file,
    scan_image_paths,
    scan_media_paths,
)


def test_is_image_file():
    assert is_image_file("photo.jpg") is True
    assert is_image_file("photo.JPEG") is True
    assert is_image_file("photo.png") is True
    assert is_image_file("photo.webp") is True
    assert is_image_file("photo.bmp") is True
    assert is_image_file("photo.txt") is False
    assert is_image_file("document.pdf") is False
    assert is_image_file(".DS_Store") is False


def test_is_video_file():
    assert is_video_file("video.mp4") is True
    assert is_video_file("video.MP4") is True
    assert is_video_file("video.avi") is True
    assert is_video_file("video.mov") is True
    assert is_video_file("video.mkv") is True
    assert is_video_file("video.webm") is True
    assert is_video_file("video.jpg") is False


def test_is_media_file():
    assert is_media_file("photo.jpg") is True
    assert is_media_file("video.mp4") is True
    assert is_media_file("audio.mp3") is False


def test_scan_image_paths_recursive(tmp_path: Path):
    # Setup test directory tree
    sub1 = tmp_path / "folder1"
    sub2 = tmp_path / "folder1" / "nested"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    img1 = tmp_path / "img1.png"
    img2 = sub1 / "img2.jpg"
    vid1 = sub2 / "clip.mp4"
    txt = sub1 / "readme.txt"

    img1.write_bytes(b"dummy")
    img2.write_bytes(b"dummy")
    vid1.write_bytes(b"dummy")
    txt.write_text("dummy")

    scanned = scan_media_paths(tmp_path)
    scanned_filenames = [p.name for p in scanned]

    assert len(scanned) == 3
    assert "img1.png" in scanned_filenames
    assert "img2.jpg" in scanned_filenames
    assert "clip.mp4" in scanned_filenames
    assert "readme.txt" not in scanned_filenames


def test_extract_video_keyframes(tmp_path: Path):
    # Create a synthetic video with 2 distinct scene cuts (60 frames total, 30fps = 2 sec)
    video_file = tmp_path / "test_scene.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (160, 120))

    # Scene 1: 30 frames of green
    for _ in range(30):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = [0, 255, 0]
        out.write(frame)

    # Scene 2: 30 frames of red
    for _ in range(30):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]
        out.write(frame)

    out.release()

    keyframes = list(
        extract_video_keyframes(
            video_file,
            scene_threshold=0.3,
            min_interval_sec=0.2,
            max_interval_sec=2.0,
        )
    )

    # Should detect initial frame (frame 0) and scene cut (frame 30)
    assert len(keyframes) >= 2
    assert keyframes[0].frame_idx == 0
    assert keyframes[0].is_scene_cut is True

    cut_frames = [k for k in keyframes if k.frame_idx >= 30 and k.is_scene_cut]
    assert len(cut_frames) >= 1


def test_scan_image_paths_invalid_path():
    with pytest.raises(FileNotFoundError):
        scan_image_paths("/non/existent/path/for/sure/12345")
