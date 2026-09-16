# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

from pathlib import Path

import cv2
import numpy as np
import pytest
from app.cache import ThumbnailCache
from app.db import Database
from app.server import create_app
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def video_test_env(tmp_path: Path):
    db_file = tmp_path / "test.db"
    cache_dir = tmp_path / "cache"
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    db = Database(db_file)
    cache = ThumbnailCache(cache_dir=cache_dir)

    # 1. Create a dummy synthetic video (320x240, 10 frames)
    video_path = media_dir / "sample_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for _ in range(10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (20, 120, 220)
        out.write(frame)
    out.release()

    # 2. Create a dummy image
    image_path = media_dir / "sample_photo.jpg"
    Image.new("RGB", (400, 300), color="blue").save(image_path)

    # Insert into database
    video_id = db.insert_image(
        file_path=str(video_path),
        width=320,
        height=240,
        file_size=video_path.stat().st_size,
    )
    image_id = db.insert_image(
        file_path=str(image_path),
        width=400,
        height=300,
        file_size=image_path.stat().st_size,
    )

    app = create_app(db_path=str(db_file), cache_dir=str(cache_dir))
    client = TestClient(app)

    return {
        "db": db,
        "cache": cache,
        "client": client,
        "video_path": video_path,
        "image_path": image_path,
        "video_id": video_id,
        "image_id": image_id,
        "cache_dir": cache_dir,
    }


def test_transcode_video_cache(video_test_env):
    """Verify video transcoding produces browser-compatible MP4 preserving resolution."""
    cache: ThumbnailCache = video_test_env["cache"]
    video_path = video_test_env["video_path"]

    transcoded = cache.get_transcoded_video(video_path)
    assert transcoded is not None
    assert transcoded.exists()
    assert transcoded.suffix == ".mp4"
    assert transcoded.stat().st_size > 0

    # Verify cached output video resolution matches original (320x240)
    cap = cv2.VideoCapture(str(transcoded))
    assert cap.isOpened()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert width == 320
    assert height == 240

    # Test cache retrieval on subsequent call
    cached_again = cache.get_transcoded_video(video_path)
    assert cached_again == transcoded

    # Non-video check
    assert cache.get_transcoded_video(video_test_env["image_path"]) is None


def test_cache_stats_and_clearing_with_transcoded(video_test_env):
    """Verify get_cache_stats and clear_cache handle transcoded videos."""
    cache: ThumbnailCache = video_test_env["cache"]
    video_path = video_test_env["video_path"]

    cache.get_transcoded_video(video_path)
    stats = cache.get_cache_stats()
    assert stats["transcoded_count"] >= 1
    assert stats["total_size_bytes"] > 0

    cache.clear_cache()
    stats_cleared = cache.get_cache_stats()
    assert stats_cleared["transcoded_count"] == 0


def test_video_stream_api_endpoint(video_test_env):
    """Verify /api/photos/{id}/stream route serves video with range support."""
    client: TestClient = video_test_env["client"]
    video_id = video_test_env["video_id"]
    image_id = video_test_env["image_id"]

    # Stream video
    resp = client.get(f"/api/photos/{video_id}/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "video/mp4"
    assert resp.headers.get("accept-ranges") == "bytes"

    # Stream requested on image falls back gracefully to raw
    resp_img = client.get(f"/api/photos/{image_id}/stream")
    assert resp_img.status_code == 200

    # Non-existent ID
    resp_404 = client.get("/api/photos/999999/stream")
    assert resp_404.status_code == 404


def test_modal_video_stream_ui_toggle(video_test_env):
    """Verify video mode toggle and error fallback appear only for videos."""
    client: TestClient = video_test_env["client"]
    video_id = video_test_env["video_id"]
    image_id = video_test_env["image_id"]

    # 1. Video modal should contain video toggle bar and stream endpoints
    resp_video = client.get(f"/api/photos/{video_id}/modal")
    assert resp_video.status_code == 200
    assert "video-mode-toggle" in resp_video.text
    assert "Web Stream" in resp_video.text
    assert "Direct (Original)" in resp_video.text
    assert "video-error-overlay" in resp_video.text
    assert f"/api/photos/{video_id}/stream" in resp_video.text

    # 2. Image modal must NOT contain video mode toggle
    resp_image = client.get(f"/api/photos/{image_id}/modal")
    assert resp_image.status_code == 200
    assert "video-mode-toggle" not in resp_image.text
    assert "video-error-overlay" not in resp_image.text
