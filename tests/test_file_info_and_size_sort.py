# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

import os
import shutil
import tempfile

import pytest
from app.db import Database, format_duration, format_file_size
from app.server import create_app
from fastapi.testclient import TestClient


def test_format_helpers():
    assert format_file_size(0) == "0 B"
    assert format_file_size(500) == "500 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(1048576) == "1.0 MB"
    assert format_file_size(1048576 * 2.5) == "2.5 MB"
    assert format_file_size(1073741824 * 1.5) == "1.50 GB"

    assert format_duration(0) == ""
    assert format_duration(45) == "0:45"
    assert format_duration(75) == "1:15"
    assert format_duration(3665) == "1:01:05"


def test_db_size_sort_and_metadata():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)

        f_small = os.path.join(temp_dir, "small.jpg")
        f_med = os.path.join(temp_dir, "medium.png")
        f_large = os.path.join(temp_dir, "large.mp4")

        with open(f_small, "wb") as f:
            f.write(b"a" * 100)
        with open(f_med, "wb") as f:
            f.write(b"b" * 5000)
        with open(f_large, "wb") as f:
            f.write(b"c" * 2000000)

        id1 = db.insert_image(f_small, 640, 480, file_size=100, duration=0.0)
        id2 = db.insert_image(f_med, 800, 600, file_size=5000, duration=0.0)
        id3 = db.insert_image(f_large, 1920, 1080, file_size=2000000, duration=125.4)

        # Test single image metadata
        img3 = db.get_image(id3)
        assert img3["file_size"] == 2000000
        assert img3["duration"] == pytest.approx(125.4, 0.1)
        assert "MB" in img3["formatted_size"]
        assert img3["formatted_duration"] == "2:05"
        assert img3["is_video"] is True
        assert "MP4 Video" in img3["file_format"]

        # Test sort by size asc
        res_asc = db.get_images(sort_by="size", sort_order="asc")
        ids_asc = [img["id"] for img in res_asc["images"]]
        assert ids_asc == [id1, id2, id3]

        # Test sort by size desc
        res_desc = db.get_images(sort_by="size", sort_order="desc")
        ids_desc = [img["id"] for img in res_desc["images"]]
        assert ids_desc == [id3, id2, id1]

        # Test adjacent ids with size sort
        adj = db.get_adjacent_image_ids(id2, sort_by="size", sort_order="desc")
        assert adj["prev_id"] == id3
        assert adj["next_id"] == id1

    finally:
        shutil.rmtree(temp_dir)


def test_gallery_size_sort_and_file_info_routes():
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(temp_dir, "test.db")
        cache_dir = os.path.join(temp_dir, "cache")
        db = Database(db_path)

        p1 = os.path.join(temp_dir, "video1.mp4")
        p2 = os.path.join(temp_dir, "image1.jpg")

        with open(p1, "wb") as f:
            f.write(b"x" * 1500000)
        with open(p2, "wb") as f:
            f.write(b"y" * 20000)

        id1 = db.insert_image(p1, 1920, 1080, file_size=1500000, duration=45.0)
        id2 = db.insert_image(p2, 800, 600, file_size=20000, duration=0.0)

        app = create_app(db_path=db_path, cache_dir=cache_dir)
        client = TestClient(app)

        # Test gallery view with sort=size
        resp = client.get("/?sort=size&order=desc")
        assert resp.status_code == 200
        assert "sort=size" in resp.text
        assert "video1.mp4" in resp.text
        assert "1.4 MB" in resp.text or "1.5 MB" in resp.text

        # Test modal contains formatted file info
        modal_resp = client.get(f"/api/photos/{id1}/modal")
        assert modal_resp.status_code == 200
        assert "File Size:" in modal_resp.text
        assert "Length:" in modal_resp.text
        assert "0:45" in modal_resp.text
        # Multiplication sign is the deliberate rendered separator in the modal.
        assert "1920 × 1080 px" in modal_resp.text  # noqa: RUF001
        assert "MP4 Video" in modal_resp.text

        # Test image modal
        modal_img_resp = client.get(f"/api/photos/{id2}/modal")
        assert modal_img_resp.status_code == 200
        assert "File Size:" in modal_img_resp.text
        assert "JPG Image" in modal_img_resp.text or "JPEG Image" in modal_img_resp.text

    finally:
        shutil.rmtree(temp_dir)
