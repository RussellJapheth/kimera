# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Tests for folder hierarchy view and direct file filtering vs subfolder contents.
"""

from pathlib import Path

import pytest
from app.cache import ThumbnailCache
from app.db import Database
from app.server import create_app
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def folder_test_env(tmp_path: Path):
    db_file = tmp_path / "test_folders.db"
    cache_dir = tmp_path / "cache"
    db = Database(db_file)
    ThumbnailCache(cache_dir)

    # Structure:
    # tmp_path/
    #   root_img.jpg (1 direct file in root)
    #   vacation/
    #     vac_img1.jpg (direct in vacation)
    #     vac_img2.jpg (direct in vacation)
    #     day1/
    #       day1_img.jpg (direct in day1)
    #   empty_parent/
    #     sub_empty/
    #       deep_img.jpg (direct in sub_empty, empty_parent has 0 direct files)

    root_img = tmp_path / "root_img.jpg"
    vac_dir = tmp_path / "vacation"
    vac_dir.mkdir(parents=True)
    vac_img1 = vac_dir / "vac_img1.jpg"
    vac_img2 = vac_dir / "vac_img2.jpg"

    day1_dir = vac_dir / "day1"
    day1_dir.mkdir(parents=True)
    day1_img = day1_dir / "day1_img.jpg"

    empty_parent = tmp_path / "empty_parent"
    sub_empty = empty_parent / "sub_empty"
    sub_empty.mkdir(parents=True)
    deep_img = sub_empty / "deep_img.jpg"

    for p in [root_img, vac_img1, vac_img2, day1_img, deep_img]:
        Image.new("RGB", (100, 100), color=(128, 128, 128)).save(p)

    id_root = db.insert_image(str(root_img))
    id_vac1 = db.insert_image(str(vac_img1))
    id_vac2 = db.insert_image(str(vac_img2))
    id_day1 = db.insert_image(str(day1_img))
    id_deep = db.insert_image(str(deep_img))

    app = create_app(db_path=str(db_file), cache_dir=str(cache_dir))
    client = TestClient(app)

    return {
        "db": db,
        "client": client,
        "root": tmp_path,
        "vac_dir": vac_dir,
        "day1_dir": day1_dir,
        "empty_parent": empty_parent,
        "sub_empty": sub_empty,
        "ids": {
            "root": id_root,
            "vac1": id_vac1,
            "vac2": id_vac2,
            "day1": id_day1,
            "deep": id_deep,
        },
    }


def test_db_get_images_direct_only(folder_test_env):
    db = folder_test_env["db"]
    root = folder_test_env["root"]
    vac_dir = folder_test_env["vac_dir"]
    empty_parent = folder_test_env["empty_parent"]
    sub_empty = folder_test_env["sub_empty"]

    # Root with direct_only=True should ONLY have 1 image (root_img.jpg)
    root_direct = db.get_images(folder_path=str(root), folder_direct_only=True)
    assert root_direct["total"] == 1
    assert root_direct["images"][0]["filename"] == "root_img.jpg"

    # Root with direct_only=False should have all 5 images
    root_recursive = db.get_images(folder_path=str(root), folder_direct_only=False)
    assert root_recursive["total"] == 5

    # Vacation with direct_only=True should have 2 images (vac_img1.jpg, vac_img2.jpg), NOT day1_img.jpg
    vac_direct = db.get_images(folder_path=str(vac_dir), folder_direct_only=True)
    assert vac_direct["total"] == 2
    vac_filenames = {img["filename"] for img in vac_direct["images"]}
    assert vac_filenames == {"vac_img1.jpg", "vac_img2.jpg"}

    # empty_parent with direct_only=True should have 0 images
    empty_direct = db.get_images(folder_path=str(empty_parent), folder_direct_only=True)
    assert empty_direct["total"] == 0

    # sub_empty with direct_only=True should have 1 image (deep_img.jpg)
    sub_direct = db.get_images(folder_path=str(sub_empty), folder_direct_only=True)
    assert sub_direct["total"] == 1
    assert sub_direct["images"][0]["filename"] == "deep_img.jpg"


def test_folders_view_endpoint(folder_test_env):
    client = folder_test_env["client"]
    folder_test_env["root"]
    vac_dir = folder_test_env["vac_dir"]
    empty_parent = folder_test_env["empty_parent"]
    sub_empty = folder_test_env["sub_empty"]

    # 1. Root folder view
    res = client.get("/folders")
    assert res.status_code == 200
    assert "root_img.jpg" in res.text
    # Should list subfolder cards
    assert "vacation" in res.text
    assert "empty_parent" in res.text
    # But should NOT display media files from subfolders in root media files grid
    assert "vac_img1.jpg" not in res.text
    assert "deep_img.jpg" not in res.text

    # 2. Navigate to vacation folder
    res_vac = client.get(f"/folders?path={vac_dir}")
    assert res_vac.status_code == 200
    assert "vac_img1.jpg" in res_vac.text
    assert "vac_img2.jpg" in res_vac.text
    assert "day1" in res_vac.text
    # day1_img.jpg is inside subfolder day1, should not be in vacation media grid
    assert "day1_img.jpg" not in res_vac.text
    assert "root_img.jpg" not in res_vac.text

    # 3. Navigate to empty_parent
    res_ep = client.get(f"/folders?path={empty_parent}")
    assert res_ep.status_code == 200
    assert "sub_empty" in res_ep.text
    assert "deep_img.jpg" not in res_ep.text

    # 4. Navigate to sub_empty
    res_se = client.get(f"/folders?path={sub_empty}")
    assert res_se.status_code == 200
    assert "deep_img.jpg" in res_se.text


def test_adjacent_modal_navigation_in_folders(folder_test_env):
    client = folder_test_env["client"]
    vac_dir = folder_test_env["vac_dir"]
    ids = folder_test_env["ids"]

    # In vacation folder, opening vac1 should only have vac2 as adjacent, not root_img or deep_img
    res_modal = client.get(f"/api/photos/{ids['vac1']}/modal?path={vac_dir}&sort=name&order=asc")
    assert res_modal.status_code == 200
    # Should navigate within the 2 direct files (vac1, vac2)
    assert "1 / 2" in res_modal.text
    assert f"setNeighbors(null, {ids['vac2']})" in res_modal.text
