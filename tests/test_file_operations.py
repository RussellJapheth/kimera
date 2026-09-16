# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Unit and integration tests for Kimera basic and bulk file operations.
Covers: deleting, renaming, batch renaming, moving, copying/duplicating, and batch favoriting.
"""

import numpy as np
import pytest
from app.db import Database
from app.server import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def test_env(tmp_path):
    """Fixture providing a test database, cache directory, pictures folder, and TestClient."""
    db_path = tmp_path / "test_file_ops.db"
    cache_dir = tmp_path / "cache"
    pics_dir = tmp_path / "pictures"
    pics_dir.mkdir(parents=True, exist_ok=True)

    db = Database(str(db_path))
    app = create_app(db_path=str(db_path), cache_dir=str(cache_dir))
    client = TestClient(app)

    return {
        "db": db,
        "db_path": db_path,
        "cache_dir": cache_dir,
        "pics_dir": pics_dir,
        "client": client,
    }


def test_delete_single_and_bulk_files(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    # Create dummy files
    f1 = pics_dir / "img1.jpg"
    f2 = pics_dir / "img2.jpg"
    f3 = pics_dir / "img3.jpg"
    f1.write_text("photo1")
    f2.write_text("photo2")
    f3.write_text("photo3")

    id1 = db.insert_image(str(f1), width=100, height=100, file_size=6)
    id2 = db.insert_image(str(f2), width=100, height=100, file_size=6)
    id3 = db.insert_image(str(f3), width=100, height=100, file_size=6)

    # Delete single file via JSON
    resp = client.post("/api/files/delete", json={"image_ids": [id1]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["deleted_count"] == 1
    assert not f1.exists()
    assert db.get_image(id1) is None

    # Delete multiple files via Form data
    resp2 = client.post("/api/files/delete", data={"image_ids": f"{id2},{id3}"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["deleted_count"] == 2
    assert not f2.exists()
    assert not f3.exists()
    assert db.get_image(id2) is None
    assert db.get_image(id3) is None


def test_rename_file(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    f1 = pics_dir / "sunset.jpg"
    f1.write_text("sunset photo")
    img_id = db.insert_image(str(f1), width=800, height=600, file_size=12)

    # Rename with extension omitted (auto-appends .jpg)
    resp = client.post("/api/files/rename", json={"image_id": img_id, "new_name": "golden_sunset"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["new_name"] == "golden_sunset.jpg"

    new_file = pics_dir / "golden_sunset.jpg"
    assert new_file.exists()
    assert not f1.exists()

    db_img = db.get_image(img_id)
    assert db_img["file_path"] == str(new_file.resolve())

    # Rename invalid path traversal attempt
    bad_resp = client.post("/api/files/rename", json={"image_id": img_id, "new_name": "../evil.jpg"})
    assert bad_resp.status_code == 400


def test_batch_rename(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    f1 = pics_dir / "raw_01.jpg"
    f2 = pics_dir / "raw_02.jpg"
    f1.write_text("1")
    f2.write_text("2")

    id1 = db.insert_image(str(f1), width=200, height=200, file_size=1)
    id2 = db.insert_image(str(f2), width=200, height=200, file_size=1)

    # Prefix mode
    resp = client.post(
        "/api/files/batch-rename",
        json={"image_ids": [id1, id2], "mode": "prefix", "prefix": "Trip_"},
    )
    assert resp.status_code == 200
    assert resp.json()["renamed_count"] == 2

    assert (pics_dir / "Trip_raw_01.jpg").exists()
    assert (pics_dir / "Trip_raw_02.jpg").exists()

    # Pattern mode
    resp2 = client.post(
        "/api/files/batch-rename",
        json={"image_ids": [id1, id2], "mode": "pattern", "pattern": "Holiday_{n}", "start_index": 10},
    )
    assert resp2.status_code == 200
    assert resp2.json()["renamed_count"] == 2
    assert (pics_dir / "Holiday_10.jpg").exists()
    assert (pics_dir / "Holiday_11.jpg").exists()


def test_move_files(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]
    sub_dir = pics_dir / "Vacation"

    f1 = pics_dir / "beach.jpg"
    f2 = pics_dir / "pool.jpg"
    f1.write_text("beach")
    f2.write_text("pool")

    id1 = db.insert_image(str(f1), width=500, height=500, file_size=5)
    id2 = db.insert_image(str(f2), width=500, height=500, file_size=4)

    resp = client.post(
        "/api/files/move",
        json={"image_ids": [id1, id2], "destination_folder": str(sub_dir)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["moved_count"] == 2

    assert (sub_dir / "beach.jpg").exists()
    assert (sub_dir / "pool.jpg").exists()
    assert not f1.exists()
    assert not f2.exists()

    img1 = db.get_image(id1)
    assert img1["file_path"] == str((sub_dir / "beach.jpg").resolve())


def test_copy_files_and_clone_records(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]
    dest_dir = pics_dir / "Backups"

    f1 = pics_dir / "portrait.jpg"
    f1.write_text("portrait data")

    id1 = db.insert_image(str(f1), width=600, height=800, file_size=13)
    emb = np.random.randn(512).astype(np.float32)
    p_id = db.name_person("Alice")
    db.insert_face(id1, bbox=(10, 10, 50, 50), confidence=0.95, embedding=emb, person_id=p_id)

    # 1. Copy to destination folder
    resp = client.post(
        "/api/files/copy",
        json={"image_ids": [id1], "destination_folder": str(dest_dir)},
    )
    assert resp.status_code == 200
    assert resp.json()["copied_count"] == 1
    assert (dest_dir / "portrait.jpg").exists()
    assert f1.exists()

    # 2. Duplicate in-place
    resp2 = client.post(
        "/api/files/copy",
        json={"image_ids": [id1]},
    )
    assert resp2.status_code == 200
    assert (pics_dir / "portrait_copy.jpg").exists()

    # Verify cloned faces exist for copied image
    images_in_dest = db.get_images(folder_path=str(dest_dir))
    assert images_in_dest["total"] == 1
    cloned_img = images_in_dest["images"][0]
    cloned_full = db.get_image(cloned_img["id"])
    assert len(cloned_full["faces"]) == 1
    assert cloned_full["faces"][0]["person_name"] == "Alice"


def test_batch_favorite(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    f1 = pics_dir / "fav1.jpg"
    f2 = pics_dir / "fav2.jpg"
    f1.write_text("1")
    f2.write_text("2")

    id1 = db.insert_image(str(f1))
    id2 = db.insert_image(str(f2))

    resp = client.post(
        "/api/files/batch-favorite",
        json={"image_ids": [id1, id2], "is_favorite": True},
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2
    assert db.get_image(id1)["is_favorite"] == 1
    assert db.get_image(id2)["is_favorite"] == 1


def test_folder_picker_and_modals(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    f1 = pics_dir / "item.jpg"
    f1.write_text("content")
    id1 = db.insert_image(str(f1))

    # /api/folders/list
    resp = client.get("/api/folders/list")
    assert resp.status_code == 200
    assert str(pics_dir.resolve()) in resp.json()["folders"]

    # /api/folders/picker-modal
    resp_modal = client.get(f"/api/folders/picker-modal?action=move&image_ids={id1}")
    assert resp_modal.status_code == 200
    assert "Move to Destination Folder" in resp_modal.text

    # /api/files/batch-rename-modal
    resp_rename = client.get(f"/api/files/batch-rename-modal?image_ids={id1}")
    assert resp_rename.status_code == 200
    assert "Batch Rename" in resp_rename.text


def test_rename_folder(test_env):
    db = test_env["db"]
    client = test_env["client"]
    pics_dir = test_env["pics_dir"]

    season_dir = pics_dir / "summer"
    season_dir.mkdir()
    sub_dir = season_dir / "nested"
    sub_dir.mkdir()

    f1 = season_dir / "a.jpg"
    f2 = season_dir / "b.jpg"
    f3 = sub_dir / "c.jpg"
    f1.write_text("1")
    f2.write_text("2")
    f3.write_text("3")

    id1 = db.insert_image(str(f1), width=100, height=100, file_size=1)
    id2 = db.insert_image(str(f2), width=100, height=100, file_size=1)
    id3 = db.insert_image(str(f3), width=100, height=100, file_size=1)

    # Rename folder with nested subfolder
    resp = client.post(
        "/api/folders/rename",
        json={"folder_path": str(season_dir), "new_name": "beach"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["renamed_count"] == 3

    renamed_dir = pics_dir / "beach"
    assert renamed_dir.exists()
    assert (renamed_dir / "a.jpg").exists()
    assert (renamed_dir / "nested" / "c.jpg").exists()
    assert not season_dir.exists()

    assert db.get_image(id1)["file_path"] == str((renamed_dir / "a.jpg").resolve())
    assert db.get_image(id2)["file_path"] == str((renamed_dir / "b.jpg").resolve())
    assert db.get_image(id3)["file_path"] == str((renamed_dir / "nested" / "c.jpg").resolve())

    # Invalid name rejected
    bad_resp = client.post(
        "/api/folders/rename",
        json={"folder_path": str(renamed_dir), "new_name": "../evil"},
    )
    assert bad_resp.status_code == 400

    # Renaming onto an existing folder rejected
    clash = pics_dir / "other"
    clash.mkdir()
    bad_resp2 = client.post(
        "/api/folders/rename",
        json={"folder_path": str(renamed_dir), "new_name": "other"},
    )
    assert bad_resp2.status_code == 400

    # Same-name rename is a no-op success
    noop_resp = client.post(
        "/api/folders/rename",
        json={"folder_path": str(renamed_dir), "new_name": "beach"},
    )
    assert noop_resp.status_code == 200
    assert noop_resp.json()["renamed_count"] == 0
