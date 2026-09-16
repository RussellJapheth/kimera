# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Tests for the SQLite database layer.
"""

from pathlib import Path

import numpy as np
import pytest
from app.db import Database


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_faces.db"
    return Database(db_file)


def test_embedding_serialization_roundtrip():
    original = np.random.randn(128).astype(np.float32)
    blob = Database.serialize_embedding(original)
    recovered = Database.deserialize_embedding(blob)
    assert np.allclose(original, recovered)


def test_insert_image_and_idempotency(temp_db: Database):
    id1 = temp_db.insert_image("/path/to/img1.jpg")
    assert id1 > 0

    # Inserting the exact same path should return the existing ID
    id2 = temp_db.insert_image("/path/to/img1.jpg")
    assert id1 == id2

    id3 = temp_db.insert_image("/path/to/img2.jpg")
    assert id3 > id1


def test_insert_face_and_clustering(temp_db: Database):
    img_id = temp_db.insert_image("/path/to/family.jpg")
    emb1 = np.random.randn(128).astype(np.float32)
    emb2 = np.random.randn(128).astype(np.float32)

    face1_id = temp_db.insert_face(
        image_id=img_id,
        bbox=(10, 10, 50, 50),
        confidence=0.95,
        embedding=emb1,
        cluster_id=-1,
    )
    face2_id = temp_db.insert_face(
        image_id=img_id,
        bbox=(60, 20, 100, 70),
        confidence=0.88,
        embedding=emb2,
        cluster_id=-1,
    )

    faces = temp_db.get_all_unclustered_or_all_faces()
    assert len(faces) == 2
    assert faces[0]["id"] == face1_id
    assert faces[0]["bbox"] == (10, 10, 50, 50)
    assert np.allclose(faces[0]["embedding"], emb1)

    # Update clusters
    temp_db.update_face_clusters([face1_id, face2_id], [0, 1])

    stats = temp_db.get_summary_stats()
    assert stats["images_scanned"] == 1
    assert stats["faces_detected"] == 2
    assert stats["num_clusters"] == 2
    assert len(stats["clusters"]) == 2

    # Inspect cluster 0
    inspection = temp_db.inspect_cluster(0)
    assert inspection["cluster_id"] == 0
    assert inspection["total_faces"] == 1
    assert inspection["total_images"] == 1
    assert inspection["image_paths"] == ["/path/to/family.jpg"]


def test_tags_create_and_case_insensitive_reuse(temp_db: Database):
    tag_id = temp_db.add_tag(" Vacation ")
    assert tag_id is not None

    same_id = temp_db.add_tag("vacation")
    assert same_id == tag_id

    assert temp_db.add_tag("   ") is None


def test_add_remove_tags_on_image(temp_db: Database):
    img_id = temp_db.insert_image("/path/to/tagged.jpg")

    added = temp_db.add_tags_to_image(img_id, ["Travel", "family", "travel"])
    assert added == 2

    tags = temp_db.get_image_tags(img_id)
    names = [t["name"] for t in tags]
    assert "Travel" in names
    assert "family" in names

    assert temp_db.remove_tag_from_image(img_id, tags[0]["id"]) is True
    assert len(temp_db.get_image_tags(img_id)) == 1


def test_get_image_includes_tags(temp_db: Database):
    img_id = temp_db.insert_image("/path/to/tagged2.jpg")
    temp_db.add_tags_to_image(img_id, ["Header", "hero"])

    img = temp_db.get_image(img_id)
    assert {t["name"] for t in img["tags"]} == {"Header", "hero"}


def test_tag_filters_images(temp_db: Database):
    img1 = temp_db.insert_image("/path/to/a.jpg")
    img2 = temp_db.insert_image("/path/to/b.jpg")
    temp_db.insert_image("/path/to/c.jpg")
    temp_db.add_tags_to_image(img1, ["Trip"])
    temp_db.add_tags_to_image(img2, ["Trip"])

    trip_tag = temp_db.get_all_tags()[0]
    result = temp_db.get_images(tag_id=trip_tag["id"])
    assert result["total"] == 2
    assert {i["file_path"] for i in result["images"]} == {"/path/to/a.jpg", "/path/to/b.jpg"}

    adj = temp_db.get_adjacent_image_ids(image_id=img1, tag_id=trip_tag["id"])
    assert adj["current_index"] != -1
    assert adj["total_count"] == 2


def test_image_tags_cascade_on_delete_and_clone(temp_db: Database):
    img_id = temp_db.insert_image("/path/to/cascade.jpg")
    temp_db.add_tags_to_image(img_id, ["Keep"])

    cloned_id = temp_db.clone_image_record(img_id, "/path/to/cloned.jpg")
    assert {t["name"] for t in temp_db.get_image_tags(cloned_id)} == {"Keep"}

    temp_db.delete_image_records([img_id])
    assert temp_db.get_image_tags(img_id) == []


def test_duplicate_groups_by_content_hash(temp_db: Database):
    # Unique file (no duplicate)
    temp_db.insert_image("/path/to/unique.jpg", content_hash="hash-a")

    # 3 copies of the same content across different directories
    id_dup1 = temp_db.insert_image("/path/a/copy1.jpg", content_hash="hash-b")
    id_dup2 = temp_db.insert_image("/path/b/copy2.jpg", content_hash="hash-b")
    id_dup3 = temp_db.insert_image("/path/c/copy3.jpg", content_hash="hash-b")

    # Legacy file with no hash must not be grouped
    temp_db.insert_image("/path/to/legacy.jpg", content_hash="")

    results = temp_db.get_duplicate_groups()

    assert results["total_groups"] == 1
    assert results["total_duplicates"] == 2
    assert results["total_files"] == 4

    group = results["groups"][0]
    assert group["size"] == 3
    assert group["keep"]["id"] == id_dup1
    assert group["keep"]["filename"] == "copy1.jpg"
    assert [d["id"] for d in group["duplicates"]] == [id_dup2, id_dup3]


def test_exclude_duplicates_filters_to_one_per_hash(temp_db: Database):
    id_keep = temp_db.insert_image("/path/a/first.jpg", content_hash="hash-b")
    temp_db.insert_image("/path/b/second.jpg", content_hash="hash-b")
    temp_db.insert_image("/path/c/third.jpg", content_hash="hash-b")

    id_unique = temp_db.insert_image("/path/d/unique.jpg", content_hash="hash-a")
    id_legacy = temp_db.insert_image("/path/e/legacy.jpg", content_hash="")

    # Without exclusion all files are visible
    full = temp_db.get_images()
    assert full["total"] == 5

    # With exclusion only one file per content hash survives; legacy keeps showing
    deduped = temp_db.get_images(exclude_duplicates=True)
    assert deduped["total"] == 3
    kept_ids = {img["id"] for img in deduped["images"]}
    assert kept_ids == {id_keep, id_unique, id_legacy}

    # Modal navigation honours the same filter
    adj = temp_db.get_adjacent_image_ids(image_id=id_keep, exclude_duplicates=True)
    assert adj["total_count"] == 3


def test_get_hash_records_returns_meta(temp_db: Database):
    temp_db.insert_image("/path/to/a.jpg", file_size=1024, mtime=123.0, content_hash="abc")
    temp_db.insert_image("/path/to/b.jpg")

    records = temp_db.get_hash_records()
    assert len(records) == 2
    by_path = {r["file_path"]: r for r in records}
    assert by_path["/path/to/a.jpg"]["content_hash"] == "abc"
    assert by_path["/path/to/a.jpg"]["file_size"] == 1024
    assert by_path["/path/to/b.jpg"]["content_hash"] == ""


def test_add_tag_propagates_to_duplicates(temp_db: Database):
    id_orig = temp_db.insert_image("/path/a/orig.jpg", content_hash="hash-x")
    id_dup1 = temp_db.insert_image("/path/b/dup1.jpg", content_hash="hash-x")
    id_dup2 = temp_db.insert_image("/path/c/dup2.jpg", content_hash="hash-x")
    id_unique = temp_db.insert_image("/path/d/unique.jpg", content_hash="hash-y")
    id_nohash = temp_db.insert_image("/path/e/nohash.jpg", content_hash="")

    # Tagging one file propagates to all its duplicates
    added = temp_db.add_tags_to_image(id_orig, ["Trip"])
    assert added == 3  # one new link on each of the three duplicate files

    assert {t["name"] for t in temp_db.get_image_tags(id_orig)} == {"Trip"}
    assert {t["name"] for t in temp_db.get_image_tags(id_dup1)} == {"Trip"}
    assert {t["name"] for t in temp_db.get_image_tags(id_dup2)} == {"Trip"}

    # Other files (different or no hash) stay untouched
    assert temp_db.get_image_tags(id_unique) == []
    assert temp_db.get_image_tags(id_nohash) == []

    # Re-tagging does not create new links
    added_again = temp_db.add_tags_to_image(id_dup2, ["Trip"])
    assert added_again == 0


def test_tag_removal_does_not_propagate_to_duplicates(temp_db: Database):
    id_orig = temp_db.insert_image("/path/a/orig.jpg", content_hash="hash-x")
    id_dup = temp_db.insert_image("/path/b/dup.jpg", content_hash="hash-x")

    temp_db.add_tags_to_image(id_orig, ["Keep"])
    assert temp_db.remove_tag_from_image(id_orig, temp_db.get_image_tags(id_orig)[0]["id"]) is True

    # Removal stays scoped to the single file; duplicate keeps its tag
    assert temp_db.get_image_tags(id_orig) == []
    assert {t["name"] for t in temp_db.get_image_tags(id_dup)} == {"Keep"}
