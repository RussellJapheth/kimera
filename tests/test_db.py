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
    img3 = temp_db.insert_image("/path/to/c.jpg")
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
