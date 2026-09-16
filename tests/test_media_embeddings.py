# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

from pathlib import Path

import numpy as np
import pytest
from app.db import Database
from app.models import get_media_embedder
from app.server import create_app
from fastapi.testclient import TestClient


def test_media_embeddings_db_crud(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)

    id1 = db.insert_image(str(tmp_path / "img1.jpg"))
    id2 = db.insert_image(str(tmp_path / "img2.jpg"))

    # Initial stats
    stats = db.get_media_embedding_stats()
    assert stats["total_images"] == 2
    assert stats["embedded_images"] == 0
    assert stats["missing"] == 2

    # Save embedding
    vec1 = np.ones(512, dtype=np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)
    db.save_media_embedding(id1, vec1, model="clip-vit-b32")

    retrieved = db.get_media_embedding(id1)
    assert retrieved is not None
    assert retrieved.shape == (512,)
    assert np.allclose(retrieved, vec1, atol=1e-5)

    stats = db.get_media_embedding_stats()
    assert stats["embedded_images"] == 1
    assert stats["missing"] == 1

    missing_ids = db.get_unembedded_image_ids()
    assert missing_ids == [id2]


def test_suggest_tags_and_similar_images(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    db = Database(db_path)

    id1 = db.insert_image(str(tmp_path / "target.jpg"))
    id2 = db.insert_image(str(tmp_path / "similar1.jpg"))
    id3 = db.insert_image(str(tmp_path / "similar2.jpg"))
    id4 = db.insert_image(str(tmp_path / "dissimilar.jpg"))

    # Base vector
    base_v = np.zeros(512, dtype=np.float32)
    base_v[0:100] = 1.0
    base_v = base_v / np.linalg.norm(base_v)

    # Target: base_v
    db.save_media_embedding(id1, base_v)

    # Similar 1: 95% similar
    v2 = base_v.copy()
    v2[100:105] = 0.2
    v2 = v2 / np.linalg.norm(v2)
    db.save_media_embedding(id2, v2)

    # Similar 2: 90% similar
    v3 = base_v.copy()
    v3[105:115] = 0.3
    v3 = v3 / np.linalg.norm(v3)
    db.save_media_embedding(id3, v3)

    # Dissimilar: orthogonal
    v4 = np.zeros(512, dtype=np.float32)
    v4[200:300] = 1.0
    v4 = v4 / np.linalg.norm(v4)
    db.save_media_embedding(id4, v4)

    # Add tags to similar images
    db.add_tags_to_image(id2, ["nature", "forest"])
    db.add_tags_to_image(id3, ["nature", "mountain"])
    db.add_tags_to_image(id4, ["urban"])

    # Test suggest_tags_for_image on target (id1)
    suggestions = db.suggest_tags_for_image(id1, top_k=5, min_score=0.2)
    assert len(suggestions) > 0

    tag_names = [s["name"] for s in suggestions]
    assert "nature" in tag_names
    # "nature" is on both similar neighbors, should have highest score
    assert suggestions[0]["name"] == "nature"
    assert suggestions[0]["confidence_pct"] >= 70

    # If target already has "nature", it should exclude "nature" and suggest remaining tags
    db.add_tags_to_image(id1, ["nature"])
    suggestions_after = db.suggest_tags_for_image(id1, top_k=5, min_score=0.2)
    names_after = [s["name"] for s in suggestions_after]
    assert "nature" not in names_after
    assert "forest" in names_after or "mountain" in names_after

    # Test get_similar_images for target (id1) includes target at index 0 followed by similar matches
    similar_res = db.get_similar_images(id1, limit=10, threshold=0.3)
    assert similar_res["total"] == 3  # id1 (target), id2, and id3
    matched_ids = [img["id"] for img in similar_res["images"]]
    assert matched_ids == [id1, id2, id3]
    assert similar_res["images"][0]["is_target"] is True
    assert similar_res["images"][0]["similarity_pct"] == 100
    assert id4 not in matched_ids

    # Test get_adjacent_image_ids navigating through similar results
    adj_target = db.get_adjacent_image_ids(id1, similar_to=id1, threshold=0.3)
    assert adj_target["prev_id"] is None
    assert adj_target["next_id"] == id2
    assert adj_target["current_index"] == 1
    assert adj_target["total_count"] == 3

    adj_match1 = db.get_adjacent_image_ids(id2, similar_to=id1, threshold=0.3)
    assert adj_match1["prev_id"] == id1
    assert adj_match1["next_id"] == id3
    assert adj_match1["current_index"] == 2

    adj_match2 = db.get_adjacent_image_ids(id3, similar_to=id1, threshold=0.3)
    assert adj_match2["prev_id"] == id2
    assert adj_match2["next_id"] is None
    assert adj_match2["current_index"] == 3


def test_gallery_and_modal_endpoints(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    cache_dir = str(tmp_path / "cache")
    app = create_app(db_path=db_path, cache_dir=cache_dir)
    client = TestClient(app)

    db = Database(db_path)
    img_file = tmp_path / "sample.jpg"
    img_file.write_bytes(b"dummy image bytes")
    id1 = db.insert_image(str(img_file))

    img_file2 = tmp_path / "sample2.jpg"
    img_file2.write_bytes(b"dummy image bytes 2")
    id2 = db.insert_image(str(img_file2))

    vec = np.ones(512, dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    db.save_media_embedding(id1, vec)
    db.save_media_embedding(id2, vec)

    # Test gallery with similar_to parameter (rendering photo_card.html for id2)
    resp = client.get(f"/?similar_to={id1}")
    assert resp.status_code == 200
    assert "Visually Similar Media" in resp.text
    assert "Clear Filter" in resp.text
    assert "photo-card-" in resp.text
    assert "100%" in resp.text

    # Test suggested-tags endpoint
    resp = client.get(f"/api/photos/{id1}/suggested-tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "suggested_tags" in data

    # Test embeddings status endpoint
    resp = client.get("/api/embeddings/status")
    assert resp.status_code == 200
    status_data = resp.json()
    assert "stats" in status_data
    assert status_data["stats"]["embedded_images"] == 2

    # Test modal rendering includes similar button and navigates in similar_to context
    resp = client.get(f"/api/photos/{id1}/modal")
    assert resp.status_code == 200
    assert "Similar" in resp.text
    assert f"/?similar_to={id1}" in resp.text

    # Test modal rendering when browsing similar_to (next item should be id2)
    resp = client.get(f"/api/photos/{id1}/modal?similar_to={id1}")
    assert resp.status_code == 200
    assert f"openModal({id2})" in resp.text


def test_media_embedder_inference():
    embedder = get_media_embedder(auto_download=False)
    if embedder is None:
        pytest.skip("CLIP model not available locally for offline test")

    dummy_rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
    emb = embedder.embed_image(dummy_rgb)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4

    # Test video frames averaging
    emb_vid = embedder.embed_video_frames([dummy_rgb, dummy_rgb])
    assert emb_vid is not None
    assert emb_vid.shape == (512,)


def test_stricter_similarity_threshold(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    app = create_app(db_path=db_path, cache_dir=str(tmp_path / "cache"))
    client = TestClient(app)
    db = Database(db_path)

    id1 = db.insert_image(str(tmp_path / "img1.jpg"))
    id2 = db.insert_image(str(tmp_path / "img2.jpg"))
    id3 = db.insert_image(str(tmp_path / "img3.jpg"))

    # Vector 1: base unit vector
    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0

    # Vector 2: similarity ~ 0.80 (very similar)
    v2 = np.zeros(512, dtype=np.float32)
    v2[0] = 0.8
    v2[1] = 0.6

    # Vector 3: similarity ~ 0.30 (weak match, should be excluded by 0.45 default threshold)
    v3 = np.zeros(512, dtype=np.float32)
    v3[0] = 0.30
    v3[1] = 0.9539
    v3 = v3 / np.linalg.norm(v3)

    db.save_media_embedding(id1, v1)
    db.save_media_embedding(id2, v2)
    db.save_media_embedding(id3, v3)

    # Default threshold (0.60) should include id1 (target) and id2 (0.80) but exclude id3 (0.30)
    res_default = db.get_similar_images(id1)
    matched_ids = [img["id"] for img in res_default["images"]]
    assert id1 in matched_ids
    assert id2 in matched_ids
    assert id3 not in matched_ids

    # Query gallery endpoint with default threshold
    resp = client.get(f"/?similar_to={id1}")
    assert resp.status_code == 200
    assert f"photo-card-{id1}" in resp.text
    assert f"photo-card-{id2}" in resp.text
    assert f"photo-card-{id3}" not in resp.text
    assert "Main" in resp.text

    # Setting save endpoint supports similarity_threshold
    resp = client.post(
        "/api/settings/save",
        data={
            "input_dir": str(tmp_path),
            "similarity_threshold": "0.85",
        },
    )
    assert resp.status_code == 200
    assert db.get_setting("similarity_threshold") == "0.85"
