"""
Tests for "Improve Matches" active learning feature.
Verifies candidate search filtering, distance bounds, 10-cap constraint, and feedback confirmation API.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import normalize

from app.cache import ThumbnailCache
from app.db import Database
from app.recognition import MultiExemplarMatcher
from app.server import create_app


@pytest.fixture
def improve_env(tmp_path: Path):
    db_file = tmp_path / "test_improve.db"
    cache_dir = tmp_path / "test_cache"
    db = Database(db_file)
    cache = ThumbnailCache(cache_dir)

    # Base normalized embedding for Person 1 (Alice)
    rng = np.random.RandomState(42)
    base_emb = normalize(rng.randn(1, 512).astype(np.float32))[0]

    # Create dummy images
    img_files = []
    for i in range(20):
        p = tmp_path / f"img_{i:02d}.jpg"
        Image.new("RGB", (200, 200), color=(100 + i * 5, 50, 50)).save(p)
        img_id = db.insert_image(str(p), width=200, height=200)
        img_files.append((p, img_id))

    # Create Person 1 ("Alice") with cover face
    alice_face_id = db.insert_face(
        image_id=img_files[0][1],
        bbox=(10, 10, 50, 50),
        confidence=0.99,
        embedding=base_emb,
        cluster_id=1,
    )
    person_id = db.name_person("Alice", face_id=alice_face_id)

    # Insert 15 unassigned faces with varying distances to Alice's embedding
    # Noise levels chosen so cosine distance ranges from 0.1 to 0.9
    unassigned_face_ids = []
    for i in range(1, 16):
        noise = rng.randn(512).astype(np.float32)
        noise_norm = normalize(noise.reshape(1, -1))[0]
        # Mix base_emb and noise
        # alpha from 0.9 (very close, dist ~ 0.1) to 0.1 (far, dist ~ 0.9)
        alpha = max(0.05, 1.0 - (i * 0.06))
        mixed = normalize((alpha * base_emb + (1 - alpha) * noise_norm).reshape(1, -1))[0]

        fid = db.insert_face(
            image_id=img_files[i][1],
            bbox=(20, 20, 60, 60),
            confidence=0.95,
            embedding=mixed,
            cluster_id=-1,
        )
        unassigned_face_ids.append(fid)

    app = create_app(db_path=str(db_file), cache_dir=str(cache_dir))
    client = TestClient(app)

    return {
        "db": db,
        "person_id": person_id,
        "base_emb": base_emb,
        "unassigned_face_ids": unassigned_face_ids,
        "client": client,
    }


def test_matcher_find_uncertain_candidates(improve_env):
    db = improve_env["db"]
    person_id = improve_env["person_id"]

    exemplars = db.get_person_exemplars(person_id)
    assert len(exemplars) >= 1

    matcher = MultiExemplarMatcher({person_id: exemplars})
    unassigned = db.get_unassigned_faces()
    assert len(unassigned) == 15

    # Find candidates with limit 10
    candidates = matcher.find_uncertain_candidates(
        person_id=person_id,
        faces=unassigned,
        min_dist=0.30,
        max_dist=0.70,
        limit=10,
    )

    # Must be capped at 10
    assert len(candidates) <= 10
    assert len(candidates) > 0

    # Ensure sorted by distance ascending
    dists = [c["distance"] for c in candidates]
    assert dists == sorted(dists)

    # Ensure all within distance bounds
    for c in candidates:
        assert 0.30 <= c["distance"] <= 0.70
        assert 0 <= c["similarity_pct"] <= 100


def test_candidates_api_endpoint(improve_env):
    client = improve_env["client"]
    person_id = improve_env["person_id"]

    resp = client.get(f"/api/people/{person_id}/candidates?limit=10")
    assert resp.status_code == 200
    data = resp.json()

    assert data["person"]["name"] == "Alice"
    assert data["person"]["id"] == person_id
    assert "candidates" in data
    assert len(data["candidates"]) <= 10
    assert data["total_unassigned"] == 15


def test_confirm_match_yes(improve_env):
    client = improve_env["client"]
    db = improve_env["db"]
    person_id = improve_env["person_id"]

    # Get candidates
    resp = client.get(f"/api/people/{person_id}/candidates?limit=10")
    data = resp.json()
    assert len(data["candidates"]) > 0
    candidate = data["candidates"][0]
    face_id = candidate["face_id"]

    # Post "Yes" match
    match_resp = client.post(
        f"/api/people/{person_id}/confirm-match",
        json={"face_id": face_id, "matched": True},
    )
    assert match_resp.status_code == 200
    assert match_resp.json()["action"] == "assigned"

    # Verify face is now assigned to Alice
    face = db.get_face(face_id)
    assert face["person_id"] == person_id


def test_confirm_match_no_and_exclusion(improve_env):
    client = improve_env["client"]
    db = improve_env["db"]
    person_id = improve_env["person_id"]

    # Get candidates
    resp = client.get(f"/api/people/{person_id}/candidates?limit=10")
    data = resp.json()
    assert len(data["candidates"]) > 0
    candidate = data["candidates"][0]
    face_id = candidate["face_id"]

    # Post "No" match (exclusion)
    match_resp = client.post(
        f"/api/people/{person_id}/confirm-match",
        json={"face_id": face_id, "matched": False},
    )
    assert match_resp.status_code == 200
    assert match_resp.json()["action"] == "excluded"

    # Verify face is NOT assigned to person
    face = db.get_face(face_id)
    assert face["person_id"] is None

    # Verify exclusion was saved
    exclusions = db.get_person_exclusions()
    assert face_id in exclusions
    assert person_id in exclusions[face_id]

    # Re-query candidates: excluded face must not appear
    new_resp = client.get(f"/api/people/{person_id}/candidates?limit=10")
    new_candidates = new_resp.json()["candidates"]
    candidate_face_ids = [c["face_id"] for c in new_candidates]
    assert face_id not in candidate_face_ids


def test_global_candidates_api_endpoint(improve_env):
    client = improve_env["client"]
    resp = client.get("/api/people/all/candidates?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert len(data["candidates"]) <= 10
    assert data["total_unassigned"] == 15
    for c in data["candidates"]:
        assert c["person_name"] == "Alice"
        assert 0 <= c["similarity_pct"] <= 100


def test_settings_page_has_improve_matches_button(improve_env):
    client = improve_env["client"]
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "Improve Matches" in resp.text


def test_person_detail_page_button_visibility(improve_env):
    client = improve_env["client"]
    person_id = improve_env["person_id"]

    # Improve Matches button is always available on person detail page
    resp = client.get(f"/person/{person_id}")
    assert resp.status_code == 200
    assert 'id="improve-matches-btn"' in resp.text

    # Candidates endpoint returns candidates
    cands_resp = client.get(f"/api/people/{person_id}/candidates?limit=10")
    assert cands_resp.status_code == 200
    assert len(cands_resp.json()["candidates"]) > 0

    # Exclude all 15 unassigned faces
    db = improve_env["db"]
    unassigned = db.get_unassigned_faces()
    for f in unassigned:
        db.add_person_exclusion(f["id"], person_id)

    # Button remains available on page
    resp2 = client.get(f"/person/{person_id}")
    assert resp2.status_code == 200
    assert 'id="improve-matches-btn"' in resp2.text

    # But candidates endpoint now returns 0 candidates
    cands_resp2 = client.get(f"/api/people/{person_id}/candidates?limit=10")
    assert cands_resp2.status_code == 200
    assert len(cands_resp2.json()["candidates"]) == 0

    # Settings page button is ALWAYS visible
    settings_resp = client.get("/settings")
    assert settings_resp.status_code == 200
    assert 'id="global-improve-matches-btn"' in settings_resp.text
