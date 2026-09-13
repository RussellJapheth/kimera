"""
Tests for media gallery database extensions, caching, and web server endpoints.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.cache import ThumbnailCache
from app.db import Database
from app.server import create_app


@pytest.fixture
def test_env(tmp_path: Path):
    db_file = tmp_path / "test_gallery.db"
    cache_dir = tmp_path / "cache"
    db = Database(db_file)
    cache = ThumbnailCache(cache_dir)

    # Create dummy images on disk
    img1 = tmp_path / "photo1.jpg"
    img2 = tmp_path / "photo2.jpg"
    Image.new("RGB", (640, 480), color=(200, 100, 50)).save(img1)
    Image.new("RGB", (800, 600), color=(50, 100, 200)).save(img2)

    id1 = db.insert_image(str(img1), width=640, height=480)
    id2 = db.insert_image(str(img2), width=800, height=600)

    emb = np.random.randn(512).astype(np.float32)
    f1 = db.insert_face(id1, bbox=(50, 50, 150, 150), confidence=0.98, embedding=emb, cluster_id=0)
    f2 = db.insert_face(id2, bbox=(100, 100, 250, 250), confidence=0.92, embedding=emb, cluster_id=0)

    return {
        "db": db,
        "db_file": db_file,
        "cache": cache,
        "cache_dir": cache_dir,
        "img1": img1,
        "img2": img2,
        "id1": id1,
        "id2": id2,
        "f1": f1,
        "f2": f2,
    }


def test_favorites_and_filtering(test_env):
    db = test_env["db"]
    id1 = test_env["id1"]
    id2 = test_env["id2"]

    # Initially neither is favorite
    res = db.get_images(filter_type="favorites")
    assert res["total"] == 0

    # Toggle favorite
    new_fav = db.toggle_favorite(id1)
    assert new_fav is True

    res = db.get_images(filter_type="favorites")
    assert res["total"] == 1
    assert res["images"][0]["id"] == id1

    # Toggle off
    new_fav = db.toggle_favorite(id1)
    assert new_fav is False
    res = db.get_images(filter_type="favorites")
    assert res["total"] == 0


def test_get_people_hides_low_quality_clusters(test_env):
    db = test_env["db"]
    emb = np.random.randn(512).astype(np.float32)

    # Tiny, low-confidence face in cluster 7
    db.insert_face(test_env["id1"], bbox=(5, 5, 25, 25), confidence=0.55, embedding=emb, cluster_id=7)

    # Good-quality faces but all in the same single image (cluster 8)
    db.insert_face(test_env["id1"], bbox=(200, 200, 250, 250), confidence=0.95, embedding=emb, cluster_id=8)
    db.insert_face(test_env["id1"], bbox=(300, 300, 350, 350), confidence=0.93, embedding=emb, cluster_id=8)

    assert 7 in {p["id"] for p in db.get_people()}
    assert 8 in {p["id"] for p in db.get_people()}

    hidden = db.get_people(hide_low_quality=True)
    assert 7 not in {p["id"] for p in hidden}
    assert 8 not in {p["id"] for p in hidden}
    assert 0 in {p["id"] for p in hidden}


def test_people_naming_and_merging(test_env):
    db = test_env["db"]
    f1 = test_env["f1"]

    # Initially cluster 0 has no named person
    people = db.get_people()
    assert len(people) >= 1
    assert people[0]["name"] == "Person #0"

    # Name the cluster "Alice"
    p_id1 = db.name_person(name="Alice", cluster_id=0)
    assert p_id1 > 0

    people = db.get_people()
    named_alice = [p for p in people if p["name"] == "Alice"]
    assert len(named_alice) == 1
    assert named_alice[0]["photo_count"] == 2

    # Rename Alice to "Alice Smith"
    db.name_person(name="Alice Smith", person_id=p_id1)
    person = db.get_person(p_id1)
    assert person["name"] == "Alice Smith"

    # Create another person "Bob"
    p_id2 = db.name_person(name="Bob", cluster_id=-1, face_id=f1)
    
    # Merge Bob into Alice Smith
    db.merge_people(source_person_id=p_id2, target_person_id=p_id1)
    assert db.get_person(p_id2) is None


def test_thumbnail_and_face_crop_cache(test_env):
    cache = test_env["cache"]
    img1 = test_env["img1"]
    f1 = test_env["f1"]

    # Thumbnail generation
    thumb = cache.get_thumbnail(img1, max_dim=320)
    assert thumb is not None
    assert thumb.exists()
    assert thumb.suffix == ".webp"

    # Face crop generation
    crop = cache.get_face_crop(f1, img1, bbox=(50, 50, 150, 150), size=120)
    assert crop is not None
    assert crop.exists()
    assert crop.suffix == ".webp"

    # Dummy video file thumbnail generation
    import cv2
    video_file = test_env["cache_dir"].parent / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (320, 240))
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_frame[:, :] = (100, 150, 200)
    for _ in range(10):
        out.write(dummy_frame)
    out.release()

    video_thumb = cache.get_thumbnail(video_file, max_dim=320)
    assert video_thumb is not None
    assert video_thumb.exists()
    assert video_thumb.suffix == ".webp"


def test_web_gallery_endpoints(test_env):
    app = create_app(db_path=str(test_env["db_file"]), cache_dir=str(test_env["cache_dir"]))
    client = TestClient(app)

    # 1. Main gallery view
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Kimera" in resp.text
    assert "photo1.jpg" in resp.text

    # 2. People view
    resp = client.get("/people")
    assert resp.status_code == 200
    assert "People & Faces" in resp.text

    # 3. Toggle favorite endpoint
    resp = client.post(f"/api/photos/{test_env['id1']}/favorite")
    assert resp.status_code == 200

    # 4. Thumbnail endpoint
    resp = client.get(f"/api/photos/{test_env['id1']}/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"

    # 5. Face crop endpoint
    resp = client.get(f"/api/faces/{test_env['f1']}/crop")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/webp"

    # 6. Lightbox modal endpoint
    resp = client.get(f"/api/photos/{test_env['id1']}/modal")
    assert resp.status_code == 200
    assert "Detected People & Faces" in resp.text


def test_legacy_db_migration_and_auto_healing(tmp_path: Path):
    import sqlite3

    db_file = tmp_path / "legacy.db"

    # Create a legacy table without is_favorite, width, height, or people table
    conn = sqlite3.connect(str(db_file))
    conn.executescript("""
        CREATE TABLE images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            box_x1 INTEGER NOT NULL,
            box_y1 INTEGER NOT NULL,
            box_x2 INTEGER NOT NULL,
            box_y2 INTEGER NOT NULL,
            confidence REAL NOT NULL,
            embedding BLOB NOT NULL,
            cluster_id INTEGER DEFAULT -1,
            FOREIGN KEY (image_id) REFERENCES images(id)
        );
    """)
    conn.close()

    # Initializing Database on legacy DB must auto-migrate without error
    db = Database(db_file)
    id1 = db.insert_image("/path/to/test.jpg")
    assert db.toggle_favorite(id1) is True

    # Deleting DB file and re-initializing must auto-heal and recreate schema
    db_file.unlink()
    db2 = Database(db_file)
    id2 = db2.insert_image("/path/to/fresh.jpg")
    assert id2 == 1
    assert db2.toggle_favorite(id2) is True


def test_cache_configuration_and_stats(tmp_path: Path):
    """Verify cache directory configuration, stats calculation, and clearing."""
    from app.cache import DEFAULT_CACHE_DIR, ThumbnailCache

    assert DEFAULT_CACHE_DIR == Path.cwd() / ".cache"

    c_dir = tmp_path / "custom_cache"
    cache = ThumbnailCache(c_dir)
    assert cache.cache_dir == c_dir.resolve()
    assert cache.thumbs_dir.exists()
    assert cache.faces_dir.exists()

    # Generate thumbnail and test stats
    img_file = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color="red").save(img_file)
    thumb = cache.get_thumbnail(img_file)
    assert thumb is not None and thumb.exists()

    stats = cache.get_cache_stats()
    assert stats["thumbnail_count"] == 1
    assert stats["total_size_bytes"] > 0

    # Clear cache
    cache.clear_cache()
    stats_cleared = cache.get_cache_stats()
    assert stats_cleared["thumbnail_count"] == 0


def test_photo_modal_context_and_folder_link(test_env):
    """Verify modal endpoint respects folder path, person_id, and links to folder."""
    db = test_env["db"]
    id1 = test_env["id1"]
    id2 = test_env["id2"]
    img1 = test_env["img1"]

    app = create_app(db_path=str(test_env["db_file"]), cache_dir=str(test_env["cache_dir"]))
    client = TestClient(app)

    # 1. Test modal returns folder link with parent_folder
    resp = client.get(f"/api/photos/{id1}/modal")
    assert resp.status_code == 200
    expected_folder = str(img1.parent)
    assert f"/folders?path={expected_folder}" in resp.text
    assert "modal-filename-link" in resp.text

    # 2. Test modal query with path parameter (folders context)
    resp_folder = client.get(f"/api/photos/{id1}/modal?path={expected_folder}")
    assert resp_folder.status_code == 200

    # 3. Test modal query with person_id context
    p_id = db.name_person(name="Alice", cluster_id=0)
    resp_person = client.get(f"/api/photos/{id1}/modal?person_id={p_id}")
    assert resp_person.status_code == 200



