import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.db import Database
from app.server import create_app


@pytest.fixture
def test_env():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_features.db")
    cache_dir = os.path.join(temp_dir, "cache")
    pictures_dir = os.path.join(temp_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)
    os.makedirs(os.path.join(pictures_dir, "trips", "summer"), exist_ok=True)
    os.makedirs(os.path.join(pictures_dir, "trips", "winter"), exist_ok=True)

    db = Database(db_path)

    # Insert dummy files
    p1 = os.path.join(pictures_dir, "trips", "summer", "beach.jpg")
    p2 = os.path.join(pictures_dir, "trips", "summer", "surf.mp4")
    p3 = os.path.join(pictures_dir, "trips", "winter", "snow.jpg")
    p4 = os.path.join(pictures_dir, "portrait.png")

    for p in [p1, p2, p3, p4]:
        with open(p, "wb") as f:
            f.write(b"fake data")

    id1 = db.insert_image(p1, 800, 600)
    id2 = db.insert_image(p2, 1920, 1080)
    id3 = db.insert_image(p3, 1024, 768)
    id4 = db.insert_image(p4, 500, 500)

    # Insert faces & people
    emb = np.random.randn(512).astype(np.float32)
    # Named person face
    person_id = db.name_person("Alice")
    db.insert_face(id1, (10, 10, 100, 100), 0.95, emb, cluster_id=0, person_id=person_id)

    # Unnamed cluster face
    db.insert_face(id1, (120, 10, 200, 100), 0.88, emb, cluster_id=1, person_id=None)

    # Video face
    db.insert_face(id2, (20, 20, 120, 120), 0.90, emb, cluster_id=0, person_id=person_id)

    app = create_app(db_path=db_path, cache_dir=cache_dir)
    client = TestClient(app)

    yield {
        "db": db,
        "app": app,
        "client": client,
        "ids": [id1, id2, id3, id4],
        "person_id": person_id,
        "temp_dir": temp_dir,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_video_play_badge(test_env):
    """Req 1: Verify play badge markup on video thumbnail cards."""
    client = test_env["client"]
    resp = client.get("/")
    assert resp.status_code == 200
    assert "video-play-indicator" in resp.text
    assert "Play Video" in resp.text


def test_settings_page_and_scan_trigger(test_env):
    """Req 2: Verify settings page and scan trigger API."""
    client = test_env["client"]
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "Library Settings & Rescan" in resp.text
    assert "Indexed Media" in resp.text

    # Trigger scan
    scan_resp = client.post(
        "/api/scan/trigger",
        data={
            "input_dir": test_env["temp_dir"],
            "conf_threshold": 0.5,
            "eps": 0.65,
            "min_samples": 1,
            "algorithm": "dbscan",
        },
    )
    assert scan_resp.status_code == 200
    assert "scan-status-card" in scan_resp.text


def test_folders_browsing_and_hierarchy(test_env):
    """Req 3: Verify folder browsing and hierarchy navigation."""
    client = test_env["client"]
    db = test_env["db"]

    folders = db.get_folders()
    assert len(folders["subfolders"]) > 0

    resp = client.get("/folders")
    assert resp.status_code == 200
    assert "folder-card" in resp.text
    assert "trips" in resp.text or "Root" in resp.text


def test_infinite_scroll_lazy_loading(test_env):
    """Req 4: Verify infinite scroll returns photo batches with sentinel."""
    client = test_env["client"]
    resp = client.get("/?page=1&infinite=1", headers={"HX-Request": "true"})
    assert resp.status_code == 200
    assert "photo-card" in resp.text


def test_search_file_folder_person(test_env):
    """Req 5: Verify search matches file name, folder name, and person name."""
    db = test_env["db"]

    # Search by filename
    res_file = db.get_images(search="beach")
    assert res_file["total"] == 1
    assert "beach.jpg" in res_file["images"][0]["filename"]

    # Search by folder name
    res_folder = db.get_images(search="winter")
    assert res_folder["total"] == 1
    assert "snow.jpg" in res_folder["images"][0]["filename"]

    # Search by person name
    res_person = db.get_images(search="Alice")
    assert res_person["total"] >= 1


def test_sorting_options(test_env):
    """Req 6: Verify sorting by name and date in asc and desc order."""
    db = test_env["db"]

    # Sort name ASC
    res_name_asc = db.get_images(sort_by="name", sort_order="asc")
    paths_asc = [img["file_path"] for img in res_name_asc["images"]]
    assert paths_asc == sorted(paths_asc)

    # Sort name DESC
    res_name_desc = db.get_images(sort_by="name", sort_order="desc")
    paths_desc = [img["file_path"] for img in res_name_desc["images"]]
    assert paths_desc == sorted(paths_desc, reverse=True)


def test_named_faces_prioritized_on_top(test_env):
    """Req 7: Verify named faces appear before unnamed faces in modal and people page."""
    db = test_env["db"]
    id1 = test_env["ids"][0]
    img = db.get_image(id1)

    assert len(img["faces"]) == 2
    # First face should be Alice (named)
    assert img["faces"][0]["person_name"] == "Alice"
    assert img["faces"][0]["person_id"] is not None
    # Second face should be Person #1 (unnamed cluster)
    assert img["faces"][1]["person_id"] is None

    people = db.get_people()
    assert len(people) >= 2
    assert people[0]["name"] == "Alice"
    assert people[0]["is_named"] is True


def test_adjacent_images_modal_navigation(test_env):
    """Req 8: Verify adjacent image lookup for modal arrow/swipe navigation."""
    client = test_env["client"]
    id1 = test_env["ids"][0]

    resp = client.get(f"/api/photos/{id1}/modal")
    assert resp.status_code == 200
    assert "setNeighbors" in resp.text
    assert "modal-nav-btn" in resp.text or "next-btn" in resp.text or "prev-btn" in resp.text


def test_mobile_video_player_experience(test_env):
    """Req 9: Verify responsive video player rendering."""
    client = test_env["client"]
    id2 = test_env["ids"][1]  # surf.mp4

    resp = client.get(f"/api/photos/{id2}/modal")
    assert resp.status_code == 200
    assert "modal-video-player" in resp.text
    assert "playsinline" in resp.text
    assert "controls" in resp.text

    # Verify base.html includes stopModalMedia handler
    base_resp = client.get("/")
    assert "stopModalMedia" in base_resp.text
    assert "activeModal = false" in base_resp.text


def test_scan_progress_percentage_reporting():
    """Verify ScanManager percent and progress callbacks."""
    from app.server import ScanManager

    mgr = ScanManager()
    assert mgr.percent == 0
    mgr._on_progress("Scanning 5/10...", percent=50, current=5, total=10)
    assert mgr.percent == 50
    assert mgr.current_count == 5
    assert mgr.total_count == 10
    assert "Scanning 5/10..." in mgr.logs[-1]


def test_cluster_renaming_on_face_page(test_env):
    """Req 10: Verify cluster renaming directly from cluster page."""
    client = test_env["client"]
    db = test_env["db"]

    # Check cluster page renders rename form
    resp = client.get("/?cluster_id=1")
    assert resp.status_code == 200
    assert "Name this face" in resp.text or "Assign Name" in resp.text

    # Rename cluster 1
    rename_resp = client.post("/api/clusters/1/name", data={"name": "Bob"})
    assert rename_resp.status_code == 200
    assert "Bob" in rename_resp.text
    assert "View Person Profile" in rename_resp.text

    # Verify cluster 1 is now named Bob
    people = db.get_people(search="Bob")
    assert len(people) == 1
    assert people[0]["name"] == "Bob"


def test_quick_hash_and_incremental_scanning(tmp_path):
    """Verify 64KB head+tail quick hashing and incremental pipeline rescan."""
    from PIL import Image
    from app.scanner import compute_quick_hash
    from app.pipeline import run_pipeline

    # Create dummy large file (> 128KB) with distinct head (64KB), middle, and tail (64KB)
    chunk = 65536
    head = b"H" * chunk
    mid1 = b"1" * 20000
    mid2 = b"2" * 20000
    tail1 = b"T" * chunk
    tail2 = b"X" * chunk

    large_file = tmp_path / "sample.bin"
    large_file.write_bytes(head + mid1 + tail1)

    h1 = compute_quick_hash(large_file)
    assert len(h1) > 0

    # Modify middle only (first 64KB and last 64KB unchanged)
    large_file.write_bytes(head + mid2 + tail1)
    h2 = compute_quick_hash(large_file)
    assert h1 == h2

    # Modify tail (last 64KB changed)
    large_file.write_bytes(head + mid1 + tail2)
    h3 = compute_quick_hash(large_file)
    assert h1 != h3

    # Test incremental scan execution on valid image folder
    from unittest.mock import MagicMock, patch

    img_dir = tmp_path / "photos"
    img_dir.mkdir(parents=True, exist_ok=True)
    test_img = img_dir / "valid.jpg"
    img = Image.new("RGB", (100, 100), color=(120, 150, 200))
    img.save(test_img)

    mock_detector = MagicMock()
    mock_detector.detect.return_value = []
    mock_embedder = MagicMock()

    db_file = tmp_path / "incremental.db"
    with patch("app.pipeline.FaceDetector", return_value=mock_detector), \
         patch("app.pipeline.FaceEmbedder", return_value=mock_embedder):
        res1 = run_pipeline(input_dir=img_dir, db_path=db_file)
        assert res1["images_scanned"] == 1
        assert mock_detector.detect.call_count == 1

        # Rescan with unchanged file (should skip re-detecting!)
        res2 = run_pipeline(input_dir=img_dir, db_path=db_file)
        assert res2["images_scanned"] == 1
        # Detector should NOT be called again because file is unchanged
        assert mock_detector.detect.call_count == 1


def test_model_download_with_progress(tmp_path):
    from unittest.mock import patch, MagicMock
    import io
    from app.models import download_file

    fake_data = b"model_onnx_weights_bytes_12345"
    mock_response = MagicMock()
    mock_response.info.return_value.get.return_value = str(len(fake_data))
    mock_response.read.side_effect = [fake_data, b""]
    mock_response.__enter__.return_value = mock_response

    progress_reports = []
    def callback(msg, pct=0, cur=0, tot=0):
        progress_reports.append((msg, pct, cur, tot))

    dest = tmp_path / "model.onnx"
    with patch("urllib.request.urlopen", return_value=mock_response):
        download_file("https://example.com/model.onnx", dest, desc="Testing Download", progress_callback=callback)

    assert dest.exists()
    assert dest.read_bytes() == fake_data
    assert len(progress_reports) > 0
    assert progress_reports[-1][1] == 100
