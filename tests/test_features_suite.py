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


def test_settings_persistence(test_env):
    db = test_env["db"]
    client = test_env["client"]

    # Initial get
    assert db.get_setting("eps", "default") == "default"

    # Save settings via endpoint
    resp = client.post("/api/settings/save", data={
        "input_dir": "/tmp/media",
        "conf_threshold": "0.60",
        "eps": "0.38",
        "min_samples": "2",
        "algorithm": "agglomerative",
        "min_interval_sec": "45.0"
    })
    assert resp.status_code == 200
    assert "Settings saved successfully" in resp.text

    # Verify DB persistence
    all_s = db.get_all_settings()
    assert all_s["eps"] == "0.38"
    assert all_s["algorithm"] == "agglomerative"
    assert all_s["conf_threshold"] == "0.6"
    assert all_s["min_samples"] == "2"

    # Verify /settings page loads saved values
    get_resp = client.get("/settings")
    assert get_resp.status_code == 200
    assert 'value="0.38"' in get_resp.text
    assert 'value="0.6"' in get_resp.text or 'value="0.60"' in get_resp.text
    assert 'selected>Agglomerative' in get_resp.text


def test_move_face_and_unlink(test_env):
    db = test_env["db"]
    client = test_env["client"]
    id1 = test_env["ids"][0]

    img = db.get_image(id1)
    face_id = img["faces"][0]["face_id"]

    # 1. Unlink face
    unlink_resp = client.post(f"/api/faces/{face_id}/unlink", data={"image_id": id1})
    assert unlink_resp.status_code == 200
    face_rec = db.get_face(face_id)
    assert face_rec["person_id"] is None
    assert face_rec["cluster_id"] == -1

    # 2. Move face to a new person
    move_new_resp = client.post(f"/api/faces/{face_id}/move", data={
        "image_id": id1,
        "target_type": "new",
        "new_name": "Bob",
    })
    assert move_new_resp.status_code == 200
    face_rec = db.get_face(face_id)
    assert face_rec["person_name"] == "Bob"
    bob_person_id = face_rec["person_id"]

    # 3. Move face to an existing person
    alice_id = test_env["person_id"]
    move_existing_resp = client.post(f"/api/faces/{face_id}/move", data={
        "image_id": id1,
        "target_type": "person",
        "target_id": alice_id,
    })
    assert move_existing_resp.status_code == 200
    face_rec = db.get_face(face_id)
    assert face_rec["person_id"] == alice_id


def test_merge_people_and_clusters(test_env):
    db = test_env["db"]
    client = test_env["client"]

    # Create two people
    p1 = db.name_person("PersonOne")
    p2 = db.name_person("PersonTwo")

    emb = np.random.randn(512).astype(np.float32)
    f1 = db.insert_face(test_env["ids"][0], (0, 0, 10, 10), 0.9, emb, cluster_id=10, person_id=p1)
    f2 = db.insert_face(test_env["ids"][1], (0, 0, 10, 10), 0.9, emb, cluster_id=11, person_id=p2)

    # Merge p1 into p2
    resp = client.post(f"/api/people/{p1}/merge", data={"target_person_id": p2})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == f"/person/{p2}"

    # Verify p1 is deleted and f1 is now assigned to p2
    assert db.get_person(p1) is None
    assert db.get_face(f1)["person_id"] == p2

    # Merge p2 into cluster 99
    resp_cluster_target = client.post(f"/api/people/{p2}/merge", data={"target_type": "cluster", "target_id": 99})
    assert resp_cluster_target.status_code == 200
    assert resp_cluster_target.headers.get("HX-Redirect") == "/?cluster_id=99"
    assert db.get_person(p2) is None
    assert db.get_face(f1)["cluster_id"] == 99
    assert db.get_face(f1)["person_id"] is None

    # Merge cluster 12 into p3
    p3 = db.name_person("Charlie")
    f3 = db.insert_face(test_env["ids"][2], (0, 0, 10, 10), 0.9, emb, cluster_id=12, person_id=None)
    resp_cluster = client.post("/api/clusters/12/merge", data={"target_type": "person", "target_id": p3})
    assert resp_cluster.status_code == 200
    assert db.get_face(f3)["person_id"] == p3


def test_target_picker_endpoint(test_env):
    client = test_env["client"]
    resp = client.get("/api/targets/picker?mode=move_face&source_id=1&image_id=1")
    assert resp.status_code == 200
    assert "Move Face to Person or Cluster" in resp.text
    assert "target-picker-modal" in resp.text

    # Test live-search items endpoint
    resp_items = client.get("/api/targets/items?mode=move_face&source_id=1&image_id=1&search=Alice")
    assert resp_items.status_code == 200
    assert "Alice" in resp_items.text

    # Search with no match
    resp_none = client.get("/api/targets/items?mode=move_face&source_id=1&image_id=1&search=NonExistentPersonXYZ")
    assert resp_none.status_code == 200
    assert "No matching people or clusters found" in resp_none.text


def test_k_medoids_selection():
    from app.recognition import select_k_medoids

    # 1. Empty and small sets
    assert select_k_medoids([]) == []
    single_v = [np.random.randn(512).astype(np.float32)]
    res_single = select_k_medoids(single_v, k=5)
    assert len(res_single) == 1

    # 2. Synthetic cluster of 20 vectors around 3 distinct centers (e.g. frontal, profile, glasses)
    center1 = np.random.randn(512).astype(np.float32)
    center2 = np.random.randn(512).astype(np.float32)
    center3 = np.random.randn(512).astype(np.float32)

    vecs = []
    for _ in range(7):
        vecs.append(center1 + np.random.randn(512) * 0.05)
    for _ in range(7):
        vecs.append(center2 + np.random.randn(512) * 0.05)
    for _ in range(6):
        vecs.append(center3 + np.random.randn(512) * 0.05)

    medoids = select_k_medoids(vecs, k=3)
    assert len(medoids) == 3
    # Ensure vectors are normalized
    for m in medoids:
        assert np.isclose(np.linalg.norm(m), 1.0, atol=1e-4)


def test_multi_exemplar_matcher():
    from app.recognition import MultiExemplarMatcher

    base_alice = np.random.randn(512).astype(np.float32)
    base_bob = np.random.randn(512).astype(np.float32)

    base_alice /= np.linalg.norm(base_alice)
    base_bob /= np.linalg.norm(base_bob)

    exemplars = {
        1: [base_alice],
        2: [base_bob],
    }
    matcher = MultiExemplarMatcher(exemplars)

    # Similar to Alice
    query_alice = base_alice + np.random.randn(512) * 0.02
    match = matcher.match_face(query_alice, threshold=0.38)
    assert match is not None
    assert match[0] == 1

    # Similar to Alice but Alice is in exclusions
    match_excluded = matcher.match_face(query_alice, exclusions={1}, threshold=0.38)
    assert match_excluded is None

    # Batch match
    batch_faces = [
        {"id": 101, "embedding": query_alice},
        {"id": 102, "embedding": base_bob + np.random.randn(512) * 0.02},
        {"id": 103, "embedding": np.random.randn(512)},  # Random face, should not match
    ]
    batch_results = matcher.match_faces_batch(batch_faces, threshold=0.38)
    assert 101 in batch_results and batch_results[101][0] == 1
    assert 102 in batch_results and batch_results[102][0] == 2
    assert 103 not in batch_results


def test_already_named_faces_seeding_and_autotag(test_env):
    db = test_env["db"]
    client = test_env["client"]

    # 1. Existing named person "Diana" with 2 face embeddings
    p_diana = db.name_person("Diana")
    diana_emb = np.random.randn(512).astype(np.float32)
    diana_emb /= np.linalg.norm(diana_emb)

    f_d1 = db.insert_face(test_env["ids"][0], (0, 0, 10, 10), 0.95, diana_emb, cluster_id=1, person_id=p_diana)
    f_d2 = db.insert_face(test_env["ids"][1], (0, 0, 10, 10), 0.95, diana_emb + np.random.randn(512) * 0.01, cluster_id=1, person_id=p_diana)

    # Check that get_all_person_exemplars returns Diana's exemplars
    all_ex = db.get_all_person_exemplars()
    assert p_diana in all_ex
    assert len(all_ex[p_diana]) == 2

    # 2. Insert unassigned face that matches Diana
    diana_variant = diana_emb + np.random.randn(512) * 0.02
    f_unassigned = db.insert_face(test_env["ids"][2], (0, 0, 10, 10), 0.95, diana_variant, cluster_id=-1, person_id=None)

    # Run single-person auto-tag endpoint
    resp_autotag = client.post(f"/api/people/{p_diana}/autotag")
    assert resp_autotag.status_code == 200

    # Face should now be assigned to Diana
    assert db.get_face(f_unassigned)["person_id"] == p_diana

    # 3. Test Unlink + Exclusion
    # If user unlinks f_unassigned, it shouldn't be auto-tagged back to Diana
    db.move_face(f_unassigned, unlink=True)
    assert db.get_face(f_unassigned)["person_id"] is None
    excl = db.get_person_exclusions()
    assert f_unassigned in excl and p_diana in excl[f_unassigned]

    # Re-run auto-tag all: excluded face must NOT be assigned to Diana
    resp_autotag_all = client.post("/api/people/autotag-all")
    assert resp_autotag_all.status_code == 200
    assert db.get_face(f_unassigned)["person_id"] is None


def test_pipeline_preserves_named_and_auto_tags_unassigned(tmp_path, monkeypatch):
    """Verify that run_pipeline uses existing named people exemplars to auto-tag matching faces."""
    from PIL import Image
    from app.pipeline import run_pipeline
    from app.db import Database
    from unittest.mock import MagicMock

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    db_file = tmp_path / "test.db"
    db = Database(db_file)

    # 1. Pre-populate database with an existing named person "Edward"
    p_edward = db.name_person("Edward")
    edward_emb = np.random.randn(512).astype(np.float32)
    edward_emb /= np.linalg.norm(edward_emb)

    # Image 1 is already in library with Edward
    img1_path = media_dir / "edward_photo.jpg"
    Image.new("RGB", (100, 100), color="blue").save(img1_path)
    img1_id = db.insert_image(str(img1_path.resolve()), 100, 100, file_size=img1_path.stat().st_size, mtime=img1_path.stat().st_mtime)
    f_orig = db.insert_face(img1_id, (10, 10, 50, 50), 0.95, edward_emb, cluster_id=1, person_id=p_edward)

    # Image 2 is a new photo with 2 faces: one matching Edward, one stranger
    img2_path = media_dir / "group_photo.jpg"
    Image.new("RGB", (200, 200), color="green").save(img2_path)

    stranger_emb = np.random.randn(512).astype(np.float32)
    stranger_emb /= np.linalg.norm(stranger_emb)

    # Mock FaceDetector and FaceEmbedder so run_pipeline doesn't load heavy ONNX models
    class MockFace:
        def __init__(self, bbox, conf):
            self.bbox = bbox
            self.confidence = conf
            self.raw_face = None

    class MockDetector:
        def __init__(self, **kwargs):
            pass
        def detect(self, img_rgb):
            # Returns 2 faces for group photo
            return [MockFace((10, 10, 50, 50), 0.95), MockFace((60, 60, 100, 100), 0.92)]

    class MockEmbedder:
        def __init__(self, **kwargs):
            self.call_count = 0
        def extract_embedding(self, img_rgb, raw_face):
            self.call_count += 1
            if self.call_count % 2 == 1:
                # Returns vector close to Edward
                return edward_emb + np.random.randn(512).astype(np.float32) * 0.02
            else:
                # Returns stranger vector
                return stranger_emb

    monkeypatch.setattr("app.pipeline.FaceDetector", MockDetector)
    monkeypatch.setattr("app.pipeline.FaceEmbedder", MockEmbedder)

    # Run the pipeline
    stats = run_pipeline(
        input_dir=str(media_dir),
        db_path=str(db_file),
        cache_dir=str(tmp_path / "cache"),
        match_threshold=0.38,
    )

    # Check assertions:
    # 1. Edward's original face is untouched and still assigned to Edward
    assert db.get_face(f_orig)["person_id"] == p_edward

    # 2. Image 2 faces:
    meta_map = db.get_all_images_file_meta_map()
    img2_id = meta_map[str(img2_path.resolve())]["id"]
    img2_record = db.get_image(img2_id)
    assert img2_record is not None
    img2_faces = img2_record["faces"]
    assert len(img2_faces) == 2

    # One face should be auto-assigned to Edward
    edward_faces_in_img2 = [f for f in img2_faces if f["person_id"] == p_edward]
    assert len(edward_faces_in_img2) == 1
    assert edward_faces_in_img2[0]["person_name"] == "Edward"

    # The stranger face should have person_id=None and a valid cluster_id >= 0
    stranger_faces_in_img2 = [f for f in img2_faces if f["person_id"] is None]
    assert len(stranger_faces_in_img2) == 1
    assert stranger_faces_in_img2[0]["cluster_id"] >= 0




