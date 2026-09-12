"""
Tests for directory and image scanning logic.
"""

from pathlib import Path
import pytest
from app.scanner import is_image_file, scan_image_paths


def test_is_image_file():
    assert is_image_file("photo.jpg") is True
    assert is_image_file("photo.JPEG") is True
    assert is_image_file("photo.png") is True
    assert is_image_file("photo.webp") is True
    assert is_image_file("photo.bmp") is True
    assert is_image_file("photo.txt") is False
    assert is_image_file("document.pdf") is False
    assert is_image_file(".DS_Store") is False


def test_scan_image_paths_recursive(tmp_path: Path):
    # Setup test directory tree
    sub1 = tmp_path / "folder1"
    sub2 = tmp_path / "folder1" / "nested"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    img1 = tmp_path / "img1.png"
    img2 = sub1 / "img2.jpg"
    img3 = sub2 / "img3.WEBP"
    txt = sub1 / "readme.txt"

    img1.write_bytes(b"dummy")
    img2.write_bytes(b"dummy")
    img3.write_bytes(b"dummy")
    txt.write_text("dummy")

    scanned = scan_image_paths(tmp_path)
    scanned_filenames = [p.name for p in scanned]

    assert len(scanned) == 3
    assert "img1.png" in scanned_filenames
    assert "img2.jpg" in scanned_filenames
    assert "img3.WEBP" in scanned_filenames
    assert "readme.txt" not in scanned_filenames


def test_scan_image_paths_invalid_path():
    with pytest.raises(FileNotFoundError):
        scan_image_paths("/non/existent/path/for/sure/12345")
