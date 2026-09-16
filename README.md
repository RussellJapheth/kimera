<p align="center">
  <img src="app/static/logo.svg" alt="Kimera logo" width="120">
</p>

<h1 align="center">Kimera</h1>

<p align="center">
  Fully offline, self-hosted media gallery with state-of-the-art face recognition and clustering.
  <br>
  Google Photos–style people organization — no cloud, no telemetry, 100% yours.
</p>

<p align="center">
  <strong>Python 3.10+</strong> · <strong>InsightFace / ArcFace</strong> · <strong>CLIP visual embeddings</strong> · <strong>FastAPI + HTMX</strong>
</p>

## Overview

Kimera is a self-contained local media library and face-clustering application. Point it at a folder of photos and videos and it:

- recursively indexes and thumbnails the library (WebP cache with `ffmpeg` fallback),
- detects every face and computes 512-dimensional ArcFace embeddings,
- clusters faces into unnamed identity groups and watches for the same person across angles, lighting, photos, and videos,
- and gives you a polished web gallery to review, name, merge, tag, and search everything.

Your media and your database never leave your machine. All model inference (face detection, face embedding, visual embedding) runs locally through ONNX Runtime.

## Key Features

### Face recognition & clustering
- **State-of-the-art models** — InsightFace `buffalo_l`: SCRFD-10G face detector + ArcFace ResNet-50 producing 512-d embeddings. Automatic CPU/GPU selection with YuNet/SFace fallback.
- **Multi-exemplar person matching** — named people are matched against up to five medoid reference faces per identity, so a person is still recognized across significant pose/lighting variation.
- **Multi-algorithm clustering** — Agglomerative (default), DBSCAN, Chinese Whispers, or HDBSCAN on a cosine distance metric.
- **Cluster-reference matching** — unclustered faces also match against existing *unnamed* clusters, not only named people, and merge into them.
- **Intra-video merge** — near-identical faces detected in the same video are linked so cutaways and lighting shifts don't split one person into duplicate identities.
- **Review workflows** — interactive "Is this person X?" review with yes/no/skip keyboard shortcuts, per-person scoring, and library-wide uncertain-match sweeps.
- **Auto-tagging** — match any unassigned library faces against all named people in one pass.

### Media gallery
- **Twilight Slate theme** — glassmorphism dark UI with custom CSS, no CDN dependencies (Alpine.js and HTMX vendored locally).
- **Photos & videos** — responsive grid, cached WebP thumbnails, inline video playback, lightbox viewer with face-detection overlays and EXIF metadata.
- **People pages** — Google Photos–style face bubbles, avatar crops, inline naming, search, and person merging.
- **Organize** — hierarchical folders (copy/move/batch-rename), one-click favorites, bulk select/action toolbar.
- **Search & tags** — file/folder/person search, manual tag picker, automatic tag suggestions, and visual similarity search powered by CLIP vector embeddings.
- **Duplicate detection** — content-fingerprint scan with optional duplicate exclusion from the gallery.

### Local persistence
- **SQLite** stores image metadata, dimensions, favorites, tags, face bounding boxes, binary embeddings, cluster assignments, and named people.
- **Content-addressed cache** — rescanning reuses fingerprints and thumbnails; only changed files are re-read.

## Technology

| Concern          | Choice                                                              |
| ---------------- | ------------------------------------------------------------------- |
| Language         | Python 3.10+                                                        |
| Web framework    | FastAPI + Uvicorn, Jinja2 templates, HTMX + Alpine.js (vendored)    |
| Face detection   | InsightFace SCRFD-10G (`buffalo_l`), ONNX Runtime, YuNet fallback   |
| Face embedding   | ArcFace ResNet-50, 512-d, L2-normalized                             |
| Visual embedding | Quantized CLIP ViT-B/32 (~89 MB, ONNX)                              |
| Clustering       | scikit-learn / agglomerative · dbscan · chinese_whispers · hdbscan  |
| Storage          | SQLite (binary face/visual embeddings)                              |
| Media            | OpenCV + Pillow, WebP cache, FFmpeg video keyframe sampling         |
| Tests            | pytest                                                              |

## Installation

```bash
# Prerequisite: Python 3.10+
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Face-detection model weights auto-download on first use. For NVIDIA GPU acceleration, install `onnxruntime-gpu==1.18.0` instead of the default CPU runtime (already noted in `requirements.txt`).

## Quick Start

### Web gallery (recommended)

```bash
# Index a directory, then open the gallery
python -m app serve /path/to/photos

# Launch with an existing database
python -m app serve --port 8000

# Force a rescan before serving
python -m app serve /path/to/photos --rescan
```

Open http://127.0.0.1:8000. Start on the **Settings** page: set your media directory, tune thresholds, and trigger a scan. Rescans run in the background with live progress on the same page.

### CLI scan & cluster

```bash
# Recursive scan, face detection, clustering, and summary table
python -m app scan /path/to/photos

# Tune the database path and clustering strictness
python -m app scan /path/to/photos --db face_clusters.db --threshold 0.43 --conf 0.60
```

### Inspect a person / cluster

```bash
python -m app inspect 0 --db face_clusters.db
```

## CLI Reference

| Subcommand | Purpose                                        |
| ---------- | ---------------------------------------------- |
| `serve`    | Launch the local web gallery (optionally indexing) |
| `scan`     | Scan, detect, embed, and cluster a directory   |
| `inspect`  | Print details for a person/cluster ID          |

### `scan` options

| Flag                       | Default | Description                                                                 |
| -------------------------- | ------- | --------------------------------------------------------------------------- |
| `--db`                     | `face_clusters.db` | SQLite database path                                        |
| `--threshold` (eps)        | `0.43`  | Cosine distance threshold for clustering (lower = stricter)                |
| `--conf`                   | `0.60`  | Face detection confidence threshold                                        |
| `--algo`                   | `agglomerative` | Clustering algorithm (`dbscan`, `agglomerative`, `chinese_whispers`, `hdbscan`) |
| `--export`                 | (none)  | Directory to export cropped face cutouts grouped by person                 |
| `--scene-thresh`           | `0.35`  | Video scene-cut threshold (0.0 – 1.0)                                      |
| `--min-interval`           | `60.0`  | Minimum seconds between video keyframe samples                             |
| `--max-interval`           | `90.0`  | Maximum seconds between video keyframe samples                             |
| `--no-cluster-refs`        | off     | Disable matching new faces against unnamed clusters                        |
| `--intra-merge-threshold`  | `0.30`  | Cosine distance for intra-video face merge (lower = stricter)             |

## Configuration

All settings are editable in the web UI (**Settings** page) and persisted in SQLite:

| Setting                            | Default | Meaning                                                        |
| ---------------------------------- | ------- | -------------------------------------------------------------- |
| Detection confidence               | `0.60`  | Face detector threshold                                        |
| Clustering epsilon                 | `0.43`  | Distance threshold for grouping faces into identities          |
| Min cluster size                   | `1`     | Minimum faces to form a cluster                                |
| Clustering method                  | agglomerative | Embedding grouping algorithm                            |
| Person recognition threshold       | `0.42`  | Multi-exemplar match threshold for named people                |
| Intra-video merge threshold        | `0.30`  | Face-merge strictness within a single video                    |
| Match against unnamed clusters     | on      | Let new faces merge into existing unnamed clusters             |
| Visual similarity threshold        | `0.60`  | Minimum cosine similarity for visual search / tag suggestions  |
| Video keyframe interval            | `60.0s` | Minimum seconds between sampled video frames (max `90.0s`)     |
| Hide low-quality faces             | off     | Hide small/low-confidence unassigned faces on People page      |
| Exclude duplicates from gallery    | off     | Show one file per duplicate group                              |
| Cache directory                    | `./.cache` | Thumbnail and face-crop storage                             |

## Project Structure

```
├── app/
│   ├── __init__.py      # Package metadata
│   ├── __main__.py      # python -m app entrypoint
│   ├── cli.py           # Command-line interface & summary formatting
│   ├── scanner.py       # Recursive media discovery, decoding, video keyframes
│   ├── models.py        # ONNX face detector/embedder, CLIP visual embedder
│   ├── recognition.py   # Multi-exemplar matching, intra-video grouping
│   ├── clustering.py    # Agglomerative / DBSCAN / Chinese Whispers / HDBSCAN
│   ├── pipeline.py      # scan → detect → embed → cluster → merge pipeline
│   ├── cache.py         # Content-addressed WebP thumbnail & face-crop cache
│   ├── db.py            # SQLite schema, indices, CRUD & aggregation
│   ├── server.py        # FastAPI web app & API routes
│   ├── static/          # CSS, vendored JS, logo
│   └── templates/       # Jinja2 templates (base, gallery, people, settings…)
├── tests/               # pytest suite (non-ML and API tests)
├── start.sh.example     # Launcher template (copy to start.sh, set MEDIA_DIR)
├── requirements.txt     # Pinned dependencies
└── README.md
```

## Running Tests

```bash
pytest
```

## Privacy

Kimera is fully offline: no external CDNs, no cloud APIs, no telemetry. The web UI binds to `127.0.0.1` by default (`--host 0.0.0.0` to expose on your LAN). Model weights are fetched once at first use; everything after that runs locally.

## License

[GNU AGPL-3.0](LICENSE) — copyright Russell Japheth. Free for personal and research use: you may use, modify, and share it, but any modified version running as a public-facing service must make its source code available to its users.

**Commercial use requires a separate commercial license.** To use Kimera commercially, or to ship a closed-source modified version, email [license@riveady.com.ng](mailto:license@riveady.com.ng) for licensing terms.