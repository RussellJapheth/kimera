# Kimera: Offline Photo Face-Clustering Tool POC

A lightweight, fully offline Python CLI tool to recursively scan a photo directory, detect human faces, extract facial embeddings, cluster identical individuals together, and persist all detections and metadata into a local SQLite database.

## Key Features

- **100% Offline & Private**: Zero external cloud APIs or web services required.
- **Lightweight Inference**: Powered by ONNX Runtime with CPU fallback and optional CUDA GPU acceleration for systems with $\le$ 4 GB VRAM.
  - **Detector**: YuNet ONNX (~336 KB)
  - **Embedder**: SFace ONNX (~37 MB)
- **Automatic Cosine Clustering**: Groups identical faces into cluster IDs using DBSCAN or Agglomerative clustering on L2-normalized embeddings.
- **Local SQLite Persistence**: Stores image paths, detected face bounding boxes, confidence scores, embeddings (binary BLOB), and assigned cluster/person IDs.
- **Detailed Summaries & Inspection**: Provides cluster breakdowns and lets you inspect any cluster to list its associated image files.

---

## Installation

### 1. Prerequisites
- Python 3.10+

### 2. Setup Virtual Environment
```bash
# Clone or navigate to the repository
cd /path/to/kimera

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Scan and Cluster Photos

Recursively scan an input directory, detect faces, compute embeddings, cluster individuals, and view a summary table:

```bash
# Basic scan
python -m app /path/to/photos

# Or explicitly using the 'scan' subcommand
python -m app scan /path/to/photos

# Customize SQLite database file and clustering thresholds:
python -m app scan /path/to/photos --db face_clusters.db --threshold 0.45 --conf 0.6
```

#### Example Output:
```text
Scanning & detecting faces: 100%|██████████| 12/12 [00:01<00:00,  8.42it/s]

=======================================================
           FACE CLUSTERING SUMMARY
=======================================================
  Images Scanned:       12
  Faces Detected:       18
  Clusters / People:    4
-------------------------------------------------------
Person / Cluster ID      Faces Count    Images Count
---------------------  -------------  --------------
Cluster 0                          8               7
Cluster 1                          5               4
Cluster 2                          3               3
Cluster 3                          2               2
=======================================================
```

---

### 2. Inspect a Person / Cluster

View all images and face bounding boxes associated with a specific person/cluster ID:

```bash
python -m app inspect 0 --db face_clusters.db
```

#### Example Output:
```text
============================================================
              CLUSTER 0 DETAILS
============================================================
  Total Faces:  8
  Total Images: 7
------------------------------------------------------------
  Associated Image Paths:
    • /path/to/photos/vacation/img_001.jpg
    • /path/to/photos/vacation/img_003.jpg
    • /path/to/photos/family/portrait.png
------------------------------------------------------------
  Face Bounding Boxes:
    [Face #1] Conf: 0.98 | BBox: (120, 80, 240, 210) -> img_001.jpg
    [Face #4] Conf: 0.94 | BBox: (300, 150, 420, 290) -> img_003.jpg
============================================================
```

---

## Running Tests

Run the test suite for all non-ML components (scanner, SQLite database layer, embedding clustering):

```bash
pytest
```

---

## Architecture & Project Structure

```
├── app/
│   ├── __init__.py      # Package metadata
│   ├── __main__.py      # python -m app entrypoint
│   ├── cli.py           # Command line interface & summary formatting
│   ├── scanner.py       # Recursive directory traversal & image decoding
│   ├── models.py        # ONNX Runtime FaceDetector & FaceEmbedder
│   ├── clustering.py   # DBSCAN & Agglomerative clustering on cosine metric
│   ├── db.py            # SQLite schema, indices, CRUD & aggregation queries
│   └── pipeline.py      # Complete scan -> detect -> embed -> cluster pipeline
├── tests/
│   ├── test_scanner.py  # Unit tests for image discovery & filtering
│   ├── test_db.py       # Unit tests for SQLite operations & serialization
│   └── test_clustering.py # Unit tests for face clustering logic
├── requirements.txt     # Pinned dependencies
└── README.md
```
