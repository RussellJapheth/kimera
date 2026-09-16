# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Tests for embedding clustering logic.
"""

import numpy as np
from app.clustering import cluster_embeddings


def test_empty_clustering():
    labels = cluster_embeddings([])
    assert labels == []


def test_single_embedding():
    emb = np.random.randn(128).astype(np.float32)
    labels = cluster_embeddings([emb])
    assert labels == [0]


def test_distinct_clusters_dbscan():
    # Create 2 synthetic clusters of 128D embeddings
    rng = np.random.RandomState(42)

    # Base person A and person B vectors (orthogonal / far apart)
    person_a = np.zeros(128, dtype=np.float32)
    person_a[0] = 1.0

    person_b = np.zeros(128, dtype=np.float32)
    person_b[1] = 1.0

    # Slight perturbations around A and B
    faces_a = [person_a + rng.normal(0, 0.05, 128).astype(np.float32) for _ in range(3)]
    faces_b = [person_b + rng.normal(0, 0.05, 128).astype(np.float32) for _ in range(3)]

    all_faces = faces_a + faces_b
    labels = cluster_embeddings(all_faces, eps=0.3, min_samples=1, algorithm="dbscan")

    assert len(labels) == 6
    # First 3 should share the same cluster ID
    assert labels[0] == labels[1] == labels[2]
    # Next 3 should share the same cluster ID
    assert labels[3] == labels[4] == labels[5]
    # Cluster A should be different from Cluster B
    assert labels[0] != labels[3]


def test_agglomerative_clustering():
    rng = np.random.RandomState(123)
    p1 = np.zeros(128, dtype=np.float32)
    p1[10] = 1.0

    p2 = np.zeros(128, dtype=np.float32)
    p2[20] = 1.0

    faces = [
        p1 + rng.normal(0, 0.02, 128).astype(np.float32),
        p1 + rng.normal(0, 0.02, 128).astype(np.float32),
        p2 + rng.normal(0, 0.02, 128).astype(np.float32),
    ]

    labels = cluster_embeddings(faces, eps=0.3, algorithm="agglomerative")
    assert len(labels) == 3
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
