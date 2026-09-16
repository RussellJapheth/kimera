# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Russell Japheth
#
# This file is part of Kimera. See the LICENSE file for details.

"""
Face embedding clustering algorithms using cosine distance.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize


def cluster_embeddings(
    embeddings: list[np.ndarray] | np.ndarray,
    eps: float = 0.45,
    min_samples: int = 1,
    algorithm: str = "dbscan",
) -> list[int]:
    """
    Cluster face embeddings using cosine distance.

    Parameters:
        embeddings: List or array of 1D face embedding vectors.
        eps: Distance threshold (cosine distance: 1 - cosine_similarity).
             Typical matching threshold for normalized embeddings is between 0.35 and 0.50.
        min_samples: Minimum samples in a neighborhood for a core point in DBSCAN.
                     If 1, every face will be placed in a cluster (no noise unless isolated).
        algorithm: 'dbscan' or 'agglomerative'

    Returns:
        List of integer cluster IDs corresponding to each input embedding.
        Outliers (if any) are labeled as -1.
    """
    if len(embeddings) == 0:
        return []

    X = np.array(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Normalize vectors to unit length so Euclidean distance directly relates to Cosine distance
    X_norm = normalize(X, norm="l2", axis=1)

    if len(X_norm) == 1:
        # Single face is its own cluster
        return [0]

    if algorithm == "agglomerative":
        # Agglomerative clustering with cosine distance metric and average/complete linkage
        # distance_threshold is eps
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=eps,
        )
        labels = clustering.fit_predict(X_norm)
        return [int(lbl) for lbl in labels]

    # Default: DBSCAN with metric='cosine'
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(X_norm)
    return [int(lbl) for lbl in labels]
