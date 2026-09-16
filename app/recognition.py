"""
Multi-exemplar face recognition engine for semi-supervised person matching.
Uses K-Medoids exemplar selection to maintain diverse representations per person
and fast vectorized cosine distance matching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.preprocessing import normalize


def select_k_medoids(embeddings: np.ndarray | List[np.ndarray], k: int = 5) -> List[np.ndarray]:
    """
    Select up to k diverse, representative medoid embeddings for a person using
    cosine distance diversity (farthest-first traversal + medoid refinement).

    Parameters:
        embeddings: Array of shape (N, D) or list of 1D vectors.
        k: Maximum number of medoids to return.

    Returns:
        List of 1D float32 normalized exemplar embedding vectors.
    """
    if len(embeddings) == 0:
        return []

    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_norm = normalize(X, norm="l2", axis=1)
    n_samples = len(X_norm)

    if n_samples <= k:
        return [vec for vec in X_norm]

    # Compute pairwise cosine distance matrix: D[i, j] = 1 - (u . v)
    sim_matrix = np.clip(np.matmul(X_norm, X_norm.T), -1.0, 1.0)
    dist_matrix = 1.0 - sim_matrix

    # 1. First medoid: point with minimum average distance to all others (central medoid)
    mean_dists = np.mean(dist_matrix, axis=1)
    selected_indices = [int(np.argmin(mean_dists))]

    # 2. Greedily select next k-1 medoids to maximize diversity (farthest point from current set)
    for _ in range(1, k):
        # Distance of each point to its nearest already-selected medoid
        sub_dist = dist_matrix[:, selected_indices]
        min_dist_to_selected = np.min(sub_dist, axis=1)
        # Exclude already selected
        min_dist_to_selected[selected_indices] = -1.0
        next_idx = int(np.argmax(min_dist_to_selected))
        selected_indices.append(next_idx)

    # 3. Refine: For each cluster assigned to a medoid, pick the true cluster medoid
    cluster_assignments = np.argmin(dist_matrix[:, selected_indices], axis=1)
    refined_indices = []
    for m_idx in range(len(selected_indices)):
        members = np.where(cluster_assignments == m_idx)[0]
        if len(members) == 0:
            refined_indices.append(selected_indices[m_idx])
            continue
        sub_matrix = dist_matrix[np.ix_(members, members)]
        best_member_idx = members[np.argmin(np.mean(sub_matrix, axis=1))]
        refined_indices.append(best_member_idx)

    return [X_norm[idx] for idx in refined_indices]


def group_intra_video(
    video_faces: List[Dict[str, Any]],
    threshold: float = 0.30,
) -> List[Dict[str, Any]]:
    """
    Group near-identical faces found within the same video into merge components.

    Faces are linked when their cosine distance is <= threshold. Components are
    reported with their known identities so callers can resolve merges:

        [{"persons": {person_id: count}, "clusters": {cluster_id: count},
          "noise_ids": [face_id, ...]}, ...]

    Named faces count toward "persons", faces already inside an unnamed cluster
    count toward "clusters", and unassigned/unclustered faces are "noise".
    Only components with 2+ faces are returned.
    """
    if not video_faces:
        return []

    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for f in video_faces:
        by_image.setdefault(f["image_id"], []).append(f)

    components: List[Dict[str, Any]] = []

    for image_faces in by_image.values():
        if len(image_faces) < 2:
            continue

        n = len(image_faces)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        E = np.asarray([f["embedding"] for f in image_faces], dtype=np.float32)
        if E.ndim == 1:
            E = E.reshape(1, -1)
        E_norm = normalize(E, norm="l2", axis=1)
        sim_matrix = np.clip(np.matmul(E_norm, E_norm.T), -1.0, 1.0)
        dist_matrix = 1.0 - sim_matrix

        for i in range(n):
            for j in range(i + 1, n):
                if float(dist_matrix[i, j]) <= threshold:
                    union(i, j)

        comps: Dict[int, Dict[str, Any]] = {}
        for idx, f in enumerate(image_faces):
            root = find(idx)
            if root not in comps:
                comps[root] = {"persons": {}, "clusters": {}, "noise_ids": []}
            comp = comps[root]
            if f.get("person_id") is not None:
                pid = int(f["person_id"])
                comp["persons"][pid] = comp["persons"].get(pid, 0) + 1
                comp["noise_ids"].append(f["id"])
            elif f.get("cluster_id", -1) >= 0:
                cid = int(f["cluster_id"])
                comp["clusters"][cid] = comp["clusters"].get(cid, 0) + 1
                comp["noise_ids"].append(f["id"])
            else:
                comp["noise_ids"].append(f["id"])

        for root, comp in comps.items():
            count = len(comp["noise_ids"])
            if count >= 2:
                components.append(comp)

    return components


class MultiExemplarMatcher:
    """
    Fast matrix-based multi-exemplar matcher for face recognition.
    Matches unknown query faces against verified person exemplar galleries.
    """

    def __init__(self, person_exemplars: Optional[Dict[int, List[np.ndarray]]] = None):
        self.person_ids: List[int] = []
        self.exemplar_matrix: Optional[np.ndarray] = None
        self.exemplar_person_map: np.ndarray = np.array([], dtype=np.int64)
        if person_exemplars:
            self.fit(person_exemplars)

    def fit(self, person_exemplars: Dict[int, List[np.ndarray]]) -> None:
        """
        Build normalized exemplar search index from a dictionary of {person_id: [exemplar_vecs]}.
        """
        all_vecs: List[np.ndarray] = []
        p_ids: List[int] = []

        for person_id, exemplars in person_exemplars.items():
            for vec in exemplars:
                all_vecs.append(vec)
                p_ids.append(int(person_id))

        if not all_vecs:
            self.person_ids = []
            self.exemplar_matrix = None
            self.exemplar_person_map = np.array([], dtype=np.int64)
            return

        X = np.asarray(all_vecs, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.exemplar_matrix = normalize(X, norm="l2", axis=1)
        self.exemplar_person_map = np.asarray(p_ids, dtype=np.int64)
        self.person_ids = sorted(list(person_exemplars.keys()))

    @property
    def is_empty(self) -> bool:
        return self.exemplar_matrix is None or len(self.exemplar_matrix) == 0

    def match_face(
        self,
        embedding: np.ndarray,
        exclusions: Optional[Set[int]] = None,
        threshold: float = 0.42,
    ) -> Optional[Tuple[int, float]]:
        """
        Match a single face embedding against all active exemplars.

        Parameters:
            embedding: 1D face embedding vector.
            exclusions: Set of person_ids that this face cannot be linked to.
            threshold: Cosine distance threshold (default: 0.42).

        Returns:
            (person_id, distance) if a match meeting threshold is found, else None.
        """
        if self.is_empty or self.exemplar_matrix is None:
            return None

        v = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        v_norm = normalize(v, norm="l2", axis=1)

        sims = np.clip(np.matmul(self.exemplar_matrix, v_norm.T).ravel(), -1.0, 1.0)
        dists = 1.0 - sims

        if exclusions:
            mask = np.isin(self.exemplar_person_map, list(exclusions))
            dists[mask] = 999.0

        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist <= threshold:
            matched_person_id = int(self.exemplar_person_map[best_idx])
            return matched_person_id, best_dist

        return None

    def match_faces_batch(
        self,
        faces: List[Dict[str, Any]],
        exclusions: Optional[Dict[int, Set[int]]] = None,
        threshold: float = 0.42,
    ) -> Dict[int, Tuple[int, float]]:
        """
        Match a batch of unassigned face records against exemplars using vectorized operations.

        Parameters:
            faces: List of face dicts containing 'id' and 'embedding'.
            exclusions: Mapping of {face_id: set_of_excluded_person_ids}.
            threshold: Maximum allowed cosine distance for a positive match.

        Returns:
            Dictionary of {face_id: (matched_person_id, cosine_distance)}.
        """
        if not faces or self.is_empty or self.exemplar_matrix is None:
            return {}

        face_ids = [f["id"] for f in faces]
        embeddings = [f["embedding"] for f in faces]

        Q = np.asarray(embeddings, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Q_norm = normalize(Q, norm="l2", axis=1)

        # Batch cosine similarities: (N_faces, M_exemplars)
        sim_matrix = np.clip(np.matmul(Q_norm, self.exemplar_matrix.T), -1.0, 1.0)
        dist_matrix = 1.0 - sim_matrix

        results: Dict[int, Tuple[int, float]] = {}

        for i, face_id in enumerate(face_ids):
            row_dists = dist_matrix[i].copy()
            if exclusions and face_id in exclusions:
                exc_set = exclusions[face_id]
                mask = np.isin(self.exemplar_person_map, list(exc_set))
                row_dists[mask] = 999.0

            best_idx = int(np.argmin(row_dists))
            best_dist = float(row_dists[best_idx])

            if best_dist <= threshold:
                matched_person_id = int(self.exemplar_person_map[best_idx])
                results[face_id] = (matched_person_id, best_dist)

        return results

    def find_uncertain_candidates(
        self,
        person_id: int,
        faces: List[Dict[str, Any]],
        exclusions: Optional[Dict[int, Set[int]]] = None,
        min_dist: float = 0.38,
        max_dist: float = 0.60,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find candidate unassigned faces within the uncertainty distance band [min_dist, max_dist]
        for a specific person, sorted by closest match first, capped at limit.
        """
        if not faces or self.is_empty or self.exemplar_matrix is None:
            return []

        # Filter exemplars belonging to target person
        p_mask = (self.exemplar_person_map == int(person_id))
        if not np.any(p_mask):
            return []

        p_exemplars = self.exemplar_matrix[p_mask]

        face_ids = [f["id"] for f in faces]
        embeddings = [f["embedding"] for f in faces]

        Q = np.asarray(embeddings, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Q_norm = normalize(Q, norm="l2", axis=1)

        sim_matrix = np.clip(np.matmul(Q_norm, p_exemplars.T), -1.0, 1.0)
        dist_matrix = 1.0 - sim_matrix

        candidates: List[Dict[str, Any]] = []

        for i, f in enumerate(faces):
            face_id = f["id"]
            if exclusions and face_id in exclusions and person_id in exclusions[face_id]:
                continue

            min_dist_val = float(np.min(dist_matrix[i]))
            if min_dist <= min_dist_val <= max_dist:
                sim_pct = max(0, min(100, int(round((1.0 - min_dist_val) * 100))))
                cand = dict(f)
                cand["distance"] = round(min_dist_val, 4)
                cand["similarity_pct"] = sim_pct
                candidates.append(cand)

        # Sort by best match (lowest distance) first
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:limit]

    def find_uncertain_candidates_all(
        self,
        faces: List[Dict[str, Any]],
        exclusions: Optional[Dict[int, Set[int]]] = None,
        min_dist: float = 0.38,
        max_dist: float = 0.60,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find candidate unassigned faces within uncertainty distance band [min_dist, max_dist]
        across all active person exemplars, sorted by closest match first, capped at limit.
        """
        if not faces or self.is_empty or self.exemplar_matrix is None:
            return []

        face_ids = [f["id"] for f in faces]
        embeddings = [f["embedding"] for f in faces]

        Q = np.asarray(embeddings, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        Q_norm = normalize(Q, norm="l2", axis=1)

        sim_matrix = np.clip(np.matmul(Q_norm, self.exemplar_matrix.T), -1.0, 1.0)
        dist_matrix = 1.0 - sim_matrix

        candidates: List[Dict[str, Any]] = []

        for i, f in enumerate(faces):
            face_id = f["id"]
            row_dists = dist_matrix[i].copy()
            if exclusions and face_id in exclusions:
                exc_set = exclusions[face_id]
                mask = np.isin(self.exemplar_person_map, list(exc_set))
                row_dists[mask] = 999.0

            best_idx = int(np.argmin(row_dists))
            best_dist = float(row_dists[best_idx])

            if min_dist <= best_dist <= max_dist:
                matched_person_id = int(self.exemplar_person_map[best_idx])
                sim_pct = max(0, min(100, int(round((1.0 - best_dist) * 100))))
                cand = dict(f)
                cand["target_person_id"] = matched_person_id
                cand["distance"] = round(best_dist, 4)
                cand["similarity_pct"] = sim_pct
                candidates.append(cand)

        candidates.sort(key=lambda x: x["distance"])
        return candidates[:limit]
