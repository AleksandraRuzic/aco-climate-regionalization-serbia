"""
Shared nearest-neighbor graph construction.

The SNN graph connects candidate node pairs whose feature-space nearest-neighbor
sets overlap. Candidate pairs can come from the spatial neighbor graph, which
keeps the graph sparse and aligned with raster adjacency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _candidate_pairs_from_neighbors(neighbors: np.ndarray) -> set[tuple[int, int]]:
    neighbors = np.asarray(neighbors, dtype=np.int64)

    if neighbors.ndim != 2:
        raise ValueError("spatial_neighbors must be a 2D array.")

    pairs = set()
    n_nodes = neighbors.shape[0]

    for i in range(n_nodes):
        for j in neighbors[i]:
            j = int(j)
            if j < 0 or j >= n_nodes:
                raise ValueError("spatial_neighbors contains invalid node indices.")
            if i == j:
                continue

            a, b = sorted((int(i), j))
            pairs.add((a, b))

    return pairs


def _candidate_pairs_from_feature_neighbors(feature_neighbors: np.ndarray) -> set[tuple[int, int]]:
    pairs = set()

    for i in range(feature_neighbors.shape[0]):
        for j in feature_neighbors[i]:
            if i == j:
                continue

            a, b = sorted((int(i), int(j)))
            pairs.add((a, b))

    return pairs


def build_snn_graph(
    X: np.ndarray,
    k: int = 20,
    min_shared_neighbors: int = 1,
    spatial_neighbors: Optional[np.ndarray] = None,
) -> sparse.csr_matrix:
    """
    Build a sparse weighted shared nearest-neighbor graph.

    Steps:
        1. Find each node's k nearest neighbors in feature space using X.
        2. For candidate pairs, compute shared-neighbor overlap:
               shared(i, j) = |N(i) intersection N(j)|
               snn(i, j) = shared(i, j) / k
        3. Return a symmetric sparse graph with SNN weights.

    Parameters
    ----------
    spatial_neighbors:
        Optional spatial neighbor graph used as candidate pairs. If omitted,
        feature-nearest-neighbor pairs are used as a sparse fallback.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_nodes, n_features).")

    if k <= 0:
        raise ValueError("k must be positive.")

    if min_shared_neighbors < 0:
        raise ValueError("min_shared_neighbors must be non-negative.")

    n_nodes = len(X)
    if n_nodes <= 1:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    k_eff = min(k, n_nodes - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
    nbrs.fit(X)
    feature_neighbors = nbrs.kneighbors(X, return_distance=False)

    cleaned_neighbors = np.empty((n_nodes, k_eff), dtype=np.int64)
    for i in range(n_nodes):
        row = feature_neighbors[i]
        row = row[row != i]
        cleaned_neighbors[i] = row[:k_eff]

    neighbor_sets = [set(row.tolist()) for row in cleaned_neighbors]

    if spatial_neighbors is None:
        candidate_pairs = _candidate_pairs_from_feature_neighbors(cleaned_neighbors)
    else:
        candidate_pairs = _candidate_pairs_from_neighbors(spatial_neighbors)

    rows = []
    cols = []
    data = []

    for i, j in candidate_pairs:
        shared = len(neighbor_sets[i].intersection(neighbor_sets[j]))
        if shared < min_shared_neighbors:
            continue

        weight = shared / k_eff
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])

    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_nodes, n_nodes),
        dtype=np.float64,
    )
