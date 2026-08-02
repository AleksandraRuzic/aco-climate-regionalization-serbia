"""
Fitness functions for ant-based clustering.

This module intentionally keeps fitness simple and algorithm-neutral.

Current design:
    compactness is always normalized:
        within_cluster_variance / total_variance

    silhouette compactness is converted to a minimization term:
        (1 - mean_silhouette) / 2

    spatial disagreement is:
        fraction of neighbor links whose endpoints have different labels

    final fitness is:
        w_compactness * compactness + w_spatial * spatial

Vegetation is intentionally not included here. Use vegetation only as an
external validation / success metric after clustering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse


def compactness(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Normalized within-cluster compactness.

    Formula:
        compactness = within_cluster_variance / total_variance

    Interpretation:
        lower = more compact clusters

    This is dimensionless and comparable across:
        - ACO vs ACO + refinement
        - different feature sets
        - PCA vs non-PCA
        - pixel vs superpixel mode
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float64)

    if weights is None:
        weights = np.ones(len(X), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != len(X):
        raise ValueError("weights must have the same length as X.")

    global_centroid = np.average(X, axis=0, weights=weights)

    total_var = np.sum(
        weights[:, None] * (X - global_centroid) ** 2
    )

    within_var = np.sum(
        weights * np.sum((X - centroids[labels]) ** 2, axis=1)
    )

    return float(within_var / (total_var + 1e-12))


def spatial_disagreement(
    labels: np.ndarray,
    neighbors: np.ndarray,
    symmetric: bool = False,
) -> float:
    """
    Fraction of neighbor links connecting different labels.

    Interpretation:
        lower = smoother / less fragmented cluster map

    Parameters
    ----------
    symmetric:
        False:
            Treat neighbor graph as directed. This matches kNN output directly.

        True:
            Count each undirected pair only once. Useful if your neighbor graph
            is symmetric or approximately symmetric and you want to avoid
            double-counting.
    """
    labels = np.asarray(labels)
    neighbors = np.asarray(neighbors)

    disagreements = 0.0
    total_links = 0.0

    if not symmetric:
        for i in range(len(labels)):
            neigh = neighbors[i]
            disagreements += np.sum(labels[i] != labels[neigh])
            total_links += len(neigh)

        return float(disagreements / (total_links + 1e-12))

    seen = set()

    for i in range(len(labels)):
        for j in neighbors[i]:
            a, b = sorted((int(i), int(j)))

            if (a, b) in seen:
                continue

            seen.add((a, b))
            disagreements += labels[a] != labels[b]
            total_links += 1

    return float(disagreements / (total_links + 1e-12))


def _deterministic_sample_indices(
    n_nodes: int,
    sample_size: Optional[int],
    use_full_dataset: bool,
    random_state: Optional[int] = 0,
) -> np.ndarray:
    if use_full_dataset or sample_size is None or sample_size >= n_nodes:
        return np.arange(n_nodes, dtype=np.int64)

    if sample_size <= 0:
        raise ValueError("sample_size must be positive when provided.")

    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_nodes, size=sample_size, replace=False))


def silhouette_spatial_term(
    X: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None,
    use_full_dataset: bool = True,
    random_state: Optional[int] = 0,
    min_same_cluster_sample: int = 1,
    full_cluster_fallback: bool = True,
) -> tuple[float, dict]:
    """
    Weighted silhouette term for minimization.

    Silhouette itself is in [-1, 1], where higher is better. The returned term
    maps it to [0, 1], where lower is better:

        silhouette_term = (1 - mean_silhouette) / 2

    Parameters
    ----------
    use_full_dataset:
        True evaluates all nodes. False evaluates a deterministic subset of
        sample_size nodes selected with random_state, while distances may fall
        back to full own-cluster membership for nodes with too few same-cluster
        points in the subset.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if weights is None:
        weights = np.ones(len(X), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if len(labels) != len(X):
        raise ValueError("labels must have the same length as X.")

    if len(weights) != len(X):
        raise ValueError("weights must have the same length as X.")

    if min_same_cluster_sample < 1:
        raise ValueError("min_same_cluster_sample must be at least 1.")

    sample_idx = _deterministic_sample_indices(
        n_nodes=len(X),
        sample_size=sample_size,
        use_full_dataset=use_full_dataset,
        random_state=random_state,
    )
    unique_labels = np.unique(labels)

    silhouettes = []
    silhouette_weights = []
    fallback_count = 0
    skipped_count = 0

    for i in sample_idx:
        label_i = labels[i]
        own_sample_idx = sample_idx[labels[sample_idx] == label_i]
        own_sample_idx = own_sample_idx[own_sample_idx != i]

        if len(own_sample_idx) < min_same_cluster_sample:
            own_full_idx = np.flatnonzero(labels == label_i)
            own_full_idx = own_full_idx[own_full_idx != i]

            if len(own_full_idx) == 0:
                silhouettes.append(0.0)
                silhouette_weights.append(weights[i])
                continue

            if full_cluster_fallback:
                own_idx = own_full_idx
                fallback_count += 1
            else:
                skipped_count += 1
                continue
        else:
            own_idx = own_sample_idx

        own_dist = np.linalg.norm(X[own_idx] - X[i], axis=1)
        own_weight = weights[own_idx]
        own_weight_sum = np.sum(own_weight)

        if own_weight_sum <= 0.0:
            silhouettes.append(0.0)
            silhouette_weights.append(weights[i])
            continue

        a_i = float(np.average(own_dist, weights=own_weight))
        b_i = np.inf

        for other_label in unique_labels:
            if other_label == label_i:
                continue

            other_idx = sample_idx[labels[sample_idx] == other_label]
            if len(other_idx) == 0:
                continue

            other_weight = weights[other_idx]
            other_weight_sum = np.sum(other_weight)
            if other_weight_sum <= 0.0:
                continue

            other_dist = np.linalg.norm(X[other_idx] - X[i], axis=1)
            other_avg = float(np.average(other_dist, weights=other_weight))
            b_i = min(b_i, other_avg)

        if not np.isfinite(b_i):
            silhouettes.append(0.0)
            silhouette_weights.append(weights[i])
            continue

        denom = max(a_i, b_i)
        if denom <= 0.0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom

        silhouettes.append(float(s_i))
        silhouette_weights.append(weights[i])

    if len(silhouettes) == 0 or np.sum(silhouette_weights) <= 0.0:
        mean_silhouette = 0.0
    else:
        mean_silhouette = float(
            np.average(
                np.asarray(silhouettes, dtype=np.float64),
                weights=np.asarray(silhouette_weights, dtype=np.float64),
            )
        )

    silhouette_term = (1.0 - mean_silhouette) / 2.0

    terms = {
        "silhouette": mean_silhouette,
        "silhouette_term": silhouette_term,
        "silhouette_sample_size": len(sample_idx),
        "silhouette_n_evaluated": len(silhouettes),
        "silhouette_fallback_count": fallback_count,
        "silhouette_skipped_count": skipped_count,
        "silhouette_full_dataset": bool(use_full_dataset),
        "silhouette_random_state": random_state,
    }

    return float(silhouette_term), terms


def silhouette_spatial_fitness(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: Optional[np.ndarray],
    neighbors: np.ndarray,
    w_silhouette: float = 0.7,
    w_spatial: float = 0.3,
    spatial_symmetric: bool = False,
    sample_size: Optional[int] = None,
    use_full_dataset: bool = True,
    random_state: Optional[int] = 0,
    min_same_cluster_sample: int = 1,
    full_cluster_fallback: bool = True,
) -> tuple[float, dict]:
    """
    Weighted-sum fitness using silhouette compactness and spatial disagreement.

    This matches fitness_weighted_sum's optimizer-facing signature. Centroids
    are accepted for interchangeability, but silhouette is computed from node
    distances and labels.
    """
    _ = centroids

    silhouette_term, terms = silhouette_spatial_term(
        X=X,
        labels=labels,
        weights=weights,
        sample_size=sample_size,
        use_full_dataset=use_full_dataset,
        random_state=random_state,
        min_same_cluster_sample=min_same_cluster_sample,
        full_cluster_fallback=full_cluster_fallback,
    )

    S = spatial_disagreement(
        labels=labels,
        neighbors=neighbors,
        symmetric=spatial_symmetric,
    )

    fitness = w_silhouette * silhouette_term + w_spatial * S

    terms.update(
        {
            "spatial": S,
        }
    )

    return float(fitness), terms


def snn_cut_term(
    labels: np.ndarray,
    snn_graph: sparse.spmatrix,
) -> tuple[float, dict]:
    """
    Fraction of weighted SNN relationships cut by the clustering.

    Formula:
        snn_cut = external_snn / total_snn

    where internal edges connect nodes with the same label and external edges
    connect nodes with different labels.
    """
    labels = np.asarray(labels, dtype=np.int64)

    if not sparse.issparse(snn_graph):
        snn_graph = sparse.csr_matrix(snn_graph, dtype=np.float64)
    else:
        snn_graph = snn_graph.tocsr().astype(np.float64)

    if snn_graph.shape != (len(labels), len(labels)):
        raise ValueError(
            "snn_graph must have shape (n_nodes, n_nodes), matching labels."
        )

    graph = sparse.triu(snn_graph, k=1).tocoo()

    if graph.nnz == 0:
        terms = {
            "snn_cut": 0.0,
            "internal_snn": 0.0,
            "external_snn": 0.0,
            "total_snn": 0.0,
            "snn_edges": 0,
        }
        return 0.0, terms

    same_cluster = labels[graph.row] == labels[graph.col]
    internal_snn = float(np.sum(graph.data[same_cluster]))
    external_snn = float(np.sum(graph.data[~same_cluster]))
    total_snn = internal_snn + external_snn

    if total_snn <= 0.0:
        snn_cut = 0.0
    else:
        snn_cut = external_snn / total_snn

    terms = {
        "snn_cut": snn_cut,
        "internal_snn": internal_snn,
        "external_snn": external_snn,
        "total_snn": total_snn,
        "snn_edges": int(graph.nnz),
    }

    return float(snn_cut), terms


def snn_spatial_fitness(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: Optional[np.ndarray],
    neighbors: np.ndarray,
    snn_graph: sparse.spmatrix,
    w_snn: float = 0.7,
    w_spatial: float = 0.3,
    spatial_symmetric: bool = False,
) -> tuple[float, dict]:
    """
    Weighted-sum fitness using SNN cut and spatial disagreement.

    This matches the AntClusteringOptimizer fitness signature. X, centroids,
    and weights are accepted for interchangeability with other fitness
    functions; the SNN term itself only needs labels and snn_graph.
    """
    _ = X, centroids, weights

    snn_cut, terms = snn_cut_term(
        labels=labels,
        snn_graph=snn_graph,
    )

    S = spatial_disagreement(
        labels=labels,
        neighbors=neighbors,
        symmetric=spatial_symmetric,
    )

    fitness = w_snn * snn_cut + w_spatial * S
    terms["spatial"] = S

    return float(fitness), terms


def fitness_weighted_sum(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    weights: Optional[np.ndarray],
    neighbors: np.ndarray,
    w_compactness: float = 0.7,
    w_spatial: float = 0.3,
    spatial_symmetric: bool = False,
) -> tuple[float, dict]:
    """
    Weighted-sum clustering fitness.

    Formula:
        fitness = w_compactness * C + w_spatial * S

    where:
        C = normalized compactness
        S = spatial disagreement

    Returns:
        fitness, terms

    The terms dictionary is used for logging and diagnostics.
    """
    C = compactness(
        X=X,
        labels=labels,
        centroids=centroids,
        weights=weights,
    )

    S = spatial_disagreement(
        labels=labels,
        neighbors=neighbors,
        symmetric=spatial_symmetric,
    )

    fitness = w_compactness * C + w_spatial * S

    terms = {
        "compactness": C,
        "spatial": S,
    }

    return float(fitness), terms
