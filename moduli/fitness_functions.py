"""
Fitness functions for ant-based clustering.

This module intentionally keeps fitness simple and algorithm-neutral.

Current design:
    compactness is always normalized:
        within_cluster_variance / total_variance

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
