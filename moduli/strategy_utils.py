"""
Utility functions used by construction strategies.

This module is intentionally focused on solution-construction helpers only.
It should not contain fitness functions, pheromone updates, or solution
post-processing utilities.
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np


def normalize_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.clip(scores, 0.0, None)

    if scores.sum() <= 1e-15:
        return np.full(len(scores), 1 / len(scores))

    probs = scores / scores.sum()

    probs[-1] = max(0.0, 1.0 - probs[:-1].sum())

    return probs / probs.sum()


def spatial_support(
    assigned_neighbor_labels,
    n_clusters,
):
    """
    Compute local neighbor support for each cluster.

    Returns:
        support[k] = fraction of assigned neighbors that belong to cluster k

    Example:
        assigned_neighbor_labels = [2, 2, 2, 3]
        n_clusters = 5

        support = [0.00, 0.00, 0.75, 0.25, 0.00]
    """
    assigned = np.asarray(assigned_neighbor_labels)
    assigned = assigned[assigned >= 0]

    if len(assigned) == 0:
        return np.zeros(n_clusters, dtype=np.float64)

    counts = np.bincount(
        assigned.astype(np.int64),
        minlength=n_clusters,
    ).astype(np.float64)

    support = counts / len(assigned)

    return support


def graph_feature_score(
    X_i,
    X_neighbors,
    assigned_neighbor_labels,
    n_clusters,
    empty_score=1.0,
):
    """
    Feature similarity score based only on already assigned neighbors.

    For each cluster k:
        score[k] = inverse mean feature distance from X_i to neighbors
                   that are already assigned to k

    If no assigned neighbor belongs to k, use empty_score.
    """
    scores = np.full(n_clusters, empty_score, dtype=np.float64)

    assigned_neighbor_labels = np.asarray(assigned_neighbor_labels)

    for k in range(n_clusters):
        mask = assigned_neighbor_labels == k

        if np.any(mask):
            dists = np.linalg.norm(X_neighbors[mask] - X_i, axis=1)
            scores[k] = 1.0 / (dists.mean() + 1e-6)

    return normalize_probabilities(scores)