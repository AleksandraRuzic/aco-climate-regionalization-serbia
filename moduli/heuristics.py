"""
Heuristic strategies for ant-based clustering.
This file should contain only heuristics that all strategies can use.

Construction strategies may need small adjustments when switching heuristics,
because not every heuristic naturally uses the same context fields.
"""

from __future__ import annotations

import numpy as np

from .strategy_utils import (
    normalize_probabilities,
    spatial_support,
)

    
def centroid_heuristic(context, params):
    X_i = context["X_i"]
    centroids = context["centroids"]
    n_clusters = context["n_clusters"]
    beta_feature = params.get("beta_feature", 1.0)

    d2 = np.sum((centroids - X_i) ** 2, axis=1)
    feature_score = 1.0 / (d2 + 1e-6)
    zero_rows_mask = np.all(centroids == 0, axis=1)
    feature_score[zero_rows_mask] = 0.85
    feature_score = np.power(feature_score + 1e-12, beta_feature)
    feature_score = normalize_probabilities(feature_score)

    return feature_score
    

def graph_heuristic(context, params):
    X_i = context["X_i"]
    X_neighbors = context["X_neighbors"]
    assigned_neighbor_labels = context["assigned_labels"]
    centroids = context["centroids"]
    n_clusters = context["n_clusters"]
    beta_graph = params.get("beta_graph", 1.0)
    n_neighbours = context["n_neighbours"]

    graph_score = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)
    if len(assigned_neighbor_labels) > 0:

        for k in range(n_clusters):
            mask = assigned_neighbor_labels == k

            if np.any(mask):
                dists = np.linalg.norm(X_neighbors[mask] - X_i, axis=1)
                graph_score[k] = 1.0 / (dists.mean() + 1e-6)

        graph_score = normalize_probabilities(graph_score)

    dists_to_centroids = np.linalg.norm(centroids - X_i, axis=1)
    centroid_score = 1.0 / (dists_to_centroids + 1e-6)
    centroid_score = normalize_probabilities(centroid_score)

    m = len(assigned_neighbor_labels)
    graph_weight = np.sqrt(m / n_neighbours)
    graph_score = graph_weight * graph_score + (1.0 - graph_weight) * centroid_score
    graph_score = np.power(graph_score + 1e-12, beta_graph)
    graph_score = normalize_probabilities(graph_score)
    
    return graph_score


def spatial_heuristic(context, params):
    """
    Spatial-only colony heuristic.

    Higher score for clusters that are already common among assigned neighbors.
    No feature distance. No centroid distance.
    """
    assigned_neighbor_labels = context["assigned_labels"]
    n_clusters = context["n_clusters"]
    beta_spatial = params.get("beta_spatial", 1.5)
    
    if len(assigned_neighbor_labels) == 0:
        score = np.full(n_clusters, 0.75, dtype=np.float64)
    else:
        support = spatial_support(
            assigned_neighbor_labels=assigned_neighbor_labels,
            n_clusters=n_clusters,
        )

        score = 1.0 + support

    score = normalize_probabilities(score)
    score = np.power(score + 1e-12, beta_spatial)

    return score


def graph_cohesion_heuristic(context, params):
    X_i = context["X_i"]
    X = context["X"]
    labels = context["labels"]
    n_clusters = context["n_clusters"]
    rng = context["rng"]
    beta_graph = params.get("beta_graph", 1.0)

    scores = np.full(n_clusters, 1.0/n_clusters, dtype=np.float64)
    sample_size = 40

    for k in range(n_clusters):
        members = np.flatnonzero(labels == k)
        if len(members) > 15:
            if len(members) > sample_size:
                members = rng.choice(members, size=sample_size, replace=False)
            dists = np.linalg.norm(X[members] - X_i, axis=1)
            scores[k] = 1.0 / (dists.mean() + 1e-6)

    scores = normalize_probabilities(scores)
    scores = np.power(scores + 1e-12, beta_graph)
    scores = normalize_probabilities(scores)

    return scores
