"""
Construction strategies for ant-based clustering.

This file should contain only solution-construction strategies.

Each strategy must match the signature expected by AntClusteringOptimizer:

    labels, centroids = construct_solution_fn(
        X=X,
        pheromone=pheromone,
        neighbors=neighbors,
        weights=weights,
        rng=rng,
        n_clusters=n_clusters,
        **construct_kwargs,
    )
"""

from __future__ import annotations


import numpy as np

from .strategy_utils import (
    normalize_probabilities,
)
from .cluster_centroid_manipulation import (
    compute_centroids,
    initialize_centroids
)
from .heuristics import (
    centroid_heuristic,
    graph_heuristic,
    graph_cohesion_heuristic,
    spatial_heuristic
)

    
def construct_aco_solution(
    X: np.ndarray,
    pheromone: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    n_clusters: int,
    ant_idx: int,
    n_ants: int,
    iteration: int,
    neighbors: np.ndarray = None,
    alpha: float = 1.0,
    beta_feature: float = 1.0,
    centroid: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plain ACO-style constructive clustering.

    One ant builds one complete label assignment progressively.

    Probability for assigning node i to cluster k is based on:
        pheromone[i, k]^alpha
        similarity to the temporary centroid of cluster k

    This strategy does not use refinement steps.
    """
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_nodes, n_features = X.shape

    labels = np.full(n_nodes, -1, dtype=np.int64)

    cluster_sums = np.zeros((n_clusters, n_features), dtype=np.float64)
    cluster_weights = np.zeros(n_clusters, dtype=np.float64)
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)

    order = rng.permutation(n_nodes)

    for i in order:
        X_i = X[i]
        pheromone_i = pheromone[i]
        
        tau = normalize_probabilities(pheromone_i)
        tau_score = np.power(tau + 1e-12, alpha)

        context = {
            "X_i": X[i],
            "X": X,
            "centroids": centroids,
            "n_clusters": n_clusters,
            "labels": labels,
            "rng": rng,
        }

        if centroid:
            heur = centroid_heuristic(
                context,
                {
                    "beta_feature": beta_feature,
                },
            )
        else:
            heur = graph_cohesion_heuristic(
                context,
                {
                    "beta_graph": beta_feature,
                }
            )

        probs = tau_score * heur
        probs = normalize_probabilities(probs)

        selected = rng.choice(n_clusters, p=probs)
        labels[i] = selected

        wi = weights[i]
        cluster_sums[selected] += wi * X_i
        cluster_weights[selected] += wi
        centroids[selected] = cluster_sums[selected]/cluster_weights[selected]

    centroids = compute_centroids(
        X=X,
        labels=labels,
        n_clusters=n_clusters,
        weights=weights,
        rng=rng,
    )

    return labels, centroids


def construct_aco_refinement_solution(
    X: np.ndarray,
    pheromone: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    n_clusters: int,
    ant_idx: int,
    n_ants: int,
    iteration: int,
    neighbors: np.ndarray = None,
    alpha: float = 0.8,
    beta_feature: float = 1.0,
    refinement_steps: int = 3,
    centroid: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    ACO + refinement-style construction.

    One ant builds a complete clustering by repeatedly assigning nodes to
    competing cluster/colony centroids.

    Probability for assigning node i to cluster k is based on:
        pheromone[i, k]^alpha
        either feature distance to centroid k or global cluster cohesion,
        depending on the centroid flag.

    Refinement:
        labels -> centroids -> labels -> centroids
    """
    X = np.asarray(X, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_nodes = len(X)

    labels = np.full(n_nodes, -1, dtype=np.int64)
    n_nodes, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    
    for step in range(refinement_steps):
        order = rng.permutation(n_nodes)

        new_labels = labels.copy()

        for i in order:
        
            context = {
                "X_i": X[i],
                "X": X,
                "centroids": centroids,
                "n_clusters": n_clusters,
                "labels": new_labels,
                "rng": rng,
            }
            
            if centroid:
                heur = centroid_heuristic(
                    context,
                    {
                        "beta_feature": beta_feature,
                    },
                )
            else:
                heur = graph_cohesion_heuristic(
                    context,
                    {
                        "beta_graph": beta_feature,
                    }
                )

            tau = normalize_probabilities(pheromone[i])
            tau_score = np.power(tau + 1e-12, alpha)

            probs = tau_score * heur
            probs = normalize_probabilities(probs)

            new_labels[i] = rng.choice(n_clusters, p=probs)

        labels = new_labels

        centroids = compute_centroids(
            X=X,
            labels=labels,
            n_clusters=n_clusters,
            weights=weights,
            rng=rng,
        )

    return labels, centroids


def construct_graph_aco_solution(
    X,
    pheromone,
    neighbors,
    weights,
    rng,
    n_clusters,
    ant_idx: int,
    n_ants: int,
    iteration: int,
    alpha=1.0,
    beta_graph=2.0,
):
    X = np.asarray(X, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    n_nodes = len(X)
    labels = np.full(n_nodes, -1, dtype=np.int64)

    order = rng.permutation(n_nodes)
    centroids = initialize_centroids(
        X=X,
        n_clusters=n_clusters,
        rng=rng,
        base_centroids=None,
        base_prob=0.0,
        noise_scale=0.0,
    )
    for i in order:
        neigh = neighbors[i]
        n_neighbours = neigh.shape[0]
        neigh_labels = labels[neigh]

        valid = neigh_labels >= 0

        assigned_nodes = neigh[valid]
        assigned_labels = neigh_labels[valid]

        tau = normalize_probabilities(pheromone[i])
        tau_score = np.power(tau + 1e-12, alpha)

        context = {
            "X_i": X[i],
            "X_neighbors": X[assigned_nodes],
            "assigned_labels": assigned_labels,
            "centroids": centroids,
            "n_clusters": n_clusters,
            "n_neighbours": n_neighbours,
        }
        
        heuristic = graph_heuristic(
            context,
            {
                "beta_graph": beta_graph,
            },
        )

        probs = tau_score * heuristic
        probs = normalize_probabilities(probs)

        labels[i] = rng.choice(n_clusters, p=probs)

    centroids = compute_centroids(
        X=X,
        labels=labels,
        n_clusters=n_clusters,
        weights=weights,
        rng=rng,
    )

    return labels, centroids


def construct_multicolony_solution(
    X,
    pheromone,
    neighbors,
    weights,
    rng,
    n_clusters,
    ant_idx=0,
    n_ants=0,
    iteration=0,
    colony_heuristics=None,
    alpha=1.0,
):
    if colony_heuristics is None:
        colony_heuristics = [
        {
            "name": "centroid",
            "fn": centroid_heuristic,
            "params": {
                "beta_feature": 1.0,
            },
        },
        {
            "name": "graph",
            "fn": graph_heuristic,
            "params": {
                "beta_graph": 0.8,
            },
        },
        {
            "name": "spatial",
            "fn": spatial_heuristic,
            "params": {
                "beta_spatial": 1.5,
            },
        },
    ]
    n_colonies = len(colony_heuristics)
    ants_per_colony = int(np.ceil(n_ants / n_colonies))
    colony_id = min(ant_idx // ants_per_colony, n_colonies - 1)
    colony_heuristic = colony_heuristics[colony_id]

    labels = np.full(len(X), -1, dtype=np.int64)
    centroids = initialize_centroids(
        X=X,
        n_clusters=n_clusters,
        rng=rng,
        base_centroids=None,
        base_prob=0.0,
        noise_scale=0.0,
    )
    mutation_num = 0
    for i in rng.permutation(len(X)):
        neigh = neighbors[i]
        assigned_labels = labels[neigh]

        valid = assigned_labels >= 0
        assigned_labels = assigned_labels[valid]
        assigned_nodes = neigh[valid]
        n_neighbours = neigh.shape[0]

        tau = normalize_probabilities(pheromone[i])
        tau_score = np.power(tau + 1e-12, alpha)

        context = {
            "X": X,
            "X_i": X[i],
            "labels": labels,
            "neighbors_i": neigh,
            "assigned_nodes": assigned_nodes,
            "assigned_labels": assigned_labels,
            "X_neighbors": X[assigned_nodes],
            "centroids": centroids,
            "n_clusters": n_clusters,
            "n_neighbours": n_neighbours,
            "rng": rng,
        }
        heuristic = colony_heuristic["fn"](context, colony_heuristic.get("params", {}))
        heuristic = normalize_probabilities(heuristic)

        probs = normalize_probabilities(tau_score * heuristic)
        if rng.random() < 0.01 and ant_idx % 3 == 0 and mutation_num < 0.02*len(X):
            selected = rng.integers(n_clusters)
            mutation_num += 1
        else:
            selected = rng.choice(n_clusters, p=probs)
        labels[i] = selected

    centroids = compute_centroids(X, labels, n_clusters, weights, rng)
    return labels, centroids
