"""
Pheromone initialization and update functions for ant-based clustering.

Shared convention:
    pheromone.shape == (n_nodes, n_clusters)

Meaning:
    pheromone[i, k] = learned preference for assigning node i to cluster k

This convention is suitable for clustering because both ACO variants ultimately
assign each node/pixel/superpixel to one cluster.
"""

from __future__ import annotations

import numpy as np


def initialize_node_cluster_pheromone(
    n_nodes: int,
    n_clusters: int,
    initial_value: float = 1.0,
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Initialize node-cluster pheromone matrix.

    Shape:
        (n_nodes, n_clusters)

    If normalize_rows=True, every row sums to 1.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    pheromone = np.full(
        (n_nodes, n_clusters),
        initial_value,
        dtype=np.float64,
    )

    if normalize_rows:
        pheromone /= pheromone.sum(axis=1, keepdims=True)

    return pheromone


def _validate_pheromone_inputs(
    pheromone: np.ndarray,
    ant_solutions: list,
    n_nodes: int,
) -> None:
    if pheromone.ndim != 2:
        raise ValueError("pheromone must be a 2D array.")

    if pheromone.shape[0] != n_nodes:
        raise ValueError(
            f"pheromone has {pheromone.shape[0]} rows, expected n_nodes={n_nodes}."
        )

    if len(ant_solutions) == 0:
        raise ValueError("ant_solutions must contain at least one solution.")

    for solution in ant_solutions:
        if len(solution) < 2:
            raise ValueError(
                "Each ant solution must contain at least (fitness, labels)."
            )

        labels = np.asarray(solution[1])

        if labels.shape != (n_nodes,):
            raise ValueError(
                f"Solution labels must have shape ({n_nodes},), got {labels.shape}."
            )


def _deposit_solution(
    pheromone: np.ndarray,
    labels: np.ndarray,
    fitness: float,
    q: float,
    n_nodes: int,
) -> None:
    """
    Deposit pheromone for one complete clustering solution.

    For every node i, reinforce the chosen label labels[i].
    """
    labels = np.asarray(labels, dtype=np.int64)

    deposit = q / (fitness + 1e-9)

    pheromone[np.arange(n_nodes), labels] += deposit


def _finalize_pheromone(
    pheromone: np.ndarray,
    clip_min: float,
    clip_max: float,
    normalize_rows: bool,
) -> np.ndarray:
    pheromone = np.clip(pheromone, clip_min, clip_max)

    if normalize_rows:
        row_sums = pheromone.sum(axis=1, keepdims=True)
        pheromone = pheromone / (row_sums + 1e-12)

    return pheromone


def update_pheromone_all_ants(
    pheromone: np.ndarray,
    ant_solutions: list,
    evaporation: float,
    q: float,
    n_nodes: int,
    clip_min: float = 1e-6,
    clip_max: float = 1e6,
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Evaporate pheromone, then let every ant deposit pheromone.

    ant_solutions entries are expected to have:
        solution[0] = fitness
        solution[1] = labels

    Lower fitness deposits more:
        deposit = q / fitness
    """
    pheromone = np.asarray(pheromone, dtype=np.float64).copy()

    _validate_pheromone_inputs(
        pheromone=pheromone,
        ant_solutions=ant_solutions,
        n_nodes=n_nodes,
    )

    pheromone *= 1.0 - evaporation

    for solution in ant_solutions:
        fitness = float(solution[0])
        labels = solution[1]

        _deposit_solution(
            pheromone=pheromone,
            labels=labels,
            fitness=fitness,
            q=q,
            n_nodes=n_nodes,
        )

    return _finalize_pheromone(
        pheromone=pheromone,
        clip_min=clip_min,
        clip_max=clip_max,
        normalize_rows=normalize_rows,
    )


def update_pheromone_best_ant(
    pheromone: np.ndarray,
    ant_solutions: list,
    evaporation: float,
    q: float,
    n_nodes: int,
    clip_min: float = 1e-6,
    clip_max: float = 1e6,
    normalize_rows: bool = True,
) -> np.ndarray:
    """
    Evaporate pheromone, then let only the iteration-best ant deposit pheromone.

    This is more exploitative than update_pheromone_all_ants.
    """
    pheromone = np.asarray(pheromone, dtype=np.float64).copy()

    _validate_pheromone_inputs(
        pheromone=pheromone,
        ant_solutions=ant_solutions,
        n_nodes=n_nodes,
    )

    pheromone *= 1.0 - evaporation

    fitness_values = [float(solution[0]) for solution in ant_solutions]
    best_idx = int(np.argmin(fitness_values))

    best_solution = ant_solutions[best_idx]
    best_fitness = float(best_solution[0])
    best_labels = best_solution[1]

    _deposit_solution(
        pheromone=pheromone,
        labels=best_labels,
        fitness=best_fitness,
        q=q,
        n_nodes=n_nodes,
    )

    return _finalize_pheromone(
        pheromone=pheromone,
        clip_min=clip_min,
        clip_max=clip_max,
        normalize_rows=normalize_rows,
    )


def update_pheromone_quality_weighted(
    pheromone,
    ant_solutions,
    evaporation,
    q,
    n_nodes,
    clip_min=1e-9,
    clip_max=1e9,
    normalize_rows=True,
    eps=1e-12,
):
    """
    Quality-weighted pheromone update.

    Lower fitness is better.

    Deposit is based on how much better each solution is than the
    batch mean:

        quality = (mean_fit - fit) / (mean_fit - best_fit)

    Best solution gets quality close to 1.
    Average-or-worse solutions get quality 0.
    """
    pheromone = np.asarray(pheromone, dtype=np.float64)
    pheromone = pheromone * (1.0 - evaporation)

    fits = np.array(
        [fit for fit, _, _, _ in ant_solutions],
        dtype=np.float64,
    )

    best_fit = np.min(fits)
    mean_fit = np.mean(fits)

    denom = mean_fit - best_fit + eps

    for fit, labels, _, _ in ant_solutions:
        quality = (mean_fit - fit) / denom
        quality = np.power(quality + 1e-12, 2)
        quality = np.clip(quality, 0.0, 1.0)

        deposit = q * quality

        if deposit <= 0:
            continue

        pheromone[np.arange(n_nodes), labels] += deposit

    pheromone = np.clip(pheromone, clip_min, clip_max)

    if normalize_rows:
        row_sum = pheromone.sum(axis=1, keepdims=True)

        pheromone = np.divide(
            pheromone,
            row_sum,
            out=np.full_like(
                pheromone,
                1.0 / pheromone.shape[1],
            ),
            where=row_sum > eps,
        )

    return pheromone