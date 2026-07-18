"""
Generic ant-based clustering optimizer.

This module intentionally does not implement ACO or ACO + refinement solution construction.
Instead, it provides a reusable optimization engine that receives:

    construct_solution_fn
    fitness_fn
    pheromone_update_fn
    pheromone_init_fn
    ant_idx=ant_idx
    iteration=iteration

This lets you compare ACO, ACO + refinement, and future variants fairly by changing only
the construction strategy while keeping the same fitness/update loop.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def default_pheromone_init(
    n_nodes: int,
    n_clusters: int,
    normalized: bool = True,
    initial_value: float = 1.0,
) -> np.ndarray:
    """
    Default pheromone initialization.

    Shape:
        (n_nodes, n_clusters)

    Meaning:
        pheromone[i, k] = preference for assigning node i to cluster k
    """
    pheromone = np.full(
        (n_nodes, n_clusters),
        initial_value,
        dtype=np.float64,
    )

    if normalized:
        pheromone /= pheromone.sum(axis=1, keepdims=True)

    return pheromone


class AntClusteringOptimizer:
    """
    Generic optimization engine for ant-based clustering.

    The optimizer controls:
        - outer iteration loop
        - ant loop
        - fitness evaluation
        - pheromone update
        - best solution tracking
        - logging/history

    It does NOT control:
        - how a single ant constructs a clustering solution

    That is supplied through construct_solution_fn.

    Expected construct_solution_fn signature:

        labels, centroids = construct_solution_fn(
            X=X,
            pheromone=pheromone,
            neighbors=neighbors,
            weights=weights,
            rng=rng,
            n_clusters=n_clusters,
            ant_idx=ant_idx,
            iteration=iteration,
            **construct_kwargs,
        )

    Expected fitness_fn signature:

        fit, terms = fitness_fn(
            X=X,
            labels=labels,
            centroids=centroids,
            weights=weights,
            neighbors=neighbors,
            **fitness_kwargs,
        )

    Expected pheromone_update_fn signature:

        pheromone = pheromone_update_fn(
            pheromone=pheromone,
            ant_solutions=ant_solutions,
            evaporation=evaporation,
            q=q,
            n_nodes=n_nodes,
            **pheromone_update_kwargs,
        )

    ant_solutions entries have the structure:

        (fit, labels, centroids, terms)
    """

    def __init__(
        self,
        n_clusters: int,
        n_ants: int,
        n_iterations: int,
        construct_solution_fn: Callable,
        fitness_fn: Callable,
        pheromone_update_fn: Callable,
        evaporation: float = 0.3,
        q: float = 1.0,
        random_state: Optional[int] = 0,
        verbose: bool = True,
        pheromone_init_fn: Optional[Callable] = None,
        pheromone_init_kwargs: Optional[dict] = None,
        construct_kwargs: Optional[dict] = None,
        fitness_kwargs: Optional[dict] = None,
        pheromone_update_kwargs: Optional[dict] = None,
        solution_postprocess_fn: Optional[Callable] = None,
        solution_postprocess_kwargs: Optional[dict] = None,
        pheromone_update_batch_size: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.n_ants = n_ants
        self.n_iterations = n_iterations

        self.construct_solution_fn = construct_solution_fn
        self.fitness_fn = fitness_fn
        self.pheromone_update_fn = pheromone_update_fn
        self.pheromone_init_fn = pheromone_init_fn or default_pheromone_init

        self.evaporation = evaporation
        self.q = q
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        self.pheromone_init_kwargs = (
            pheromone_init_kwargs.copy()
            if pheromone_init_kwargs is not None
            else {}
        )

        self.construct_kwargs = (
            construct_kwargs.copy()
            if construct_kwargs is not None
            else {}
        )

        self.fitness_kwargs = (
            fitness_kwargs.copy()
            if fitness_kwargs is not None
            else {}
        )

        self.pheromone_update_kwargs = (
            pheromone_update_kwargs.copy()
            if pheromone_update_kwargs is not None
            else {}
        )

        self.solution_postprocess_fn = solution_postprocess_fn
        self.solution_postprocess_kwargs = (
            solution_postprocess_kwargs.copy()
            if solution_postprocess_kwargs is not None
            else {}
        )

        self.labels_ = None
        self.centroids_ = None
        self.best_fitness_ = None
        self.history_ = []
        self.term_history_ = []
        self.pheromone_ = None
        self.pheromone_update_batch_size = pheromone_update_batch_size

        self._validate_constructor()

    def _validate_constructor(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if self.n_ants <= 0:
            raise ValueError("n_ants must be positive.")

        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")

        if not callable(self.construct_solution_fn):
            raise TypeError("construct_solution_fn must be callable.")

        if not callable(self.fitness_fn):
            raise TypeError("fitness_fn must be callable.")

        if not callable(self.pheromone_update_fn):
            raise TypeError("pheromone_update_fn must be callable.")

        if not callable(self.pheromone_init_fn):
            raise TypeError("pheromone_init_fn must be callable.")

        if self.solution_postprocess_fn is not None and not callable(self.solution_postprocess_fn):
            raise TypeError("solution_postprocess_fn must be callable if provided.")

        if self.pheromone_update_batch_size is not None and self.pheromone_update_batch_size <= 0:
            raise ValueError("pheromone_update_batch_size must be positive or None.")

    def _validate_fit_inputs(
        self,
        X: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_nodes, n_features).")

        n_nodes = len(X)

        if len(weights) != n_nodes:
            raise ValueError("weights must have the same length as X.")

        if neighbors.ndim != 2:
            raise ValueError("neighbors must be a 2D array with shape (n_nodes, k_neighbors).")

        if neighbors.shape[0] != n_nodes:
            raise ValueError("neighbors must have the same number of rows as X.")

        if np.any(neighbors < 0) or np.any(neighbors >= n_nodes):
            raise ValueError("neighbors contains invalid node indices.")

    def _postprocess_solution(
        self,
        labels: np.ndarray,
        centroids: np.ndarray,
        X: np.ndarray,
        neighbors: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Optional hook for label alignment or solution repair.

        Example use:
            align ACO + refinement labels to KMeans base_centroids.

        If no solution_postprocess_fn is supplied, solution is returned unchanged.
        """
        if self.solution_postprocess_fn is None:
            return labels, centroids

        return self.solution_postprocess_fn(
            labels=labels,
            centroids=centroids,
            X=X,
            neighbors=neighbors,
            weights=weights,
            rng=self.rng,
            n_clusters=self.n_clusters,
            **self.solution_postprocess_kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        neighbors: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "AntClusteringOptimizer":
        """
        Run the ant-based clustering optimization.
        """
        X = np.asarray(X, dtype=np.float64)
        neighbors = np.asarray(neighbors, dtype=np.int64)

        n_nodes = len(X)

        if weights is None:
            weights = np.ones(n_nodes, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)

        self._validate_fit_inputs(X, neighbors, weights)

        pheromone = self.pheromone_init_fn(
            n_nodes=n_nodes,
            n_clusters=self.n_clusters,
            **self.pheromone_init_kwargs,
        )

        best_labels = None
        best_centroids = None
        best_fitness = np.inf

        history = []
        term_history = []
        batch_size = self.pheromone_update_batch_size or self.n_ants
        for iteration in range(self.n_iterations):
            ant_solutions = []
            iteration_solutions = []

            for ant_idx in range(self.n_ants):
                labels, centroids = self.construct_solution_fn(
                    X=X,
                    pheromone=pheromone,
                    neighbors=neighbors,
                    weights=weights,
                    rng=self.rng,
                    n_clusters=self.n_clusters,
                    ant_idx=ant_idx,
                    n_ants=self.n_ants,
                    iteration=iteration,
                    **self.construct_kwargs,
                )

                labels = np.asarray(labels, dtype=np.int64)
                centroids = np.asarray(centroids, dtype=np.float64)

                if labels.shape != (n_nodes,):
                    raise ValueError(
                        "construct_solution_fn must return labels with shape "
                        f"({n_nodes},), got {labels.shape}."
                    )

                if centroids.shape != (self.n_clusters, X.shape[1]):
                    raise ValueError(
                        "construct_solution_fn must return centroids with shape "
                        f"({self.n_clusters}, {X.shape[1]}), got {centroids.shape}."
                    )

                labels, centroids = self._postprocess_solution(
                    labels=labels,
                    centroids=centroids,
                    X=X,
                    neighbors=neighbors,
                    weights=weights,
                )

                fit, terms = self.fitness_fn(
                    X=X,
                    labels=labels,
                    centroids=centroids,
                    weights=weights,
                    neighbors=neighbors,
                    **self.fitness_kwargs,
                )

                if terms is None:
                    terms = {}

                solution = (fit, labels, centroids, terms)
                ant_solutions.append(solution)
                iteration_solutions.append(solution)
                
                should_update = (
                    len(ant_solutions) == batch_size
                    or ant_idx == self.n_ants - 1
                )
                if should_update:
                    pheromone = self.pheromone_update_fn(
                        pheromone=pheromone,
                        ant_solutions=ant_solutions,
                        evaporation=self.evaporation,
                        q=self.q,
                        n_nodes=n_nodes,
                        **self.pheromone_update_kwargs,
                    )
                    ant_solutions = []

                if fit < best_fitness:
                    best_fitness = float(fit)
                    best_labels = labels.copy()
                    best_centroids = centroids.copy()


            history.append(best_fitness)

            fits = np.array([fit for fit, _, _, _ in iteration_solutions], dtype=np.float64)
            best_idx = int(np.argmin(fits))
            iter_best_fit, _, _, iter_best_terms = iteration_solutions[best_idx]

            term_record = {
                "iteration": iteration + 1,
                "iteration_best": float(iter_best_fit),
                "global_best": float(best_fitness),
                "mean": float(fits.mean()),
                "std": float(fits.std()),
            }

            if pheromone is not None:
                term_record["pheromone_min"] = float(np.min(pheromone))
                term_record["pheromone_max"] = float(np.max(pheromone))

            term_record.update(
                {
                    key: float(value)
                    for key, value in iter_best_terms.items()
                    if np.isscalar(value)
                }
            )

            term_history.append(term_record)

            if self.verbose:
                parts = [
                    f"Iter {iteration + 1}/{self.n_iterations}",
                    f"iter_best={fits.min():.4f}",
                    f"iter_mean={fits.mean():.4f}",
                    f"iter_std={fits.std():.4f}",
                    f"global_best={best_fitness:.4f}",
                ]

                for key, value in iter_best_terms.items():
                    if np.isscalar(value):
                        parts.append(f"{key}={float(value):.4f}")

                if pheromone is not None:
                    parts.append(f"pheromone_min={np.min(pheromone):.4e}")
                    parts.append(f"pheromone_max={np.max(pheromone):.4e}")

                print(" | ".join(parts))

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.best_fitness_ = best_fitness
        self.history_ = history
        self.term_history_ = term_history
        self.pheromone_ = pheromone

        return self

    def fit_predict(
        self,
        X: np.ndarray,
        neighbors: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.fit(X, neighbors, weights)
        return self.labels_
