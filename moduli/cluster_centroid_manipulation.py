"""
Centroid functions used for strategies with scores based on the distance from cluster center.
"""

from __future__ import annotations

from typing import Optional
from .strategy_utils import normalize_probabilities
import numpy as np


def compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    weights: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute weighted cluster centroids.

    Empty clusters are reinitialized with a random data point.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if weights is None:
        weights = np.ones(len(X), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if len(weights) != len(X):
        raise ValueError("weights must have the same length as X.")

    if rng is None:
        rng = np.random.default_rng()

    centroids = np.zeros((n_clusters, X.shape[1]), dtype=np.float64)

    for k in range(n_clusters):
        mask = labels == k

        if np.any(mask):
            centroids[k] = np.average(
                X[mask],
                axis=0,
                weights=weights[mask],
            )
        else:
            idx = rng.integers(len(X))
            centroids[k] = X[idx]

    return centroids


def kmeanspp_centroids(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    KMeans++ style centroid initialization.

    This chooses initial centroids that are spread out in feature space.

    Important:
        This is only initialization. It does not run KMeans optimization.
    """
    X = np.asarray(X, dtype=np.float64)

    n = len(X)

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive.")

    if n < n_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} cannot be larger than number of samples={n}."
        )

    centroids = np.empty((n_clusters, X.shape[1]), dtype=np.float64)

    # First centroid: random point.
    idx = rng.integers(n)
    centroids[0] = X[idx]

    # Remaining centroids: sample proportional to squared distance
    # from nearest selected centroid.
    for k in range(1, n_clusters):
        d2 = np.min(
            np.sum(
                (X[:, None, :] - centroids[:k][None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )

        total = d2.sum()

        if not np.isfinite(total) or total <= 1e-15:
            idx = rng.integers(n)
        else:
            probs = normalize_probabilities(d2 / total)
            idx = rng.choice(n, p=probs)

        centroids[k] = X[idx]

    return centroids


def initialize_centroids(
    X: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    base_centroids: Optional[np.ndarray] = None,
    base_prob: float = 0.0,
    noise_scale: float = 0.05,
) -> np.ndarray:
    """
    Unified centroid initialization.

    Behavior:
        base_prob = 0.0:
            always use KMeans++ initialization

        base_prob = 1.0:
            always use base_centroids + noise, if base_centroids is provided

        0.0 < base_prob < 1.0:
            probabilistic mixture:
                with probability base_prob, use base_centroids + noise
                otherwise use KMeans++ initialization

    If base_centroids is None, this always falls back to KMeans++.
    """
    X = np.asarray(X, dtype=np.float64)

    if not 0.0 <= base_prob <= 1.0:
        raise ValueError("base_prob must be in [0, 1].")

    use_base = (
        base_centroids is not None
        and rng.random() < base_prob
    )

    if use_base:
        base_centroids = np.asarray(base_centroids, dtype=np.float64)

        expected_shape = (n_clusters, X.shape[1])
        if base_centroids.shape != expected_shape:
            raise ValueError(
                f"base_centroids must have shape {expected_shape}, "
                f"got {base_centroids.shape}."
            )

        noise = rng.normal(
            scale=noise_scale,
            size=base_centroids.shape,
        )

        return base_centroids + noise

    return kmeanspp_centroids(
        X=X,
        n_clusters=n_clusters,
        rng=rng,
    )

