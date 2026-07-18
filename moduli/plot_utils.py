
"""
Plotting utilities for comparing ACO, ACO + refinement, and KMeans results.

These functions work for both pixel-level and superpixel-level mode,
as long as df contains a node mapping column such as:
    - "pixel_id"
    - "superpixel_compact"
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def labels_to_raster(
    df,
    node_labels,
    node_col: str,
    row_col: str = "row",
    col_col: str = "col",
):
    n_rows = int(df[row_col].max()) + 1
    n_cols = int(df[col_col].max()) + 1

    raster = np.full((n_rows, n_cols), np.nan)

    rows = df[row_col].to_numpy(dtype=int)
    cols = df[col_col].to_numpy(dtype=int)
    node_ids = df[node_col].to_numpy(dtype=int)

    raster[rows, cols] = np.asarray(node_labels)[node_ids]

    return raster


def plot_cluster_comparison(
    df,
    label_dict: dict,
    node_col: str,
    n_clusters: int = 5,
    colors=None,
    figsize_per_plot=(7, 7),
    title: str = "Cluster comparison",
    row_col: str = "row",
    col_col: str = "col",
    save_path=None,
    dpi: int = 300,
):
    """
    Plot several cluster maps side by side.

    Example:
        plot_cluster_comparison(
            df=data["df"],
            label_dict={
                "ACO": aco.labels_,
                "ACO + refinement": aco_refinement.labels_,
                "KMeans": kmeans.labels_,
            },
            node_col=data["node_col"],
            n_clusters=5,
        )
    """
    if colors is None:
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    if len(colors) < n_clusters:
        raise ValueError("Provide at least n_clusters colors.")

    cmap = ListedColormap(colors[:n_clusters])
    cmap.set_bad(color="white")

    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), cmap.N)

    n_plots = len(label_dict)

    fig, axes = plt.subplots(
        1,
        n_plots + 1,
        figsize=(figsize_per_plot[0] * n_plots + 1.0, figsize_per_plot[1]),
        gridspec_kw={"width_ratios": [1] * n_plots + [0.04]},
    )

    plot_axes = axes[:-1]
    cax = axes[-1]

    first_img = None

    for ax, (name, labels) in zip(plot_axes, label_dict.items()):
        raster = labels_to_raster(
            df=df,
            node_labels=labels,
            node_col=node_col,
            row_col=row_col,
            col_col=col_col,
        )

        img = ax.imshow(
            raster,
            cmap=cmap,
            norm=norm,
            origin="upper",
            interpolation="nearest",
        )

        if first_img is None:
            first_img = img

        ax.set_title(name)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_aspect("equal")

    cbar = fig.colorbar(first_img, cax=cax, ticks=np.arange(n_clusters))
    cbar.set_label("Cluster")

    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.05)
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def plot_single_cluster_map(
    df,
    node_labels,
    node_col: str,
    n_clusters: int = 5,
    title: str = "Clusters",
    colors=None,
    figsize=(10, 10),
    row_col: str = "row",
    col_col: str = "col",
):
    if colors is None:
        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

    cmap = ListedColormap(colors[:n_clusters])
    cmap.set_bad(color="white")
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5, 1), cmap.N)

    raster = labels_to_raster(
        df=df,
        node_labels=node_labels,
        node_col=node_col,
        row_col=row_col,
        col_col=col_col,
    )

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(raster, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    cbar = fig.colorbar(img, ax=ax, ticks=np.arange(n_clusters), shrink=0.8)
    cbar.set_label("Cluster")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_aspect("equal")

    plt.show()
