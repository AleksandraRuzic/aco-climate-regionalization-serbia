"""
Vegetation / land-cover homogeneity evaluation for clustering results.

Expected vegetation CSV columns:
    row, col, easting, northing, clc_class

Typical usage:

    import pandas as pd
    from vegetation_eval import (
        load_clc_legend,
        attach_node_labels_to_pixels,
        merge_clusters_with_vegetation,
        add_clc_metadata,
        vegetation_quality_report,
        plot_landcover_groups,
        plot_cluster_landcover,
    )

    veg = pd.read_csv("vegetation.csv")
    legend = load_clc_legend("CLC2018_CLC2018_V2018_20_QGIS.txt")

    # If you already have df with one row per pixel and a cluster column:
    merged = merge_clusters_with_vegetation(df_clusters, veg)

    # Or if you have node labels from ACO, ACO + refinement, or KMeans:
    df_clusters = attach_node_labels_to_pixels(
        df=data["df"],
        node_labels=aco_refinement.labels_,
        node_col=data["node_col"],
        cluster_col="cluster",
    )
    merged = merge_clusters_with_vegetation(df_clusters, veg)

    merged = add_clc_metadata(merged, legend=legend)

    report = vegetation_quality_report(
        merged,
        cluster_col="cluster",
        class_col="clc_group",
    )

    print(report["quality"])
    print(report["overall"])
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


DEFAULT_GROUP_ORDER = [
    "Artificial",
    "Agriculture",
    "Broadleaf_forest",
    "Coniferous_forest",
    "Mixed_forest",
    "Shrubland",
    "Grassland_Bare",
    "Wetlands",
    "Water",
    "NODATA",
    "Unknown",
]

DEFAULT_GROUP_COLORS = {
    "Artificial": "black",
    "Agriculture": "orange",
    "Broadleaf_forest": "blue",
    "Coniferous_forest": "purple",
    "Mixed_forest": "green",
    "Shrubland": "red",
    "Grassland_Bare": "white",
    "Wetlands": "#80cdc1",
    "Water": "#4575b4",
    "NODATA": "lightgray",
    "Unknown": "gray",
}

ARTIFICIAL_GROUPS = {"Artificial"}
AGRICULTURE_GROUPS = {"Agriculture"}

NATURAL_GROUPS = {
    "Broadleaf_forest",
    "Coniferous_forest",
    "Mixed_forest",
    "Shrubland",
    "Grassland_Bare",
    "Wetlands",
    "Water",
}

EXCLUDED_GROUPS = {"Artificial", "NODATA", "Unknown"}

def load_clc_legend(path: str) -> pd.DataFrame:
    """
    Load a CORINE Land Cover legend file.

    Expected txt format:
        code,r,g,b,a,name

    Returns columns:
        clc_class, r, g, b, a, clc_name, color
    """
    legend = pd.read_csv(
        path,
        header=None,
        names=["clc_class", "r", "g", "b", "a", "clc_name"],
    )

    legend["clc_class"] = legend["clc_class"].astype(int)

    legend["color"] = legend.apply(
        lambda x: "#{:02x}{:02x}{:02x}".format(
            int(x["r"]),
            int(x["g"]),
            int(x["b"]),
        ),
        axis=1,
    )

    return legend


def clc_group_from_code(code) -> str:
    """
    Convert CLC code to broader vegetation/land-cover group.

    Supports both:
    - original CORINE codes: 111, 112, ..., 523, 999
    - sequential codes used in some rasters/notebooks: 1, 2, ..., 44

    The sequential-code mapping follows the user's original vegetation notebook:
        1-11   Artificial
        12-22  Agriculture
        23     Broadleaf_forest
        24     Coniferous_forest
        25     Mixed_forest
        26-29  Shrubland
        30-34  Grassland_Bare
        35-39  Wetlands
        40-44  Water
    """
    if pd.isna(code):
        return "Unknown"

    code = int(code)

    if code in (0, 999):
        return "NODATA"

    # Sequential/reclassified codes from the original notebook.
    if 1 <= code <= 44:
        if 1 <= code <= 11:
            return "Artificial"
        if 12 <= code <= 22:
            return "Agriculture"
        if code == 23:
            return "Broadleaf_forest"
        if code == 24:
            return "Coniferous_forest"
        if code == 25:
            return "Mixed_forest"
        if 26 <= code <= 29:
            return "Shrubland"
        if 30 <= code <= 34:
            return "Grassland_Bare"
        if 35 <= code <= 39:
            return "Wetlands"
        if 40 <= code <= 44:
            return "Water"

    # Original CORINE Land Cover codes.
    if 111 <= code <= 142:
        return "Artificial"

    if 211 <= code <= 244:
        return "Agriculture"

    if code == 311:
        return "Broadleaf_forest"

    if code == 312:
        return "Coniferous_forest"

    if code == 313:
        return "Mixed_forest"

    if 321 <= code <= 324:
        return "Shrubland"

    if 331 <= code <= 335:
        return "Grassland_Bare"

    if 411 <= code <= 423:
        return "Wetlands"

    if 511 <= code <= 523:
        return "Water"

    return "Unknown"


def attach_node_labels_to_pixels(
    df: pd.DataFrame,
    node_labels,
    node_col: str,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Attach node-level cluster labels to a pixel dataframe.

    Works for:
        node_col="pixel_id"
        node_col="superpixel_compact"
    """
    out = df.copy()
    labels = np.asarray(node_labels)

    node_ids = out[node_col].to_numpy(dtype=int)
    out[cluster_col] = labels[node_ids]

    return out


def merge_clusters_with_vegetation(
    clusters_df: pd.DataFrame,
    vegetation_df: pd.DataFrame,
    cluster_col: str = "cluster",
    row_col: str = "row",
    col_col: str = "col",
    class_col: str = "clc_class",
    nodata_values: Iterable[int] = (0, 999),
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge pixel-level cluster assignments with vegetation / CLC classes.
    """
    needed_cluster_cols = [row_col, col_col, cluster_col]
    missing_cluster = [c for c in needed_cluster_cols if c not in clusters_df.columns]
    if missing_cluster:
        raise ValueError(f"clusters_df is missing columns: {missing_cluster}")

    needed_veg_cols = [row_col, col_col, class_col]
    missing_veg = [c for c in needed_veg_cols if c not in vegetation_df.columns]
    if missing_veg:
        raise ValueError(f"vegetation_df is missing columns: {missing_veg}")

    merged = clusters_df[needed_cluster_cols].merge(
        vegetation_df[[row_col, col_col, class_col]],
        on=[row_col, col_col],
        how=how,
    )

    if nodata_values is not None:
        merged = merged[~merged[class_col].isin(nodata_values)].copy()

    return merged


def add_clc_metadata(
    df: pd.DataFrame,
    legend: Optional[pd.DataFrame] = None,
    class_col: str = "clc_class",
) -> pd.DataFrame:
    """
    Add clc_group and, if a legend is supplied, clc_name/color columns.
    """
    out = df.copy()

    out["clc_group"] = out[class_col].apply(clc_group_from_code)

    if legend is not None:
        legend_cols = ["clc_class", "clc_name", "color"]
        legend_small = legend[legend_cols].copy()

        out = out.merge(
            legend_small,
            left_on=class_col,
            right_on="clc_class",
            how="left",
            suffixes=("", "_legend"),
        )

        # If both merge keys exist with suffixes in edge cases, keep the original class column.
        if "clc_class_legend" in out.columns:
            out = out.drop(columns=["clc_class_legend"])

    return out


def vegetation_counts(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    class_col: str = "clc_group",
) -> pd.DataFrame:
    """
    Count land-cover classes inside each cluster.
    """
    counts = (
        df.groupby([cluster_col, class_col], dropna=False)
        .size()
        .reset_index(name="count")
    )

    counts["cluster_total"] = counts.groupby(cluster_col)["count"].transform("sum")
    counts["proportion"] = counts["count"] / counts["cluster_total"]

    return counts


def vegetation_quality_metrics(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    class_col: str = "clc_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute vegetation homogeneity metrics per cluster.

    Lower entropy / Simpson diversity = more homogeneous.
    Higher dominant_class_share = more homogeneous.

    Returns:
        quality, counts
    """
    counts = vegetation_counts(
        df,
        cluster_col=cluster_col,
        class_col=class_col,
    )

    entropy = (
        counts.groupby(cluster_col)
        .apply(lambda x: -np.sum(x["proportion"] * np.log(x["proportion"] + 1e-12)))
        .reset_index(name="shannon_entropy")
    )

    n_classes = (
        counts.groupby(cluster_col)[class_col]
        .nunique()
        .reset_index(name="n_classes")
    )

    entropy = entropy.merge(n_classes, on=cluster_col)

    entropy["normalized_entropy"] = np.where(
        entropy["n_classes"] > 1,
        entropy["shannon_entropy"] / np.log(entropy["n_classes"]),
        0.0,
    )

    simpson = (
        counts.groupby(cluster_col)
        .apply(lambda x: 1.0 - np.sum(x["proportion"] ** 2))
        .reset_index(name="simpson_diversity")
    )

    dominant = (
        counts.sort_values([cluster_col, "proportion"], ascending=[True, False])
        .groupby(cluster_col)
        .first()
        .reset_index()
        .rename(
            columns={
                class_col: f"dominant_{class_col}",
                "proportion": "dominant_class_share",
            }
        )
    )

    quality = (
        entropy
        .merge(simpson, on=cluster_col)
        .merge(
            dominant[
                [
                    cluster_col,
                    f"dominant_{class_col}",
                    "dominant_class_share",
                    "cluster_total",
                ]
            ],
            on=cluster_col,
        )
    )

    return quality, counts


def overall_homogeneity_summary(
    quality: pd.DataFrame,
    cluster_total_col: str = "cluster_total",
) -> dict:
    """
    Aggregate per-cluster quality into global summary metrics.

    Weighted values are weighted by cluster area/pixel count.
    """
    weights = quality[cluster_total_col].to_numpy(dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    return {
        "mean_dominant_class_share": float(quality["dominant_class_share"].mean()),
        "weighted_dominant_class_share": float(np.sum(weights * quality["dominant_class_share"])),
        "mean_normalized_entropy": float(quality["normalized_entropy"].mean()),
        "weighted_normalized_entropy": float(np.sum(weights * quality["normalized_entropy"])),
        "mean_simpson_diversity": float(quality["simpson_diversity"].mean()),
        "weighted_simpson_diversity": float(np.sum(weights * quality["simpson_diversity"])),
    }


def vegetation_quality_report(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    class_col: str = "clc_group",
    exclude_classes: Optional[Iterable[str]] = None,
) -> dict:
    """
    Convenience wrapper returning counts, per-cluster quality, and overall summary.

    Example:
        report = vegetation_quality_report(df, "cluster", "clc_group")
        report_natural = vegetation_quality_report(
            df,
            "cluster",
            "clc_group",
            exclude_classes=["Artificial", "Agriculture"],
        )
    """
    work = df.copy()

    if exclude_classes is not None:
        work = work[~work[class_col].isin(exclude_classes)].copy()

    quality, counts = vegetation_quality_metrics(
        work,
        cluster_col=cluster_col,
        class_col=class_col,
    )

    overall = overall_homogeneity_summary(quality)

    return {
        "quality": quality,
        "counts": counts,
        "overall": overall,
        "filtered_df": work,
    }


def cluster_class_distribution(
    df: pd.DataFrame,
    cluster_id,
    cluster_col: str = "cluster",
    class_col: str = "clc_group",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Return class distribution for one cluster.
    """
    subset = df[df[cluster_col] == cluster_id]

    counts = subset[class_col].value_counts(normalize=normalize).reset_index()

    if normalize:
        counts.columns = [class_col, "proportion"]
    else:
        counts.columns = [class_col, "count"]

    return counts


def vegetation_contingency_table(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    class_col: str = "clc_group",
    normalize: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build cluster x vegetation-class table.

    normalize:
        None      -> counts
        "index"   -> rows sum to 1, distribution within each cluster
        "columns" -> columns sum to 1
        "all"     -> whole table sums to 1
    """
    return pd.crosstab(
        df[cluster_col],
        df[class_col],
        normalize=normalize,
    )


def _grid_from_values(
    df: pd.DataFrame,
    value_col: str,
    row_col: str = "row",
    col_col: str = "col",
):
    n_rows = int(df[row_col].max()) + 1
    n_cols = int(df[col_col].max()) + 1

    grid = np.full((n_rows, n_cols), np.nan)

    rows = df[row_col].to_numpy(dtype=int)
    cols = df[col_col].to_numpy(dtype=int)

    grid[rows, cols] = df[value_col].to_numpy()

    return grid


def plot_landcover_groups(
    df: pd.DataFrame,
    group_col: str = "clc_group",
    row_col: str = "row",
    col_col: str = "col",
    group_order: Optional[list[str]] = None,
    colors: Optional[dict] = None,
    title: str = "Grouped land cover",
    figsize: tuple[int, int] = (15, 10),
):
    """
    Plot grouped land-cover classes.
    """
    if group_order is None:
        group_order = DEFAULT_GROUP_ORDER

    if colors is None:
        colors = DEFAULT_GROUP_COLORS

    present_groups = [g for g in group_order if g in set(df[group_col].dropna())]

    group_to_int = {g: i for i, g in enumerate(present_groups)}

    plot_df = df.copy()
    plot_df["_group_id"] = plot_df[group_col].map(group_to_int)

    grid = _grid_from_values(
        plot_df.dropna(subset=["_group_id"]),
        value_col="_group_id",
        row_col=row_col,
        col_col=col_col,
    )

    cmap = ListedColormap([colors.get(g, "gray") for g in present_groups])

    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap=cmap, origin="upper", interpolation="nearest")
    plt.title(title)
    plt.axis("off")

    patches = [
        mpatches.Patch(color=cmap(i), label=present_groups[i])
        for i in range(len(present_groups))
    ]

    plt.legend(
        handles=patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


def plot_cluster_landcover(
    df: pd.DataFrame,
    cluster_id,
    cluster_col: str = "cluster",
    group_col: str = "clc_group",
    row_col: str = "row",
    col_col: str = "col",
    group_order: Optional[list[str]] = None,
    colors: Optional[dict] = None,
    figsize: tuple[int, int] = (15, 10),
):
    """
    Plot land-cover groups only inside a selected cluster.
    """
    subset = df[df[cluster_col] == cluster_id].copy()

    plot_landcover_groups(
        subset,
        group_col=group_col,
        row_col=row_col,
        col_col=col_col,
        group_order=group_order,
        colors=colors,
        title=f"Cluster {cluster_id} land-cover composition",
        figsize=figsize,
    )


def land_use_component(row):
    """Return broad component used by the combined score."""
    group = row["clc_group"]

    if group == "Agriculture":
        return "agriculture"

    if group in NATURAL_GROUPS:
        return "natural"

    if group == "Artificial":
        return "artificial"

    return "excluded"


def agriculture_class_label(row):
    """Detailed class used inside agriculture component."""
    if row["land_use_component"] != "agriculture":
        return np.nan

    if pd.notna(row.get("clc_name", np.nan)):
        return row["clc_name"]

    return f"CLC_{int(row['clc_class'])}"


def natural_class_label(row):
    """Grouped class used inside natural/ecological component."""
    if row["land_use_component"] != "natural":
        return np.nan

    return row["clc_group"]

def land_use_adjusted_ecological_consistency(
    merged,
    cluster_col="cluster",
):
    """Compute the combined agriculture + natural/ecological consistency metric."""
    work = merged.copy()

    if "land_use_component" not in work.columns:
        work["land_use_component"] = work.apply(land_use_component, axis=1)

    if "agriculture_class" not in work.columns:
        work["agriculture_class"] = work.apply(agriculture_class_label, axis=1)

    if "natural_class" not in work.columns:
        work["natural_class"] = work.apply(natural_class_label, axis=1)

    # Component counts per cluster
    component_counts = (
        work.groupby([cluster_col, "land_use_component"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["agriculture", "natural", "artificial", "excluded"]:
        if col not in component_counts.columns:
            component_counts[col] = 0

    component_counts["total_pixels"] = component_counts[
        ["agriculture", "natural", "artificial", "excluded"]
    ].sum(axis=1)

    component_counts["valid_scored_pixels"] = (
        component_counts["agriculture"] + component_counts["natural"]
    )

    component_counts["agriculture_share_total"] = (
        component_counts["agriculture"] / component_counts["total_pixels"].replace(0, np.nan)
    )

    component_counts["natural_share_total"] = (
        component_counts["natural"] / component_counts["total_pixels"].replace(0, np.nan)
    )

    component_counts["artificial_share_total"] = (
        component_counts["artificial"] / component_counts["total_pixels"].replace(0, np.nan)
    )

    component_counts["agriculture_weight_scored"] = (
        component_counts["agriculture"] / component_counts["valid_scored_pixels"].replace(0, np.nan)
    )

    component_counts["natural_weight_scored"] = (
        component_counts["natural"] / component_counts["valid_scored_pixels"].replace(0, np.nan)
    )

    # Agriculture homogeneity: detailed agriculture classes
    agri_df = work[work["land_use_component"] == "agriculture"].dropna(subset=["agriculture_class"])
    agri_quality = entropy_quality_for_subset(
        agri_df,
        cluster_col=cluster_col,
        class_col="agriculture_class",
    )
    agri_quality = agri_quality.rename(
        columns={
            "normalized_entropy": "agriculture_normalized_entropy",
            "homogeneity": "agriculture_homogeneity",
            "dominant_class_share": "agriculture_dominant_share",
            "dominant_class": "agriculture_dominant_class",
            "n_classes": "agriculture_n_classes",
            "cluster_total": "agriculture_total",
        }
    )

    # Natural homogeneity: grouped natural/ecological classes
    natural_df = work[work["land_use_component"] == "natural"].dropna(subset=["natural_class"])
    natural_quality = entropy_quality_for_subset(
        natural_df,
        cluster_col=cluster_col,
        class_col="natural_class",
    )
    natural_quality = natural_quality.rename(
        columns={
            "normalized_entropy": "natural_normalized_entropy",
            "homogeneity": "natural_homogeneity",
            "dominant_class_share": "natural_dominant_share",
            "dominant_class": "natural_dominant_class",
            "n_classes": "natural_n_classes",
            "cluster_total": "natural_total",
        }
    )

    keep_agri = [
        cluster_col,
        "agriculture_normalized_entropy",
        "agriculture_homogeneity",
        "agriculture_dominant_share",
        "agriculture_dominant_class",
        "agriculture_n_classes",
        "agriculture_total",
    ]

    keep_natural = [
        cluster_col,
        "natural_normalized_entropy",
        "natural_homogeneity",
        "natural_dominant_share",
        "natural_dominant_class",
        "natural_n_classes",
        "natural_total",
    ]

    result = component_counts.merge(
        agri_quality[[c for c in keep_agri if c in agri_quality.columns]],
        on=cluster_col,
        how="left",
    )

    result = result.merge(
        natural_quality[[c for c in keep_natural if c in natural_quality.columns]],
        on=cluster_col,
        how="left",
    )

    # If a component is absent, it should not contribute to the score.
    result["agriculture_homogeneity"] = result["agriculture_homogeneity"].fillna(0.0)
    result["natural_homogeneity"] = result["natural_homogeneity"].fillna(0.0)

    result["combined_ecological_consistency"] = (
        result["agriculture_weight_scored"].fillna(0.0) * result["agriculture_homogeneity"]
        + result["natural_weight_scored"].fillna(0.0) * result["natural_homogeneity"]
    )

    # Clusters with no scored land-cover pixels get NaN.
    result.loc[
        result["valid_scored_pixels"] == 0,
        "combined_ecological_consistency"
    ] = np.nan

    return result


def overall_land_use_summary(per_cluster, cluster_col="cluster"):
    """Area-weighted and unweighted summary for the combined metric."""
    valid = per_cluster.dropna(subset=["combined_ecological_consistency"]).copy()

    if len(valid) == 0:
        return {}

    weights = valid["valid_scored_pixels"].to_numpy(dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    return {
        "mean_combined_ecological_consistency": float(valid["combined_ecological_consistency"].mean()),
        "weighted_combined_ecological_consistency": float(
            np.sum(weights * valid["combined_ecological_consistency"])
        ),
        "mean_agriculture_homogeneity": float(valid["agriculture_homogeneity"].mean()),
        "weighted_agriculture_homogeneity": float(
            np.sum(weights * valid["agriculture_homogeneity"])
        ),
        "mean_natural_homogeneity": float(valid["natural_homogeneity"].mean()),
        "weighted_natural_homogeneity": float(
            np.sum(weights * valid["natural_homogeneity"])
        ),
        "mean_artificial_share_total": float(valid["artificial_share_total"].mean()),
        "weighted_artificial_share_total": float(
            np.sum(weights * valid["artificial_share_total"])
        ),
    }


def entropy_quality_for_subset(
    df,
    cluster_col="cluster",
    class_col="clc_group",
):
    """Return per-cluster homogeneity based on 1 - normalized entropy."""
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                cluster_col,
                "shannon_entropy",
                "normalized_entropy",
                "homogeneity",
                "dominant_class_share",
                "dominant_class",
                "cluster_total",
                "n_classes",
            ]
        )

    quality, counts = vegetation_quality_metrics(
        df,
        cluster_col=cluster_col,
        class_col=class_col,
    )

    quality = quality.copy()
    quality["homogeneity"] = 1.0 - quality["normalized_entropy"]

    dominant_col = f"dominant_{class_col}"
    if dominant_col in quality.columns:
        quality = quality.rename(columns={dominant_col: "dominant_class"})

    return quality


def evaluate_label_set(
    method_name,
    labels,
    data_df,
    node_col,
    vegetation,
    legend,
):
    df_clusters = attach_node_labels_to_pixels(
        df=data_df,
        node_labels=labels,
        node_col=node_col,
        cluster_col="cluster",
    )

    merged = merge_clusters_with_vegetation(
        clusters_df=df_clusters,
        vegetation_df=vegetation,
        cluster_col="cluster",
        row_col="row",
        col_col="col",
        class_col="clc_class",
        nodata_values=(0, 999),
        how="inner",
    )

    merged = add_clc_metadata(merged, legend=legend, class_col="clc_class")
    merged["land_use_component"] = merged.apply(land_use_component, axis=1)
    merged["agriculture_class"] = merged.apply(agriculture_class_label, axis=1)
    merged["natural_class"] = merged.apply(natural_class_label, axis=1)

    per_cluster = land_use_adjusted_ecological_consistency(
        merged,
        cluster_col="cluster",
    )

    overall = overall_land_use_summary(per_cluster)
    overall["method"] = method_name

    return overall, per_cluster, merged
