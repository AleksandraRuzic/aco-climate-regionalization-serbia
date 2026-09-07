#!/usr/bin/env python3
"""Create Top-N diagnostic plots for Stage 1 sweep results."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


RESULTS_DIR = Path("experiments/results")
OUTPUT_DIR = RESULTS_DIR / "stage1_topn_analysis"
OPPOSITE_OUTPUT_DIR = RESULTS_DIR / "stage1_topn_analysis_opposite_ranking"
ECOLOGY_COL = "mean_weighted_combined_ecological_consistency"
FITNESS_COL = "mean_recomputed_fitness"
FITNESS_FUNCTIONS = ("centroid", "silhouette", "snn")
FEATURE_ORDER = ("koppen", "raw", "pca")


def load_stage1_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("*/*/stage1_sweep_summary.csv")):
        df = pd.read_csv(path, sep=";")
        feature_mode = path.parts[-3]
        fitness_name = path.parts[-2]

        if "feature_mode" not in df.columns:
            df["feature_mode"] = feature_mode
        if "fitness_name" not in df.columns:
            df["fitness_name"] = fitness_name

        df["feature_mode"] = df["feature_mode"].fillna(feature_mode)
        df["fitness_name"] = df["fitness_name"].fillna(fitness_name)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No stage1_sweep_summary.csv files found under {results_dir}"
        )

    df_all = pd.concat(frames, ignore_index=True, sort=False)
    required = {"fitness_name", "feature_mode", FITNESS_COL, ECOLOGY_COL}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df_all


def minmax(values: pd.Series) -> pd.Series:
    vmin = values.min()
    vmax = values.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - vmin) / (vmax - vmin)


def add_normalized_columns(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    df["active_fitness_norm"] = np.nan
    df["ecology_norm_by_fitness"] = np.nan

    for fitness_name in FITNESS_FUNCTIONS:
        mask = df["fitness_name"].eq(fitness_name)
        df.loc[mask, "active_fitness_norm"] = minmax(df.loc[mask, FITNESS_COL])
        df.loc[mask, "ecology_norm_by_fitness"] = minmax(df.loc[mask, ECOLOGY_COL])

    df["ecology_norm_global"] = minmax(df[ECOLOGY_COL])
    return df


def metric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fitness_name in FITNESS_FUNCTIONS:
        sub = df[df["fitness_name"].eq(fitness_name)]
        for metric_name, col in (
            ("active_fitness", FITNESS_COL),
            ("ecology", ECOLOGY_COL),
        ):
            rows.append(
                {
                    "scope": "fitness_group",
                    "fitness_function": fitness_name,
                    "metric": metric_name,
                    "min": sub[col].min(),
                    "max": sub[col].max(),
                    "range": sub[col].max() - sub[col].min(),
                    "std": sub[col].std(ddof=1),
                }
            )

    rows.append(
        {
            "scope": "global",
            "fitness_function": "all",
            "metric": "ecology",
            "min": df[ECOLOGY_COL].min(),
            "max": df[ECOLOGY_COL].max(),
            "range": df[ECOLOGY_COL].max() - df[ECOLOGY_COL].min(),
            "std": df[ECOLOGY_COL].std(ddof=1),
        }
    )
    return pd.DataFrame(rows)


def topn_std_curves(
    df: pd.DataFrame,
    n_min: int = 10,
    fitness_ascending: bool = True,
    ecology_ascending: bool = False,
) -> pd.DataFrame:
    rows = []

    for fitness_name in FITNESS_FUNCTIONS:
        sub = (
            df[df["fitness_name"].eq(fitness_name)]
            .sort_values(FITNESS_COL, ascending=fitness_ascending)
            .reset_index(drop=True)
        )
        for n in range(n_min, len(sub) + 1):
            top = sub.iloc[:n]
            for metric, col in (
                ("active_fitness_norm", "active_fitness_norm"),
                ("ecology_norm", "ecology_norm_by_fitness"),
            ):
                rows.append(
                    {
                        "graph": f"rank_fitness_{fitness_name}",
                        "ranking": "active_fitness",
                        "fitness_function": fitness_name,
                        "n": n,
                        "metric": metric,
                        "std": top[col].std(ddof=1),
                    }
                )

    sub = df.sort_values(ECOLOGY_COL, ascending=ecology_ascending).reset_index(drop=True)
    for n in range(n_min, len(sub) + 1):
        top = sub.iloc[:n]
        rows.append(
            {
                "graph": "rank_ecology_global",
                "ranking": "ecology",
                "fitness_function": "all",
                "n": n,
                "metric": "ecology_norm",
                "std": top["ecology_norm_global"].std(ddof=1),
            }
        )

    return pd.DataFrame(rows)


def topn_corr_curves(
    df: pd.DataFrame, n_min: int = 20, fitness_ascending: bool = True
) -> pd.DataFrame:
    rows = []
    for fitness_name in FITNESS_FUNCTIONS:
        sub = (
            df[df["fitness_name"].eq(fitness_name)]
            .sort_values(FITNESS_COL, ascending=fitness_ascending)
            .reset_index(drop=True)
        )
        for n in range(n_min, len(sub) + 1):
            top = sub.iloc[:n]
            pearson = pearsonr(top[FITNESS_COL], top[ECOLOGY_COL]).statistic
            spearman = spearmanr(top[FITNESS_COL], top[ECOLOGY_COL]).statistic
            rows.append(
                {
                    "fitness_function": fitness_name,
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def plot_std_curves(std_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved = []
    metric_labels = {
        "active_fitness_norm": "Active fitness",
        "ecology_norm": "Ecological consistency",
    }
    titles = {
        "rank_fitness_centroid": "Top-N variation ranked by centroid fitness",
        "rank_fitness_silhouette": "Top-N variation ranked by silhouette fitness",
        "rank_fitness_snn": "Top-N variation ranked by SNN fitness",
        "rank_ecology_global": "Top-N variation ranked by ecological consistency",
    }
    files = {
        "rank_fitness_centroid": "topn_std_rank_fitness_centroid.png",
        "rank_fitness_silhouette": "topn_std_rank_fitness_silhouette.png",
        "rank_fitness_snn": "topn_std_rank_fitness_snn.png",
        "rank_ecology_global": "topn_std_rank_ecology_global.png",
    }

    for graph, filename in files.items():
        sub = std_df[std_df["graph"].eq(graph)]
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric, metric_sub in sub.groupby("metric"):
            ax.plot(
                metric_sub["n"],
                metric_sub["std"],
                label=metric_labels.get(metric, metric),
                linewidth=2,
            )
        ax.set_title(titles[graph])
        ax.set_xlabel("Top N configurations")
        ax.set_ylabel("Ecological consistency std. after min-max normalization")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        saved.append(path)

    return saved


def plot_snn_std_comparison(
    normal_std_df: pd.DataFrame, opposite_std_df: pd.DataFrame, output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    panels = (
        (axes[0], normal_std_df, "Best to worst SNN fitness"),
        (axes[1], opposite_std_df, "Worst to best SNN fitness"),
    )
    labels = {
        "active_fitness_norm": "Active fitness",
        "ecology_norm": "Ecological consistency",
    }

    for ax, std_df, title in panels:
        sub = std_df[std_df["graph"].eq("rank_fitness_snn")]
        for metric, metric_sub in sub.groupby("metric"):
            ax.plot(
                metric_sub["n"],
                metric_sub["std"],
                label=labels.get(metric, metric),
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Top N configurations")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Standard deviation after min-max normalization")
    axes[0].legend()
    fig.suptitle("Top-N variation for SNN-ranked configurations")
    fig.tight_layout()
    path = output_dir / "topn_std_snn_ranking_comparison.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_corr_curves(corr_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved = []
    for fitness_name in FITNESS_FUNCTIONS:
        sub = corr_df[corr_df["fitness_function"].eq(fitness_name)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["n"], sub["pearson"], label="Pearson", linewidth=2)
        ax.plot(sub["n"], sub["spearman"], label="Spearman", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"Top-N fitness-ecology correlation ranked by {fitness_name} fitness"
        )
        ax.set_xlabel("Top N configurations")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"topn_corr_fitness_ecology_{fitness_name}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        saved.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, fitness_name in zip(axes, FITNESS_FUNCTIONS):
        sub = corr_df[corr_df["fitness_function"].eq(fitness_name)]
        ax.plot(sub["n"], sub["pearson"], label="Pearson", linewidth=2)
        ax.plot(sub["n"], sub["spearman"], label="Spearman", linewidth=2)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{fitness_name.capitalize()} fitness")
        ax.set_xlabel("Top N configurations")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Correlation")
    axes[0].legend()
    fig.suptitle("Top-N fitness-ecology correlation by active fitness ranking")
    fig.tight_layout()
    path = output_dir / "topn_corr_fitness_ecology_combined.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    saved.append(path)

    return saved


def print_summary(
    df: pd.DataFrame, ranges: pd.DataFrame, saved: list[Path], output_dir: Path
) -> None:
    print(f"Loaded rows: {len(df)}")
    print("\nRows by feature set and fitness function:")
    print(
        df.groupby(["feature_mode", "fitness_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=FEATURE_ORDER, columns=FITNESS_FUNCTIONS)
        .to_string()
    )
    print("\nMetric ranges:")
    print(ranges.round(4).to_string(index=False))
    print(f"\nOutput directory: {output_dir}")
    print("\nSaved PNG files:")
    for path in saved:
        print(f"- {path}")


def main() -> None:
    args = set(sys.argv[1:])
    opposite_ranking = "--opposite-ranking" in args
    std_only = "--std-only" in args
    output_dir = OPPOSITE_OUTPUT_DIR if opposite_ranking else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_stage1_summaries(RESULTS_DIR)
    df = add_normalized_columns(df)

    ranges = metric_ranges(df)
    std_curves = topn_std_curves(
        df,
        fitness_ascending=not opposite_ranking,
        ecology_ascending=opposite_ranking,
    )

    ranges.to_csv(output_dir / "metric_ranges.csv", index=False)
    std_curves.to_csv(output_dir / "topn_std_curves.csv", index=False)

    saved = []
    saved.extend(plot_std_curves(std_curves, output_dir))
    if not opposite_ranking:
        opposite_std_curves = topn_std_curves(
            df,
            fitness_ascending=False,
            ecology_ascending=True,
        )
        saved.append(
            plot_snn_std_comparison(std_curves, opposite_std_curves, output_dir)
        )
    if not std_only:
        corr_curves = topn_corr_curves(df, fitness_ascending=not opposite_ranking)
        corr_curves.to_csv(output_dir / "topn_corr_curves.csv", index=False)
        saved.extend(plot_corr_curves(corr_curves, output_dir))

    print_summary(df, ranges, saved, output_dir)


if __name__ == "__main__":
    main()
