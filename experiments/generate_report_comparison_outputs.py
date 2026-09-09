#!/usr/bin/env python3
"""Generate report-ready reference and regionalization comparison outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moduli.ant_clustering_optimizer import AntClusteringOptimizer
from moduli.cluster_centroid_manipulation import compute_centroids
from moduli.fitness_functions import (
    fitness_weighted_sum,
    silhouette_spatial_fitness,
    snn_spatial_fitness,
)
from moduli.plot_utils import labels_to_raster
from moduli.preprocessing import build_koppen_features, prepare_model_data
from moduli.snn_graph import build_snn_graph
from moduli.vegetation_eval import evaluate_label_set, load_clc_legend

from experiments.stage1_sweep import (
    ALPHAS,
    BETAS,
    CENTROID_FITNESS_KWARGS,
    COMMON_FITNESS_KWARGS,
    COMMON_PHEROMONE_INIT_KWARGS,
    COMMON_PHEROMONE_UPDATE_KWARGS,
    EVAPORATIONS,
    K_NEIGHBORS,
    LEGEND_PATH,
    MODE,
    N_ANTS,
    N_CLUSTERS,
    N_ITERATIONS,
    QS,
    RESULT_DIR,
    SILHOUETTE_FITNESS_KWARGS,
    SNN_FITNESS_KWARGS,
    SNN_GRAPH_KWARGS,
    TARGET_SIZE,
    UPDATE_CONFIGS,
    VEGETATION_PATH,
    build_algorithm_configs,
    build_fitness_configs,
)


OUTPUT_DIR = RESULT_DIR / "report_comparison_outputs"
FEATURE_MODE = "koppen"
FITNESS_COL = "mean_recomputed_fitness"
ECO_COL = "mean_weighted_combined_ecological_consistency"
REPRESENTATIVE_SEED = 0


def load_koppen_data():
    df = pd.read_csv(DATA_ROOT / "df_valid.csv", index_col=0)
    koppen = pd.read_csv(DATA_ROOT / "koppen_df_valid.csv", index_col=0)
    df = df.merge(koppen[["row", "col", "koppen_class"]], on=["row", "col"], how="inner")
    df = df.dropna(subset=["koppen_class"]).copy()

    x_features, feature_names = build_koppen_features(df)
    data = prepare_model_data(
        df=df,
        X_features=x_features,
        feature_names=feature_names,
        mode=MODE,
        target_size=TARGET_SIZE,
        k_neighbors=K_NEIGHBORS,
        scale_final=True,
        random_state=0,
    )
    return data


def aggregate_koppen_labels(data) -> np.ndarray:
    df_model = data["df"]
    node_col = data["node_col"]
    node_koppen = (
        df_model.dropna(subset=["koppen_class"])
        .groupby(node_col)["koppen_class"]
        .agg(lambda x: x.value_counts().idxmax())
    )
    codes = sorted(node_koppen.unique())
    code_to_label = {code: i for i, code in enumerate(codes)}
    return node_koppen.sort_index().map(code_to_label).to_numpy(dtype=int)


def all_fitness_values(data, labels, snn_graph) -> dict[str, float]:
    x = data["X_model"]
    weights = data["weights"]
    neighbors = data["neighbors"]
    centroids = compute_centroids(
        X=x,
        labels=labels,
        n_clusters=N_CLUSTERS,
        weights=weights,
        rng=np.random.default_rng(0),
    )

    centroid, _ = fitness_weighted_sum(
        X=x,
        labels=labels,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        **CENTROID_FITNESS_KWARGS,
        **COMMON_FITNESS_KWARGS,
    )
    silhouette, _ = silhouette_spatial_fitness(
        X=x,
        labels=labels,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        **SILHOUETTE_FITNESS_KWARGS,
        **COMMON_FITNESS_KWARGS,
    )
    snn, _ = snn_spatial_fitness(
        X=x,
        labels=labels,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        snn_graph=snn_graph,
        **SNN_FITNESS_KWARGS,
        **COMMON_FITNESS_KWARGS,
    )
    return {
        "centroid_fitness": float(centroid),
        "silhouette_fitness": float(silhouette),
        "snn_fitness": float(snn),
    }


def ecological_consistency(data, labels, vegetation, legend) -> float:
    overall, _, _ = evaluate_label_set(
        method_name="method",
        labels=labels,
        data_df=data["df"],
        node_col=data["node_col"],
        vegetation=vegetation,
        legend=legend,
    )
    return float(overall["weighted_combined_ecological_consistency"])


def reference_rows(data, snn_graph, vegetation, legend):
    x = data["X_model"]
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit(x)
    koppen_labels = aggregate_koppen_labels(data)

    refs = [
        ("KMeans", kmeans.labels_),
        ("Köppen-Geiger", koppen_labels),
    ]
    rows = []
    for name, labels in refs:
        rows.append(
            {
                "regionalization": name,
                **all_fitness_values(data, labels, snn_graph),
                "ecology": ecological_consistency(data, labels, vegetation, legend),
                "ari_vs_kmeans": float(adjusted_rand_score(kmeans.labels_, labels)),
            }
        )
    return rows, kmeans.labels_, koppen_labels


def load_koppen_summaries() -> pd.DataFrame:
    frames = []
    for fitness_name in ("centroid", "silhouette", "snn"):
        path = RESULT_DIR / FEATURE_MODE / fitness_name / "stage1_sweep_summary.csv"
        df = pd.read_csv(path, sep=";")
        df["fitness_name"] = fitness_name
        df["feature_mode"] = FEATURE_MODE
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def selected_summary_rows(summary: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for fitness_name in ("centroid", "silhouette", "snn"):
        sub = summary[summary["fitness_name"].eq(fitness_name)]
        selected.append(
            sub.sort_values([FITNESS_COL, ECO_COL], ascending=[True, False]).iloc[0]
        )

    ecology_best = summary.sort_values([ECO_COL, FITNESS_COL], ascending=[False, True]).iloc[0]
    selected_keys = {
        tuple(row[c] for c in config_cols()) for row in selected
    }
    ecology_key = tuple(ecology_best[c] for c in config_cols())
    if ecology_key not in selected_keys:
        selected.append(ecology_best)

    out = pd.DataFrame(selected).reset_index(drop=True)
    labels = ["Best centroid fitness", "Best silhouette fitness", "Best SNN fitness"]
    if len(out) == 4:
        labels.append("Best vegetation metric")
    out.insert(0, "selection", labels)
    return out


def config_cols() -> list[str]:
    return [
        "fitness_name",
        "algorithm",
        "pheromone_update",
        "alpha",
        "beta",
        "evaporation",
        "q",
        "feature_mode",
        "mode",
    ]


def build_config_lookup(snn_graph) -> dict[tuple, dict]:
    fitness_configs = build_fitness_configs(snn_graph)
    lookup = {}
    update_lookup = {cfg["pheromone_update"]: cfg for cfg in UPDATE_CONFIGS}

    for fit_cfg in fitness_configs:
        for beta in BETAS:
            for alg_cfg in build_algorithm_configs(beta):
                for alpha in ALPHAS:
                    for evaporation in EVAPORATIONS:
                        for q in QS:
                            for update_name, upd_cfg in update_lookup.items():
                                construct_kwargs = dict(alg_cfg["construct_kwargs"])
                                construct_kwargs["alpha"] = alpha
                                key = (
                                    fit_cfg["fitness_name"],
                                    alg_cfg["algorithm"],
                                    update_name,
                                    float(alpha),
                                    float(beta),
                                    float(evaporation),
                                    float(q),
                                    FEATURE_MODE,
                                    MODE,
                                )
                                lookup[key] = {
                                    "fitness_name": fit_cfg["fitness_name"],
                                    "fitness_fn": fit_cfg["fitness_fn"],
                                    "fitness_kwargs": dict(fit_cfg["fitness_kwargs"]),
                                    "algorithm": alg_cfg["algorithm"],
                                    "construct_solution_fn": alg_cfg["construct_solution_fn"],
                                    "construct_kwargs": construct_kwargs,
                                    "pheromone_update": update_name,
                                    "pheromone_update_fn": upd_cfg["pheromone_update_fn"],
                                    "pheromone_update_kwargs": dict(upd_cfg["pheromone_update_kwargs"]),
                                    "pheromone_update_batch_size": upd_cfg["pheromone_update_batch_size"],
                                    "alpha": alpha,
                                    "beta": beta,
                                    "evaporation": evaporation,
                                    "q": q,
                                    "seed": REPRESENTATIVE_SEED,
                                }
    return lookup


def rerun_for_labels(data, selected: pd.DataFrame, snn_graph) -> dict[str, np.ndarray]:
    lookup = build_config_lookup(snn_graph)
    labels = {}
    for _, row in selected.iterrows():
        key = tuple(row[c] for c in config_cols())
        key = (
            key[0],
            key[1],
            key[2],
            float(key[3]),
            float(key[4]),
            float(key[5]),
            float(key[6]),
            key[7],
            key[8],
        )
        cfg = lookup[key]
        opt_kwargs = {}
        if cfg["pheromone_update_batch_size"] is not None:
            opt_kwargs["pheromone_update_batch_size"] = cfg["pheromone_update_batch_size"]

        optimizer = AntClusteringOptimizer(
            n_clusters=N_CLUSTERS,
            n_ants=N_ANTS,
            n_iterations=N_ITERATIONS,
            construct_solution_fn=cfg["construct_solution_fn"],
            construct_kwargs=cfg["construct_kwargs"],
            fitness_fn=cfg["fitness_fn"],
            fitness_kwargs=cfg["fitness_kwargs"],
            pheromone_init_fn=None,
            pheromone_init_kwargs=COMMON_PHEROMONE_INIT_KWARGS,
            pheromone_update_fn=cfg["pheromone_update_fn"],
            pheromone_update_kwargs=COMMON_PHEROMONE_UPDATE_KWARGS,
            evaporation=cfg["evaporation"],
            q=cfg["q"],
            random_state=REPRESENTATIVE_SEED,
            verbose=False,
            **opt_kwargs,
        )
        from moduli.pheromone_updates import initialize_node_cluster_pheromone

        optimizer.pheromone_init_fn = initialize_node_cluster_pheromone
        optimizer.fit(data["X_model"], data["neighbors"], weights=data["weights"])
        labels[row["selection"]] = optimizer.labels_

    return labels


def aco_summary_rows(selected: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in selected.iterrows():
        centroid_fitness = row.get("mean_centroid_fitness", np.nan)
        silhouette_fitness = row.get("mean_silhouette_fitness", np.nan)
        snn_fitness = row.get("mean_snn_fitness", np.nan)

        if row["fitness_name"] == "centroid" and pd.isna(centroid_fitness):
            centroid_fitness = row[FITNESS_COL]
        if row["fitness_name"] == "silhouette" and pd.isna(silhouette_fitness):
            silhouette_fitness = row[FITNESS_COL]
        if row["fitness_name"] == "snn" and pd.isna(snn_fitness):
            snn_fitness = row[FITNESS_COL]

        rows.append(
            {
                "regionalization": row["selection"],
                "fitness_function": row["fitness_name"],
                "algorithm": row["algorithm"],
                "pheromone_update": row["pheromone_update"],
                "alpha": row["alpha"],
                "beta": row["beta"],
                "evaporation": row["evaporation"],
                "q": row["q"],
                "centroid_fitness": centroid_fitness,
                "silhouette_fitness": silhouette_fitness,
                "snn_fitness": snn_fitness,
                "ecology": row[ECO_COL],
                "ari_vs_kmeans": row["mean_ari_vs_kmeans"],
            }
        )
    return rows


def comparison_rows_from_labels(
    data,
    selected: pd.DataFrame,
    label_dict: dict[str, np.ndarray],
    snn_graph,
    vegetation,
    legend,
) -> list[dict]:
    rows = []
    for name, labels in label_dict.items():
        row = {
            "regionalization": name,
            **all_fitness_values(data, labels, snn_graph),
            "ecology": ecological_consistency(data, labels, vegetation, legend),
            "ari_vs_kmeans": float(adjusted_rand_score(label_dict["KMeans"], labels)),
        }
        if name in set(selected["selection"]):
            selected_row = selected[selected["selection"].eq(name)].iloc[0]
            row.update(
                {
                    key: selected_row[key]
                    for key in [
                        "fitness_name",
                        "algorithm",
                        "pheromone_update",
                        "alpha",
                        "beta",
                        "evaporation",
                        "q",
                    ]
                }
            )
        rows.append(row)
    return rows


def save_labels(label_dict: dict[str, np.ndarray]) -> None:
    for name, labels in label_dict.items():
        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("ö", "o")
            .replace("-", "_")
            .replace("+", "plus")
        )
        pd.DataFrame({"label": labels}).to_csv(OUTPUT_DIR / f"{filename}_labels.csv", index=False)


def label_filename(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("ö", "o")
        .replace("-", "_")
        .replace("+", "plus")
    )


def load_saved_labels(names: list[str]) -> dict[str, np.ndarray]:
    labels = {}
    for name in names:
        path = OUTPUT_DIR / f"{label_filename(name)}_labels.csv"
        if not path.exists() and name == "Best vegetation metric":
            path = OUTPUT_DIR / "best_ecological_consistency_labels.csv"
        labels[name] = pd.read_csv(path)["label"].to_numpy(dtype=int)
    return labels


def plot_label_grid(
    data,
    label_dict: dict[str, np.ndarray],
    save_path: Path,
    *,
    ncols: int,
    title: str,
) -> None:
    colors = ["red", "blue", "green", "orange", "purple"]
    cmap = ListedColormap(colors)
    cmap.set_bad(color="white")
    norm = BoundaryNorm(np.arange(-0.5, N_CLUSTERS + 0.5, 1), cmap.N)

    n_items = len(label_dict)
    nrows = int(np.ceil(n_items / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()
    img = None

    for ax, (name, labels) in zip(axes, label_dict.items()):
        raster = labels_to_raster(
            df=data["df"],
            node_labels=labels,
            node_col=data["node_col"],
        )
        img = ax.imshow(raster, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in axes[n_items:]:
        ax.axis("off")

    cbar = fig.colorbar(img, ax=axes, shrink=0.85, ticks=np.arange(N_CLUSTERS))
    cbar.set_label("Cluster")
    fig.suptitle(title, fontsize=14)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_report_comparison(data, label_dict: dict[str, np.ndarray]) -> None:
    reference_names = ["KMeans", "Köppen-Geiger"]
    reference_labels = {name: label_dict[name] for name in reference_names}
    aco_labels = {name: labels for name, labels in label_dict.items() if name not in reference_names}

    plot_label_grid(
        data=data,
        label_dict=reference_labels,
        save_path=OUTPUT_DIR / "reference_regionalizations_koppen_features.png",
        ncols=2,
        title="Reference regionalizations on Köppen-inspired features",
    )
    plot_label_grid(
        data=data,
        label_dict=aco_labels,
        save_path=OUTPUT_DIR / "selected_aco_regionalizations_koppen_features.png",
        ncols=2,
        title="Selected ACO regionalizations on Köppen-inspired features",
    )


def main() -> None:
    plot_only = "--plot-only" in sys.argv[1:]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_koppen_data()

    if plot_only:
        selected = pd.read_csv(OUTPUT_DIR / "selected_aco_configurations.csv")
        names = ["KMeans", "Köppen-Geiger", *selected["selection"].tolist()]
        label_dict = load_saved_labels(names)
        plot_report_comparison(data=data, label_dict=label_dict)
        print(f"Saved plot to {OUTPUT_DIR / 'reference_regionalizations_koppen_features.png'}")
        print(f"Saved plot to {OUTPUT_DIR / 'selected_aco_regionalizations_koppen_features.png'}")
        return

    vegetation = pd.read_csv(VEGETATION_PATH)
    legend = load_clc_legend(LEGEND_PATH)
    snn_graph = build_snn_graph(
        X=data["X_model"],
        spatial_neighbors=data["neighbors"],
        **SNN_GRAPH_KWARGS,
    )

    reference, kmeans_labels, koppen_labels = reference_rows(data, snn_graph, vegetation, legend)
    reference_df = pd.DataFrame(reference)
    reference_df.to_csv(OUTPUT_DIR / "reference_regionalizations_koppen_features.csv", index=False)

    summary = load_koppen_summaries()
    selected = selected_summary_rows(summary)
    selected.to_csv(OUTPUT_DIR / "selected_aco_configurations.csv", index=False)

    aco_labels = rerun_for_labels(data, selected, snn_graph)
    label_dict = {
        "KMeans": kmeans_labels,
        "Köppen-Geiger": koppen_labels,
        **aco_labels,
    }
    save_labels(label_dict)

    regionalization_rows = comparison_rows_from_labels(
        data=data,
        selected=selected,
        label_dict=label_dict,
        snn_graph=snn_graph,
        vegetation=vegetation,
        legend=legend,
    )
    regionalization_df = pd.DataFrame(regionalization_rows)
    regionalization_df.to_csv(OUTPUT_DIR / "regionalization_comparison_koppen_features.csv", index=False)

    plot_report_comparison(data=data, label_dict=label_dict)

    print("Reference regionalizations")
    print(reference_df.round(4).to_string(index=False))
    print("\nSelected ACO configurations")
    print(selected[["selection", *config_cols(), FITNESS_COL, ECO_COL]].round(4).to_string(index=False))
    print("\nRegionalization comparison")
    print(regionalization_df.round(4).to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
