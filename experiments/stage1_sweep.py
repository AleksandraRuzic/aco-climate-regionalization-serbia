"""
Stage 1 + ecological validation experiment sweep.

This script merges the previous Stage 1 sweep, centroid supplement,
and Stage 2 land-use ecological validation into one overnight run.

It runs all algorithm families directly and logs both:
    - optimization metrics:
        best_fitness
        compactness
        spatial
        ARI vs KMeans
        runtime
    - land-use-adjusted ecological consistency:
        weighted_combined_ecological_consistency
        mean_combined_ecological_consistency
        weighted_agriculture_homogeneity
        weighted_natural_homogeneity
        weighted_artificial_share_total

Expected folder structure:

    parent_folder/
    ├── df_valid.csv
    ├── vegetacija.csv
    ├── Vegetacija/
    │   └── u2018_clc2018_v2020_20u1_raster100m/
    │       └── Legend/
    │           └── CLC2018_CLC2018_V2018_20_QGIS.txt
    └── kod/
        ├── moduli/
        └── experiments/
            ├── stage1_sweep.py
            └── results/

Run from kod/experiments:

    python stage1_sweep.py --save-every 1

Useful:

    python stage1_sweep.py --dry-run
    python stage1_sweep.py --max-runs 20 --save-every 1
    python stage1_sweep.py --no-resume

Notes:
    - Lower fitness is better.
    - Higher ecological consistency is better.
    - ARI vs KMeans is not a quality metric; it is a similarity-to-KMeans diagnostic.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
import traceback
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# =============================================================================
# Project paths
# =============================================================================

# This file is expected to live inside kod/experiments/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent       # kod/
DATA_ROOT = PROJECT_ROOT.parent                             # parent folder containing data

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from moduli.preprocessing import (
    build_koppen_features,
    build_raw_features,
    prepare_model_data,
)

from moduli.cluster_centroid_manipulation import compute_centroids

from moduli.ant_clustering_optimizer import AntClusteringOptimizer

from moduli.construction_strategies import (
    construct_aco_solution,
    construct_graph_aco_solution,
    construct_aco_refinement_solution,
    construct_multicolony_solution,
)

from moduli.fitness_functions import fitness_weighted_sum

from moduli.pheromone_updates import (
    initialize_node_cluster_pheromone,
    update_pheromone_all_ants,
    update_pheromone_best_ant,
    update_pheromone_quality_weighted,
)

from moduli.heuristics import (
    centroid_heuristic,
    graph_heuristic,
    graph_cohesion_heuristic,
    spatial_heuristic,
)

from moduli.vegetation_eval import (
    load_clc_legend,
    attach_node_labels_to_pixels,
    merge_clusters_with_vegetation,
    add_clc_metadata,
    land_use_component,
    agriculture_class_label,
    natural_class_label,
    land_use_adjusted_ecological_consistency,
    overall_land_use_summary,
)


# =============================================================================
# Default experiment settings
# =============================================================================

DATA_PATH = DATA_ROOT / "df_valid.csv"
VEGETATION_PATH = DATA_ROOT / "vegetacija.csv"
LEGEND_PATH = (
    DATA_ROOT
    / "Vegetacija"
    / "u2018_clc2018_v2020_20u1_raster100m"
    / "Legend"
    / "CLC2018_CLC2018_V2018_20_QGIS.txt"
)

RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
RESULT_PATH = RESULT_DIR / "stage1_sweep_runs.csv"
FAILURE_PATH = RESULT_DIR / "stage1_sweep_failures.csv"
SUMMARY_PATH = RESULT_DIR / "stage1_sweep_summary.csv"
LOG_PATH = RESULT_DIR / "stage1_sweep.log"
KMEANS_PATH = RESULT_DIR / "stage1_kmeans_baseline.csv"

MODE = "superpixels"
FEATURE_MODE = "koppen"

N_CLUSTERS = 5
TARGET_SIZE = 150
K_NEIGHBORS = 16

N_ANTS = 15
N_ITERATIONS = 20

SEEDS = [0, 1]
ALPHAS = [0.75, 1.0, 2.0]
BETAS = [1.0, 3.0]
EVAPORATIONS = [0.02, 0.1]
QS = [0.05]

MAX_RUNS = None
SAVE_EVERY = 1

CSV_SEP = ";"

COMMON_FITNESS_KWARGS = {
    "w_compactness": 0.7,
    "w_spatial": 0.3,
    "spatial_symmetric": False,
}

COMMON_PHEROMONE_INIT_KWARGS = {
    "initial_value": 1.0,
    "normalize_rows": True,
}

COMMON_PHEROMONE_UPDATE_KWARGS = {
    "clip_min": 1e-6,
    "clip_max": 1e6,
    "normalize_rows": True,
}


# =============================================================================
# Basic utilities
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage1_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read comma or semicolon CSV automatically."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=";")


def append_csv(path: Path, rows: list[dict], sep: str = CSV_SEP) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    file_exists = path.exists()
    header = not file_exists

    if file_exists and path.stat().st_size > 0:
        with open(path, "rb+") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            if last_char != b"\n":
                f.write(b"\n")

    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        sep=sep,
        lineterminator="\n",
    )


def sanitize_for_json(obj):
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return f"array(shape={obj.shape})"
    return obj


def config_key(cfg: dict) -> tuple:
    return (
        str(cfg["algorithm"]),
        str(cfg["pheromone_update"]),
        float(cfg["alpha"]),
        float(cfg["beta"]),
        float(cfg["evaporation"]),
        float(cfg["q"]),
        int(cfg["seed"]),
        str(cfg.get("feature_mode", FEATURE_MODE)),
        str(cfg.get("mode", MODE)),
    )


def load_completed_keys(result_path: Path) -> set:
    if not result_path.exists():
        return set()

    try:
        df = read_csv_auto(result_path)
    except Exception:
        return set()

    required = {
        "algorithm",
        "pheromone_update",
        "alpha",
        "beta",
        "evaporation",
        "q",
        "seed",
    }

    if not required.issubset(df.columns):
        return set()

    completed = set()
    for _, row in df.iterrows():
        completed.add(
            (
                str(row["algorithm"]),
                str(row["pheromone_update"]),
                float(row["alpha"]),
                float(row["beta"]),
                float(row["evaporation"]),
                float(row["q"]),
                int(row["seed"]),
                str(row.get("feature_mode", FEATURE_MODE)),
                str(row.get("mode", MODE)),
            )
        )

    return completed


# =============================================================================
# Data preparation
# =============================================================================

def load_problem(
    data_path: Path,
    feature_mode: str,
    mode: str,
    target_size: int,
    k_neighbors: int,
):
    df = pd.read_csv(data_path, index_col=0)

    if feature_mode == "koppen":
        X_features, feature_names = build_koppen_features(df)
    elif feature_mode == "raw":
        X_features, feature_names = build_raw_features(df)
    else:
        raise ValueError("feature_mode must be 'koppen' or 'raw'.")

    data = prepare_model_data(
        df=df,
        X_features=X_features,
        feature_names=feature_names,
        mode=mode,
        target_size=target_size,
        k_neighbors=k_neighbors,
        scale_final=True,
        random_state=0,
    )

    return data


def compute_kmeans_baseline(X, neighbors, weights):
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=10,
        random_state=0,
    ).fit(X)

    centroids = compute_centroids(
        X=X,
        labels=kmeans.labels_,
        n_clusters=N_CLUSTERS,
        weights=weights,
        rng=np.random.default_rng(0),
    )

    fit, terms = fitness_weighted_sum(
        X=X,
        labels=kmeans.labels_,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        **COMMON_FITNESS_KWARGS,
    )

    return kmeans, centroids, float(fit), terms


# =============================================================================
# Algorithm configuration
# =============================================================================

def make_centroid_colony(beta):
    return {
        "name": "centroid",
        "fn": centroid_heuristic,
        "params": {"beta_feature": beta},
    }


def make_graph_colony(beta):
    return {
        "name": "graph",
        "fn": graph_heuristic,
        "params": {"beta_graph": beta},
    }


def make_cohesion_colony(beta):
    return {
        "name": "cohesion",
        "fn": graph_cohesion_heuristic,
        "params": {"beta_graph": beta},
    }


def make_spatial_colony(beta):
    return {
        "name": "spatial",
        "fn": spatial_heuristic,
        "params": {"beta_spatial": beta},
    }


def build_algorithm_configs(beta: float) -> list[dict]:
    """
    Algorithm families included in the sweep.

    Current construction convention:
        construct_aco_solution(..., centroid=True)  = true centroid ACO
        construct_aco_solution(..., centroid=False) = cohesion ACO
        construct_aco_refinement_solution(..., centroid=True) = centroid refinement
        construct_aco_refinement_solution(..., centroid=False)= cohesion refinement
    """
    configs = []

    configs.append({
        "algorithm": "aco_centroid",
        "construct_solution_fn": construct_aco_solution,
        "construct_kwargs": {
            "alpha": None,
            "beta_feature": beta,
            "centroid": True,
        },
    })

    configs.append({
        "algorithm": "aco_cohesion",
        "construct_solution_fn": construct_aco_solution,
        "construct_kwargs": {
            "alpha": None,
            "beta_feature": beta,
            "centroid": False,
        },
    })

    configs.append({
        "algorithm": "aco_centroid_refinement",
        "construct_solution_fn": construct_aco_refinement_solution,
        "construct_kwargs": {
            "alpha": None,
            "beta_feature": beta,
            "refinement_steps": 3,
            "centroid": True,
        },
    })

    configs.append({
        "algorithm": "aco_cohesion_refinement",
        "construct_solution_fn": construct_aco_refinement_solution,
        "construct_kwargs": {
            "alpha": None,
            "beta_feature": beta,
            "refinement_steps": 3,
            "centroid": False,
        },
    })

    configs.append({
        "algorithm": "aco_graph_local",
        "construct_solution_fn": construct_graph_aco_solution,
        "construct_kwargs": {
            "alpha": None,
            "beta_graph": beta,
        },
    })

    multi_variants = [
        (
            "multi_centroid_graph_spatial",
            [
                make_centroid_colony(beta),
                make_graph_colony(beta),
                make_spatial_colony(beta),
            ],
        ),
        (
            "multi_cohesion_graph_spatial",
            [
                make_cohesion_colony(beta),
                make_graph_colony(beta),
                make_spatial_colony(beta),
            ],
        ),
    ]

    for algorithm_name, colony_heuristics in multi_variants:
        configs.append({
            "algorithm": algorithm_name,
            "construct_solution_fn": construct_multicolony_solution,
            "construct_kwargs": {
                "colony_heuristics": colony_heuristics,
                "alpha": None,
            },
        })

    return configs


UPDATE_CONFIGS = [
    {
        "pheromone_update": "all_ants",
        "pheromone_update_fn": update_pheromone_all_ants,
        "pheromone_update_kwargs": COMMON_PHEROMONE_UPDATE_KWARGS.copy(),
        "pheromone_update_batch_size": None,
    },
    {
        "pheromone_update": "best_ant",
        "pheromone_update_fn": update_pheromone_best_ant,
        "pheromone_update_kwargs": COMMON_PHEROMONE_UPDATE_KWARGS.copy(),
        "pheromone_update_batch_size": None,
    },
    {
        "pheromone_update": "quality_weighted",
        "pheromone_update_fn": update_pheromone_quality_weighted,
        "pheromone_update_kwargs": COMMON_PHEROMONE_UPDATE_KWARGS.copy(),
        "pheromone_update_batch_size": None,
    },
]


def iter_experiment_configs(
    alphas: list[float],
    betas: list[float],
    evaporations: list[float],
    qs: list[float],
    seeds: list[int],
    feature_mode: str,
    mode: str,
):
    run_id = 0

    for beta in betas:
        for alg_cfg in build_algorithm_configs(beta):
            for upd_cfg in UPDATE_CONFIGS:
                for alpha, evaporation, q, seed in itertools.product(
                    alphas,
                    evaporations,
                    qs,
                    seeds,
                ):
                    run_id += 1

                    construct_kwargs = dict(alg_cfg["construct_kwargs"])
                    construct_kwargs["alpha"] = alpha

                    yield {
                        "run_id": run_id,
                        "algorithm": alg_cfg["algorithm"],
                        "construct_solution_fn": alg_cfg["construct_solution_fn"],
                        "construct_kwargs": construct_kwargs,
                        "pheromone_update": upd_cfg["pheromone_update"],
                        "pheromone_update_fn": upd_cfg["pheromone_update_fn"],
                        "pheromone_update_kwargs": dict(upd_cfg["pheromone_update_kwargs"]),
                        "pheromone_update_batch_size": upd_cfg["pheromone_update_batch_size"],
                        "alpha": alpha,
                        "beta": beta,
                        "evaporation": evaporation,
                        "q": q,
                        "seed": seed,
                        "feature_mode": feature_mode,
                        "mode": mode,
                    }


# =============================================================================
# Land-use ecological validation
# =============================================================================

def prepare_land_use_eval(vegetation_path: Path, legend_path: Path):
    vegetation = pd.read_csv(vegetation_path)
    legend = load_clc_legend(legend_path)
    return vegetation, legend


def evaluate_land_use(labels, data_df, node_col, vegetation, legend) -> dict:
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

    merged = add_clc_metadata(
        merged,
        legend=legend,
        class_col="clc_class",
    )

    merged["land_use_component"] = merged.apply(land_use_component, axis=1)
    merged["agriculture_class"] = merged.apply(agriculture_class_label, axis=1)
    merged["natural_class"] = merged.apply(natural_class_label, axis=1)

    per_cluster = land_use_adjusted_ecological_consistency(
        merged,
        cluster_col="cluster",
    )

    overall = overall_land_use_summary(per_cluster)
    return overall


# =============================================================================
# Single run
# =============================================================================

def run_one_config(
    cfg,
    data,
    kmeans,
    kmeans_fit,
    kmeans_terms,
    vegetation,
    legend,
):
    X = data["X_model"]
    neighbors = data["neighbors"]
    weights = data["weights"]

    t0 = time.perf_counter()

    opt_kwargs = {}
    if cfg["pheromone_update_batch_size"] is not None:
        opt_kwargs["pheromone_update_batch_size"] = cfg["pheromone_update_batch_size"]

    optimizer = AntClusteringOptimizer(
        n_clusters=N_CLUSTERS,
        n_ants=N_ANTS,
        n_iterations=N_ITERATIONS,
        construct_solution_fn=cfg["construct_solution_fn"],
        construct_kwargs=cfg["construct_kwargs"],
        fitness_fn=fitness_weighted_sum,
        fitness_kwargs=COMMON_FITNESS_KWARGS,
        pheromone_init_fn=initialize_node_cluster_pheromone,
        pheromone_init_kwargs=COMMON_PHEROMONE_INIT_KWARGS,
        pheromone_update_fn=cfg["pheromone_update_fn"],
        pheromone_update_kwargs=cfg["pheromone_update_kwargs"],
        evaporation=cfg["evaporation"],
        q=cfg["q"],
        random_state=cfg["seed"],
        verbose=False,
        **opt_kwargs,
    )

    optimizer.fit(X, neighbors, weights=weights)

    labels = optimizer.labels_
    centroids = optimizer.centroids_

    recomputed_fitness, terms = fitness_weighted_sum(
        X=X,
        labels=labels,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        **COMMON_FITNESS_KWARGS,
    )

    ecological_overall = evaluate_land_use(
        labels=labels,
        data_df=data["df"],
        node_col=data["node_col"],
        vegetation=vegetation,
        legend=legend,
    )

    row = {
        "run_id": cfg["run_id"],
        "algorithm": cfg["algorithm"],
        "pheromone_update": cfg["pheromone_update"],
        "alpha": cfg["alpha"],
        "beta": cfg["beta"],
        "evaporation": cfg["evaporation"],
        "q": cfg["q"],
        "seed": cfg["seed"],
        "n_ants": N_ANTS,
        "n_iterations": N_ITERATIONS,
        "k_neighbors": K_NEIGHBORS,
        "target_size": TARGET_SIZE,
        "mode": cfg["mode"],
        "feature_mode": cfg["feature_mode"],
        "best_fitness": float(optimizer.best_fitness_),
        "recomputed_fitness": float(recomputed_fitness),
        "ari_vs_kmeans": float(adjusted_rand_score(kmeans.labels_, labels)),
        "runtime_sec": float(time.perf_counter() - t0),
        "final_pheromone_min": float(np.min(optimizer.pheromone_)),
        "final_pheromone_max": float(np.max(optimizer.pheromone_)),
        "final_iter_mean": float(optimizer.term_history_[-1]["mean"]),
        "final_iter_std": float(optimizer.term_history_[-1]["std"]),
        "construct_kwargs_json": json.dumps(
            sanitize_for_json(cfg["construct_kwargs"]),
            sort_keys=True,
        ),
        "kmeans_fitness": float(kmeans_fit),
    }

    for key, value in terms.items():
        if np.isscalar(value):
            row[key] = float(value)

    for key, value in kmeans_terms.items():
        if np.isscalar(value):
            row[f"kmeans_{key}"] = float(value)

    for key, value in ecological_overall.items():
        if np.isscalar(value):
            row[key] = float(value)

    return row


# =============================================================================
# Summary
# =============================================================================

def rebuild_summary(result_path: Path, summary_path: Path, sep: str = CSV_SEP):
    if not result_path.exists():
        return None

    results = read_csv_auto(result_path)

    group_cols = [
        "algorithm",
        "pheromone_update",
        "alpha",
        "beta",
        "evaporation",
        "q",
        "feature_mode",
        "mode",
    ]

    metric_cols = [
        "best_fitness",
        "recomputed_fitness",
        "compactness",
        "spatial",
        "ari_vs_kmeans",
        "weighted_combined_ecological_consistency",
        "mean_combined_ecological_consistency",
        "weighted_agriculture_homogeneity",
        "weighted_natural_homogeneity",
        "weighted_artificial_share_total",
        "runtime_sec",
    ]

    agg_dict = {}
    for col in metric_cols:
        if col in results.columns:
            agg_dict[f"mean_{col}"] = (col, "mean")
            agg_dict[f"std_{col}"] = (col, "std")
            if col in {"best_fitness", "recomputed_fitness"}:
                agg_dict[f"best_{col}"] = (col, "min")
            if col == "weighted_combined_ecological_consistency":
                agg_dict[f"best_{col}"] = (col, "max")

    summary = (
        results
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # Primary sorting: ecological consistency desc, then fitness asc.
    if "mean_weighted_combined_ecological_consistency" in summary.columns:
        summary = summary.sort_values(
            ["mean_weighted_combined_ecological_consistency", "mean_best_fitness"],
            ascending=[False, True],
        )
    else:
        summary = summary.sort_values("mean_best_fitness")

    summary.to_csv(summary_path, index=False, sep=sep)
    return summary


def write_kmeans_baseline(
    path: Path,
    data,
    kmeans,
    kmeans_fit,
    kmeans_terms,
    vegetation,
    legend,
    sep: str = CSV_SEP,
):
    ecological_overall = evaluate_land_use(
        labels=kmeans.labels_,
        data_df=data["df"],
        node_col=data["node_col"],
        vegetation=vegetation,
        legend=legend,
    )

    row = {
        "algorithm": "KMeans",
        "n_clusters": N_CLUSTERS,
        "k_neighbors": K_NEIGHBORS,
        "target_size": TARGET_SIZE,
        "mode": MODE,
        "feature_mode": FEATURE_MODE,
        "fitness": float(kmeans_fit),
    }

    for key, value in kmeans_terms.items():
        if np.isscalar(value):
            row[key] = float(value)

    for key, value in ecological_overall.items():
        if np.isscalar(value):
            row[key] = float(value)

    pd.DataFrame([row]).to_csv(path, index=False, sep=sep)


# =============================================================================
# CLI
# =============================================================================

def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 experiment sweep with ecological validation."
    )

    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--vegetation-path", type=str, default=str(VEGETATION_PATH))
    parser.add_argument("--legend-path", type=str, default=str(LEGEND_PATH))

    parser.add_argument("--feature-mode", type=str, default=FEATURE_MODE, choices=["koppen", "raw"])
    parser.add_argument("--mode", type=str, default=MODE, choices=["superpixels", "pixels"])

    parser.add_argument("--alphas", type=str, default=",".join(map(str, ALPHAS)))
    parser.add_argument("--betas", type=str, default=",".join(map(str, BETAS)))
    parser.add_argument("--evaporations", type=str, default=",".join(map(str, EVAPORATIONS)))
    parser.add_argument("--qs", type=str, default=",".join(map(str, QS)))
    parser.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))

    parser.add_argument("--results-path", type=str, default=str(RESULT_PATH))
    parser.add_argument("--failures-path", type=str, default=str(FAILURE_PATH))
    parser.add_argument("--summary-path", type=str, default=str(SUMMARY_PATH))
    parser.add_argument("--kmeans-path", type=str, default=str(KMEANS_PATH))
    parser.add_argument("--log-path", type=str, default=str(LOG_PATH))

    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--max-runs", type=int, default=MAX_RUNS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    result_path = Path(args.results_path)
    failure_path = Path(args.failures_path)
    summary_path = Path(args.summary_path)
    kmeans_path = Path(args.kmeans_path)
    log_path = Path(args.log_path)

    logger = setup_logging(log_path)

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)
    evaporations = parse_float_list(args.evaporations)
    qs = parse_float_list(args.qs)
    seeds = parse_int_list(args.seeds)

    logger.info("Loading problem data...")
    data = load_problem(
        data_path=Path(args.data_path),
        feature_mode=args.feature_mode,
        mode=args.mode,
        target_size=TARGET_SIZE,
        k_neighbors=K_NEIGHBORS,
    )

    logger.info("Loading land-cover data...")
    vegetation, legend = prepare_land_use_eval(
        vegetation_path=Path(args.vegetation_path),
        legend_path=Path(args.legend_path),
    )

    logger.info("Computing KMeans baseline...")
    kmeans, kmeans_centroids, kmeans_fit, kmeans_terms = compute_kmeans_baseline(
        X=data["X_model"],
        neighbors=data["neighbors"],
        weights=data["weights"],
    )
    logger.info("KMeans fitness: %.6f", kmeans_fit)
    logger.info("KMeans terms: %s", kmeans_terms)

    write_kmeans_baseline(
        path=kmeans_path,
        data=data,
        kmeans=kmeans,
        kmeans_fit=kmeans_fit,
        kmeans_terms=kmeans_terms,
        vegetation=vegetation,
        legend=legend,
    )
    logger.info("Wrote KMeans baseline to %s", kmeans_path)

    configs = list(
        iter_experiment_configs(
            alphas=alphas,
            betas=betas,
            evaporations=evaporations,
            qs=qs,
            seeds=seeds,
            feature_mode=args.feature_mode,
            mode=args.mode,
        )
    )

    total_configs = len(configs)

    if not args.no_resume:
        completed = load_completed_keys(result_path)
        before = len(configs)
        configs = [cfg for cfg in configs if config_key(cfg) not in completed]
        skipped = before - len(configs)
    else:
        skipped = 0

    if args.max_runs is not None:
        configs = configs[:args.max_runs]

    logger.info("Total grid configs: %s", total_configs)
    logger.info("Skipped by resume: %s", skipped)
    logger.info("Pending configs to run now: %s", len(configs))
    logger.info("Algorithms per beta: %s", len(build_algorithm_configs(betas[0])) if betas else 0)
    logger.info("Alphas: %s", alphas)
    logger.info("Betas: %s", betas)
    logger.info("Evaporations: %s", evaporations)
    logger.info("Qs: %s", qs)
    logger.info("Seeds: %s", seeds)
    logger.info("Results path: %s", result_path)
    logger.info("Summary path: %s", summary_path)
    logger.info("Failures path: %s", failure_path)

    if args.dry_run:
        logger.info("Dry run requested. Exiting.")
        return

    rows_buffer = []
    failures_buffer = []

    iterator = enumerate(configs, start=1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(configs), desc="Stage 1 sweep", unit="run")

    for idx, cfg in iterator:
        msg = (
            f"[{idx}/{len(configs)}] "
            f"{cfg['algorithm']} | {cfg['pheromone_update']} | "
            f"alpha={cfg['alpha']} beta={cfg['beta']} "
            f"evap={cfg['evaporation']} q={cfg['q']} seed={cfg['seed']}"
        )

        if tqdm is not None:
            tqdm.write(msg)
        else:
            logger.info(msg)

        try:
            row = run_one_config(
                cfg=cfg,
                data=data,
                kmeans=kmeans,
                kmeans_fit=kmeans_fit,
                kmeans_terms=kmeans_terms,
                vegetation=vegetation,
                legend=legend,
            )

            rows_buffer.append(row)

            logger.info(
                "SUCCESS algorithm=%s update=%s alpha=%s beta=%s seed=%s fitness=%.6f ari=%.4f eco=%.4f runtime=%.2fs",
                cfg["algorithm"],
                cfg["pheromone_update"],
                cfg["alpha"],
                cfg["beta"],
                cfg["seed"],
                row["best_fitness"],
                row["ari_vs_kmeans"],
                row.get("weighted_combined_ecological_consistency", np.nan),
                row["runtime_sec"],
            )

        except Exception as exc:
            failure = {
                "run_id": cfg["run_id"],
                "algorithm": cfg["algorithm"],
                "pheromone_update": cfg["pheromone_update"],
                "alpha": cfg["alpha"],
                "beta": cfg["beta"],
                "evaporation": cfg["evaporation"],
                "q": cfg["q"],
                "seed": cfg["seed"],
                "feature_mode": cfg["feature_mode"],
                "mode": cfg["mode"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures_buffer.append(failure)

            logger.error(
                "FAILURE run_id=%s algorithm=%s update=%s alpha=%s beta=%s seed=%s error=%s",
                cfg["run_id"],
                cfg["algorithm"],
                cfg["pheromone_update"],
                cfg["alpha"],
                cfg["beta"],
                cfg["seed"],
                repr(exc),
            )
            logger.error(failure["traceback"])

        if len(rows_buffer) >= args.save_every:
            append_csv(result_path, rows_buffer)
            logger.info("Flushed %s successful rows to %s", len(rows_buffer), result_path)
            rows_buffer = []

        if len(failures_buffer) >= args.save_every:
            append_csv(failure_path, failures_buffer)
            logger.info("Flushed %s failure rows to %s", len(failures_buffer), failure_path)
            failures_buffer = []

    if rows_buffer:
        append_csv(result_path, rows_buffer)
        logger.info("Final flush: %s successful rows to %s", len(rows_buffer), result_path)

    if failures_buffer:
        append_csv(failure_path, failures_buffer)
        logger.info("Final flush: %s failure rows to %s", len(failures_buffer), failure_path)

    logger.info("Rebuilding summary...")
    summary = rebuild_summary(result_path, summary_path)

    if summary is not None:
        logger.info("Top configurations by ecological consistency:")
        logger.info("\n%s", summary.head(20).to_string(index=False))

        if "mean_best_fitness" in summary.columns:
            by_fitness = summary.sort_values("mean_best_fitness")
            logger.info("Top configurations by fitness:")
            logger.info("\n%s", by_fitness.head(20).to_string(index=False))

    logger.info("Done.")


if __name__ == "__main__":
    main()
