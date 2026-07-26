"""
Stage 2 validation of top Stage 1 configurations.

This script:
    1. reads stage1_sweep_summary.csv,
    2. filters configurations by mean fitness,
    3. optionally keeps only the top N configurations,
    4. reruns those configurations for selected seeds,
    5. computes:
        - fitness terms,
        - ARI vs KMeans,
        - land-use-adjusted ecological consistency,
        - artificial share,
        - runtime,
    6. writes:
        - results/stage2_validation_runs.csv
        - results/stage2_validation_summary.csv
        - results/stage2_validation_stats.csv

Run from project root:

    python stage2_validate_top_configs.py --top-n 20

Run all configurations with mean_fitness < 0.36:

    python stage2_validate_top_configs.py --top-n -1 --fitness-threshold 0.36

Recommended first test:

    python stage2_validate_top_configs.py --top-n 5 --save-every 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Make imports work when this script is inside kod/experiments/.
# Expected structure:
#   kod/
#     moduli/
#     experiments/stage2_validate_top_configs.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent    
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moduli.preprocessing import (
    build_koppen_features,
    build_raw_features,
    build_temp_rain_pca_features,
    find_temperature_rainfall_columns,
    prepare_model_data,
)

from moduli.cluster_centroid_manipulation import compute_centroids
from moduli.ant_clustering_optimizer import AntClusteringOptimizer
from moduli.fitness_functions import fitness_weighted_sum
from moduli.pheromone_updates import (
    initialize_node_cluster_pheromone,
    update_pheromone_all_ants,
    update_pheromone_best_ant,
    update_pheromone_quality_weighted,
)
from moduli.strategy_utils import normalize_probabilities

from moduli.construction_strategies import (
    construct_aco_solution,
    construct_aco_refinement_solution,
    construct_graph_aco_solution,
    construct_multicolony_solution,
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
# Defaults matching the Stage 1 experiments
# =============================================================================

DATA_PATH = DATA_ROOT / "df_valid.csv"
VEGETATION_PATH = DATA_ROOT / "vegetacija.csv"
LEGEND_PATH = DATA_ROOT / "Vegetacija" / "u2018_clc2018_v2020_20u1_raster100m" / "Legend" / "CLC2018_CLC2018_V2018_20_QGIS.txt"

RESULT_DIR = PROJECT_ROOT / "experiments" / "results"
SUMMARY_PATH = "stage1_sweep_summary.csv"
OUTPUT_RUNS_PATH = "stage2_validation_runs.csv"
OUTPUT_SUMMARY_PATH = "stage2_validation_summary.csv"
OUTPUT_STATS_PATH = "stage2_validation_stats.csv"
LOG_PATH = "stage2_validation.log"

MODE = "superpixels"
FEATURE_MODE = "koppen"

N_CLUSTERS = 5
TARGET_SIZE = 150
K_NEIGHBORS = 16
N_ANTS = 15
N_ITERATIONS = 20

DEFAULT_SEEDS = "0,1,2,3,4"

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
# Utility functions
# =============================================================================

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage2_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

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


def append_csv(path: Path, rows: list[dict], sep: str = ";"):
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


def feature_result_dir(feature_mode: str) -> Path:
    return RESULT_DIR / feature_mode


def resolve_result_path(path_text: str | None, feature_mode: str, filename: str) -> Path:
    if path_text:
        return Path(path_text)
    return feature_result_dir(feature_mode) / filename


def parse_seeds(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip() != ""]


def config_key(row, seed: int):
    return (
        str(row["algorithm"]),
        str(row["pheromone_update"]),
        float(row["alpha"]),
        float(row["beta"]),
        float(row["evaporation"]),
        float(row["q"]),
        int(seed),
        str(row.get("feature_mode", FEATURE_MODE)),
        str(row.get("mode", MODE)),
    )


def load_completed_run_keys(path: Path):
    if not path.exists():
        return set()

    df = read_csv_auto(path)

    required = {"algorithm", "pheromone_update", "alpha", "beta", "evaporation", "q", "seed"}
    if not required.issubset(df.columns):
        return set()

    keys = set()
    for _, row in df.iterrows():
        keys.add(config_key(row, int(row["seed"])))

    return keys


def stage1_fitness_column(summary: pd.DataFrame) -> str:
    for col in ["mean_fitness", "mean_best_fitness", "mean_recomputed_fitness"]:
        if col in summary.columns:
            return col
    raise ValueError(
        "Stage 1 summary must contain one of: mean_fitness, "
        "mean_best_fitness, mean_recomputed_fitness."
    )


def deduplicate_validation_runs(runs: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["algorithm", "pheromone_update", "alpha", "beta", "evaporation", "q", "seed"]
    for optional_col in ["feature_mode", "mode"]:
        if optional_col in runs.columns:
            key_cols.append(optional_col)

    return runs.drop_duplicates(key_cols, keep="last").reset_index(drop=True)


def select_validation_candidates(
    summary: pd.DataFrame,
    fitness_col: str,
    fitness_threshold: float,
    top_n: int,
    top_n_per_algorithm: int,
) -> pd.DataFrame:
    candidates = summary[summary[fitness_col] < fitness_threshold].copy()
    candidates = candidates.sort_values(fitness_col)

    selected_parts = []

    if top_n is not None and top_n > 0:
        selected_parts.append(candidates.head(top_n))
    elif top_n == -1:
        selected_parts.append(candidates)

    if top_n_per_algorithm is not None and top_n_per_algorithm > 0:
        selected_parts.append(
            candidates
            .groupby("algorithm", group_keys=False, dropna=False)
            .head(top_n_per_algorithm)
        )

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = candidates

    key_cols = ["algorithm", "pheromone_update", "alpha", "beta", "evaporation", "q"]
    for optional_col in ["feature_mode", "mode"]:
        if optional_col in selected.columns:
            key_cols.append(optional_col)

    return (
        selected
        .drop_duplicates(key_cols, keep="first")
        .sort_values(fitness_col)
        .reset_index(drop=True)
    )


def load_problem(data_path: Path, feature_mode: str, mode: str, target_size: int, k_neighbors: int):
    df = pd.read_csv(data_path, index_col=0)

    if feature_mode == "koppen":
        X_features, feature_names = build_koppen_features(df)
    elif feature_mode == "raw":
        X_features, feature_names = build_raw_features(df)
    elif feature_mode == "pca":
        feature_cols = build_raw_features(df)[1]
        temp_cols, rain_cols = find_temperature_rainfall_columns(
            df,
            feature_cols=feature_cols,
        )
        X_features, feature_names, _ = build_temp_rain_pca_features(
            df,
            temp_cols=temp_cols,
            rain_cols=rain_cols,
            n_temp=2,
            n_rain=2,
            scale_method="minmax",
            final_scale=True,
        )
    else:
        raise ValueError("feature_mode must be 'koppen', 'raw', or 'pca'.")

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

    return kmeans, float(fit), terms


def pheromone_update_config(name: str):
    mapping = {
        "all_ants": update_pheromone_all_ants,
        "best_ant": update_pheromone_best_ant,
        "quality_weighted": update_pheromone_quality_weighted,
    }

    if name not in mapping:
        raise ValueError(f"Unknown pheromone update strategy: {name}")

    return mapping[name], COMMON_PHEROMONE_UPDATE_KWARGS.copy()


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


def construction_config(algorithm: str, alpha: float, beta: float):
    """
    Map algorithm names from the Stage 1 summary to construction functions.

    Current convention:
        construct_aco_solution(..., centroid=True)  -> true centroid ACO
        construct_aco_solution(..., centroid=False) -> cohesion ACO
        construct_aco_refinement_solution(..., centroid=True) -> centroid refinement
        construct_aco_refinement_solution(..., centroid=False) -> cohesion refinement

    Adjust this mapping if algorithm names in the Stage 1 summary change.
    """
    if algorithm == "aco_centroid":
        return construct_aco_solution, {
            "alpha": alpha,
            "beta_feature": beta,
            "centroid": True,
        }

    if algorithm == "aco_centroid_refinement":
        return construct_aco_refinement_solution, {
            "alpha": alpha,
            "beta_feature": beta,
            "refinement_steps": 3,
            "centroid": True,
        }

    if algorithm == "aco_cohesion":
        return construct_aco_solution, {
            "alpha": alpha,
            "beta_feature": beta,
            "centroid": False,
        }

    if algorithm == "aco_cohesion_refinement":
        return construct_aco_refinement_solution, {
            "alpha": alpha,
            "beta_feature": beta,
            "refinement_steps": 3,
            "centroid": False,
        }

    if algorithm == "aco_graph_local":
        return construct_graph_aco_solution, {
            "alpha": alpha,
            "beta_graph": beta,
        }

    if algorithm == "multi_centroid_graph_spatial":
        return construct_multicolony_solution, {
            "colony_heuristics": [
                make_centroid_colony(beta),
                make_graph_colony(beta),
                make_spatial_colony(beta),
            ],
            "alpha": alpha,
        }

    if algorithm == "multi_cohesion_graph_spatial":
        return construct_multicolony_solution, {
            "colony_heuristics": [
                make_cohesion_colony(beta),
                make_graph_colony(beta),
                make_spatial_colony(beta),
            ],
            "alpha": alpha,
        }

    raise ValueError(f"Unknown algorithm name: {algorithm}")

def evaluate_land_use(labels, data_df, node_col, vegetation, legend):
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

    return overall, per_cluster


def run_one(row, seed, data, kmeans, vegetation, legend, feature_mode: str, mode: str):
    X = data["X_model"]
    neighbors = data["neighbors"]
    weights = data["weights"]

    algorithm = str(row["algorithm"])
    update_name = str(row["pheromone_update"])
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    evaporation = float(row["evaporation"])
    q = float(row["q"])

    construct_fn, construct_kwargs = construction_config(
        algorithm=algorithm,
        alpha=alpha,
        beta=beta,
    )

    update_fn, update_kwargs = pheromone_update_config(update_name)

    t0 = time.perf_counter()

    opt = AntClusteringOptimizer(
        n_clusters=N_CLUSTERS,
        n_ants=N_ANTS,
        n_iterations=N_ITERATIONS,
        construct_solution_fn=construct_fn,
        construct_kwargs=construct_kwargs,
        fitness_fn=fitness_weighted_sum,
        fitness_kwargs=COMMON_FITNESS_KWARGS,
        pheromone_init_fn=initialize_node_cluster_pheromone,
        pheromone_init_kwargs=COMMON_PHEROMONE_INIT_KWARGS,
        pheromone_update_fn=update_fn,
        pheromone_update_kwargs=update_kwargs,
        evaporation=evaporation,
        q=q,
        random_state=seed,
        verbose=False,
    )

    opt.fit(X, neighbors, weights=weights)

    labels = opt.labels_
    centroids = opt.centroids_

    recomputed_fitness, terms = fitness_weighted_sum(
        X=X,
        labels=labels,
        centroids=centroids,
        weights=weights,
        neighbors=neighbors,
        **COMMON_FITNESS_KWARGS,
    )

    ecological_overall, _ = evaluate_land_use(
        labels=labels,
        data_df=data["df"],
        node_col=data["node_col"],
        vegetation=vegetation,
        legend=legend,
    )

    out = {
        "algorithm": algorithm,
        "pheromone_update": update_name,
        "alpha": alpha,
        "beta": beta,
        "evaporation": evaporation,
        "q": q,
        "feature_mode": feature_mode,
        "mode": mode,
        "seed": seed,
        "stage1_mean_fitness": float(row.get("mean_fitness", row.get("mean_best_fitness", np.nan))),
        "stage1_best_fitness": float(row.get("best_fitness", row.get("best_best_fitness", np.nan))),
        "best_fitness": float(opt.best_fitness_),
        "recomputed_fitness": float(recomputed_fitness),
        "ari_vs_kmeans": float(adjusted_rand_score(kmeans.labels_, labels)),
        "runtime_sec": float(time.perf_counter() - t0),
        "final_pheromone_min": float(np.min(opt.pheromone_)),
        "final_pheromone_max": float(np.max(opt.pheromone_)),
    }

    for k, v in terms.items():
        if np.isscalar(v):
            out[k] = float(v)

    for k, v in ecological_overall.items():
        if np.isscalar(v):
            out[k] = float(v)

    return out


def rebuild_validation_summary(runs_path: Path, summary_path: Path, sep: str = ";"):
    runs = deduplicate_validation_runs(read_csv_auto(runs_path))

    group_cols = ["algorithm", "pheromone_update", "alpha", "beta", "evaporation", "q"]
    for optional_col in ["feature_mode", "mode"]:
        if optional_col in runs.columns:
            group_cols.append(optional_col)

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
    for c in metric_cols:
        if c in runs.columns:
            agg_dict[f"mean_{c}"] = (c, "mean")
            agg_dict[f"std_{c}"] = (c, "std")

    summary = (
        runs
        .groupby(group_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    sort_col = "mean_weighted_combined_ecological_consistency"
    if sort_col in summary.columns:
        summary = summary.sort_values(
            [sort_col, "mean_best_fitness"],
            ascending=[False, True],
        )
    else:
        summary = summary.sort_values("mean_best_fitness")

    summary.to_csv(summary_path, index=False, sep=sep)
    return summary


def _paired_test_values(a: pd.Series, b: pd.Series):
    diffs = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs)]

    if len(diffs) < 2:
        return {
            "n_pairs": int(len(diffs)),
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "std_diff": np.nan,
            "cohen_dz": np.nan,
            "paired_t_stat": np.nan,
            "paired_t_p": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
        }

    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan
    mean_diff = float(np.mean(diffs))

    try:
        t_res = stats.ttest_rel(a, b, nan_policy="omit")
        paired_t_stat = float(t_res.statistic)
        paired_t_p = float(t_res.pvalue)
    except Exception:
        paired_t_stat = np.nan
        paired_t_p = np.nan

    try:
        if np.allclose(diffs, 0.0):
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
        else:
            w_res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
    except Exception:
        wilcoxon_stat = np.nan
        wilcoxon_p = np.nan

    return {
        "n_pairs": int(len(diffs)),
        "mean_diff": mean_diff,
        "median_diff": float(np.median(diffs)),
        "std_diff": std_diff,
        "cohen_dz": mean_diff / std_diff if std_diff and np.isfinite(std_diff) else np.nan,
        "paired_t_stat": paired_t_stat,
        "paired_t_p": paired_t_p,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
    }


def rebuild_significance_stats(runs_path: Path, stats_path: Path, sep: str = ";"):
    runs = deduplicate_validation_runs(read_csv_auto(runs_path))

    config_cols = ["algorithm", "pheromone_update", "alpha", "beta", "evaporation", "q"]
    for optional_col in ["feature_mode", "mode"]:
        if optional_col in runs.columns:
            config_cols.append(optional_col)

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
    metric_cols = [c for c in metric_cols if c in runs.columns]

    configs = (
        runs[config_cols]
        .drop_duplicates()
        .sort_values(config_cols)
        .reset_index(drop=True)
    )

    rows = []
    for i in range(len(configs)):
        cfg_a = configs.iloc[i]
        mask_a = np.ones(len(runs), dtype=bool)
        for col in config_cols:
            mask_a &= runs[col] == cfg_a[col]

        for j in range(i + 1, len(configs)):
            cfg_b = configs.iloc[j]
            mask_b = np.ones(len(runs), dtype=bool)
            for col in config_cols:
                mask_b &= runs[col] == cfg_b[col]

            a = runs.loc[mask_a, ["seed", *metric_cols]]
            b = runs.loc[mask_b, ["seed", *metric_cols]]
            paired = a.merge(b, on="seed", suffixes=("_a", "_b"))

            if paired.empty:
                continue

            base = {}
            for col in config_cols:
                base[f"{col}_a"] = cfg_a[col]
                base[f"{col}_b"] = cfg_b[col]

            for metric in metric_cols:
                values = _paired_test_values(paired[f"{metric}_a"], paired[f"{metric}_b"])
                rows.append({
                    **base,
                    "metric": metric,
                    **values,
                })

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df = stats_df.sort_values(["metric", "wilcoxon_p", "paired_t_p"], na_position="last")

    stats_df.to_csv(stats_path, index=False, sep=sep)
    return stats_df


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 validation for top Stage 1 configurations.")

    parser.add_argument("--stage1-summary", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=20, help="Number of top configs to validate. Use -1 for all after threshold.")
    parser.add_argument("--top-n-per-algorithm", type=int, default=0, help="Also include the top N configs within each algorithm family.")
    parser.add_argument("--fitness-threshold", type=float, default=0.36)
    parser.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)

    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--vegetation-path", type=str, default=str(VEGETATION_PATH))
    parser.add_argument("--legend-path", type=str, default=str(LEGEND_PATH))

    parser.add_argument("--mode", type=str, default=MODE)
    parser.add_argument("--feature-mode", type=str, default=FEATURE_MODE)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)

    parser.add_argument("--output-runs", type=str, default=None)
    parser.add_argument("--output-summary", type=str, default=None)
    parser.add_argument("--output-stats", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sep-output", type=str, default=";")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    stage1_summary_path = resolve_result_path(args.stage1_summary, args.feature_mode, SUMMARY_PATH)
    output_runs_path = resolve_result_path(args.output_runs, args.feature_mode, OUTPUT_RUNS_PATH)
    output_summary_path = resolve_result_path(args.output_summary, args.feature_mode, OUTPUT_SUMMARY_PATH)
    output_stats_path = resolve_result_path(args.output_stats, args.feature_mode, OUTPUT_STATS_PATH)
    log_path = resolve_result_path(args.log_path, args.feature_mode, LOG_PATH)

    logger = setup_logging(log_path)

    seeds = parse_seeds(args.seeds)

    logger.info("Reading Stage 1 summary: %s", stage1_summary_path)
    summary = read_csv_auto(stage1_summary_path)

    fitness_col = stage1_fitness_column(summary)
    logger.info("Using Stage 1 fitness column for filtering/sorting: %s", fitness_col)

    candidates = select_validation_candidates(
        summary=summary,
        fitness_col=fitness_col,
        fitness_threshold=args.fitness_threshold,
        top_n=args.top_n,
        top_n_per_algorithm=args.top_n_per_algorithm,
    )

    logger.info("Candidate configs after threshold/top-n: %s", len(candidates))
    if args.top_n_per_algorithm > 0:
        logger.info("Included up to %s configs per algorithm family.", args.top_n_per_algorithm)
    logger.info("Seeds: %s", seeds)

    expanded = []
    for _, row in candidates.iterrows():
        for seed in seeds:
            expanded.append((row, seed))

    if not args.no_resume:
        completed = load_completed_run_keys(output_runs_path)
        before = len(expanded)
        expanded = [
            (row, seed)
            for row, seed in expanded
            if config_key(row, seed) not in completed
        ]
        logger.info("Skipped by resume: %s", before - len(expanded))

    logger.info("Total runs to execute: %s", len(expanded))

    if args.dry_run:
        logger.info("Dry run requested. Exiting.")
        return

    logger.info("Loading problem data...")
    data = load_problem(
        data_path=Path(args.data_path),
        feature_mode=args.feature_mode,
        mode=args.mode,
        target_size=args.target_size,
        k_neighbors=args.k_neighbors,
    )

    logger.info("Loading vegetation and legend...")
    vegetation = pd.read_csv(args.vegetation_path)
    legend = load_clc_legend(args.legend_path)

    logger.info("Computing KMeans baseline...")
    kmeans, kmeans_fit, kmeans_terms = compute_kmeans_baseline(
        X=data["X_model"],
        neighbors=data["neighbors"],
        weights=data["weights"],
    )
    logger.info("KMeans fitness: %.6f", kmeans_fit)
    logger.info("KMeans terms: %s", kmeans_terms)

    buffer = []

    iterator = enumerate(expanded, start=1)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(expanded), desc="Stage 2 validation", unit="run")

    for idx, (row, seed) in iterator:
        msg = (
            f"[{idx}/{len(expanded)}] {row['algorithm']} | {row['pheromone_update']} | "
            f"alpha={row['alpha']} beta={row['beta']} evap={row['evaporation']} q={row['q']} seed={seed}"
        )
        if tqdm is not None:
            tqdm.write(msg)
        else:
            logger.info(msg)

        try:
            result = run_one(
                row=row,
                seed=seed,
                data=data,
                kmeans=kmeans,
                vegetation=vegetation,
                legend=legend,
                feature_mode=args.feature_mode,
                mode=args.mode,
            )
            buffer.append(result)

            logger.info(
                "SUCCESS algorithm=%s seed=%s fitness=%.6f ari=%.4f eco=%.4f",
                result["algorithm"],
                seed,
                result["best_fitness"],
                result["ari_vs_kmeans"],
                result.get("weighted_combined_ecological_consistency", np.nan),
            )

        except Exception as e:
            logger.exception("FAILED config=%s seed=%s error=%s", dict(row), seed, repr(e))
            fail_row = {
                "algorithm": row.get("algorithm"),
                "pheromone_update": row.get("pheromone_update"),
                "alpha": row.get("alpha"),
                "beta": row.get("beta"),
                "evaporation": row.get("evaporation"),
                "q": row.get("q"),
                "feature_mode": args.feature_mode,
                "mode": args.mode,
                "seed": seed,
                "error": repr(e),
            }
            append_csv(
                output_runs_path.with_name(output_runs_path.stem + "_failures.csv"),
                [fail_row],
                sep=args.sep_output,
            )

        if len(buffer) >= args.save_every:
            append_csv(output_runs_path, buffer, sep=args.sep_output)
            logger.info("Flushed %s rows to %s", len(buffer), output_runs_path)
            buffer = []

    if buffer:
        append_csv(output_runs_path, buffer, sep=args.sep_output)
        logger.info("Final flush: %s rows to %s", len(buffer), output_runs_path)

    logger.info("Rebuilding validation summary...")
    validation_summary = rebuild_validation_summary(
        runs_path=output_runs_path,
        summary_path=output_summary_path,
        sep=args.sep_output,
    )

    logger.info("Top validation summary rows:")
    logger.info("\n%s", validation_summary.head(20).to_string(index=False))

    logger.info("Rebuilding significance statistics...")
    stats_summary = rebuild_significance_stats(
        runs_path=output_runs_path,
        stats_path=output_stats_path,
        sep=args.sep_output,
    )
    logger.info("Wrote %s statistical comparison rows to %s", len(stats_summary), output_stats_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
