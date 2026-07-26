"""
Run raw/PCA Stage 1 sweeps and balanced Stage 2 validations with checkpoints.

The supervisor stops on failures or sanity-check anomalies and can be resumed by
rerunning the same command. The underlying experiment scripts handle completed
config resume through their CSV outputs.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT_ROOT / "experiments" / "results"
LOG_PATH = RESULT_ROOT / "overnight_supervisor.log"
CHECK_INTERVAL_SEC = 60
CHECKPOINT_EVERY = 50
STAGE1_EXPECTED_UNIQUE = 504


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {message}"
    print(text, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=";")


def unique_stage1_runs(feature: str) -> pd.DataFrame:
    runs = read_csv(RESULT_ROOT / feature / "stage1_sweep_runs.csv")
    if runs.empty:
        return runs
    key_cols = [
        "algorithm",
        "pheromone_update",
        "alpha",
        "beta",
        "evaporation",
        "q",
        "seed",
        "feature_mode",
        "mode",
    ]
    return runs.drop_duplicates([c for c in key_cols if c in runs.columns], keep="last")


def failure_count(path: Path) -> int:
    failures = read_csv(path)
    return len(failures)


def check_stage1(feature: str, force: bool = False) -> int:
    base = RESULT_ROOT / feature
    runs = unique_stage1_runs(feature)
    n = len(runs)
    failures_path = base / "stage1_sweep_failures.csv"
    failures = failure_count(failures_path)

    if failures:
        raise RuntimeError(f"{feature} Stage 1 has {failures} failure rows in {failures_path}.")

    if runs.empty:
        if force:
            log(f"{feature} Stage 1 checkpoint: rows=0 failures=0")
        return 0

    required = ["best_fitness", "weighted_combined_ecological_consistency"]
    for col in required:
        if col not in runs.columns:
            raise RuntimeError(f"{feature} Stage 1 missing required column: {col}")
        values = runs[col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{feature} Stage 1 has non-finite values in {col}.")

    if "final_pheromone_min" in runs.columns:
        mins = runs["final_pheromone_min"].to_numpy(dtype=float)
        if not np.isfinite(mins).all():
            raise RuntimeError(f"{feature} Stage 1 has non-finite final_pheromone_min.")

    if "final_pheromone_max" in runs.columns:
        maxs = runs["final_pheromone_max"].to_numpy(dtype=float)
        if not np.isfinite(maxs).all():
            raise RuntimeError(f"{feature} Stage 1 has non-finite final_pheromone_max.")

    if force or n % CHECKPOINT_EVERY == 0:
        by_alg = (
            runs.groupby("algorithm")["best_fitness"]
            .agg(["count", "mean", "min", "max"])
            .round(6)
        )
        log(
            f"{feature} Stage 1 checkpoint: rows={n} "
            f"fitness=({runs['best_fitness'].min():.6f}, {runs['best_fitness'].max():.6f}) "
            f"eco=({runs['weighted_combined_ecological_consistency'].min():.6f}, "
            f"{runs['weighted_combined_ecological_consistency'].max():.6f})"
        )
        log(f"{feature} Stage 1 by_algorithm:\n{by_alg.to_string()}")

    return n


def check_stage2(feature: str, force: bool = False) -> int:
    base = RESULT_ROOT / feature
    runs = read_csv(base / "stage2_validation_runs.csv")
    failures_path = base / "stage2_validation_runs_failures.csv"
    failures = failure_count(failures_path)

    if failures:
        raise RuntimeError(f"{feature} Stage 2 has {failures} failure rows in {failures_path}.")

    if runs.empty:
        if force:
            log(f"{feature} Stage 2 checkpoint: rows=0 failures=0")
        return 0

    if "best_fitness" in runs.columns:
        values = runs["best_fitness"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{feature} Stage 2 has non-finite best_fitness values.")

    if force or len(runs) % CHECKPOINT_EVERY == 0:
        log(
            f"{feature} Stage 2 checkpoint: rows={len(runs)} "
            f"fitness=({runs['best_fitness'].min():.6f}, {runs['best_fitness'].max():.6f})"
        )

    return len(runs)


def run_command(name: str, command: list[str], console_path: Path, check_fn) -> None:
    log(f"START {name}: {' '.join(command)}")
    console_path.parent.mkdir(parents=True, exist_ok=True)

    with open(console_path, "a", encoding="utf-8") as console:
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
        )

        last_checkpoint_bucket = -1

        while True:
            rc = proc.poll()
            rows = check_fn(force=False)
            bucket = math.floor(rows / CHECKPOINT_EVERY)
            if bucket > last_checkpoint_bucket and rows > 0:
                check_fn(force=True)
                last_checkpoint_bucket = bucket

            if rc is not None:
                if rc != 0:
                    raise RuntimeError(f"{name} exited with code {rc}. See {console_path}.")
                check_fn(force=True)
                log(f"DONE {name}")
                return

            time.sleep(CHECK_INTERVAL_SEC)


def main() -> int:
    py = str(PROJECT_ROOT / ".venv" / "bin" / "python")

    stages = [
        (
            "raw Stage 1",
            "raw",
            [
                py,
                "experiments/stage1_sweep.py",
                "--feature-mode",
                "raw",
                "--save-every",
                "1",
            ],
            RESULT_ROOT / "raw" / "stage1_console.log",
            lambda force=False: check_stage1("raw", force=force),
            STAGE1_EXPECTED_UNIQUE,
        ),
        (
            "pca Stage 1",
            "pca",
            [
                py,
                "experiments/stage1_sweep.py",
                "--feature-mode",
                "pca",
                "--save-every",
                "1",
            ],
            RESULT_ROOT / "pca" / "stage1_console.log",
            lambda force=False: check_stage1("pca", force=force),
            STAGE1_EXPECTED_UNIQUE,
        ),
        (
            "raw Stage 2",
            "raw",
            [
                py,
                "experiments/stage2_validate_top_configs.py",
                "--feature-mode",
                "raw",
                "--top-n",
                "5",
                "--top-n-per-algorithm",
                "3",
                "--save-every",
                "1",
            ],
            RESULT_ROOT / "raw" / "stage2_console.log",
            lambda force=False: check_stage2("raw", force=force),
            None,
        ),
        (
            "pca Stage 2",
            "pca",
            [
                py,
                "experiments/stage2_validate_top_configs.py",
                "--feature-mode",
                "pca",
                "--top-n",
                "5",
                "--top-n-per-algorithm",
                "3",
                "--save-every",
                "1",
            ],
            RESULT_ROOT / "pca" / "stage2_console.log",
            lambda force=False: check_stage2("pca", force=force),
            None,
        ),
    ]

    try:
        for name, feature, command, console_path, check_fn, expected in stages:
            if expected is not None and check_fn(force=False) >= expected:
                log(f"SKIP {name}: expected rows already present.")
                continue
            run_command(name, command, console_path, check_fn)
        log("ALL STAGES COMPLETE")
        return 0
    except Exception as exc:
        log(f"STOPPED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
