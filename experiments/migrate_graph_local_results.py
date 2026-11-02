#!/usr/bin/env python3
"""Replace legacy graph-local Stage 1 rows with the generic ACO rerun.

The migration is deliberately one-shot.  It snapshots every canonical input
file and writes the extracted legacy graph-local rows separately before it
changes the canonical Stage 1 files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from stage1_sweep import rebuild_summary


FEATURE_MODES = ("koppen", "raw", "pca")
FITNESS_NAMES = ("centroid", "silhouette", "snn")
CANONICAL_RUNS = "stage1_sweep_runs.csv"
CANONICAL_SUMMARY = "stage1_sweep_summary.csv"
RERUN_DIR = "graph_local_rerun"
RERUN_RUNS = "stage1_runs.csv"
LEGACY_ALGORITHM = "aco_graph_local"
RERUN_ONLINE_ALGORITHM = "aco_graph_local_online"
REFINEMENT_ALGORITHM = "aco_graph_local_refinement"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=";")


def paths(results_dir: Path, mode: str, fitness: str) -> dict[str, Path]:
    canonical_dir = results_dir / mode / fitness
    archive_dir = results_dir / "legacy_graph_local" / mode / fitness
    rerun_dir = results_dir / mode / RERUN_DIR / fitness
    return {
        "runs": canonical_dir / CANONICAL_RUNS,
        "summary": canonical_dir / CANONICAL_SUMMARY,
        "rerun": rerun_dir / RERUN_RUNS,
        "archive_runs": archive_dir / "legacy_graph_local_runs.csv",
        "archive_summary": archive_dir / "legacy_graph_local_summary.csv",
        "snapshot_runs": archive_dir / "stage1_sweep_runs_before_replacement.csv",
        "snapshot_summary": archive_dir / "stage1_sweep_summary_before_replacement.csv",
    }


def validate_sources(results_dir: Path) -> None:
    errors: list[str] = []
    for mode in FEATURE_MODES:
        for fitness in FITNESS_NAMES:
            item = paths(results_dir, mode, fitness)
            for key in ("runs", "summary", "rerun"):
                if not item[key].is_file():
                    errors.append(f"Missing {key}: {item[key]}")
            if errors:
                continue
            canonical = read_csv(item["runs"])
            rerun = read_csv(item["rerun"])
            old_count = canonical["algorithm"].eq(LEGACY_ALGORITHM).sum()
            expected = rerun["algorithm"].isin(
                [RERUN_ONLINE_ALGORITHM, REFINEMENT_ALGORITHM]
            ).sum()
            if old_count == 0:
                errors.append(f"No legacy rows in {item['runs']}; migration may already have run.")
            if expected == 0:
                errors.append(f"No new graph-local rows in {item['rerun']}.")
            if item["snapshot_runs"].exists() or item["snapshot_summary"].exists():
                errors.append(f"Archive already exists for {mode}/{fitness}; refusing to overwrite it.")
    if errors:
        raise RuntimeError("\n".join(errors))


def migrate(results_dir: Path) -> list[dict[str, object]]:
    validate_sources(results_dir)
    manifest: list[dict[str, object]] = []

    for mode in FEATURE_MODES:
        for fitness in FITNESS_NAMES:
            item = paths(results_dir, mode, fitness)
            canonical = read_csv(item["runs"])
            rerun = read_csv(item["rerun"])
            legacy = canonical[canonical["algorithm"].eq(LEGACY_ALGORITHM)].copy()
            replacement = rerun[rerun["algorithm"].isin(
                [RERUN_ONLINE_ALGORITHM, REFINEMENT_ALGORITHM]
            )].copy()
            replacement.loc[
                replacement["algorithm"].eq(RERUN_ONLINE_ALGORITHM), "algorithm"
            ] = LEGACY_ALGORITHM

            # The complete originals preserve every historical column and row;
            # the separate legacy file makes the retired graph-local data easy
            # to inspect without polluting report analysis.
            item["snapshot_runs"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["runs"], item["snapshot_runs"])
            shutil.copy2(item["summary"], item["snapshot_summary"])
            write_csv(legacy, item["archive_runs"])
            # Some historical centroid rows omitted this column.  Preserve the
            # raw archive exactly, but recover its known path-level metadata
            # for a usable legacy-only summary.
            legacy_for_summary = legacy.copy()
            if "fitness_name" not in legacy_for_summary:
                legacy_for_summary["fitness_name"] = fitness
            else:
                legacy_for_summary["fitness_name"] = legacy_for_summary[
                    "fitness_name"
                ].fillna(fitness)
            write_csv(legacy_for_summary, item["archive_summary"].with_suffix(".tmp"))
            rebuild_summary(
                item["archive_summary"].with_suffix(".tmp"), item["archive_summary"]
            )
            item["archive_summary"].with_suffix(".tmp").unlink()

            retained = canonical[~canonical["algorithm"].eq(LEGACY_ALGORITHM)].copy()
            merged = pd.concat([retained, replacement], ignore_index=True, sort=False)
            write_csv(merged, item["runs"])
            rebuild_summary(item["runs"], item["summary"])
            manifest.append({
                "feature_mode": mode,
                "fitness_name": fitness,
                "legacy_rows_archived": int(len(legacy)),
                "online_rows_relabelled": int(replacement["algorithm"].eq(LEGACY_ALGORITHM).sum()),
                "refinement_rows_added": int(replacement["algorithm"].eq(REFINEMENT_ALGORITHM).sum()),
                "canonical_rows_after": int(len(merged)),
            })

    manifest_path = results_dir / "legacy_graph_local" / "migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    args = parser.parse_args()

    if not args.apply:
        validate_sources(args.results_dir)
        print("Validation passed. Re-run with --apply to perform the migration.")
        return
    for row in migrate(args.results_dir):
        print(row)


if __name__ == "__main__":
    main()
