# Climate Regionalization using Ant Colony Optimization

This repository contains the implementation developed as part of a PhD project on climate regionalization of Serbia using Ant Colony Optimization (ACO).

The project includes:
- preprocessing of climate raster data,
- graph and superpixel construction,
- multiple ACO construction strategies,
- evaluation using optimization and ecological metrics,
- experiment scripts and analysis notebooks.

## Repository structure

```
moduli/                 Core implementation
experiments/            Experiment scripts
experiments/results/    Saved experiment summaries and run outputs
notebooks/              Data preparation, experiments, and analysis
```

Experiment outputs are grouped by feature representation:

```
experiments/results/koppen/    Results using Koppen-derived features
experiments/results/raw/       Results using raw climate features
experiments/results/pca/       Results using PCA-transformed features
```

## Main workflow

1. Prepare climate raster data.
2. Build the graph representation.
3. Run parameter sweeps or individual experiments.
4. Evaluate generated regionalizations.
5. Analyze and visualize results.

The systematic experiment runs are executed from `experiments/`, while reusable functionality is implemented in `moduli/`. The notebooks are kept for data preparation, checks, and analysis.

## Experiment scripts

The main sweep scripts are:

- `experiments/stage1_sweep.py`: runs the general parameter sweep for one feature representation. Use `--feature-mode koppen`, `--feature-mode raw`, or `--feature-mode pca`.
- `experiments/stage2_validate_top_configs.py`: reruns selected top configurations with multiple seeds and writes validation summaries and statistical significance checks.
- `experiments/run_feature_pipeline.py`: orchestrates the long raw/PCA protocol by running raw Stage 1, PCA Stage 1, raw Stage 2, and PCA Stage 2 in sequence.

`run_feature_pipeline.py` is a supervisor script, not a separate experiment definition. It calls the Stage 1 and Stage 2 scripts with the agreed raw/PCA settings, writes console logs under each feature-set result folder, checks progress every 50 completed runs, stops if failures or non-finite metrics appear, and can be resumed by rerunning the same command because the underlying scripts skip completed CSV entries.

To run the raw/PCA pipeline:

```bash
.venv/bin/python experiments/run_feature_pipeline.py
```

For a single feature set or a custom protocol, run `stage1_sweep.py` and `stage2_validate_top_configs.py` directly.

## Notes

This repository is an active research project. The code is organized to prioritize experimentation and extensibility rather than providing a production-ready package.
