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

## Main workflow

1. Prepare climate raster data.
2. Build the graph representation.
3. Run parameter sweeps or individual experiments.
4. Evaluate generated regionalizations.
5. Analyze and visualize results.

The systematic experiment runs are executed from `experiments/`, while reusable functionality is implemented in `moduli/`. The notebooks are kept for data preparation, checks, and analysis.

## Notes

This repository is an active research project. The code is organized to prioritize experimentation and extensibility rather than providing a production-ready package.
