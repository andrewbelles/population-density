# Population Density Prediction

County-level population density modeling with multi‑modal features, stacking, and graph
post‑processing.

## Current Status

The project now spans three primary feature sources (VIIRS nighttime lights, TIGER roads, NLCD land
cover) and a late‑fusion pipeline:
- Expert models per dataset (VIIRS/TIGER/NLCD).
- A stacked classifier over OOF probabilities.
- Correct‑and‑Smooth graph post‑processing on stacked outputs (Queen or mobility adjacency).

Hyperparameter search uses Optuna with early stopping and YAML config caching (`analysis/
model_config.yaml`, `testbench/model_config.yaml`).

## Data Processing Pipeline

Dataset compilers live in `preprocessing/` and emit `.mat` files under `data/`:
- VIIRS nighttime lights texture/entropy features.
- TIGER road topology statistics.
- NLCD land cover configuration metrics.
- Mobility / travel‑time adjacency sources.
- Optional climate and SAIPE baselines.

See `preprocessing/README.md` for the per‑dataset build steps.

## Evaluation and Optimization

Core tools:
- `analysis/cross_validation.py` for repeated CV with fold‑safe scaling.
- `analysis/hyperparameter.py` for Optuna optimization and nested CV.
- `testbench/stacking.py` for the full stacking + C+S pipeline.

Quick start (stacking pipeline):
```bash
nix-shell
python testbench/stacking.py --resume
```

## Repository Layout

- `analysis/`: CV harness, Optuna, graph metrics, diagnostics.
- `preprocessing/`: dataset compilation and loaders.
- `models/`: estimators, metric learning, graph utilities, post‑processing.
- `testbench/`: stacking, downstream metric tests, graph evaluation.
- `support/`: C++ geospatial backend and helpers.
- `scripts/`: data fetch and parsing scripts.

## Supporting Modules

The C++ geospatial backend (support/) provides fast graph construction and distance utilities used by
graph models and post‑processing.

Build instructions are in `support/README.md`.
