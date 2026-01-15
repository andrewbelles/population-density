# Population Density Prediction

County-level population regression using a two-stage, hierarchical mixture of experts model, with label propagation and smoothing. 

The first stage targets the NCHS 6-class urban-rural classification scheme using multi-modal geospatial features. 

## Current Focus 

- Improving performance of expert models on scalar datasets (NLCD, SAIPE)
- Shifting satellite based data into spatial, and frequency based imaging for use in CNN (VIIRS)
- Correct-and-Smooth improvement through learned graph topologies. 

## Data Processing Pipeline

Dataset compilers live in `preprocessing/` and emit `.mat` files under `data/`:
- VIIRS nighttime lights into images for use in convolutional neural networks.
- TIGER road topology statistics.
- NLCD land cover configuration metrics.
- SAIPE socioeconomic (specifically poverty) information. 

See `preprocessing/README.md` for the per‑dataset build steps.

## Evaluation and Optimization

Core tools:
- `analysis/cross_validation.py` for repeated CV with fold‑safe scaling.
- `analysis/hyperparameter.py` for Optuna optimization and nested CV.
- `testbench/stacking.py` for the full stacking + C+S pipeline.

For the analysis of the full pipeline: 
```bash
nix-shell
python testbench/stacking.py --tests pipeline --cross both
```

## Repository Layout

- `analysis/`: CV harness, Optuna, graph metrics, diagnostics.
- `preprocessing/`: dataset compilation and loaders.
- `models/`: estimators, metric learning, graph utilities, post‑processing.
- `testbench/`: end-to-end testing and visualization. 
- `scripts/`: data fetch and parsing scripts.
- `utils/`: shared helpers (paths  metrics, interfaces).
- `data/`: datasets and outputs. 
