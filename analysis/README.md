# Analysis

This directory contains evaluation and diagnostic harnesses.

## Cross-Validation

`analysis/cross_validation.py` runs repeated cross-validation over one of the datasets and one or more models.

Key properties:
- Scaling is fit on the training split only (per fold) to avoid transductive leakage.
- Supports both `kfold` and repeated random splits (via `CVConfig`).

Example:
```bash
python analysis/cross_validation.py --decade 2020 --folds 5 --repeats 20 --models xgb rf linear
```

Outputs:
- `data/models/raw/cv_results_*.csv`
- `data/models/raw/cv_summary_*.csv`

## Feature Ablation

`analysis/climate_analysis.py` runs group-based ablations for the climate geospatial dataset.

Example:
```bash
python analysis/climate_analysis.py --target lat --folds 5 --repeats 10
```

Outputs:
- `data/models/raw/feature_ablation_results.csv`
- `analysis/images/<target>/ablation_*.png`

## Residual Workflow

`analysis/residual_analysis.py`:
1) Fits a coordinate→climate model (multi-output) with collection enabled.
2) Exports a residual dataset (`.mat`).
3) Re-runs CV on the residual-augmented feature space.

Example:
```bash
python analysis/residual_analysis.py --folds 5 --repeats 20
```

