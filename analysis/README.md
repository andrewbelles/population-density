# Analysis

Evaluation harnesses, Optuna optimization, and graph diagnostics.

## Cross‑Validation

`analysis/cross_validation.py` runs repeated CV over a dataset and models.

Key properties:
- Fold‑safe scaling (fit on train only).
- `kfold` or repeated random splits (`CVConfig`).
- Supports classification and regression tasks.

Example:
```bash
python analysis/cross_validation.py --decade 2020 --folds 5 --repeats 20 --models xgb rf linear
```

Outputs:

- `data/models/raw/cv_results_*.csv`
- `data/models/raw/cv_summary_*.csv`

## Hyperparameter Optimization

`analysis/hyperparameter.py` provides Optuna wrappers with:

- Standard evaluator (single model).
- Nested CV helper (run_nested_cv).
- Early stopping by stagnation.

## Graph Metrics

`analysis/graph_metrics.py` computes adjacency diagnostics:

- Degree stats, largest component ratio.
- Homophily / confidence edge difference.
- Train‑neighbor coverage.

For quantifying and understanding Correct‑and‑Smooth behavior.

## Benchmarks

Specialized benchmarks and diagnostics:

- `climate_benchmarks.py`
- `saipe_benchmarks.py`
- `gating_benchmarks.py`
- `optimizer.py`

Although `testbench/` includes more comprehensive tests and in the future will be where all benchmakrs. 
