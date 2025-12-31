# Models

Model implementations used by the analysis and testbench harnesses.

## Model Interface

Most estimators follow `support.helpers.ModelInterface`:

`fit_and_predict((X_train, X_test), (y_train, y_test), (coords_train, coords_test), **kwargs) ->
y_hat_scaled`

## Estimators

Classification/regression estimators live in `models/estimators.py`:
- Logistic, RandomForest, XGBoost, SVM (classification).
- Linear / RF / XGB / (regression).

## Graph Post‑Processing

`models/post_processing.py`:
- `CorrectAndSmooth` for graph‑aware refinement of OOF probabilities.

## Metric Learning + Graph Utilities

- `models/metric.py` — metric learner used to build adaptive adjacency matrices.
- `models/graph_utils.py` — adjacency builders (Queen, mobility, k‑NN) and helpers.
