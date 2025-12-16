# Models

This directory contains model implementations used by the analysis harnesses.

## Model Interface

Models intended for cross-validation implement `support.helpers.ModelInterface`:

- `fit_and_predict((X_train, X_test), (y_train, y_test), (coords_train, coords_test), **kwargs) -> y_hat_scaled`

Where:
- `X_train/X_test` are feature matrices.
- `y_train/y_test` are scaled labels (1D or 2D for multi-output).
- `coords_*` are raw coordinates (typically `(lat, lon)`), passed for models that optionally use them.
- The return value must be predictions in the *scaled* label space; the CV harness handles inverse-transform.

## Implemented Models

- `linear_model.py` — closed-form linear regression baseline.
- `random_forest_model.py` — `sklearn.ensemble.RandomForestRegressor`.
- `xgboost_model.py` — `xgboost.XGBRegressor`, supports multi-output by looping over targets.
- `gp_xgboost_model.py` — GPBoost-style model: XGBoost random forest + GP residual model using coordinates.

## GNN Baseline (standalone)

- `gnn_models.py` is a standalone harness using `models/geospatial.py` and `torch_geometric`.
- It is not currently wired into the cross-validation interface.

