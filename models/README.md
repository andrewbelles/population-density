# Models

Model implementations used by the analysis and testbench harnesses.

## Model Interface

Most estimators follow `support.helpers.ModelInterface`:

`fit_and_predict((X_train, X_test), (y_train, y_test), (coords_train, coords_test), **kwargs) ->
y_hat_scaled`

Neural models expose sklearn-like `fit` / `predict_proba` in `models/estimators.py`. 

## Estimators

`models/estimators.py`: 
- Classical ML: Logistic, RandomForest, XGBoost, SVM. 
- CNN estimator that wraps backbone from `models/networks.py`
- Implements optimizer/CV contracts used by `analysis/` and `testbench/`

## CNN Backbones 

`models/networks.py` defines reusable image backbones: 
- `ConvBackbone` for generic image inputs. 
- Configurable channels, kernel size, pooling, dropout, and FC head. 

The estimator selects the backbone and handles training, evaluation, and prediction 

## Graph Post‑Processing

`models/post_processing.py`:
- `CorrectAndSmooth` for graph‑aware refinement of OOF probabilities.

## Metric Learning + Graph Utilities

- `models/metric.py` — metric learner used to build adaptive adjacency matrices.
- `models/graph_utils.py` — adjacency builders (Queen, mobility, k‑NN) and helpers.
