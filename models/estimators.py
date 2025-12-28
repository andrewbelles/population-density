#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

import numpy as np 
# from typing import Any 

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 

from sklearn.utils import class_weight
from xgboost import XGBClassifier, XGBRegressor  

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch_geometric.nn import GCNConv 

from models.graph_utils import build_knn_graph_from_coords, to_pyg_data

# import torch, gpytorch 

# ---------------------------------------------------------
# Regressors 
# ---------------------------------------------------------

class LinearRegressor(BaseEstimator, RegressorMixin): 

    '''Linear Regression with optional Regularization'''

    def __init__(self, alpha: float = 1.0): 
        self.alpha = alpha 

    def fit(self, X, y): 

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        X_b = np.hstack([np.ones((X.shape[0], 1)), X])

        # with regularization 
        if self.alpha > 0: 
            I = np.eye(X_b.shape[1])
            I[0, 0] = 0 # bias does not get regularized 
            self.coef_ = np.linalg.solve(
                X_b.T @ X_b + self.alpha * I, X_b.T @ y
            )
        # without (psuedo-inv)
        else: 
            self.coef_ = np.linalg.pinv(X_b) @ y

        return self 

    def predict(self, X): 
        X = np.asarray(X, dtype=np.float64) 
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.coef_ 


class RFRegressor(BaseEstimator, RegressorMixin): 

    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int | None = None, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        n_jobs: int = -1, 
        random_state: int | None = None, 
        **kwargs
    ): 
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y): 

        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self 

    def predict(self, X):
        return self.model_.predict(X)


class XGBRegressorWrapper(BaseEstimator, RegressorMixin): 

    '''XGBoost regressor with early stopping support'''

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        early_stopping_rounds: int | None = None,
        eval_fraction: float = 0.2,
        gpu: bool = False,
        random_state: int | None = None,
        n_jobs: int = -1,  
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.gpu = gpu
        self.random_state = random_state
        self.n_jobs = n_jobs 
        self.kwargs = kwargs

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64) 

        device = "cuda" if self.gpu else "cpu"

        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            device=device,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs
        )

        if self.early_stopping_rounds and self.eval_fraction > 0: 
            n = X.shape[0]
            n_eval = int(n * self.eval_fraction)
            idx = np.random.default_rng(self.random_state).permutation(n)
            train_idx, eval_idx = idx[n_eval:], idx[:n_eval]

            self.model_.fit(
                X[train_idx],
                y[train_idx], 
                eval_set=[(X[eval_idx], y[eval_idx])],
                verbose=False 
            )
        else: 
            self.model_.fit(X, y)

        return self 

    def predict(self, X): 
        return self.model_.predict(np.asarray(X, dtype=np.float64))


class MultiOutputRegressor(BaseEstimator, RegressorMixin): 

    def __init__(self, base_estimator: BaseEstimator): 
        self.base_estimator = base_estimator 

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 1: 
            y = y.reshape(-1, 1)

        self.n_outputs_ = y.shape[1]
        self.estimators_ = []

        for i in range(self.n_outputs_):
            est = clone(self.base_estimator)
            est.fit(X, y[:, i])
            self.estimators_.append(est)

        return self 

    def predict(self, X): 
        X = np.asarray(X, dtype=np.float64)
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return preds if self.n_outputs_ > 1 else preds.ravel() 


class _GCNRegressorNet(nn.Module): 

    def __init__(
        self, 
        in_dim: int, 
        hidden_dims: tuple[int, ...]
    ): 
        super().__init__() 
        dims = (in_dim,) + tuple(hidden_dims) + (1,)
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(
        self, 
        x, 
        edge_index, 
        edge_weight=None 
    ): 
        for conv in self.convs[:-1]: 
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x.squeeze(-1) 

class GCNGraphRegressor(BaseEstimator, RegressorMixin): 

    def __init__(
        self,
        hidden_dims: tuple[int, ...], 
        *, 
        k: int = 5, 
        epochs: int = 1000, 
        lr: float = 1e-3,
        weight_decay: float = 0.0, 
        dropout: float = 0.0, 
        bandwidth_km: float = 250.0, 
        directed: bool = False, 
        device: str = "cpu", 
        random_state: int = 0
    ): 
        self.hidden_dims = hidden_dims
        self.k = k 
        self.epochs = epochs 
        self.lr = lr           
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.bandwidth_km = bandwidth_km
        self.directed = directed
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, coords=None): 

        if coords is None: 
            raise ValueError("GCNGraphRegressor required coords=(lat,lon) passed by CrossValidator")

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if y.ndim == 2 and y.shape[1] == 1: 
            y = y.ravel() 
        if y.ndim != 1: 
            raise ValueError(f"only 1d regression supported, got shape {y.shape}")

        torch.manual_seed(self.random_state)
        graph = build_knn_graph_from_coords(coords, k=self.k, directed=self.directed)
        data  = to_pyg_data(graph, x=X, y=y, undirect_mean=False)

        device = torch.device(self.device)
        self.model_ = _GCNRegressorNet(in_dim=X.shape[1], hidden_dims=tuple(self.hidden_dims)).to(device)

        data = data.to(self.device)
        if data is None or not hasattr(data, "edge_attr") or not hasattr(data, "edge_index"): 
            raise ValueError("stupid fucking type check fuck you lsp")
        if data.edge_attr is None or data.edge_index is None: 
            raise ValueError("also a stupid fucking type check")
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.lr), 
                                     weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss() 

        self.model_.train()
        for epoch in range(int(self.epochs)): 
            optimizer.zero_grad() 

            d_km = data.edge_attr.view(-1)
            edge_weight = torch.exp(-((d_km / float(self.bandwidth_km)) ** 2))
             
            out = self.model_(data.x, data.edge_index, edge_weight=edge_weight)
            loss = loss_fn(out, data.y)

            loss.backward() 
            optimizer.step() 

        return self 

    def predict(self, X, coords=None):
        if coords is None: 
            raise ValueError("GCNGraphRegressor required coords=(lat,lon) passed by CrossValidator")

        X = np.asarray(X, dtype=np.float64)

        graph = build_knn_graph_from_coords(coords, k=self.k, directed=self.directed)
        data  = to_pyg_data(graph, x=X, undirect_mean=False)

        data   = data.to(self.device)
        if data is None or not hasattr(data, "edge_attr") or not hasattr(data, "edge_index"): 
            raise ValueError("stupid fucking type check fuck you lsp")
        if data.edge_attr is None or data.edge_index is None: 
            raise ValueError("also a stupid fucking type check")

        self.model_.eval() 
        with torch.no_grad(): 
            d_km = data.edge_attr.view(-1)
            edge_weight = torch.exp(-((d_km / float(self.bandwidth_km)) ** 2))
            out = self.model_(data.x, data.edge_index, edge_weight=edge_weight)
        return out.detach().cpu().numpy() 


# ---------------------------------------------------------
# Classifiers 
# ---------------------------------------------------------

class RFClassifierWrapper(BaseEstimator, ClassifierMixin): 

    def __init__(
        self, 
        n_estimators: int = 100, 
        max_depth: int | None = None, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        n_jobs: int = -1, 
        random_state: int | None = None, 
        **kwargs
    ): 
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y): 

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_ 
        return self 

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X): 
        return self.model_.predict_proba(X)

    @property 
    def feature_importances_(self):
        return self.model_.feature_importances_


class XGBClassifierWrapper(BaseEstimator, RegressorMixin): 

    '''XGBoost classifier with early stopping support'''

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        early_stopping_rounds: int | None = None,
        eval_fraction: float = 0.2,
        gpu: bool = False,
        random_state: int | None = None,
        n_jobs: int = -1,  
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.gpu = gpu
        self.random_state = random_state
        self.n_jobs = n_jobs 
        self.kwargs = kwargs

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64) 

        device = "cuda" if self.gpu else "cpu"

        n_classes = len(np.unique(y))
        is_multi  = n_classes > 2 

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            device=device,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="mlogloss" if is_multi else "logloss",
            objective="multi:softprob" if is_multi else "binary:logistic", 
            num_class=n_classes if is_multi else None, 
            **self.kwargs
        )

        if self.early_stopping_rounds and self.eval_fraction > 0: 
            n = X.shape[0]
            n_eval = int(n * self.eval_fraction)
            idx = np.random.default_rng(self.random_state).permutation(n)
            train_idx, eval_idx = idx[n_eval:], idx[:n_eval]

            self.model_.fit(
                X[train_idx],
                y[train_idx], 
                eval_set=[(X[eval_idx], y[eval_idx])],
                verbose=False 
            )
        else: 
            self.model_.fit(X, y)

        self.classes_ = self.model_.classes_ 
        return self 

    def predict(self, X): 
        return self.model_.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X): 
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))


class LogisticWrapper(BaseEstimator, ClassifierMixin): 
    
    '''Logistic regression wrapper for contrastive learning'''

    def __init__(
        self, 
        C: float = 1.0, 
        max_iter: int = 1000, 
        random_state: int | None = None, 
        **kwargs 
    ): 
        self.C = C 
        self.max_iter = max_iter 
        self.random_state = random_state 
        self.kwargs = kwargs 

    def fit(self, X, y): 
        self.model_ = LogisticRegression(
            C=self.C ,
            max_iter=self.max_iter ,
            random_state=self.random_state ,
            **self.kwargs
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_ 
        return self 

    def predict(self, X): 
        return self.model_.predict(X)

    def predict_proba(self, X): 
        return self.model_.predict_proba(X)


class SVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        *, 
        C: float = 1.0, 
        kernel: str = "rbf", 
        gamma: str = "scale", 
        probability: bool = False, 
        class_weight: str | dict | None = None, 
        random_state: int | None = None, 
        **kwargs
    ): 
        self.C = C 
        self.kernel       = kernel 
        self.gamma        = gamma 
        self.probability  = probability 
        self.class_weight = class_weight 
        self.random_state = random_state 
        self.kwargs       = kwargs

    def fit(self, X, y, coords=None): 
        self.model_ = SVC(
            C=self.C, 
            kernel=self.kernel, 
            gamma=self.gamma, 
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state,
            **self.kwargs
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_ 
        return self 

    def predict(self, X, coords=None):
        return self.model_.predict(X)

    def predict_proba(self, X, coords=None):
        if not self.probability: 
            raise AttributeError("SVMClassifier was instantiated with probability=False")
        return self.model_.predict_proba(X)


# ---------------------------------------------------------
# Factory Functions (backwards compatibility with CrossValidator)
# ---------------------------------------------------------

def make_linear(alpha: float = 1.0):
    return lambda: LinearRegressor(alpha=alpha)

def make_rf_regressor(n_estimators: int = 400, **kwargs): 
    return lambda: RFRegressor(n_estimators=n_estimators, **kwargs) 

def make_xgb_regressor(
    n_estimators: int = 300, 
    early_stopping_rounds: int = 200, 
    gpu: bool = False, 
    **kwargs 
): 
    return lambda: XGBRegressorWrapper(
        n_estimators=n_estimators, 
        early_stopping_rounds=early_stopping_rounds,
        gpu=gpu,
        **kwargs 
    )

def make_gcn_regressor(**kwargs):
    return lambda: GCNGraphRegressor(**kwargs)

def make_rf_classifier(n_estimators: int = 400, **kwargs):
    return lambda: RFClassifierWrapper(n_estimators=n_estimators, **kwargs)

def make_xgb_classifier(
    n_estimators: int = 400,
    early_stopping_rounds: int = 200,
    gpu: bool = False,
    **kwargs,
):
    return lambda: XGBClassifierWrapper(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        gpu=gpu,
        **kwargs,
    )


def make_logistic(C: float = 1.0, **kwargs):
    return lambda: LogisticWrapper(C=C, **kwargs)

def make_svm_classifier(probability=True, **kwargs): 
    return lambda: SVMClassifier(probability=probability, **kwargs)
