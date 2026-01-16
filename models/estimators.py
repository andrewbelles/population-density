#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

from typing import Callable
from joblib import pool
import numpy as np 

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 

from xgboost import XGBClassifier, XGBRegressor  

from utils.loss_funcs import focal_loss_obj, focal_mlogloss_eval 

import torch 

from torch import Tensor, from_numpy, nn 

from torch.utils.data import DataLoader, TensorDataset 

from models.networks import ConvBackbone

from torch.utils.data import WeightedRandomSampler

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
        self.n_estimators          = n_estimators
        self.max_depth             = max_depth
        self.learning_rate         = learning_rate
        self.subsample             = subsample
        self.colsample_bytree      = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction         = eval_fraction
        self.gpu                   = gpu
        self.random_state          = random_state
        self.n_jobs                = n_jobs 
        self.kwargs                = kwargs

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
        focal_gamma: float = 2.0, 
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
        self.focal_gamma = focal_gamma
        self.kwargs = kwargs

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64) 

        device = "cuda" if self.gpu else "cpu"

        n_classes = int(np.max(y)) + 1 if y.size else 0  
        is_multi  = n_classes > 2 

        if is_multi:
            alpha     = _alpha_from_counts(y, n_classes)
            objective = focal_loss_obj(n_classes, alpha, self.focal_gamma)
            metric    = focal_mlogloss_eval(n_classes)
            disable_default_eval_metric = True 
        else: 
            objective = "binary:logistic"
            metric    = "logloss"
            disable_default_eval_metric = False  

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
            eval_metric=metric,
            objective=objective, 
            disable_default_eval_metric=disable_default_eval_metric,
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
# CNN, Supervised Feature Extraction  
# ---------------------------------------------------------

class CNNClassifier(BaseEstimator, ClassifierMixin): 

    def __init__(
        self, 
        *,
        input_shape: tuple[int, int, int] | None = None, 
        aux_input_shape: tuple[int, int, int] | None = None, 
        conv_channels: tuple[int, ...] = (32, 64, 128), 
        kernel_size: int = 3, 
        pool_size: int = 2, 
        fc_dim: int = 128, 
        dropout: float = 0.2, 
        use_bn: bool = True, 
        epochs: int = 10, 
        batch_size: int = 32, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        random_state: int = 0,
        device: str | None = None, 
        normalize_main: bool = True,
        normalize_aux: bool = False, 
        input_adapter: Callable | None = None, 
        mask_channel: int | None = 1, 
        merge_fn: Callable | None = None 
    ): 
        self.input_shape     = input_shape
        self.aux_input_shape = aux_input_shape
        self.conv_channels   = conv_channels
        self.kernel_size     = kernel_size
        self.pool_size       = pool_size
        self.fc_dim          = fc_dim
        self.dropout         = dropout
        self.use_bn          = use_bn
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.random_state    = random_state 
        self.device          = device
        self.normalize_main  = normalize_main 
        self.normalize_aux   = normalize_aux 
        self.input_adapter   = input_adapter
        self.mask_channel    = mask_channel
        self.merge_fn        = merge_fn

    def _resolve_device(self): 
        if self.device is not None: 
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _adapt_inputs(self, X: NDArray): 
        if self.input_adapter is not None: 
            return self.input_adapter(X)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 4: 
            return X 

        if self.input_shape is None: 
            raise ValueError("input_shape required when X is flat")

        n = X.shape[0]
        c, h, w = self.input_shape 
        return X.reshape(n, c, h, w)

    def _fit_norm_stats(self, x_main, x_aux): 
        def _stats(x): 
            mu = x.mean(axis=(0,2,3), keepdims=True)
            sd = x.std(axis=(0,2,3), keepdims=True) + 1e-6 
            if self.mask_channel is not None and x.shape[1] > self.mask_channel: 
                mu[:, self.mask_channel, :, :] = 0.0 
                sd[:, self.mask_channel, :, :] = 1.0 
            return (mu, sd)

        self._norm_main = _stats(x_main) if self.normalize_main else None 
        self._norm_aux  = _stats(x_aux) if (self.normalize_aux and x_aux is not None) else None

    def _apply_norm(self, x: NDArray, stats): 
        if stats is None: 
            return x 
        mu, sd = stats 
        return (x - mu) / sd 

    def _prepare_inputs(self, X, training: bool): 
        adapted = self._adapt_inputs(X)
        if isinstance(adapted, tuple): 
            x_main, x_aux = adapted 
        else: 
            x_main, x_aux = adapted, None 

        if training: 
            self._fit_norm_stats(x_main, x_aux)

        x_main = self._apply_norm(x_main, self._norm_main)
        if x_aux is not None: 
            x_aux = self._apply_norm(x_aux, self._norm_aux)

        return x_main, x_aux 

    def _make_dataset(self, x_main, x_aux, y_idx=None): 
        if y_idx is None: 
            return (TensorDataset(torch.from_numpy(x_main)) if x_aux is None else 
                    TensorDataset(torch.from_numpy(x_main), torch.from_numpy(x_aux)))
        return (TensorDataset(
                    torch.from_numpy(x_main), 
                    torch.from_numpy(y_idx)
                ) if x_aux is None else 
                TensorDataset(
                    torch.from_numpy(x_main), 
                    torch.from_numpy(x_aux), 
                    torch.from_numpy(y_idx)
                ))
    
    def _make_loader(self, ds, shuffle: bool, sample_weights=None): 
        pin = (self.device_ is not None and self.device_.type == "cuda")
        if sample_weights is not None: 
            weights = torch.as_tensor(sample_weights, dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True
            )
            return DataLoader(
                ds, 
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=2,
                pin_memory=pin
            )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False,
                          num_workers=2, pin_memory=pin)

    def _forward_features(self, xb, xa=None): 
        f_main = self.model_.backbone_main(xb)
        if xa is None: 
            return f_main 
        f_aux = self.model_.backbone_aux(xa)
        return (self.merge_fn(f_main, f_aux) 
                if self.merge_fn else torch.cat([f_main, f_aux], dim=1)) 

    def fit(self, X, y, coords=None): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_      = len(self.classes_)

        torch.manual_seed(self.random_state)
        self.device_ = self._resolve_device()

        x_main, x_aux = self._prepare_inputs(X, training=True)

        self.backbone_main_ = ConvBackbone(
            in_channels=x_main.shape[1],
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            use_bn=self.use_bn
        )

        self.backbone_aux_  = None 
        if x_aux is not None: 
            self.backbone_aux_ = ConvBackbone(
                in_channels=x_aux.shape[1],
                conv_channels=self.conv_channels,
                kernel_size=self.kernel_size,
                pool_size=self.pool_size,
                use_bn=self.use_bn
            )
            feat_dim = self.backbone_main_.out_dim + self.backbone_aux_.out_dim 
        else: 
            feat_dim = self.backbone_main_.out_dim

        self.head_ = nn.Sequential(
            nn.Linear(feat_dim, self.fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.fc_dim, self.n_classes_)
        )

        self.model_ = nn.Module() 
        self.model_.backbone_main = self.backbone_main_ 
        self.model_.backbone_aux  = self.backbone_aux_ 
        self.model_.head          = self.head_ 
        self.model_.to(self.device_)

        class_counts   = np.bincount(y_idx, minlength=self.n_classes_).astype(np.float32)
        class_weights  = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights  = class_weights / class_weights.mean()

        sample_weights = class_weights[y_idx] 

        ds      = self._make_dataset(x_main, x_aux, y_idx)
        dl      = self._make_loader(ds, sample_weights=sample_weights, shuffle=False)
        opt     = torch.optim.AdamW(
            self.model_.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        loss_fn = nn.CrossEntropyLoss()

        self.model_.train() 
        for _ in range(self.epochs): 
            for batch in dl: 
                if x_aux is None: 
                    xb, yb = batch 
                    feats  = self._forward_features(xb.to(self.device_))
                else: 
                    xb, xa, yb = batch 
                    feats  = self._forward_features(xb.to(self.device_), xa.to(self.device_))

                logits = self.model_.head(feats)
                loss   = loss_fn(logits, yb.to(self.device_))
                opt.zero_grad() 
                loss.backward() 
                opt.step() 

        return self

    def predict_proba(self, X, coords=None): 
        X = np.asarray(X, dtype=np.float32)

        x_main, x_aux = self._prepare_inputs(X, training=False)
        ds = self._make_dataset(x_main, x_aux)
        dl = self._make_loader(ds, shuffle=False)

        self.model_.eval() 
        probs = []
        with torch.no_grad(): 
            for batch in dl: 
                if x_aux is None: 
                    xb     = batch[0]
                    feats  = self._forward_features(xb.to(self.device_))
                else: 
                    xb, xa = batch 
                    feats  = self._forward_features(xb.to(self.device_), xa.to(self.device_))

                logits = self.model_.head(feats)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        return np.vstack(probs)

    def predict(self, X, coords=None): 
        proba = self.predict_proba(X, coords=coords)
        idx   = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def extract_features(self, X, coords=None): 
        X = np.asarray(X, dtype=np.float32)
        x_main, x_aux = self._prepare_inputs(X, training=False)
        ds = self._make_dataset(x_main, x_aux)
        dl = self._make_loader(ds, shuffle=False)

        self.model_.eval() 
        feats = []
        with torch.no_grad(): 
            for batch in dl: 
                if x_aux is None: 
                    xb = batch[0]
                    f  = self._forward_features(xb.to(self.device_))
                else: 
                    xb, xa = batch 
                    f = self._forward_features(xb.to(self.device_), xa.to(self.device_))
                feats.append(f.cpu().numpy())

        return np.vstack(feats)

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

def make_image_cnn(
    *,
    input_shape=None, 
    aux_input_shape=None,
    input_adapter=None,
    merge_fn=None,
    **fixed 
): 
    def _factory(**params): 
        merged = dict(fixed)
        merged.update(params)
        return CNNClassifier(
            input_shape=input_shape,
            aux_input_shape=aux_input_shape,
            input_adapter=input_adapter,
            merge_fn=merge_fn,
            **merged 
        )
    return _factory 

# ---------------------------------------------------------
# Helpers 
# ---------------------------------------------------------

def _alpha_from_counts(y, n_classes): 
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv    = (counts.sum() / (n_classes * counts))
    alpha  = inv / inv.mean()
    return alpha 
