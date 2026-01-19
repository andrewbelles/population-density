#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

from os import wait
import sys
import time
from typing import Callable

import numpy as np 

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 

from sklearn.utils import shuffle
from xgboost import XGBClassifier, XGBRegressor  

from utils.loss_funcs import (
    focal_loss_obj, 
    focal_mlogloss_eval, 
    WassersteinLoss
)

import torch, copy 

from torch import nn 

from torch.utils.data import DataLoader, TensorDataset, random_split, Subset 

from models.networks import SpatialBackbone

from sklearn.model_selection import StratifiedShuffleSplit 



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


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin): 

    '''XGBoost classifier with early stopping support'''

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int = 1, 
        reg_alpha: float = 0.0, 
        reg_lambda: float = 1.0, 
        gamma: float = 0.0, 
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
        self.min_child_weight      = min_child_weight
        self.reg_alpha             = reg_alpha
        self.reg_lambda            = reg_lambda
        self.gamma                 = gamma 
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction         = eval_fraction
        self.gpu                   = gpu
        self.random_state          = random_state
        self.n_jobs                = n_jobs 
        self.kwargs                = kwargs

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
# Decision Optimizer Adapter for Ordinal Regression 
# --------------------------------------------------------- 

class OrdinalCutModel: 
    '''
    Learns monotonic thresholds on ordinal logits to optimize target metric 
    '''

    def __init__(
        self, 
        scorer=None, 
        n_grid: int = 25, 
        max_iter: int = 2
    ): 
        self.scorer   = scorer 
        self.n_grid   = n_grid 
        self.max_iter = max_iter 

    def fit(
        self, 
        logits: NDArray,
        y: NDArray,
        classes_: NDArray
    ): 
        if logits.ndim == 1: 
            logits = logits.reshape(-1, 1)

        y_idx     = np.searchsorted(classes_, y)
        p         = 1.0 / (1.0 + np.exp(-logits))
        n_classes = len(classes_)
        if p.shape[1] != n_classes - 1: 
            raise ValueError("logits must have shape (n, k-1)")

        scorer = self.scorer 
        if scorer is None: 
            scorer = lambda yt, yp: f1_score(yt, yp, average="macro")

        thresholds = np.full(n_classes - 1, 0.5, dtype=np.float64)

        for _ in range(self.max_iter): 
            for k in range(n_classes - 1): 
                cand   = np.unique(np.quantile(p[:, k], np.linspace(0.05, 0.95, self.n_grid)))
                best_t = thresholds[k]
                best_s = -np.inf 
                for t in cand: 
                    tmp    = thresholds.copy() 
                    tmp[k] = t 
                    y_pred = self._predict_from_p(p, tmp)
                    score  = scorer(y_idx, y_pred)
                    if score > best_s: 
                        best_s = score 
                        best_t = t 
                thresholds[k] = best_t 

        self.thresholds_ = thresholds 
        return self 

    def predict(
        self,
        logits: NDArray,
        classes_: NDArray
    ): 
        if logits.ndim == 1: 
            logits = logits.reshape(-1, 1)
        p = 1.0 / (1.0 + np.exp(-logits))
        preds = self._predict_from_p(p, self.thresholds_)
        return np.asarray(classes_)[preds]

    @staticmethod 
    def _predict_from_p(p: NDArray, thresholds: NDArray) -> NDArray:
        return (p > thresholds).sum(axis=1)

class OrdinalDecisionAdapter(BaseEstimator, ClassifierMixin): 

    '''
    Generic adapter that calibrates ordinal logits into discrete classes. 
    
    Works with any base model that exposes the predict_logits() method. 
    '''

    def __init__(
        self, 
        base_model,
        cut_model: OrdinalCutModel | None = None, 
        scorer=None,
        fit_base: bool = True 
    ): 
        self.base_model = base_model 
        self.cut_model  = cut_model 
        self.scorer     = scorer 
        self.fit_base   = fit_base 

    def fit(self, X, y, coords=None): 
        if self.fit_base: 
            try: 
                self.base_model.fit(X, y, coords)
            except TypeError: 
                self.base_model.fit(X, y)

        if not hasattr(self.base_model, "predict_logits"): 
            raise AttributeError("base_model must implement predict_logits")

        eval_X = self._as_eval_loader(X)
        logits = self._predict_logits(eval_X, coords)

        self.classes_ = getattr(self.base_model, "classes_", None)
        if self.classes_ is None: 
            self.classes_ = np.unique(np.asarray(y))

        cut = self.cut_model or OrdinalCutModel(scorer=self.scorer)
        self.cut_model_ = cut.fit(logits, y, self.classes_)
        return self 

    def predict(self, X, coords=None): 
        eval_X = self._as_eval_loader(X)
        logits = self._predict_logits(eval_X, coords)
        return self.cut_model_.predict(logits, np.asarray(self.classes_))
    
    def _predict_logits(self, X, coords): 
        try: 
            return self.base_model.predict_logits(X, coords)
        except TypeError: 
            return self.base_model.predict_logits(X)

    def _as_eval_loader(self, X): 
        if isinstance(X, DataLoader):
            return DataLoader(
                X.dataset, 
                batch_size=X.batch_size,
                shuffle=False,
                num_workers=X.num_workers,
                pin_memory=X.pin_memory,
                drop_last=False,
                collate_fn=X.collate_fn
            )
        return X 

# ---------------------------------------------------------
# Supervised Feature Extraction based Models  
# ---------------------------------------------------------

class SpatialClassifier(BaseEstimator, ClassifierMixin): 

    def __init__(
        self,
        *,
        conv_channels: tuple[int, ...] = (32, 64, 128, 256), 
        kernel_size: int = 3, 
        pool_size: int = 2, 
        fc_dim: int = 128, 
        dropout: float = 0.2, 
        use_bn: bool = True, 
        roi_output_size: int | tuple[int, int] = 7, 
        sampling_ratio: int = 2, 
        aligned: bool = False, 
        epochs: int = 100, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        random_state: int = 0, 
        device: str | None = None,
        early_stopping_rounds: int | None = 15, 
        eval_fraction: float = 0.1,
        min_delta: float = 1e-4,
        batch_size: int = 16,
        accum_steps: int = 4, 
        pin_memory: bool | None = None, 
        shuffle: bool = True, 
        collate_fn=None
    ): 
        self.conv_channels   = conv_channels
        self.kernel_size     = kernel_size
        self.pool_size       = pool_size
        self.fc_dim          = fc_dim
        self.dropout         = dropout
        self.use_bn          = use_bn
        self.roi_output_size = roi_output_size
        self.sampling_ratio  = sampling_ratio
        self.aligned         = aligned
        self.epochs          = epochs
        self.lr              = lr
        self.weight_decay    = weight_decay
        self.random_state    = random_state
        self.accum_steps     = accum_steps
        self.device          = self._resolve_device(device)

        self.early_stopping_rounds = early_stopping_rounds 
        self.min_delta             = min_delta
        self.eval_fraction         = eval_fraction

        self.pin_memory = pin_memory 
        self.shuffle    = shuffle 
        self.batch_size = batch_size 
        self.collate_fn = collate_fn

        self.classes_: NDArray | None = None 
        self.n_classes_: int | None   = None 

    def fit(self, X, y=None, val_loader=None): 
        torch.manual_seed(self.random_state)

        loader = self._ensure_loader(X, shuffle=self.shuffle)

        if y is None: 
            label_loader = self._as_eval_loader(X)
            all_labels   = []
            for batch in label_loader: 
                all_labels.append(np.asarray(batch[3]).reshape(-1))
            y_full = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
        else: 
            y_full = np.asarray(y).reshape(-1)

        if y_full.size == 0: 
            raise ValueError("empty loader")

        self.classes_   = np.unique(y_full)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2: 
            raise ValueError("ordinal regression requires at least 2 classes")

        if val_loader is None: 
            loader, val_loader = self._split_loader(loader, y_full)

        for batch in loader: 
            packed = batch[0]
            self._build_model(in_channels=packed.shape[1])
            break

        loss_fn       = WassersteinLoss(n_classes=self.n_classes_)
        self.loss_fn_ = loss_fn 

        opt     = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_state = None 
        best_val   = float("inf")
        patience   = 0 
        run_es     = self.early_stopping_rounds is not None and val_loader is not None 

        for ep in range(self.epochs): 
            t0 = time.perf_counter()
            self.model_.train() 

            accum = max(1, int(self.accum_steps))
            opt.zero_grad()

            step_idx = -1 
            total_loss  = 0.0
            total_count = 0

            for step_idx, batch in enumerate(loader): 
                loss, bsz = self._process_batch(batch)
                total_loss += loss.item() * bsz 
                total_count += bsz
                
                loss = loss / accum 
                loss.backward() 

                if (step_idx + 1) % accum == 0: 
                    opt.step() 
                    opt.zero_grad() 
            
            if step_idx >= 0 and (step_idx + 1) % accum != 0: 
                opt.step() 
                opt.zero_grad()

            val_loss = None 
            if run_es: 
                if self.early_stopping_rounds is None: 
                    raise TypeError
                val_loss = self._eval_loss(val_loader)
                if val_loss < best_val - self.min_delta: 
                    best_val   = val_loss 
                    best_state = copy.deepcopy(self.model_.state_dict())
                    patience   = 0 
                else:
                    patience  += 1
                    if patience >= self.early_stopping_rounds: 
                        break 

            avg_loss = total_loss / max(total_count, 1)
            dt       = time.perf_counter() - t0 
            msg      = f"[epoch {ep}] {dt:.2f}s, training_loss={avg_loss:.4f}"
            if ep % 5 == 0: 
                if val_loss is not None: 
                    msg += f" val_loss={val_loss:.4f}"
                print(msg, file=sys.stderr, flush=True)

        if best_state is not None: 
            self.model_.load_state_dict(best_state)
        return self 

    def loss(self, X) -> NDArray: 
        loader = self._ensure_loader(X, shuffle=False)
        if not hasattr(self, "loss_fn_"):
            self.loss_fn_ = WassersteinLoss(n_classes=self.n_classes_)
        return self._eval_loss(loader)

    def extract(self, X) -> NDArray: 
        loader = self._ensure_loader(X, shuffle=False)
        outs   = []

        self.model_.eval() 
        with torch.no_grad(): 
            for batch in loader: 
                packed, masks, rois, labels, *extra = batch 
                group_ids      = extra[1] if len(extra) > 1 else None 
                group_weights  = extra[2] if len(extra) > 2 else None 

                xb = torch.as_tensor(packed, dtype=torch.float32, device=self.device)
                mb = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
    
                feats  = self.model_.backbone(xb, mask=mb, rois=rois)
                if group_ids is not None: 
                    feats = self._aggregate_tiles(
                        feats, group_ids, group_weights, n_groups=len(labels)
                    )

                emb = self.fc_(feats)
                emb = self.act_(emb)
                outs.append(emb.cpu().numpy())

        return np.vstack(outs)

    def _resolve_device(self, device: str | None): 
        if device is not None: 
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_loader(self, dataset, shuffle: bool): 
        pin = self.pin_memory 
        if pin is None: 
            pin = self.device is not None and self.device.type == "cuda" 
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=pin,
            collate_fn=self.collate_fn
        )

    def _ensure_loader(self, X, shuffle: bool): 
        if isinstance(X, DataLoader): 
            return X 
        return self._make_loader(X, shuffle=shuffle)

    def _split_loader(self, loader, y_full): 
        run_es = self.early_stopping_rounds is not None and self.eval_fraction > 0 
        if not run_es: 
            return loader, None 

        ds = getattr(loader, "dataset", None)
        if ds is None or len(ds) < 2: 
            return loader, None 

        if y_full is None or len(y_full) != len(ds): 
            return loader, None 

        try: 
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.eval_fraction, 
                random_state=self.random_state 
            )
            train_idx, val_idx = next(splitter.split(np.zeros(len(y_full)), y_full))
            train_ds           = Subset(ds, train_idx)
            val_ds             = Subset(ds, val_idx)
        except ValueError:
            n_val   = max(1, int(len(ds) * self.eval_fraction))
            n_train = len(ds) - n_val 
            if n_train < 1: 
                return loader, None 

            gen              = torch.Generator().manual_seed(self.random_state)
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

        train_loader     = self._make_loader(train_ds, shuffle=True)
        val_loader       = self._make_loader(val_ds, shuffle=False)
        return train_loader, val_loader 

    def _build_model(self, in_channels: int): 
        if self.n_classes_ is None: 
            raise ValueError("n_classes not set before model build")

        self.backbone_ = SpatialBackbone(
            in_channels=in_channels,
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            use_bn=self.use_bn,
            roi_output_size=self.roi_output_size,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )
        self.fc_       = nn.Linear(self.backbone_.out_dim, self.fc_dim)
        self.act_      = nn.ReLU(inplace=True)
        self.drop_     = nn.Dropout(self.dropout)
        self.out_      = nn.Linear(self.fc_dim, self.n_classes_)
        self.head_     = nn.Sequential(self.fc_, self.act_, self.drop_, self.out_) 

        self.model_          = nn.Module() 
        self.model_.backbone = self.backbone_ 
        self.model_.head     = self.head_ 
        self.model_.to(self.device)

    def _aggregate_tiles(
        self, 
        feats, 
        group_ids,
        group_weights,
        n_groups
    ): 
        gids = torch.as_tensor(group_ids, dtype=torch.int64, device=feats.device)
        
        dim      = feats.size(1) // 2 
        avg_part = feats[:, :dim] 
        max_part = feats[:, dim:]

        if group_weights is None: 
            ones      = torch.ones((avg_part.size(0), 1), device=feats.device)
            sum_avg   = torch.zeros((n_groups, dim), device=feats.device)
            counts    = torch.zeros((n_groups, 1), device=feats.device)
            sum_avg.index_add_(0, gids, avg_part)
            counts.index_add_(0, gids, ones)
            agg_avg   = sum_avg / counts.clamp(min = 1.0)
        else: 

            w = torch.as_tensor(
                group_weights, dtype=avg_part.dtype, device=avg_part.device
            ).view(-1, 1)
            sum_avg   = torch.zeros((n_groups, dim), device=feats.device)
            sum_w     = torch.zeros((n_groups, 1), device=feats.device)
            sum_avg.index_add_(0, gids, avg_part * w)
            sum_w.index_add_(0, gids, w)
            agg_avg   = sum_avg / sum_w.clamp(min=1e-6)

        agg_max = torch.full((n_groups, dim), torch.finfo(max_part.dtype).min, 
                             device=feats.device)
        if hasattr(agg_max, "scatter_reduce_"): 
            idx = gids.view(-1, 1).expand(-1, dim) 
            agg_max.scatter_reduce_(0, idx, max_part, reduce="amax", include_self=True)
        else: 
            for i in range(max_part.size(0)):
                g = gids[i].item() 
                agg_max[g] = torch.maximum(agg_max[g], max_part[i])

        has_group = torch.zeros((n_groups, 1), dtype=torch.bool, device=feats.device)
        has_group.index_fill_(0, gids, True)
        agg_max   = torch.where(has_group, agg_max, torch.zeros_like(agg_max))

        return torch.cat([agg_avg, agg_max], dim=1)

    def _eval_loss(self, loader): 
        self.model_.eval() 
        total = 0.0 
        count = 0 
        with torch.no_grad(): 
            for batch in loader: 
                loss, bsz = self._process_batch(batch)
                total    += loss.item() * bsz 
                count    += bsz 

        return total / max(count, 1)

    def _process_batch(self, batch):
        if self.n_classes_ is None: 
            raise TypeError
        if self.classes_ is None: 
            raise TypeError

        packed, masks, rois, labels, *extra = batch
        group_ids     = extra[1] if len(extra) > 1 else None 
        group_weights = extra[2] if len(extra) > 2 else None 

        xb = torch.as_tensor(packed, dtype=torch.float32, device=self.device)
        mb = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
    
        y_np  = np.asarray(labels).reshape(-1)
        y_idx = np.searchsorted(self.classes_, y_np)
        yb    = torch.as_tensor(y_idx, dtype=torch.int64, device=self.device)

        feats  = self.model_.backbone(xb, mask=mb, rois=rois)
        if group_ids is not None: 
            feats = self._aggregate_tiles(feats, group_ids, group_weights, n_groups=len(labels))
        logits = self.model_.head(feats)
        loss   = self.loss_fn_(logits, yb)

        bsz    = yb.size(0)
        return loss, bsz  

    def _as_eval_loader(self, X): 
        if isinstance(X, DataLoader):
            return DataLoader(
                X.dataset, 
                batch_size=X.batch_size,
                shuffle=False,
                num_workers=X.num_workers,
                pin_memory=X.pin_memory,
                drop_last=False,
                collate_fn=X.collate_fn
            )
        return X 

    @staticmethod 
    def _ordinal_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor: 
        thresholds = torch.arange(n_classes - 1, device=y.device).view(1, -1)
        return (y.view(-1, 1) > thresholds).to(dtype=torch.float32)

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

def make_spatial_ordinal(
    *,
    collate_fn=None, 
    cut_model: OrdinalCutModel | None = None,
    scorer=None, 
    fit_base: bool = True,
    **fixed 
): 
    def _factory(**params):
        merged  = dict(fixed)
        merged.update(params)
        collate = merged.pop("collate_fn", collate_fn)
        base    = SpatialClassifier(collate_fn=collate, **merged)
        cut     = cut_model() if isinstance(cut_model, type) else cut_model
        return OrdinalDecisionAdapter(
            base_model=base,
            cut_model=cut,
            scorer=scorer,
            fit_base=fit_base
        )
    return _factory 

def make_spatial_sfe(
    *,
    collate_fn=None,
    **fixed 
): 
    def _factory(**params): 
        merged = dict(fixed)
        merged.update(params) 
        collate = merged.pop("collate_fn", collate_fn) 
        return SpatialClassifier(collate_fn=collate, **merged)
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
