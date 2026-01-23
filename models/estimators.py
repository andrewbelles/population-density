#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

import numpy as np 

import time, copy, time, sys 

from utils.resources import ComputeStrategy

from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 

from xgboost import XGBClassifier, XGBRegressor  

from utils.loss_funcs import (
    WassersteinLoss
)

from scipy.sparse import csr_matrix

from torch import nn 

import torch 

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
        ordinal: bool = True,
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
        self.ordinal               = ordinal 
        self.kwargs                = kwargs

        self.models_ = []
        self.model_  = None 

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64) 

        y_idx, classes  = self._encode_labels(y)
        self.classes_   = classes 
        self.n_classes_ = classes.size 

        if self.n_classes_ <= 2 or not self.ordinal: 
            self.model_ = self._build_model(self.n_classes_)
            train_idx, eval_idx = self._split_indices(X.shape[0])

            if eval_idx is not None: 
                self.model_.fit(
                    X[train_idx], y_idx[train_idx],
                    eval_set=[(X[eval_idx], y_idx[eval_idx])],
                    verbose=False
                )
            else: 
                self.model_.fit(X, y_idx)
            return self 

        train_idx, eval_idx = self._split_indices(X.shape[0])
        self.models_ = []
        for k in range(self.n_classes_ - 1): 
            y_bin = (y_idx > k).astype(np.int64)
            model = self._build_model(2)
            if eval_idx is not None: 
                model.fit(
                    X[train_idx], y_bin[train_idx],
                    eval_set=[(X[eval_idx], y_bin[eval_idx])],
                    verbose=False
                )
            else: 
                model.fit(X, y_bin)
            self.models_.append(model)

        return self 

    def predict(self, X): 
        proba = self.predict_proba(X)
        idx   = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X): 
        X = np.asarray(X, dtype=np.float32)

        if self.model_ is not None: 
            return self.model_.predict_proba(X)

        prob_gt = self._prob_gt(X)
        n       = prob_gt.shape[0]
        K       = self.n_classes_ 
        probs   = np.zeros((n, K), dtype=np.float32)

        probs[:, 0] = 1.0 - prob_gt[:, 0]
        for k in range(1, K - 1): 
            probs[:, k] = prob_gt[:, k - 1] - prob_gt[:, k]
        probs[:, -1] = prob_gt[:, -1]

        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    @property 
    def feature_importances_(self): 
        if self.model_ is not None: 
            return self.model_.feature_importances_ 
        if self.models_:
            return np.mean(
                [m.feature_importances_ for m in self.models_], axis=0
            )

    def _build_model(self, n_classes: int):
        if n_classes <= 2: 
            objective   = "binary:logistic"
            eval_metric = "logloss"
        else: 
            objective   = "multi:softprob"
            eval_metric = "mlogloss" 

        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            objective=objective,
            eval_metric=eval_metric,
            num_class=n_classes if n_classes > 2 else None,
            device="cuda" if self.gpu else "cpu",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.kwargs 
        )
    
    def _encode_labels(self, y): 
        classes = np.unique(y)
        classes = np.sort(classes)
        y_idx   = np.searchsorted(classes, y)
        if not np.all(classes[y_idx] == y):
            raise ValueError("y contains values not in sorted classes")
        return y_idx, classes 

    def _split_indices(self, n): 
        if not (self.early_stopping_rounds and self.eval_fraction > 0):
            return np.arange(n), None 
        n_eval = max(1, int(n * self.eval_fraction))
        rng    = np.random.default_rng(self.random_state)
        idx    = rng.permutation(n)
        return idx[n_eval:], idx[:n_eval]

    def _prob_gt(self, X): 
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])

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
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        weight_decay: float = 0.0, 
        random_state: int = 0, 
        early_stopping_rounds: int | None = 15, 
        eval_fraction: float = 0.1,
        min_delta: float = 1e-4,
        batch_size: int = 2,
        accum_steps: int = 2, 
        shuffle: bool = True, 
        in_channels: int | None = None, 
        categorical_input: bool = False, 
        collate_fn=None
    ): 
        self.conv_channels     = conv_channels
        self.kernel_size       = kernel_size
        self.pool_size         = pool_size
        self.fc_dim            = fc_dim
        self.dropout           = dropout
        self.use_bn            = use_bn
        self.roi_output_size   = roi_output_size
        self.sampling_ratio    = sampling_ratio
        self.aligned           = aligned
        self.epochs            = epochs
        self.lr                = lr
        self.weight_decay      = weight_decay
        self.random_state      = random_state
        self.accum_steps       = accum_steps
        self.in_channels       = in_channels
        self.categorical_input = categorical_input
        self.device            = self._resolve_device(compute_strategy.device)

        self.early_stopping_rounds = early_stopping_rounds 
        self.min_delta             = min_delta
        self.eval_fraction         = eval_fraction

        self.compute_strategy = compute_strategy 
        self.shuffle          = shuffle 
        self.batch_size       = batch_size 
        self.collate_fn       = collate_fn

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
            in_ch = self.in_channels or packed.shape[1]
            self._build_model(in_channels=in_ch)
            break

        loss_fn       = WassersteinLoss(n_classes=self.n_classes_)
        self.loss_fn_ = loss_fn 

        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        opt     = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        best_state = None 
        best_val   = float("inf")
        patience   = 0 
        run_es     = self.early_stopping_rounds is not None and val_loader is not None 

        start      = time.perf_counter()
        for ep in range(self.epochs): 
            t0 = time.perf_counter()
            self.model_.train() 

            accum = max(1, int(self.accum_steps))
            opt.zero_grad()

            step_idx = -1 
            total_loss  = 0.0
            total_count = 0

            for step_idx, batch in enumerate(loader): 
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"): 
                    loss, bsz = self._process_batch(batch)
                    total_loss += loss.item() * bsz 
                    total_count += bsz
                    loss = loss / accum 

                scaler.scale(loss).backward() 

                if (step_idx + 1) % accum == 0: 
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad() 
            
            if step_idx >= 0 and (step_idx + 1) % accum != 0: 
                scaler.step(opt)
                scaler.update()
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
            avg_dt   = (time.perf_counter() - start) / (ep + 1)
            msg      = f"[epoch {ep}] {avg_dt:.2f}s avg, training_loss={avg_loss:.4f}"
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
                if packed.ndim == 3: 
                    packed = packed[:, None, ...]
                if masks.ndim == 3: 
                    masks  = masks[:, None, ...]

                if rois is not None and len(rois) == 0: 
                    rois = None 

                group_ids = group_weights = None 
                if len(extra) >= 2: 
                    group_ids, group_weights = extra[-2], extra[-1]

                if self.categorical_input: 
                    xb = torch.as_tensor(packed, dtype=torch.uint8, device=self.device)
                else: 
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
        pin = self.compute_strategy.device == "cuda" 
        if self.compute_strategy.n_jobs == -1: 
            num_workers = 8 
        else: 
            num_workers = self.compute_strategy.n_jobs

        base       = getattr(dataset, "dataset", dataset)
        is_packed  = hasattr(base, "is_packed") and base.is_packed  

        worker_override = getattr(base, "prefetch_workers", None)
        if worker_override is not None: 
            num_workers = int(worker_override)

        batch_size = 1 if is_packed else self.batch_size 

        prefetch_factor = 4 if num_workers > 0 else None 
        pf_override     = getattr(base, "prefetch_factor", None)
        if pf_override is not None and num_workers > 0: 
            prefetch_factor = int(pf_override)

        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
            collate_fn=self.collate_fn,
            persistent_workers=(num_workers > 0), 
            prefetch_factor=prefetch_factor  
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
            categorical_input=self.categorical_input,
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
        if self.n_classes_ is None or self.classes_ is None: 
            raise TypeError

        packed, masks, rois, labels, *extra = batch
        if packed.ndim == 3: 
            packed = packed[:, None, ...]
        if masks.ndim == 3: 
            masks  = masks[:, None, ...]

        if rois is not None and len(rois) == 0: 
            rois = None 

        group_ids = group_weights = None 
        if len(extra) >= 2: 
            group_ids, group_weights = extra[-2], extra[-1]

        if self.categorical_input:
            xb = torch.as_tensor(packed, dtype=torch.uint8, device=self.device)
        else: 
            xb = torch.as_tensor(packed, dtype=torch.float32, device=self.device)
        mb = torch.as_tensor(masks, dtype=torch.float32, device=self.device)

        if xb.shape[-2:] != mb.shape[-2:]:
            raise RuntimeError(f"pre-backbone mismatch: xb={xb.shape}, mb={mb.shape}")
    
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

class EmbeddingProjector(BaseEstimator): 

    '''
    Shallow non-linear projection for embedding compression. Trains with Wasserstein loss 
    on a small classification head, returning only the projected embeddings. 
    '''

    def __init__(
        self,
        in_dim: int, 
        out_dim: int = 64, 
        hidden_dim: int | None = None, 
        dropout: float = 0.1, 
        epochs: int = 200, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-4, 
        batch_size: int = 128, 
        early_stopping_rounds: int = 30, 
        eval_fraction: int = 0, 
        random_state: int = 0, 
        device: str | None = None 
    ): 
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.random_state = random_state
        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def _build(self, n_classes: int): 
        hidden = self.hidden_dim or max(self.out_dim, min(self.in_dim, 128))
        self.proj_ = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, self.out_dim)
        )

        self.head_  = nn.Linear(self.out_dim, n_classes)
        self.model_ = nn.Module() 
        self.model_.proj = self.proj_ 
        self.model_.head = self.head_ 
        self.model_.to(self.device)

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        
        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self._build(self.n_classes_)
        loss_fn  = WassersteinLoss(n_classes=self.n_classes_)
        opt      = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_fn_ = loss_fn

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.eval_fraction, random_state=self.random_state
        ) 

        train_idx, val_idx = next(splitter.split(X, y))
        X_train, y_train   = X[train_idx], y[train_idx] 
        X_val, y_val       = X[val_idx], y[val_idx]

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )

        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_state   = None 
        best_val     = float("inf")
        patience     = 0 

        for _ in range(self.epochs): 
            self.model_.train() 
            for xb, yb in train_loader: 
                xb     = xb.to(self.device)
                yb     = yb.to(self.device)
                emb    = self.proj_(xb)
                logits = self.head_(emb)
                loss   = loss_fn(logits, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

            self.model_.eval()
            total = 0.0 
            count = 0 
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    emb = self.proj_(xb)
                    logits = self.head_(emb)
                    loss = loss_fn(logits, yb)
                    total += loss.item() * yb.size(0)
                    count += yb.size(0)
            val_loss = total / max(count, 1)

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_rounds:
                    break    

        if best_state is not None: 
            self.model_.load_state_dict(best_state)
        return self 

    def transform(self, X): 
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval() 
        outs = []
        with torch.no_grad(): 
            for i in range(0, X.shape[0], self.batch_size): 
                xb  = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)
                emb = self.proj_(xb).cpu().numpy() 
                outs.append(emb)
        return np.vstack(outs)

    def fit_transform(self, X, y): 
        return self.fit(X, y).transform(X)

    def loss(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                            batch_size=self.batch_size, shuffle=False)
        self.model_.eval() 
        total = 0.0 
        count = 0 
        with torch.no_grad(): 
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                emb = self.proj_(xb)
                logits = self.head_(emb)
                loss = self.loss_fn_(logits, yb)
                total += loss.item() * yb.size(0)
                count += yb.size(0)
        return total / max(count, 1)


class XGBOrdinalRegressor(BaseEstimator, ClassifierMixin): 

    '''
    Implementation of Frank-Hall method for ordinal multi-classification using the 
    XGBoost model. 
    '''
    
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

        self.models_ = []

    def fit(self, X, y): 
        X                   = np.asarray(X, dtype=np.float32)
        y_idx, classes      = self._encode_labels(y)
        self.classes_       = classes 
        self.n_classes_     = classes.size  
        train_idx, eval_idx = self._split_indices(X.shape[0])
        self.models_        = []

        for k in range(self.n_classes_ - 1): 
            y_bin = (y_idx > k).astype(np.int64)
            model = self._build_model()

            if eval_idx is not None: 
                model.fit(
                    X[train_idx],
                    y_bin[train_idx],
                    eval_set=[(X[eval_idx], y_bin[eval_idx])],
                    verbose=False
                )
            else: 
                model.fit(X, y_bin)

            self.models_.append(model)

        return self 

    def loss(self, X, y):
        y     = np.asarray(y, dtype=np.int64).reshape(-1)
        y_idx = np.searchsorted(self.classes_, y) 
        if not np.all(self.classes_[y_idx] == y): 
            raise ValueError("y contains values not in fitted classes")

        prob_gt = self._prob_gt(X)
        eps     = 1e-12 
        total   = 0.0 

        for k in range(prob_gt.shape[1]): 
            y_bin  = (y_idx > k).astype(np.float32)
            p      = np.clip(prob_gt[:, k], eps, 1.0 - eps)
            ll     = -(y_bin * np.log(p) + (1.0 - y_bin) * np.log(1.0 - p)).mean() 
            total += ll

        return float(total / prob_gt.shape[1])

    def transform(
        self,
        X,
        *,
        embed_dim: int = 0, 
        out_dim: int = 64, 
        pooling: str = "mean"
    ): 
        '''
        Bag of Leaves embedding. computes lookup table of leaf embeddings, pools across trees 
        then projects to out_dim
        '''
    
        self._init_leaf_projection(embed_dim, out_dim)

        leaf_ids = self.leaves(X)
        emb      = self.leaf_embed_table_[leaf_ids]

        if pooling == "sum": 
            pooled = emb.sum(axis=1)
        elif pooling == "mean": 
            pooled = emb.mean(axis=1)
        else: 
            raise ValueError("invalid pooling method")

        return pooled @ self.leaf_proj_ + self.leaf_proj_bias_

    def leaf_matrix(
        self,
        X,
        *,
        dense: bool = False 
    ): 
        leaf_ids = self.leaves(X)
        n, t     = leaf_ids.shape 
        n_cols   = int(self.leaf_table_size_)

        if dense: 
            mat = np.zeros((n, n_cols), dtype=np.float32)
            mat[np.arange(n)[:, None], leaf_ids] = 1.0 
            return mat 

        rows = np.repeat(np.arange(n), t)
        cols = leaf_ids.reshape(-1)
        data = np.ones(rows.shape[0], dtype=np.float32)
        return csr_matrix((data, (rows, cols)), shape=(n, n_cols))

    def leaves(self, X): 
        X      = np.asarray(X, dtype=np.float32)
        dmat   = xgboost.DMatrix(X, nthread=self.n_jobs) 

        if not hasattr(self, "leaf_table_size_"):
            self._build_leaf_table()

        leaves = []

        for model_idx, model in enumerate(self.models_): 
            booster   = model.get_booster() 
            best_iter = getattr(model, "best_iteration", None)
            if best_iter is None: 
                raw = booster.predict(dmat, pred_leaf=True)
            else: 
                raw = booster.predict(dmat, pred_leaf=True, iteration_range=(0, best_iter + 1))

            raw     = raw.astype(np.int64)
            maps    = self.leaf_maps_[model_idx]
            offsets = self.leaf_offsets_[model_idx]
            mapped  = np.empty_like(raw, dtype=np.int64)
            for t in range(raw.shape[1]): 
                ids = raw[:, t] 
                map_arr = maps[t]
                if ids.max() >= map_arr.size: 
                    raise ValueError("leaf id out of range for tree")
                mapped[:, t] = map_arr[ids] + offsets[t]

            leaves.append(mapped)

        return np.hstack(leaves)

    def  _build_leaf_table(self):
        '''
        Builds a per-tree leaf ID map w/ global offests. 
        '''

        self.leaf_maps_    = []
        self.leaf_offsets_ = []
        offset             = 0 

        for model in self.models_: 
            booster = model.get_booster() 
            df      = booster.trees_to_dataframe() 
            n_trees = self._num_trees(model)

            model_maps    = []
            model_offsets = []
            for t in range(n_trees): 
                leaf_nodes = (df.loc[(df["Tree"] == t) & (df["Feature"] == "Leaf"), "Node"]
                    .to_numpy())
                leaf_nodes = np.unique(leaf_nodes.astype(np.int64))

                if leaf_nodes.size == 0: 
                    map_arr = np.array([0], dtype=np.int64)
                    n_leaf  = 1
                else: 
                    max_id  = int(leaf_nodes.max())
                    map_arr = np.full(max_id + 1, -1, dtype=np.int64)
                    map_arr[leaf_nodes] = np.arange(leaf_nodes.size, dtype=np.int64) 
                    n_leaf  = leaf_nodes.size 

                model_maps.append(map_arr)
                model_offsets.append(offset)
                offset += n_leaf 

            self.leaf_maps_.append(model_maps)
            self.leaf_offsets_.append(np.asarray(model_offsets, dtype=np.int64))

        self.leaf_table_size_ = int(offset)

    def _build_model(self): 
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            objective="binary:logistic",
            eval_metric="logloss",
            device="cuda" if self.gpu else "cpu",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.kwargs,
        )

    def _init_leaf_projection(self, embed_dim: int, out_dim: int): 
        if (getattr(self, "leaf_embed_dim_", None) == embed_dim and 
            getattr(self, "leaf_out_dim_", None) == out_dim): 
            return 

        if not hasattr(self, "leaf_table_size_"): 
            self._build_leaf_table()

        rng = np.random.default_rng(self.random_state)

        # Zero initialization for bag of leaves projection 
        self.leaf_embed_dim_   = int(embed_dim)
        self.leaf_out_dim_     = int(out_dim)
        self.leaf_embed_table_ = rng.normal(
            0.0, 0.1, size=(self.leaf_table_size_, self.leaf_embed_dim_)
        ).astype(np.float32)
        self.leaf_proj_        = rng.normal(
            0.0, 0.1, size=(self.leaf_embed_dim_, self.leaf_out_dim_)
        ).astype(np.float32)
        self.leaf_proj_bias_   = np.zeros((self.leaf_out_dim_,), dtype=np.float32)

    def _encode_labels(self, y): 
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        classes = np.unique(y)
        classes = np.sort(classes)
        if classes.size <= 2: 
            raise ValueError("expected n_classes > 2")
        y_idx = np.searchsorted(classes, y)
        if not np.all(classes[y_idx] == y): 
            raise ValueError("y contains values not in sorted classes")
        return y_idx, classes 

    def _num_trees(self, model): 
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None: 
            return model.get_booster().num_boosted_rounds() 
        return int(best_iter) + 1 

    def _split_indices(self, n): 
        if not (self.early_stopping_rounds and self.eval_fraction > 0): 
            return np.arange(n), None 
        n_eval = max(1, int(n * self.eval_fraction))
        rng    = np.random.default_rng(self.random_state)
        idx    = rng.permutation(n)
        return idx[n_eval:], idx[:n_eval]

    def _prob_gt(self, X): 
        X = np.asarray(X, dtype=np.float32)
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])

# ---------------------------------------------------------
# Factory Functions (backwards compatibility with CrossValidator)
# ---------------------------------------------------------

def make_linear(alpha: float = 1.0):
    return lambda: LinearRegressor(alpha=alpha)

def make_rf_regressor(
    n_estimators: int = 400, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs
): 
    return lambda: RFRegressor(
        n_estimators=n_estimators, 
        n_jobs=compute_strategy.n_jobs,
        **kwargs
    ) 

def make_xgb_regressor(
    n_estimators: int = 300, 
    early_stopping_rounds: int = 200, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs 
): 
    return lambda: XGBRegressorWrapper(
        n_estimators=n_estimators, 
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=compute_strategy.n_jobs,
        gpu=compute_strategy.gpu_id is not None,
        **kwargs 
    )

def make_rf_classifier(
    n_estimators: int = 400, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
    **kwargs
):
    return lambda: RFClassifierWrapper(
        n_estimators=n_estimators, 
        n_jobs=compute_strategy.n_jobs,
        **kwargs
    )

def make_xgb_classifier(
    n_estimators: int = 400,
    early_stopping_rounds: int = 200,
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs,
):
    return lambda: XGBClassifierWrapper(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=compute_strategy.n_jobs,
        gpu=compute_strategy.gpu_id is not None,
        **kwargs,
    )


def make_logistic(
    C: float = 1.0, 
    compute_strategy: ComputeStrategy | None = None, # doesn't require 
    **kwargs
):
    return lambda: LogisticWrapper(C=C, **kwargs)

def make_svm_classifier(
    probability=True, 
    compute_strategy: ComputeStrategy | None = None, # doesn't require 
    **kwargs
): 
    return lambda: SVMClassifier(probability=probability, **kwargs)

def make_spatial_sfe(
    *,
    collate_fn=None,
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **fixed 
): 
    def _factory(**params): 
        merged = dict(fixed)
        merged.update(params) 
        collate = merged.pop("collate_fn", collate_fn) 
        return SpatialClassifier(
            collate_fn=collate, 
            compute_strategy=compute_strategy,
            **merged
        )
    return _factory 

def make_xgb_sfe(
    n_estimators: int = 400, 
    early_stopping_rounds: int = 200, 
    eval_fraction: float = 0.2, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs
):
    return lambda: XGBOrdinalRegressor(
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_fraction=eval_fraction,
        n_jobs=compute_strategy.n_jobs, 
        gpu=compute_strategy.gpu_id is not None,
        **kwargs
    )
