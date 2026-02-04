#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

import numpy as np 


from sklearn.preprocessing   import StandardScaler

import time, copy, time, sys, torch   

from abc                     import abstractmethod

from utils.resources         import ComputeStrategy

from numpy.typing            import NDArray

from sklearn.base            import (
    BaseEstimator, 
    RegressorMixin, 
    ClassifierMixin,
    check_is_fitted, 
    clone
)

from sklearn.ensemble        import (
    RandomForestRegressor, 
    RandomForestClassifier 
)

from sklearn.svm             import SVC

from sklearn.linear_model    import LogisticRegression 

from xgboost                 import (
    XGBClassifier, 
    XGBRegressor  
)

from utils.loss_funcs        import (
    HybridOrdinalLoss,
    LogitScaler,
    MixedLoss,
    compute_ens_weights
)

from scipy.sparse            import csr_matrix

from torch                   import nn 

from torch.utils.data        import (
    DataLoader, 
    TensorDataset,
    non_deterministic, 
    random_split, 
    Subset 
)

from models.networks         import (
    GatedAttentionPooling,
    ResNetMIL,
    MILOrdinalHead,
    HypergraphBackbone,
    TransformerProjector,
    ResidualMLP,
    Mixer
) 

from models.graph.construction import (
    LOGRADIANCE_GATE_HIGH,
    LOGRADIANCE_GATE_LOW
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.metrics         import (
    cohen_kappa_score
)

from utils.helpers           import align_on_fips, bind 

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
        max_iter: int = 5000, 
        solver: str = "lbfgs", 
        random_state: int | None = None, 
        **kwargs 
    ): 
        self.C            = C 
        self.max_iter     = max_iter 
        self.solver       = solver 
        self.random_state = random_state 
        self.kwargs       = kwargs 

    def fit(self, X, y): 
        self.model_ = LogisticRegression(
            C=self.C ,
            max_iter=self.max_iter ,
            solver=self.solver, 
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
        ordinal: bool = True, 
        calibration_cv: int = 5, 
        calibration_max_iter: int = 5000, 
        calibration_solver: str = "newton-cg", 
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        **kwargs
    ): 
        self.C = C 
        self.kernel         = kernel 
        self.gamma          = gamma 
        self.probability    = probability 
        self.class_weight   = class_weight 
        self.random_state   = random_state 
        self.ordinal        = ordinal 
        self.kwargs         = kwargs

        self.model_         = None 
        self.models_        = None 
        self.classes_       = None 

        self.calibration_scaler_  = None 
        self.calibrator_          = None 
        self.calibration_cv       = calibration_cv
        self.calibration_max_iter = calibration_max_iter 
        self.calibration_solver   = calibration_solver

        self.compute_strategy = compute_strategy 

    def fit(self, X, y, coords=None): 
        X             = np.asarray(X, dtype=np.float64)
        y             = np.asarray(y).reshape(-1)
        self.classes_ = np.sort(np.unique(y))

        if not self.ordinal: 
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
            return self 

        if not self.probability: 
            raise ValueError("ordinal SVM requires probability=True")

        y_idx     = np.searchsorted(self.classes_, y)
        n_classes = len(self.classes_)
        if n_classes < 2: 
            raise ValueError("ordinal SVM requires multiclass")

        self.models_ = []
        for k in range(n_classes - 1): 
            y_bin = (y_idx > k).astype(np.int64)
            clf = SVC(
                C=self.C, 
                kernel=self.kernel, 
                gamma=self.gamma, 
                probability=False,
                class_weight=self.class_weight,
                random_state=self.random_state,
                **self.kwargs
            )
            clf.fit(X, y_bin)
            self.models_.append(clf) 

        if self.probability:

            cv = StratifiedKFold(
                n_splits=self.calibration_cv,
                shuffle=True,
                random_state=self.random_state
            )

            X_scores = np.zeros((X.shape[0], len(self.models_)), dtype=np.float64)

            for k, model in enumerate(self.models_):
                y_bin = (y_idx > k).astype(np.int64)
                cv_clf = clone(model)
                scores = cross_val_predict(
                    cv_clf,
                    X,
                    y_bin,
                    cv=cv,
                    method="decision_function",
                    n_jobs=self.compute_strategy.n_jobs
                )
                scores = np.asarray(scores)
                if scores.ndim > 1: 
                    scores = scores[:, -1]
                X_scores[:, k] = scores 

            self.calibration_scaler_ = StandardScaler() 
            X_scores_scaled          = self.calibration_scaler_.fit_transform(X_scores)

            self.calibrator_ = LogisticRegression(
                solver=self.calibration_solver,
                penalty=None, 
                random_state=self.random_state,
                max_iter=self.calibration_max_iter
            )

            counts  = np.bincount(y_idx, minlength=len(self.classes_))
            weights = counts.max() / np.clip(counts, 1, None)
            sample_weight = weights[y_idx]

            self.calibrator_.fit(X_scores_scaled, y, sample_weight=sample_weight) 

        return self 

    def predict(self, X, coords=None):
        check_is_fitted(self, ["classes_"])

        if not self.ordinal: 
            return self.model_.predict(X)

        if self.probability and self.calibrator_ is not None: 
            probs = self.predict_proba(X)
            idx   = np.argmax(probs, axis=1)
            return self.classes_[idx]
        else: 
            X     = np.asarray(X, dtype=np.float64)
            votes = np.zeros(X.shape[0], dtype=np.int64)
            for clf in self.models_:
                votes += clf.predict(X)

            return self.classes_[votes]

    def predict_proba(self, X, coords=None):
        if not self.probability: 
            raise AttributeError("SVMClassifier was instantiated with probability=False")

        if not self.ordinal:
            return self.model_.predict_proba(X)

        check_is_fitted(self, ["calibrator_", "models_", "calibration_scaler_"])

        X = np.asarray(X, dtype=np.float64)

        n_samples = X.shape[0]
        n_models  = len(self.models_)
        scores    = np.zeros((n_samples, n_models), dtype=np.float64)

        for k, clf in enumerate(self.models_):
            scores[:, k] = clf.decision_function(X)

        scores_scaled = self.calibration_scaler_.transform(scores)
        
        return self.calibrator_.predict_proba(scores_scaled)

# ---------------------------------------------------------
# Spatial Estimator contract 
# ---------------------------------------------------------

class BaseSpatialEstimator(BaseEstimator, ClassifierMixin):

    '''
    Contract Spatial based models must fulfill. 
    '''

    def __init__(
        self, 
        *, 
        in_channels: int,
        fc_dim: int, 
        dropout: float, 
        supcon_dim: int,
        epochs: int,
        lr: float,
        weight_decay: float,
        alpha_rps: float, 
        beta_supcon: float,
        ens: float = 0.999, 
        supcon_temperature: float,
        random_state: int,
        early_stopping_rounds: int | None = 15,
        eval_fraction: float = 0.2,
        min_delta: float, 
        batch_size: int,
        target_global_batch: int,
        shuffle: bool,
        collate_fn,
        class_values,
        compute_strategy,
        compile_model 
    ):
        self.in_channels = in_channels

        self.fc_dim = fc_dim
        self.dropout = dropout
        self.supcon_dim = supcon_dim

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha_rps = alpha_rps
        self.beta_supcon = beta_supcon
        self.ens = ens 
        self.supcon_temperature = supcon_temperature

        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.target_global_batch = target_global_batch
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.class_values = class_values
        self.compute_strategy = compute_strategy
        self.compile_model = compile_model

        self.device     = self.resolve_device(compute_strategy.device)
        self.classes_   = None
        self.n_classes_ = None
        self.model_     = nn.Module()

    def eval(self):
        if hasattr(self.model_, "eval"): 
            self.model_.eval() 
        return self 

    def train(self): 
        if hasattr(self.model_, "train"): 
            self.model_.train() 
        return self 

    def fit(self, X, y=None, val_loader=None, callbacks=None): 

        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
             torch.backends.cudnn.benchmark = True
             torch.backends.cuda.matmul.allow_tf32 = True
             torch.backends.cudnn.allow_tf32 = True

        if callbacks is None: 
            callbacks = []
        elif callable(callbacks): 
            callbacks = [callbacks]

        loader = self.ensure_loader(X, shuffle=self.shuffle)
        effective_batch  = loader.batch_size
        world_size       = getattr(self, "world_size_", 1)
        self.accum_steps = self.resolve_accum_steps(effective_batch, world_size) 

        print(f"[data specs] Batch Size={self.batch_size}, "
              f"Target Batch Size={self.target_global_batch} "
              f"Accumulation Steps={self.accum_steps}")

        if y is None: 
            label_loader = self.as_eval_loader(X)
            all_labels   = []
            for batch in label_loader: 
                labels = batch[1]
                all_labels.append(np.asarray(labels).reshape(-1))
            y_full = np.concatenate(all_labels) if all_labels else np.asarray([], dtype=np.int64)
        else: 
            y_full = np.asarray(y).reshape(-1)

        if y_full.size == 0: 
            raise ValueError("empty loader")

        if self.class_values is not None: 
            classes = np.asarray(self.class_values, dtype=np.int64)
            if classes.ndim != 1 or classes.size == 0: 
                raise ValueError("class_values must be non-empty 1D list/array")
            classes = np.unique(classes)
            if not np.all(np.isin(y_full, classes)):
                missing = np.setdiff1d(np.unique(y_full), classes)
                raise ValueError(f"labels not in class_values: {missing}")
            self.classes_   = classes 
            self.n_classes_ = len(classes)
        else: 
            self.classes_   = np.unique(y_full)
            self.n_classes_ = len(self.classes_)

        y_idx           = np.searchsorted(self.classes_, y_full)
        class_counts    = np.bincount(y_idx, minlength=self.n_classes_)
        self.class_counts_ = class_counts

        if self.n_classes_ < 2: 
            raise ValueError("ordinal regression requires at least 2 classes")

        if val_loader is None: 
            loader, val_loader = self.split_loader(loader, y_full)

        self.build_model()

        class_weights = compute_ens_weights(class_counts, self.ens) 
        self.loss_fn_ = HybridOrdinalLoss(
            n_classes=self.n_classes_,
            class_weights=class_weights, 
            alpha_rps=self.alpha_rps, 
            beta_supcon=self.beta_supcon,
            temperature=self.supcon_temperature
        ).to(self.device)

        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        opt     = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.epochs,
            eta_min=1e-6
        )

        best_state = None 
        best_val   = float("inf")
        patience   = 0 
        run_es     = self.early_stopping_rounds is not None and val_loader is not None 

        start      = time.perf_counter()
        for ep in range(self.epochs): 
            base_ds = getattr(loader, "dataset", None)
            if base_ds is not None and hasattr(base_ds, "set_epoch"):
                base_ds.set_epoch(ep)

            self.model_.train() 

            accum = max(1, int(self.accum_steps))
            opt.zero_grad()

            step_idx    = -1 
            total_loss  = 0.0
            total_count = 0

            total_corn  = 0.0 
            total_rps   = 0.0 
            total_sup   = 0.0 

            for step_idx, batch in enumerate(loader): 
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"): 
                    loss, bsz, lc, lrps, ls = self.process_batch(batch)
                    
                    total_loss += loss.item() * bsz 
                    total_corn += lc.item() * bsz  
                    total_sup  += ls.item() * bsz 
                    total_rps  += lrps.item() * bsz 

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

            scheduler.step() 

            avg_sup  = total_sup / max(total_count, 1)
            val_loss = None 
            corn     = None 
            rps      = None 
            if run_es: 
                if self.early_stopping_rounds is None: 
                    raise TypeError
                val_loss, corn, rps = self.eval_loss(val_loader)
                if val_loss < best_val - self.min_delta: 
                    best_val   = val_loss 
                    self.best_val_score_ = best_val  
                    best_state = copy.deepcopy(self.model_.state_dict())
                    patience   = 0 
                else:
                    patience  += 1
                    if patience >= self.early_stopping_rounds: 
                        break 

                if callbacks: 
                    metrics = {"val_loss": val_loss} 
                    for cb in callbacks: 
                        cb(ep, metrics)

            avg_loss = total_loss / max(total_count, 1)

            avg_dt   = (time.perf_counter() - start) / (ep + 1)
            msg      = (f"[epoch {ep:3d}] {avg_dt:.2f}s avg | training_loss={avg_loss:.4f} | " 
                        f"supcon={avg_sup:.4f}") 

            if ep % 5 == 0: 
                if val_loss is not None: 
                    msg += (f" | val_loss={val_loss:.4f} | val_corn={corn:.4f} | "
                            f"val_rps={rps:.4f}")
                    print(msg, file=sys.stderr, flush=True)

        if best_state is not None: 
            self.model_.load_state_dict(best_state)
        return self 

    def predict(self, batch) -> torch.Tensor: 
        self.model_.eval() 
        inputs, labels, batch_indices = self.unpack_batch(batch)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"): 
            feats        = self.forward_feats(inputs, batch_indices)
            _, logits, _ = self.forward_logits(feats, with_supcon=False)
            probs        = self.logits_to_probs(logits)
            idx          = torch.argmax(probs, dim=1)
        return torch.as_tensor(self.classes_, device=idx.device)[idx]

    def predict_proba(self, batch) -> torch.Tensor: 
        self.model_.eval() 
        inputs, labels, batch_indices = self.unpack_batch(batch)
        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"): 
            feats        = self.forward_feats(inputs, batch_indices)
            _, logits, _ = self.forward_logits(feats, with_supcon=False)
            probs        = self.logits_to_probs(logits)
        return probs 

    def extract(self, X) -> NDArray: 
        loader = self.ensure_loader(X, shuffle=False)
        outs   = []
        self.model_.eval() 
        with torch.no_grad(): 
            for batch in loader: 
                inputs, labels, batch_indices = self.unpack_batch(batch)
                feats     = self.forward_feats(inputs, batch_indices)
                emb, _, _ = self.forward_logits(feats, with_supcon=False)
                outs.append(emb.cpu().numpy())
        return np.vstack(outs)

    def process_batch(self, batch, *, with_supcon=True): 
        if self.n_classes_ is None or self.classes_ is None: 
            raise TypeError 

        inputs, labels, batch_indices = self.unpack_batch(batch)

        y_np  = np.asarray(labels).reshape(-1)
        y_idx = np.searchsorted(self.classes_, y_np)
        yb    = torch.as_tensor(y_idx, dtype=torch.int64, device=self.device)

        if self.model_.training: 
            inputs = self.augment_inputs(inputs)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"): 
            feats = self.forward_feats(inputs, batch_indices)
            _, logits, proj = self.forward_logits(feats, with_supcon=with_supcon)
            loss, loss_corn, loss_rps, loss_sup = self.loss_fn_(logits, proj, yb)

        bsz = yb.size(0)
        return loss, bsz, loss_corn, loss_rps, loss_sup 

    def augment_inputs(self, inputs):
        k = np.random.randint(0, 4)
        x = torch.rot90(inputs, k, dims=[2, 3])
        if np.random.random() > 0.5: 
            x = torch.flip(x, dims=[3])
        return x 

    def unpack_batch(self, batch): 
        inputs, labels, batch_indices = batch 
        return inputs, labels, batch_indices   

    def prepare_inputs(self, tiles, batch_indices): 
        xb   = tiles.to(self.device, non_blocking=True)
        bidx = batch_indices.to(self.device, non_blocking=True)
        return xb, bidx

    def forward_feats(self, tiles, batch_indices):
        xb, bidx = self.prepare_inputs(tiles, batch_indices)
        return self.model_.backbone(xb, bidx)

    def forward_logits(self, feats, with_supcon: bool): 
        if not hasattr(self.model_, "head"): 
            raise RuntimeError("must build and model before calling forward")
        emb, logits, proj = self.model_.head(feats)
        return emb, logits, (proj if with_supcon else None)

    def build_model(self): 
        self.backbone_ = self.build_backbone()
        self.head_     = self.build_head(self.backbone_.out_dim, self.n_classes_)
        self.model_    = nn.Module() 
        self.model_.backbone = self.backbone_ 
        self.model_.head     = self.head_ 
        self.model_.to(self.device)
        if self.device.type == "cuda" and self.compile_model:
            self.model_ = torch.compile(self.model_, mode="reduce-overhead", fullgraph=False)

    @abstractmethod 
    def build_backbone(self):
        raise NotImplementedError

    @abstractmethod 
    def build_head(self, in_dim, n_classes): 
        raise NotImplementedError

    def build_loss(self, class_counts): 
        if self.n_classes_ is None: 
            raise ValueError("n_classes not set")
        class_weight = (class_counts.max() / np.clip(class_counts, 1, None))
        return HybridOrdinalLoss(self.n_classes_, class_weight)

    # -----------------------------------------------------
    # Universal Helper Functions 
    # -----------------------------------------------------

    def logits_to_probs(self, logits):
        prob_gt = torch.sigmoid(logits)
        n, k    = prob_gt.shape
        probs   = torch.empty((n, k + 1), device=logits.device, dtype=logits.dtype)
        probs[:, 0] = 1.0 - prob_gt[:, 0]
        for i in range(1, k): 
            probs[:, i] = prob_gt[:, i - 1] - prob_gt[:, i]
        probs[:, -1] = prob_gt[:, -1]
        probs = torch.clamp(probs, min=1e-9)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs 

    def resolve_device(self, device: str | None): 
        if device is not None: 
            return torch.device(device)
        return torch.device(str(device) if torch.cuda.is_available() else "cpu") 

    def make_loader(self, dataset, shuffle: bool):
        pin = self.compute_strategy.device == "cuda"
        if self.compute_strategy.n_jobs == -1:
            num_workers = 8
        else:
            num_workers = min(self.compute_strategy.n_jobs, 8)

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

    def resolve_accum_steps(self, world_size, effective_batch):
        world_size   = max(1, int(world_size))
        local_global = max(1, int(effective_batch) * world_size)
        return max(1, int(np.ceil(self.target_global_batch / local_global)))

    def ensure_loader(self, X, shuffle: bool):
        if isinstance(X, DataLoader):
            return X
        return self.make_loader(X, shuffle=shuffle)

    def split_loader(self, loader, y_full):
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

        train_loader = self.make_loader(train_ds, shuffle=True)
        val_loader   = self.make_loader(val_ds, shuffle=False)
        return train_loader, val_loader

    def as_eval_loader(self, X): 
        if isinstance(X, DataLoader): 
            ds = getattr(X, "dataset", None)
            if ds is None: 
                return X 
            return self.make_loader(ds, shuffle=False)
        return self.make_loader(X, shuffle=False)

    def eval_loss(self, loader): 
        self.model_.eval() 
        total_metric = 0.0
        total_corn   = 0.0 
        total_rps    = 0.0 
        count        = 0 

        with torch.no_grad(): 
            for batch in loader: 
                _, bsz, lc, lrps, _ = self.process_batch(batch, with_supcon=False)
                total_corn += lc.item() * bsz
                total_rps  += (lrps.item() / self.alpha_rps) * bsz
                count      += bsz 

        total_metric = total_corn + total_rps
        denom        = max(count, 1)
        return total_metric / denom, total_corn / denom, total_rps / denom  

# ---------------------------------------------------------
# Supervised Feature Extraction based Models  
# ---------------------------------------------------------

class SpatialClassifier(BaseSpatialEstimator): 

    def __init__(
        self,
        *,
        # backbone 
        in_channels: int, 
        attn_dim: int = 256, 
        attn_dropout: float = 0.0, 
        resnet_weights=None, 

        # head 
        fc_dim: int = 128, 
        dropout: float = 0.2, 
        supcon_dim: int = 128, 

        # training 
        epochs: int = 120, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        alpha_rps: float = 0.5, 
        beta_supcon: float = 0.5, 
        ens: float = 0.999, 
        supcon_temperature: float = 0.07, 
        random_state: int = 0, 
        early_stopping_rounds: int = 15, 
        eval_fraction: float = 0.15, 
        min_delta: float = 1e-3, 
        batch_size: int = 8, 
        target_global_batch: int = 8, 
        shuffle: bool = True, 
        collate_fn=None, 
        class_values: list[int] | None = None, 

        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        compile_model: bool = True 
    ): 
        super().__init__( 
            in_channels=in_channels,
            fc_dim=fc_dim,
            dropout=dropout,
            supcon_dim=supcon_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            alpha_rps=alpha_rps,
            beta_supcon=beta_supcon,
            supcon_temperature=supcon_temperature,
            ens=ens,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_fraction=eval_fraction,
            min_delta=min_delta,
            batch_size=batch_size,
            target_global_batch=target_global_batch,
            shuffle=shuffle,
            collate_fn=collate_fn,
            class_values=class_values,
            compute_strategy=compute_strategy,
            compile_model=compile_model,
        )
        self.attn_dim = attn_dim
        self.attn_dropout = attn_dropout
        self.resnet_weights = resnet_weights

    # -----------------------------------------------------
    # Internal Helpers 
    # -----------------------------------------------------

    def augment_inputs(self, inputs):
        k = np.random.randint(0, 4)
        x = torch.rot90(inputs, k, dims=[2, 3])
        if np.random.random() > 0.5: 
            x = torch.flip(x, dims=[3])
        return x 

    def build_backbone(self): 
        return ResNetMIL(
            in_channels=self.in_channels,
            attn_dim=self.attn_dim,
            attn_dropout=self.attn_dropout,
            weights=self.resnet_weights
        )

    def build_head(self, in_dim, n_classes):
        return MILOrdinalHead(
            in_dim=in_dim,
            fc_dim=self.fc_dim,
            n_classes=n_classes,
            dropout=self.dropout,
            supcon_dim=self.supcon_dim,
            use_logit_scaler=True
        )

# ---------------------------------------------------------
# HyperGraph based Spatial Classifier  
# ---------------------------------------------------------

class SpatialGATClassifier(BaseSpatialEstimator): 

    def __init__(
        self,
        *,
        # backbone 
        tile_size: int = 256, 
        patch_size: int = 32, 
        embed_dim: int = 64, 
        gnn_dim: int = 128, 
        gnn_layers: int = 1, 
        gnn_heads: int = 1, 
        gap_attn_dim: int = 64, 
        attn_dropout: float = 0.0, 
        patch_stat: str = "p95", 
        patch_quantile: float = 0.95, 
        thresh_low: float = LOGRADIANCE_GATE_LOW, 
        thresh_high: float = LOGRADIANCE_GATE_HIGH,
        max_bag_frac: float = 1.0, 

        # head 
        fc_dim: int = 128, 
        dropout: float = 0.2, 
        supcon_dim: int = 128, 

        # training 
        epochs: int = 120, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        alpha_rps: float = 0.5, 
        beta_supcon: float = 0.5, 
        supcon_temperature: float = 0.07, 
        ens: float = 0.999,
        random_state: int = 0, 
        early_stopping_rounds: int = 15, 
        eval_fraction: float = 0.15, 
        min_delta: float = 1e-3, 
        batch_size: int = 8, 
        target_global_batch: int = 8, 
        shuffle: bool = True, 
        collate_fn=None, 
        class_values: list[int] | None = None, 

        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        compile_model: bool = True 
    ): 
        super().__init__( 
            in_channels=1,
            fc_dim=fc_dim,
            dropout=dropout,
            supcon_dim=supcon_dim,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            alpha_rps=alpha_rps,
            beta_supcon=beta_supcon,
            supcon_temperature=supcon_temperature,
            ens=ens,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            eval_fraction=eval_fraction,
            min_delta=min_delta,
            batch_size=batch_size,
            target_global_batch=target_global_batch,
            shuffle=shuffle,
            collate_fn=collate_fn,
            class_values=class_values,
            compute_strategy=compute_strategy,
            compile_model=compile_model,
        )

        self.tile_size      = tile_size 
        self.patch_size     = patch_size 
        self.embed_dim      = embed_dim 
        self.gap_attn_dim   = gap_attn_dim
        self.gnn_dim        = gnn_dim 
        self.thresh_low     = thresh_low 
        self.thresh_high    = thresh_high
        self.max_bag_frac   = max_bag_frac
        self.gnn_layers     = gnn_layers 
        self.gnn_heads      = gnn_heads 
        self.attn_dropout   = attn_dropout
        self.patch_stat     = patch_stat 
        self.patch_quantile = patch_quantile
    
    def build_backbone(self):
        return HypergraphBackbone(
            tile_size=self.tile_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            gnn_dim=self.gnn_dim,
            gnn_layers=self.gnn_layers,
            gnn_heads=self.gnn_heads,
            attn_dropout=self.attn_dropout,
            dropout=self.dropout,
            max_bag_frac=self.max_bag_frac,
            patch_stat=self.patch_stat,
            patch_quantile=self.patch_quantile,
            thresh_low=self.thresh_low,
            thresh_high=self.thresh_high
        )

    def build_head(self, in_dim, n_classes):
        return MILOrdinalHead(
            in_dim=in_dim,
            fc_dim=self.fc_dim,
            n_classes=n_classes,
            dropout=self.dropout,
            supcon_dim=self.supcon_dim,
            use_logit_scaler=True 
        )

    def build_model(self):
        self.backbone_ = self.build_backbone()
        self.head_     = self.build_head(self.backbone_.out_dim, self.n_classes_)
        
        # pool across tiles per bag via attention 
        self.pool_     = GatedAttentionPooling(
            in_dim=self.backbone_.out_dim,
            attn_dim=self.gap_attn_dim,
            attn_dropout=self.attn_dropout
        )

        self.model_          = nn.Module() 
        self.model_.backbone = self.backbone_ 
        self.model_.head     = self.head_
        self.model_.pool     = self.pool_ 

        self.model_.to(self.device)
        
        if self.device.type == "cuda" and self.compile_model:
            self.model_ = torch.compile(self.model_, mode="reduce-overhead", fullgraph=False)

    def forward_feats(self, tiles, batch_indices):
        xb, bidx   = self.prepare_inputs(tiles, batch_indices)
        tile_feats = self.model_.backbone(xb)

        if bidx.numel() == 0: 
            return tile_feats 

        batch_size = int(bidx.max().item()) + 1 
        pooled     = self.model_.pool(tile_feats, bidx, batch_size)
        return pooled 

# ---------------------------------------------------------
# Mixing Tabular, Residual MLP Model 
# ---------------------------------------------------------

class TFTabular(BaseEstimator, ClassifierMixin): 
    
    def __init__(
        self, 
        *,
        in_dim: int, 
        hidden_dim: int = 256, 
        depth: int = 6, 
        dropout: float = 0.1, 

        mix_alpha: float = 0.2, 
        mix_mult: int = 2, 
        max_mix: int | None = None,
        anchor_power: float = 1.0, 

        alpha_rps: float = 0.5, 
        beta_supcon: float = 0.5, 
        supcon_temperature: float = 0.07, 
        supcon_dim: int = 128,
        ens: float = 0.999, 

        transformer_dim: int,     
        transformer_tokens: int, 
        transformer_heads: int = 4, 
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1, 
        transformer_attn_dropout: float = 0.1, 
        
        epochs: int = 2000, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        random_state: int = 0, 
        early_stopping_rounds: int = 15, 
        eval_fraction: float = 0.15, 
        min_delta: float = 1e-3, 
        batch_size: int = 1024, 

        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        compile_model: bool = True 
    ):
        self.in_dim                   = in_dim
        self.hidden_dim               = hidden_dim
        self.depth                    = depth
        self.dropout                  = dropout

        self.mix_alpha                = mix_alpha
        self.mix_mult                 = mix_mult
        self.max_mix                  = max_mix
        self.anchor_power             = anchor_power

        self.alpha_rps                = alpha_rps
        self.beta_supcon              = beta_supcon
        self.supcon_temperature       = supcon_temperature
        self.supcon_dim               = supcon_dim
        self.ens                      = ens

        self.transformer_dim          = transformer_dim 
        self.transformer_tokens       = transformer_tokens
        self.transformer_heads        = transformer_heads 
        self.transformer_layers       = transformer_layers
        self.transformer_dropout      = transformer_dropout
        self.transformer_attn_dropout = transformer_attn_dropout

        self.epochs                   = epochs
        self.lr                       = lr
        self.weight_decay             = weight_decay
        self.batch_size               = batch_size
        self.early_stopping_rounds    = early_stopping_rounds
        self.eval_fraction            = eval_fraction
        self.min_delta                = min_delta
        self.random_state             = random_state
        self.device                   = self.resolve_device(compute_strategy.device)
        self.compile_model            = bool(compile_model)

        self.classes_   = None
        self.n_classes_ = None
        self.model_     = nn.Module()

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        y_idx         = np.searchsorted(self.classes_, y)
        class_counts  = np.bincount(y_idx, minlength=self.n_classes_)
        class_weights = compute_ens_weights(class_counts, self.ens)

        self.build_model(class_weights)

        if self.eval_fraction and self.eval_fraction > 0: 
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.eval_fraction, random_state=self.random_state
            )
            train_idx, val_idx = next(splitter.split(X, y))
            X_tr, y_tr   = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else: 
            X_tr, y_tr   = X, y 
            X_val, y_val = X, y 

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_soft = self.run_phase(
            name="Manifold-Mixing",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            mixing=True 
        )

        if best_soft is not None: 
            best_hard = self.run_phase(
                name="Hard-Labels",
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=max(1, self.epochs), 
                mixing=False, 
                start_state=best_soft,
                lr_scale=0.3
            )
            if best_hard is not None: 
                self.model_.load_state_dict(best_hard)

        return self 

    def predict_proba(self, X): 
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad(): 
            xb    = torch.from_numpy(X).to(self.device)
            feats = self.forward_feats(xb)
            _, logits, _ = self.forward_logits(feats, with_supcon=False)
            probs = self.logits_to_probs(logits)
        return probs.cpu().numpy() 

    def predict(self, X): 
        probs = self.predict_proba(X)
        idx   = probs.argmax(axis=1)
        return self.classes_[idx]

    def extract(self, X): 
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad(): 
            xb    = torch.from_numpy(X).to(self.device)
            feats = self.forward_feats(xb)
            emb, _, _ = self.forward_logits(feats, with_supcon=False)
        return emb.cpu().numpy() 

    def loss(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=self.batch_size, shuffle=False
        )

        loss, _, _ = self.validate(loader)
        return loss 

    # -----------------------------------------------------
    # Internal Helpers 
    # -----------------------------------------------------

    def run_phase(
        self,
        name, 
        *,
        train_loader,
        val_loader,
        epochs: int, 
        mixing: bool,
        start_state=None,
        lr_scale: float = 1.0
    ): 

        print(f"[{name}] starting...")

        phase_lr = lr_scale * self.lr 

        if start_state is not None: 
            self.model_.load_state_dict(start_state)

        self.opt_ = torch.optim.AdamW(
            self.model_.parameters(), lr=phase_lr, weight_decay=self.weight_decay
        )
        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_, T_max=epochs)

        best_val   = float("inf")
        best_state = None 
        patience   = 0
        start      = time.perf_counter()

        for ep in range(epochs): 
            _         = self.train_epoch(train_loader, mixing=mixing)
            val_score, val_corn, val_rps = self.validate(val_loader)

            if val_score < best_val - self.min_delta: 
                best_val   = val_score 
                best_state = copy.deepcopy(self.model_.state_dict())
                patience   = 0
            else: 
                patience  += 1 
                if self.early_stopping_rounds and patience >= self.early_stopping_rounds: 
                    break 

            avg_dt   = (time.perf_counter() - start) / (ep + 1)
            msg      = (f"[epoch {ep:3d}] {avg_dt:.2f}s avg") 

            if ep % 20 == 0: 
                if val_score is not None: 
                    msg += (f" | val_loss={val_score:.4f} | val_corn={val_corn:.4f} | "
                            f"val_rps={val_rps:.4f}")
                    print(msg, file=sys.stderr, flush=True)

        return best_state

    def validate(self, val_loader): 
        self.model_.eval() 
        val_rps  = 0.0 
        val_corn = 0.0 
        count    = 0 

        with torch.no_grad(): 
            for xb, yb in val_loader: 
                xb, yb = xb.to(self.device), yb.to(self.device)

                feats  = self.forward_feats(xb)
                _, logits, proj = self.forward_logits(feats, with_supcon=False)
                _, corn, rps, _ = self.loss_fn_(logits, proj, yb)

                corn = corn.mean() 
                rps  = rps.mean() 

                val_corn += corn.item() * yb.size(0) 
                val_rps  += rps.item() * yb.size(0)
                count    += yb.size(0)

        val_corn = val_corn / max(count, 1)
        val_rps  = val_rps / max(count, 1)
        return val_corn + val_rps, val_corn, val_rps  

    def train_epoch(self, train_loader, *, mixing: bool): 
        self.model_.train() 
        total_loss  = 0.0 
        total_count = 0 

        for xb, yb in train_loader: 
            xb, yb = xb.to(self.device), yb.to(self.device)

            loss, _, _, _, bsz = self.process_batch(xb, yb, with_mixing=mixing)

            self.opt_.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_.step() 

            total_loss  += loss.item() * bsz 
            total_count += bsz 

        if self.scheduler_ is not None: 
            self.scheduler_.step() 

        return total_loss / max(total_count, 1)

    def resolve_device(self, device: str | None): 
        if device is not None: 
            return torch.device(device)
        return torch.device(str(device) if torch.cuda.is_available() else "cpu") 

    def build_model(self, class_weights): 
        self.tokenizer_ = TransformerProjector(
            in_dim=self.in_dim, 
            out_dim=self.transformer_dim,
            d_model=self.transformer_dim,
            num_tokens=self.transformer_tokens,
            n_heads=self.transformer_heads,
            n_layers=self.transformer_layers,
            dropout=self.transformer_dropout,
            attn_dropout=self.transformer_attn_dropout,
            pre_norm=True
        )

        self.backbone_ = ResidualMLP(
            in_dim=self.transformer_dim, 
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
            out_dim=self.hidden_dim 
        )

        self.head_     = MILOrdinalHead(
            in_dim=self.hidden_dim,
            fc_dim=self.hidden_dim,
            n_classes=self.n_classes_,
            dropout=self.dropout,
            supcon_dim=self.supcon_dim,
            use_logit_scaler=True
        )

        self.model_           = nn.Module() 
        self.model_.tokenizer = self.tokenizer_  
        self.model_.backbone  = self.backbone_
        self.model_.head      = self.head_ 
        self.model_.to(self.device)

        self.loss_fn_ = HybridOrdinalLoss(
            n_classes=self.n_classes_,
            class_weights=class_weights,
            alpha_rps=self.alpha_rps,
            beta_supcon=self.beta_supcon,
            temperature=self.supcon_temperature,
            reduction="none"
        ).to(self.device)

        self.mix_loss_ = MixedLoss(self.loss_fn_)

        self.mixer_    = Mixer(
            class_weights=class_weights,
            alpha=self.mix_alpha,
            mix_mult=self.mix_mult,
            max_mix=self.max_mix,
            anchor_power=self.anchor_power
        )

        if self.device.type == "cuda" and self.compile_model:
            self.model_ = torch.compile(self.model_, mode="reduce-overhead", fullgraph=False)

    def process_batch(self, xb, yb, with_mixing: bool = True): 
        x_embed = self.model_.tokenizer(xb)

        if with_mixing:
            x, y_a, y_b, mix_lam = self.mixer_(x_embed, yb)
        else: 
            x = x_embed 

        feats = self.model_.backbone(x) 
        _, logits, proj = self.forward_logits(feats, with_supcon=True)

        if with_mixing:
            loss, corn, rps, sup = self.mix_loss_(logits, proj, y_a, y_b, mix_lam)
        else: 
            loss, corn, rps, sup = self.loss_fn_(logits, proj, yb)
            loss = loss.mean() 
            corn = corn.mean() 
            rps  = rps.mean() 
            sup  = sup.mean() 

        return loss, corn, rps, sup, x.size(0)

    def forward_feats(self, x): 
        x = self.model_.tokenizer(x)
        return self.model_.backbone(x)

    def forward_logits(self, feats, with_supcon=True):
        emb, logits, proj = self.model_.head(feats)
        if not with_supcon:
            proj = None 
        return emb, logits, proj 

    def logits_to_probs(self, logits):
        prob_gt = torch.sigmoid(logits)
        n, k    = prob_gt.shape
        probs   = torch.empty((n, k + 1), device=logits.device, dtype=logits.dtype)
        probs[:, 0] = 1.0 - prob_gt[:, 0]
        for i in range(1, k): 
            probs[:, i] = prob_gt[:, i - 1] - prob_gt[:, i]
        probs[:, -1] = prob_gt[:, -1]
        probs = torch.clamp(probs, min=1e-9)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs 

# ---------------------------------------------------------
# Scalar Embedding Learner 
# ---------------------------------------------------------

class ResBlock(nn.Module): 
    def __init__(self, dim: int, dropout: float): 
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim) 
        self.act  = nn.GELU() 
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 

    def forward(self, x): 
        residual = x 

        out  = self.bn1(self.fc1(x))
        out  = self.act(out)
        out  = self.fc1(out)
        out  = self.drop(out)

        out  = self.bn2(self.fc2(out))
        out  = self.act(out)
        out  = self.fc2(out)
        out  = self.drop(out)

        return out + residual  


class EmbeddingProjector(BaseEstimator): 

    '''
    Projection head for embedding compression. Trains with CORN loss on a small ordinal head 
    and returns only learned embeddings. Supports a single-layer projector or a 3-layer 
    manifold learner.  
    '''

    def __init__(
        self,
        in_dim: int, 
        out_dim: int = 5, 
        hidden_dims: tuple[int, ...] | None = None, 
        supcon_dim: int = 128, 
        mode: str = "single",
        dropout: float = 0.1, 
        epochs: int = 200, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-4, 
        use_residual: bool = False, 
        n_residual_blocks: int = 1, 
        alpha_rps: float = 5.0,
        beta_supcon: float = 0.5, 
        temperature: float = 0.1, 
        batch_size: int = 128, 
        early_stopping_rounds: int = 30, 
        eval_fraction: int = 0, 
        random_state: int = 0, 
        device: str | None = None 
    ): 
        self.in_dim                = in_dim
        self.out_dim               = out_dim
        self.hidden_dims           = hidden_dims
        self.supcon_dim            = supcon_dim
        self.mode                  = mode 
        self.dropout               = dropout
        self.epochs                = epochs
        self.lr                    = lr
        self.weight_decay          = weight_decay
        self.use_residual          = use_residual 
        self.n_residual_blocks     = n_residual_blocks
        self.alpha_rps             = alpha_rps 
        self.beta_supcon           = beta_supcon 
        self.temperature           = temperature 
        self.batch_size            = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction         = eval_fraction
        self.random_state          = random_state

        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device) 
        elif device == "cuda" and torch.cuda.is_available(): 
            self.device = torch.device("cuda")
        else: 
            self.device = torch.device("cpu")

    def fit(self, X, y): 
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        
        self.classes_   = np.unique(y)
        self.n_classes_ = len(self.classes_)

        y_idx         = np.searchsorted(self.classes_, y)
        class_counts  = np.bincount(y_idx, minlength=self.n_classes_)
        class_weights = class_counts.max() / np.clip(class_counts, 1, None)

        self._build(self.n_classes_)
        
        self.criterion_ = HybridOrdinalLoss(
            n_classes=self.n_classes_,
            class_weights=class_weights,
            alpha_rps=self.alpha_rps, 
            beta_supcon=self.beta_supcon,
            temperature=self.temperature
        ).to(self.device)

        opt      = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.eval_fraction and self.eval_fraction > 0:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.eval_fraction, random_state=self.random_state
            )
            train_idx, val_idx = next(splitter.split(X, y))
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

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

                emb    = self.model_.proj(xb)
                raw_logits = self.model_.head(emb)

                logits = self.model_.scaler(raw_logits)

                z    = self.model_.supcon(emb)
                loss, _, _, _ = self.criterion_(logits, z, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

            self.model_.eval()
            val_rps_accum = 0.0 
            count = 0 

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    emb = self.model_.proj(xb)
                    logits = self.model_.head(emb)
                    logits = self.model_.scaler(logits)
                    
                    _, _, w_rps, _ = self.criterion_(logits, None, yb)
                    val_rps_accum += w_rps.item() * yb.size(0) 
                    count += yb.size(0)

            avg_weighted_rps = (val_rps_accum / max(count, 1)) / self.alpha_rps

            if avg_weighted_rps < best_val - 1e-4:
                best_val = avg_weighted_rps 
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
                logits = self.model_.scaler(self.model_.head(emb))

                _, _, loss_rps, _ = self.criterion_(logits, None, yb)
                total += loss_rps.item() * yb.size(0)
                count += yb.size(0)
        loss = total / max(count, 1)
        loss = loss / self.alpha_rps
        return loss / (self.n_classes_ - 1)

    def predict(self, X): 
        X = np.asarray(X, dtype=np.float32) 
        self.model_.eval() 
        outs = []

        with torch.no_grad(): 
            for i in range(0, X.shape[0], self.batch_size): 
                xb = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)

                emb = self.model_.proj(xb)
                logits = self.model_.scaler(self.model_.head(emb))

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
                outs.append(preds.cpu().numpy())

        return np.concatenate(outs)
    
    def predict_proba(self, X): 
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval() 
        outs = []

        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size): 
                xb  = torch.from_numpy(X[i:i+self.batch_size]).to(self.device)
                emb = self.proj_(xb)
                logits = self.model_.scaler(self.model_.head(emb))

                prob_gt = torch.sigmoid(logits)
                n, k    = prob_gt.shape 
                probs   = torch.empty((n, k + 1), device=self.device)
                probs[:, 0] = 1.0 - prob_gt[:, 0]
                for j in range(1, k): 
                    probs[:, j] = prob_gt[:, j - 1] - prob_gt[:, j]
                probs[:, -1] = prob_gt[:, -1]
                
                probs = torch.clamp(probs, min=1e-9)
                probs = probs / probs.sum(dim=1, keepdim=True)

                outs.append(probs.cpu().numpy())

        return np.vstack(outs)

    def _build(self, n_classes: int): 

        if self.mode == "single": 
            if self.dropout > 0: 
                self.proj_ = nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(self.in_dim, self.out_dim)
                )
            else: 
                self.proj_ = nn.Linear(self.in_dim, self.out_dim)
        elif self.mode == "manifold": 

            if self.hidden_dims: 
                width = self.hidden_dims[0]
            else: 
                width = 256 

            layers = []
            layers.append(nn.Linear(self.in_dim, width))

            for _ in range(self.n_residual_blocks): 
                layers.append(ResBlock(width, self.dropout))

            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.GELU())
            layers.append(nn.Linear(width, self.out_dim))
            self.proj_ = nn.Sequential(*layers)
        else: 
            raise ValueError(f"unknown projector mode: {self.mode}")

        self.head_  = nn.Linear(self.out_dim, n_classes - 1)
        self.supcon_head_ = nn.Sequential(
            nn.Linear(self.out_dim, self.supcon_dim),
            nn.BatchNorm1d(self.supcon_dim),
            nn.GELU(), 
            nn.Linear(self.supcon_dim, self.supcon_dim)
        )

        self.scaler_ = LogitScaler(initial_value=1.0)

        self.model_        = nn.Module() 
        self.model_.supcon = self.supcon_head_
        self.model_.proj   = self.proj_ 
        self.model_.head   = self.head_ 
        self.model_.scaler = self.scaler_ 
        self.model_.to(self.device)

# ---------------------------------------------------------
# Factory Functions (backwards compatibility with CrossValidator)
# ---------------------------------------------------------

def make_linear(alpha: float = 1.0):
    return bind(LinearRegressor, alpha=alpha)

def make_rf_regressor(
    n_estimators: int = 400, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs
): 
    return bind(
        RFRegressor,
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
    return bind(
        XGBRegressorWrapper,
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
    return bind(
        RFClassifierWrapper,
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
    return bind(
        XGBClassifierWrapper,
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
    return bind(LogisticWrapper, C=C, **kwargs)

def make_svm_classifier(
    probability=True, 
    compute_strategy: ComputeStrategy | None = None, # doesn't require 
    **kwargs
): 
    return bind(SVMClassifier, probability=probability, **kwargs)

def spatial_sfe_factory(
    *,
    collate_fn, 
    compute_strategy,
    fixed,
    **params
):
    merged = dict(fixed)
    merged.update(params) 
    collate = merged.pop("collate_fn", collate_fn) 
    return SpatialClassifier(
        collate_fn=collate, 
        compute_strategy=compute_strategy,
        **merged
    )

def make_spatial_sfe(
    *,
    collate_fn=None,
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **fixed 
): 
    return bind(
        spatial_sfe_factory,
        collate_fn=collate_fn,
        compute_strategy=compute_strategy,
        fixed=fixed
    )

def make_xgb_sfe(
    n_estimators: int = 400, 
    early_stopping_rounds: int = 200, 
    eval_fraction: float = 0.2, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
    **kwargs
):
    return bind(
        XGBOrdinalRegressor,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_fraction=eval_fraction,
        n_jobs=compute_strategy.n_jobs, 
        gpu=compute_strategy.gpu_id is not None,
        **kwargs
    )
