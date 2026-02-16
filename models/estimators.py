#!/usr/bin/env python3 
# 
# estimators.py  Andrew Belles  Dec 17th, 2025 
# 
# Esimator wrappers for all available models. 
# Compatible with Sklearn via Sklearn's estimator interface 
# and usable within Sklearn's CV infrastructure 
# 

from dataclasses import dataclass

from typing import Optional

import numpy as np 

from numpy.typing import NDArray

from sklearn.preprocessing   import StandardScaler

import time, copy, time, sys, torch   

from utils.resources         import ComputeStrategy

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
    MixedLoss,
)

from torch                   import nn 

from torch.utils.data        import (
    DataLoader, 
    TensorDataset,
)

from models.networks         import (
    DeepFusionMLP,
    MILOrdinalHead,
    TransformerProjector,
    ResidualMLP,
    Mixer,
    WideRidgeRegressor
) 

from models.loss               import (
    KESConfig,
    KernelEffectiveSamples,
    MixedLossAdapter,
    build_wide_deep_loss
)

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    cross_val_predict
)

from preprocessing.loaders   import (
    DatasetLoader,
    FusionDataset
)

from utils.helpers           import (
    bind 
)

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
# Wide and Deep Hierarchical Fusion 
# ---------------------------------------------------------

@dataclass 
class PhaseStats: 
    loss: float 
    ordinal: float 
    uncertainty: float 
    ridge: float = 0.0 
    deep_aux: float = 0.0 
    wide_aux: float = 0.0 

class HierarchicalFusionModel(BaseEstimator, RegressorMixin): 

    '''
    Top-level fusion module
    '''

    def __init__(
        self, 
        *,
        expert_dims: dict[str, int], 
        wide_in_dim: int, 
        cut_edges: NDArray,
        
        d_model: int, 
        transformer_heads: int, 
        transformer_layers: int, 
        transformer_ff_mult: int, 
        transformer_dropout: float, 
        transformer_attn_dropout: float, 
        gate_floor: float = 0.05, 

        trunk_hidden_dim: int, 
        trunk_depth: int, 
        trunk_dropout: float, 
        trunk_out_dim: Optional[int], 

        head_hidden_dim: Optional[int], 
        head_dropout: float, 
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 

        wide_l2_alpha: float, 
        wide_init_log_var: float, 
        ridge_scale: float, 

        w_ordinal: float = 1.0, 
        w_uncertainty: float = 1.0, 
        var_floor: float = 1e-6, 
        prob_eps: float = 1e-9, 

        mix_alpha: float = 0.2, 
        mix_mult: int = 2, 
        mix_min_lambda: float, 
        mix_with_replacement: bool = True, 

        kes_config: Optional[KESConfig] = None, 

        soft_epochs: int = 300, 
        hard_epochs: int = 200, 
        batch_size: int = 256, 
        lr_deep: float, 
        lr_wide: float, 
        weight_decay: float, 
        eval_fraction: float = 0.2, 
        early_stopping_rounds: int = 20, 
        min_delta: float = 1e-4, 
        random_state: int = 0,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.expert_dims              = expert_dims
        self.wide_in_dim              = wide_in_dim
        self.cut_edges                = np.asarray(cut_edges, dtype=np.float32)

        self.d_model                  = d_model
        self.transformer_heads        = transformer_heads
        self.transformer_layers       = transformer_layers
        self.transformer_ff_mult      = transformer_ff_mult
        self.transformer_dropout      = transformer_dropout
        self.transformer_attn_dropout = transformer_attn_dropout
        self.gate_floor               = gate_floor

        self.trunk_hidden_dim         = trunk_hidden_dim
        self.trunk_depth              = trunk_depth
        self.trunk_dropout            = trunk_dropout
        self.trunk_out_dim            = trunk_out_dim

        self.head_hidden_dim          = head_hidden_dim
        self.head_dropout             = head_dropout
        self.log_var_min              = log_var_min
        self.log_var_max              = log_var_max

        self.wide_l2_alpha            = wide_l2_alpha
        self.wide_init_log_var        = wide_init_log_var
        self.ridge_scale              = ridge_scale

        self.w_ordinal                = w_ordinal
        self.w_uncertainty            = w_uncertainty
        self.var_floor                = var_floor
        self.prob_eps                 = prob_eps

        self.mix_alpha                = mix_alpha
        self.mix_mult                 = mix_mult
        self.mix_min_lambda           = mix_min_lambda
        self.mix_with_replacement     = mix_with_replacement

        self.kes_config               = kes_config

        self.soft_epochs              = soft_epochs
        self.hard_epochs              = hard_epochs
        self.batch_size               = batch_size 
        self.lr_deep                  = lr_deep
        self.lr_wide                  = lr_wide
        self.weight_decay             = weight_decay
        self.eval_fraction            = eval_fraction
        self.early_stopping_rounds    = early_stopping_rounds
        self.min_delta                = min_delta
        self.random_state             = random_state

        self.aux_deep_weight = 0.25 
        self.aux_wide_weight = 0.25 

        self.device     = self.resolve_device(compute_strategy.device)

    # -----------------------------------------------------
    # Public Interface  
    # -----------------------------------------------------
    
    def predict(self, X) -> NDArray: 
        check_is_fitted(self, "is_fitted_")
        comp = self.predict_components(X)
        return comp["mu_fused"]

    def predict_distribution(self, X) -> dict[str, NDArray]: 
        check_is_fitted(self, "is_fitted_")
        comp = self.predict_components(X)
        return {
            "mu_fused": comp["mu_fused"],
            "log_var_fused": comp["log_var_fused"], 
            "std_fused": np.exp(0.5 * comp["log_var_fused"])
        }

    def extract(self, X) -> NDArray:
        check_is_fitted(self, "is_fitted_")
        loader = self.build_predict_loader(X)

        self.deep_.eval() 
        chunks: list[torch.Tensor] = []

        with torch.no_grad(): 
            for experts_b, _, _, _ in loader: 
                experts_b = {k: v.to(self.device) for k, v in experts_b.items()}
                emb = self.deep_.extract(experts_b)
                chunks.append(emb.detach().cpu())

        return torch.cat(chunks, dim=0).numpy() 

    # -----------------------------------------------------
    # Training 
    # -----------------------------------------------------

    def fit(self, X, y=None, coords=None): 
        experts = X["experts"]
        wide    = X["wide"]
        y_rank  = np.asarray(X["y_rank"] if y is None else y, dtype=np.float32).reshape(-1)
        coords  = X.get("coords", coords)

        experts = {k: np.asarray(v, dtype=np.float32) for k, v in experts.items()}
        wide    = np.asarray(wide, dtype=np.float32)

        tr_idx, va_idx = self.split_indices(y_rank)

        self.expert_scalers_ = {}
        for name, mat in experts.items(): 
            sc = StandardScaler()
            sc.fit(mat[tr_idx])
            experts[name] = sc.transform(mat).astype(np.float32, copy=False)
            self.expert_scalers_[name] = sc

        if wide.ndim == 1: 
            wide = wide.reshape(-1, 1)

        self.wide_scaler_ = StandardScaler() 
        self.wide_scaler_.fit(wide[tr_idx])

        wide = self.wide_scaler_.transform(wide).astype(np.float32, copy=False)

        sw = np.ones(y_rank.shape[0], dtype=np.float32)
        if coords is None: 
            raise ValueError("coords are required for sample weight")

        coords       = np.asarray(coords, dtype=np.float64)
        sw_tr, sw_va = self.build_sample_weights(y_rank, coords, tr_idx, va_idx)
        sw[tr_idx]   = sw_tr 
        sw[va_idx]   = sw_va

        train_loader = self.build_fusion_loader(
            experts, wide, y_rank, sw, 
            tr_idx, shuffle=True, drop_last=True)
        val_loader   = self.build_fusion_loader(
            experts, wide, y_rank, sw, 
            va_idx, shuffle=False, drop_last=False)

        self.build_model() 
        self.initialize_wide(wide[tr_idx], y_rank[tr_idx], sw_tr)

        best_soft = self.run_phase(
            name="Soft-Labels",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.soft_epochs,
            mixing=True,
            lr_scale=1.0,
            start_state=None
        )

        best_hard = self.run_phase(
            name="Rank-Labels",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.hard_epochs,
            mixing=False,
            lr_scale=0.2,
            start_state=best_soft
        )

        best_state = best_hard if best_hard is not None else best_soft 
        if best_state is not None: 
            self.deep_.load_state_dict(best_state["deep"])
            self.wide_.load_state_dict(best_state["wide"])

        self.is_fitted_ = True 
        return self 

    def run_phase(
        self,
        *,
        name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        epochs: int, 
        mixing: bool, 
        lr_scale: float, 
        start_state: Optional[dict[str, dict[str, torch.Tensor]]] = None 
    ): 
        print(f"[{name}] starting...", file=sys.stderr, flush=True)

        if start_state is not None: 
            self.deep_.load_state_dict(start_state["deep"])
            self.wide_.load_state_dict(start_state["wide"])

        self.build_optimizer(lr_scale=lr_scale, ep=epochs)

        best_val   = float("inf")
        best_state = None 
        patience   = 0 
        t0         = time.perf_counter()
    
        for ep in range(epochs): 
            _ = self.train_epoch(train_loader, mixing=mixing)
            val   = self.validate(val_loader)  
            score = val.ordinal + val.uncertainty 

            if score < best_val - self.min_delta:
                best_val   = score 
                best_state = {
                    "deep": copy.deepcopy(self.deep_.state_dict()),
                    "wide": copy.deepcopy(self.wide_.state_dict()),
                } 
                self.best_val_score_ = best_val 
                patience = 0 
            else: 
                patience += 1 
                if self.early_stopping_rounds and patience >= self.early_stopping_rounds: 
                    break 

            if ep % 5 == 0: 
                avg_dt = (time.perf_counter() - t0) / (ep + 1)
                
                print(
                    f"[epoch {ep:3d}] {avg_dt:.2f}s avg | "
                    f"val_loss={val.loss:.4f} | val_ord={val.ordinal:.4f} | " 
                    f"val_unc={val.uncertainty:.4f}",
                    file=sys.stderr,
                    flush=True,
                )

        return best_state 

    def train_epoch(self, loader: DataLoader, *, mixing: bool) -> PhaseStats:
        self.deep_.train() 
        self.wide_.train() 

        total, ord_sum, unc_sum, count = 0.0, 0.0, 0.0, 0 
        for experts_b, wide_b, yb, swb in loader: 
            experts_b = {k: v.to(self.device) for k, v in experts_b.items()}
            wide_b    = wide_b.to(self.device)
            yb        = yb.to(self.device)
            swb       = swb.to(self.device)

            loss, ord_raw, unc_raw = self.process_batch(
                experts_b, wide_b, yb, swb,
                with_mixing=mixing,
            )

            self.opt_.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_.step() 

            bsz      = yb.size(0) 
            total   += loss.item() * bsz 
            ord_sum += ord_raw * bsz 
            unc_sum += unc_raw * bsz 
            count   += bsz 

        if self.scheduler_ is not None: 
            self.scheduler_.step() 

        denom = max(count, 1)
        return PhaseStats(total / denom, ord_sum / denom, unc_sum / denom)

    def validate(self, loader: DataLoader) -> PhaseStats: 
        self.deep_.eval() 
        self.wide_.eval() 

        with torch.no_grad(): 
            total, ord_sum, unc_sum, count = 0.0, 0.0, 0.0, 0 
            for experts_b, wide_b, yb, _ in loader: 
                experts_b = {k: v.to(self.device) for k, v in experts_b.items()}
                wide_b    = wide_b.to(self.device)
                yb        = yb.to(self.device)

                loss, ord_raw, unc_raw = self.process_batch(
                    experts_b, wide_b, yb, None,
                    with_mixing=False,
                )

                bsz      = yb.size(0) 
                total   += loss.item() * bsz 
                ord_sum += ord_raw * bsz 
                unc_sum += unc_raw * bsz 
                count   += bsz 

        denom = max(count, 1)
        return PhaseStats(total / denom, ord_sum / denom, unc_sum / denom)

    # -----------------------------------------------------
    # Processing  
    # -----------------------------------------------------

    def process_batch(
        self,
        experts_b: dict[str, torch.Tensor], 
        wide_b: torch.Tensor, 
        yb: torch.Tensor, 
        swb: torch.Tensor,
        *,
        with_mixing: bool = False 
    ) -> tuple[torch.Tensor, float, float]: 

        if with_mixing: 
            y_bucket = torch.floor(yb).to(torch.long)
            self.mixer_.fit(y_bucket=y_bucket)

            idx_a, idx_b, lam = self.mixer_.plan() 
            idx_a, idx_b = idx_a.to(self.device), idx_b.to(self.device) 
            lam = lam.to(self.device, dtype=yb.dtype)
        
            experts = {k: self.mixer_.transform(v) for k, v in experts_b.items()}
            wide    = self.mixer_.transform(wide_b)
        else: 
            experts = experts_b 
            wide    = wide_b 
            lam, idx_a, idx_b = None, None, None 

        deep_out = self.deep_(experts, return_features=False)
        wide_out = self.wide_(wide)

        context = {
            "mu_deep": deep_out["mu_deep"], 
            "log_var_deep": deep_out["log_var_deep"], 
            "mu_wide": wide_out["mu_wide"], 
            "log_var_wide": wide_out["log_var_wide"], 
            "w_ordinal": self.w_ordinal, 
            "w_uncertainty": self.w_uncertainty
        }

        if with_mixing: 
            # no sample weights in mixed loss 
            context.update({
                "y_rank_a": yb[idx_a], 
                "y_rank_b": yb[idx_b],
                "mix_lambda": lam, 
                "sample_weight_a": swb[idx_a],
                "sample_weight_b": swb[idx_b]
            })
        else: 
            context.update({
                "y_rank": yb, 
                "sample_weight": swb 
            })
        comp  = self.mix_loss_fn_(**context) if with_mixing else self.loss_fn_(**context)
        ridge = self.ridge_scale * self.wide_.ridge_penalty()
        total = comp.total + ridge

        ord_raw = float(comp.raw["ordinal"].detach().item())
        unc_raw = float(comp.raw["uncertainty"].detach().item())

        return total, ord_raw, unc_raw 

    # -----------------------------------------------------
    # Instantiation 
    # -----------------------------------------------------

    def initialize_wide(self, x_train, y_train, sw_train): 
        x = np.asarray(x_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        w = np.asarray(sw_train, dtype=np.float64).reshape(-1) 

        n, d = x.shape 

        xb = np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1)

        sqrt_w = np.sqrt(np.clip(w, 1e-8, None))[:, None]
        xw = xb * sqrt_w
        yw = y * sqrt_w[:, 0]

        lam = float(self.wide_l2_alpha)
        I = np.eye(d + 1, dtype=np.float64)
        I[0, 0] = 0.0  # no bias penalty

        A = xw.T @ xw + lam * I
        b = xw.T @ yw
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(A) @ b

        bias = float(theta[0])
        coef = theta[1:]

        pred = x @ coef + bias
        resid = y - pred
        var = float(np.average(resid * resid, weights=np.clip(w, 1e-8, None)))
        init_lv = float(np.log(max(var, self.var_floor)))
        init_lv = float(np.clip(init_lv, self.log_var_min, self.log_var_max))

        with torch.no_grad():
            self.wide_.linear.weight.copy_(
                torch.from_numpy(coef.reshape(1, -1)).to(self.device, dtype=torch.float32)
            )
            if self.wide_.linear.bias is not None:
                self.wide_.linear.bias.fill_(bias)
            self.wide_.log_var_param.fill_(init_lv)

    def build_model(self): 
        self.deep_ = DeepFusionMLP(
            expert_dims=self.expert_dims,
            d_model=self.d_model,
            n_heads=self.transformer_heads,
            n_layers=self.transformer_layers,
            ff_mult=self.transformer_ff_mult,
            transformer_dropout=self.transformer_dropout,
            transformer_attn_dropout=self.transformer_attn_dropout,
            pre_norm=True,
            gate_floor=self.gate_floor,
            hidden_dim=self.trunk_hidden_dim,
            depth=self.trunk_depth,
            dropout=self.trunk_dropout,
            trunk_out_dim=self.trunk_out_dim,
            head_hidden_dim=self.head_hidden_dim,
            head_dropout=self.head_dropout,
            log_var_min=self.log_var_min,
            log_var_max=self.log_var_max,
        ).to(self.device)

        self.wide_ = WideRidgeRegressor(
            in_dim=self.wide_in_dim,
            l2_alpha=self.wide_l2_alpha,
            init_log_var=self.wide_init_log_var,
            log_var_min=self.log_var_min,
            log_var_max=self.log_var_max,
        ).to(self.device)

        # self.deep_ = torch.compile(self.deep_, mode="reduce-overhead", fullgraph=False)
        # self.wide_ = torch.compile(self.wide_, mode="reduce-overhead", fullgraph=False)

        self.loss_fn_ = build_wide_deep_loss(
            cut_edges=self.cut_edges,
            log_var_min=self.log_var_min,
            log_var_max=self.log_var_max,
            var_floor=self.var_floor,
            prob_eps=self.prob_eps,
        )

        self.mix_loss_fn_ = MixedLossAdapter(
            self.loss_fn_,
            target_pairs={"y_rank": ("y_rank_a", "y_rank_b")},
        )

        self.mixer_ = Mixer(
            alpha=self.mix_alpha,
            mix_mult=self.mix_mult,
            min_lambda=self.mix_min_lambda,
            with_replacement=self.mix_with_replacement,
        )

    def ensure_loader(
        self,
        X,
        *,
        shuffle: bool = False, 
        drop_last: bool = False 
    ) -> DataLoader: 
        if isinstance(X, DataLoader): 
            return X 
        if isinstance(X, FusionDataset): 
            return self.build_loader(X, shuffle=shuffle, drop_last=drop_last)

    def build_loader(
        self,
        dataset, 
        *,
        shuffle: bool,
        drop_last: bool 
    ) -> DataLoader: 
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=drop_last 
        )

    def build_fusion_loader(
        self,
        experts: dict[str, NDArray],
        wide: NDArray,
        y_rank: NDArray,
        sw: NDArray,
        idx: NDArray,
        *,
        shuffle: bool,
        drop_last: bool = False
    ) -> DataLoader:
        experts_t = {k: torch.from_numpy(v[idx]).float() for k, v in experts.items()}
        wide_t    = torch.from_numpy(wide[idx]).float() 
        y_t       = torch.from_numpy(y_rank[idx]).float() 
        sw_t      = torch.from_numpy(sw[idx]).float() 

        ds = FusionDataset(experts_t, wide_t, y_t, sw_t)
        return self.build_loader(ds, shuffle=shuffle, drop_last=drop_last)

    def build_predict_loader(self, X) -> DataLoader: 
        experts = {k: np.asarray(v, dtype=np.float32) for k, v in X["experts"].items()}
        wide    = np.asarray(X["wide"], dtype=np.float32)
        n       = wide.shape[0]
        if wide.ndim == 1: 
            wide = wide.reshape(-1, 1)

        if hasattr(self, "expert_scalers_"): 
            experts = {
                k: self.expert_scalers_[k].transform(v).astype(np.float32, copy=False)
                for k, v in experts.items()
            }

        if hasattr(self, "wide_scaler_"):
            wide = self.wide_scaler_.transform(wide).astype(np.float32, copy=False)

        experts_t = {k: torch.from_numpy(v).float() for k, v in experts.items()}
        wide_t    = torch.from_numpy(wide).float() 
        y_t       = torch.zeros(n, dtype=torch.float32)
        sw_t      = torch.ones(n, dtype=torch.float32)

        ds = FusionDataset(experts_t, wide_t, y_t, sw_t)
        return self.ensure_loader(ds, shuffle=False, drop_last=False)

    def build_optimizer(self, *, lr_scale: float = 1.0, ep: int = 1): 
        self.opt_ = torch.optim.AdamW([
            {"params": self.deep_.parameters(), "lr": lr_scale * self.lr_deep},
            {"params": self.wide_.parameters(), "lr": lr_scale * self.lr_wide}
        ], weight_decay=self.weight_decay)

        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_, T_max=max(1, ep)
        )

    def build_sample_weights(
        self,
        y_rank: NDArray,
        coords: NDArray, 
        tr_idx: NDArray,
        va_idx: NDArray
    ) -> tuple[NDArray, NDArray]: 
        sw_train = np.ones(tr_idx.shape[0], dtype=np.float32)
        sw_val   = np.ones(va_idx.shape[0], dtype=np.float32)

        self.kes_ = KernelEffectiveSamples(self.kes_config)

        sw_train  = (self.kes_.fit_transform(y_rank[tr_idx], coords[tr_idx])
                     .astype(np.float32, copy=False)) 
        sw_val    = self.kes_.transform(y_rank[va_idx]).astype(np.float32, copy=False)
        return sw_train, sw_val 

    def split_indices(self, y_rank: NDArray) -> tuple[NDArray, NDArray]: 
        n = y_rank.shape[0]
        if self.eval_fraction <= 0.0: 
            idx = np.arange(n) 
            return idx, idx 

        y_bucket = np.floor(y_rank).astype(np.int64)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.eval_fraction,
            random_state=self.random_state
        )
        return next(splitter.split(np.zeros((n, 1)), y_bucket))

    def resolve_device(self, device: Optional[str]) -> torch.device: 
        if device is not None: 
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------
    # Predict Helpers 
    # -----------------------------------------------------

    def predict_components(self, X) -> dict[str, NDArray]: 
        check_is_fitted(self, "is_fitted_") 
        loader = self.build_predict_loader(X)

        self.deep_.eval() 
        self.wide_.eval() 

        keys = (
            "mu_deep", "log_var_deep",
            "mu_wide", "log_var_wide",
            "mu_fused", "log_var_fused",
            "alpha_deep", "alpha_wide"
        )

        acc: dict[str, list[torch.Tensor]] = {k: [] for k in keys}

        with torch.no_grad(): 
            for experts_b, wide_b, _, _ in loader: 
                experts_b = {k: v.to(self.device) for k, v in experts_b.items()} 
                wide_b    = wide_b.to(self.device)
                out_b     = self.predict_batch_components(experts_b, wide_b)
                for k in keys: 
                    acc[k].append(out_b[k].detach().cpu())
                    
        return {k: torch.cat(v, dim=0).numpy() for k, v in acc.items()}

    def predict_batch_components(
        self, 
        experts_b: dict[str, torch.Tensor], 
        wide_b: torch.Tensor
    ) -> dict[str, torch.Tensor]: 
        deep_out = self.deep_(experts_b, return_features=False)
        wide_out = self.wide_(wide_b) 

        fuse_term = self.loss_fn_.terms[0]
        mu_fused, log_var_fused, alpha_d, alpha_w = fuse_term.fuse({
            "y": deep_out["mu_deep"], # for shape 
            "mu_deep": deep_out["mu_deep"],
            "log_var_deep": deep_out["log_var_deep"],
            "mu_wide": wide_out["mu_wide"],
            "log_var_wide": wide_out["log_var_wide"],
        })

        return {
            "mu_deep": deep_out["mu_deep"],
            "log_var_deep": deep_out["log_var_deep"],
            "mu_wide": wide_out["mu_wide"],
            "log_var_wide": wide_out["log_var_wide"],
            "mu_fused": mu_fused,
            "log_var_fused": log_var_fused,
            "alpha_deep": alpha_d,
            "alpha_wide": alpha_w,
        }


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

        alpha_mae: float = 0.5, 
        beta_supcon: float = 0.5, 
        supcon_temperature: float = 0.07, 
        supcon_dim: int = 128,
        ens: float = 0.995, 

        transformer_dim: int,     
        transformer_heads: int = 4, 
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1, 
        transformer_attn_dropout: float = 0.1, 
        
        reduce_dim: int = 256, 
        reduce_depth: int = 1, 
        reduce_dropout: float = 0.0, 

        soft_epochs: int = 400,
        hard_epochs: int = 300, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        random_state: int = 0, 
        early_stopping_rounds: int = 20, 
        eval_fraction: float = 0.15, 
        min_delta: float = 1e-3, 
        batch_size: int = 256, 

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

        self.alpha_mae                = alpha_mae
        self.beta_supcon              = beta_supcon
        self.supcon_temperature       = supcon_temperature
        self.supcon_dim               = supcon_dim
        self.ens                      = ens

        self.transformer_dim          = transformer_dim 
        self.transformer_heads        = transformer_heads 
        self.transformer_layers       = transformer_layers
        self.transformer_dropout      = transformer_dropout
        self.transformer_attn_dropout = transformer_attn_dropout

        self.reduce_dim               = reduce_dim 
        self.reduce_depth             = reduce_depth
        self.reduce_dropout           = reduce_dropout

        self.soft_epochs              = soft_epochs
        self.hard_epochs              = hard_epochs
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
        print(f"[params] ens={self.ens}, batch_size={self.batch_size}")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)

        y_rank, y_bucket = soft_rank_and_bucket(y)
        self.n_classes_  = max(2, int(np.floor(np.nanmax(y_rank))) + 1) 
        self.classes_    = np.arange(self.n_classes_, dtype=np.int64)
        class_weights = torch.ones(self.n_classes_, dtype=torch.float32)

        self.build_model(class_weights=class_weights)

        if self.eval_fraction and self.eval_fraction > 0: 
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.eval_fraction, random_state=self.random_state
            )
            train_idx, val_idx = next(splitter.split(X, y_bucket))
            X_tr, y_tr   = X[train_idx], y_rank[train_idx]
            X_val, y_val = X[val_idx], y_rank[val_idx]
        else: 
            X_tr, y_tr   = X, y_rank 
            X_val, y_val = X, y_rank

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_soft = self.run_phase(
            name="Manifold-Mixing",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.soft_epochs,
            mixing=True 
        )

        if best_soft is not None: 
            best_hard = self.run_phase(
                name="Soft-Rank",
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=max(1, self.hard_epochs), 
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
        y = np.asarray(y, dtype=np.float32)

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
            val_score, val_corn, val_mae = self.validate(val_loader)

            if val_score < best_val - self.min_delta: 
                best_val   = val_score 
                best_state = copy.deepcopy(self.model_.state_dict())
                self.best_val_score_ = best_val 
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
                            f"val_mae={val_mae:.4f}")
                    print(msg, file=sys.stderr, flush=True)

        return best_state

    def validate(self, val_loader): 
        self.model_.eval() 
        val_mae  = 0.0 
        val_corn = 0.0 
        count    = 0 

        with torch.no_grad(): 
            for xb, yb in val_loader: 
                xb, yb = xb.to(self.device), yb.to(self.device)

                feats  = self.forward_feats(xb)
                _, logits, proj = self.forward_logits(feats, with_supcon=False)
                _, corn, mae, _ = self.loss_fn_(logits, proj, yb)

                corn = corn.mean() 
                mae  = mae.mean()  

                val_corn += corn.item() * yb.size(0) 
                val_mae  += mae.item() * yb.size(0)
                count    += yb.size(0)

        val_corn /= max(count, 1)
        val_mae  /= max(count, 1)
        return val_corn + val_mae, val_corn, val_mae  

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
            reduce_dim=self.reduce_dim,
            reduce_depth=self.reduce_depth,
            reduce_dropout=self.reduce_dropout,
            use_logit_scaler=True
        )

        self.model_           = nn.Module() 
        self.model_.tokenizer = self.tokenizer_  
        self.model_.backbone  = self.backbone_
        self.model_.head      = self.head_ 
        self.model_.to(self.device)

        self.loss_fn_ = HybridOrdinalLoss(
            n_classes=self.n_classes_,
            # class_weights=class_weights,
            alpha_mae=self.alpha_mae,
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
        yb = yb.to(self.device, dtype=torch.float32).view(-1)
        up = np.nextafter(float(self.n_classes_ - 1), 0.0)
        yb = yb.clamp(0.0, up)
        yb_bucket = torch.floor(yb).to(torch.int64)

        if with_mixing:
            x, idx_a, idx_b, mix_lam = self.mixer_(x_embed, yb_bucket, return_indices=True)
            y_a = yb[idx_a]
            y_b = yb[idx_b]
        else: 
            x = x_embed 

        feats = self.model_.backbone(x) 
        _, logits, proj = self.forward_logits(feats, with_supcon=True)

        if with_mixing:
            loss, corn, mae, sup = self.mix_loss_(logits, proj, y_a, y_b, mix_lam)
        else: 
            loss, corn, mae, sup = self.loss_fn_(logits, proj, yb)
            loss = loss.mean() 
            corn = corn.mean() 
            mae  = mae.mean() 
            sup  = sup.mean() 

        return loss, corn, mae, sup, x.size(0)

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
# Helper functions 
# ---------------------------------------------------------

def soft_rank_and_bucket(y):
    arr = np.asarray(y).reshape(-1)
    if np.issubdtype(arr.dtype, np.floating): 
        rank = arr.astype(np.float32)
    else: 
        rank = arr.astype(np.int64).astype(np.float32)
    rank = np.clip(rank, 0.0, None)
    bucket = np.floor(rank).astype(np.int64)
    return rank, bucket 

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
