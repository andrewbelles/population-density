#!/usr/bin/env python3 
# 
# evaluators.py  Andrew Belles  Jan 24th, 2026 
# 
# Implementation of all optimization evaluators 
# 
# 


import gc, optuna, torch
from inspect import Attribute 

import numpy as np 

from numpy.typing              import NDArray

from abc                       import (
    ABC, 
    abstractmethod 
)

from dataclasses               import replace  

from sklearn.preprocessing     import StandardScaler

from typing                    import (
    Any, 
    Callable, 
    Dict,
    Mapping,
    Optional 
)

from sklearn.metrics           import (
    accuracy_score, 
)

from sklearn.model_selection   import (
    StratifiedShuffleSplit, 
    KFold, 
)

from torch.utils.data          import (
    DataLoader, 
    Subset,
    TensorDataset
)

from optimization.engine       import (
    run_optimization, 
    NestedCVConfig,
    EngineConfig, 
    WorkerSpec
)

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    TaskSpec,
    ScaledEstimator,
    ranked_probability_score
)

from models.graph.processing   import CorrectAndSmooth 

from models.graph.construction import (
    make_queen_adjacency_factory,
)

from preprocessing.loaders     import load_oof_predictions 

from utils.helpers             import (
    make_train_mask,
    align_on_fips,
    load_probs_for_fips
)

from scipy.special             import (
    ndtr 
)

from utils.resources           import (
    ComputeStrategy,
    DevicePool
)

# ---------------------------------------------------------
# Generic Evaluator Contract 
# ---------------------------------------------------------

class OptunaEvaluator(ABC):
    @abstractmethod 
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: 
        pass 

    @abstractmethod 
    def evaluate(self, params: Dict[str, Any]) -> float: 
        pass 

# ---------------------------------------------------------
# Generic Helper Functions  
# ---------------------------------------------------------

def predict_proba_if_any(model, X, coords=None): 
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X, coords=coords)
        except TypeError: 
            proba = model.predict_proba(X)
    elif hasattr(model, "load_oof_predictions"):
        proba = model.load_oof_predictions(X, coords)
    else: 
        return None 

    proba = np.asarray(proba, np.float32)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return proba 

def select_metric_value(metrics, task, metric): 
    if metric is not None and metric in metrics and not np.isnan(metrics[metric]):
        return metrics[metric]

    if task.task_type == "classification": 
        for key in ("f1_macro", "accuracy", "roc_auc"):
            if key in metrics and not np.isnan(metrics[key]):
                return metrics[key]
    else: 
        if "r2" in metrics and not np.isnan(metrics["r2"]):
            return metrics["r2"]

    raise ValueError("no suitable metric found")

# ---------------------------------------------------------
# Hierarchical Fusion Model Evaluator  
# ---------------------------------------------------------

class HierarchicalFusionEvaluator(OptunaEvaluator): 

    def __init__(
        self,
        *,
        X: dict, 
        model_factory: Callable,
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        fixed_params: Optional[Dict[str, Any]] = None, 
        random_state: int = 0, 
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
        cv_folds: int = 3 
    ): 
        self.X                = X 
        self.model_factory    = model_factory
        self.param_space_fn   = param_space 
        self.fixed_params     = dict(fixed_params or {})
        self.random_state     = random_state 
        self.compute_strategy = compute_strategy
        self.cv_folds         = cv_folds 

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> float: 
        kwargs = dict(self.fixed_params)
        kwargs.update(params)

        y_all = np.asarray(self.X["y"], dtype=np.float64).reshape(-1)
        n     = y_all.shape[0] 

        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(n))): 
            model = self.model_factory(**kwargs)

            try: 
                model.fit(self.subset_inputs(train_idx, include_y=True)) 
                comp  = model.predict_components(self.subset_inputs(val_idx, include_y=False))
                score = self.score_from_components(y_all[val_idx], comp) 
                if not np.isfinite(score): 
                    return float("inf")
                fold_scores.append(float(score))
                if trial is not None: 
                    trial.report(float(np.mean(fold_scores)), step=fold_idx)
                    if trial.should_prune(): 
                        raise optuna.TrialPruned() 
            finally: 
                del model 
                _cleanup_cuda()

        return float(np.mean(fold_scores))

    @staticmethod
    def gaussian_crps(
        y: NDArray, 
        mu: NDArray, 
        log_var: NDArray, 
        var_floor: float = 1e-6
    ) -> float:
        y  = np.asarray(y, dtype=np.float64).reshape(-1)
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        lv = np.asarray(log_var, dtype=np.float64).reshape(-1)
        sigma = np.exp(0.5 * lv)
        sigma = np.clip(sigma, var_floor, None)

        z = (y - mu) / sigma
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
        inv_sqrt_pi  = 1.0 / np.sqrt(np.pi)
        cdf = ndtr(z)
        pdf = np.exp(-0.5 * z * z) * inv_sqrt_2pi
        per = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - inv_sqrt_pi)
        return float(np.mean(np.clip(per, 0.0, None)))

    def subset_inputs(self, idx: NDArray, *, include_y: bool) -> Dict[str, Any]:
        idx = np.asarray(idx, dtype=np.int64)
        out = {
            "experts": {k: np.asarray(v)[idx] for k, v in self.X["experts"].items()},
            "wide": np.asarray(self.X["wide"])[idx],
        }
        if include_y:
            out["y"] = np.asarray(self.X["y"])[idx]
            if "coords" in self.X and self.X["coords"] is not None:
                out["coords"] = np.asarray(self.X["coords"])[idx]
        return out

    def score_from_components(self, y_true: NDArray, comp: Dict[str, NDArray]) -> float:
        y  = np.asarray(y_true, dtype=np.float64).reshape(-1)
        mu = np.asarray(comp["mu_fused"], dtype=np.float64).reshape(-1)
        lv = np.asarray(comp["log_var_fused"], dtype=np.float64).reshape(-1)
        l1 = float(np.mean(np.abs(y - mu)))
        crps = self.gaussian_crps(y, mu, lv)
        return l1 + crps

# ---------------------------------------------------------
# Evaluator for Self-Supervised Feature Extractor 
# ---------------------------------------------------------

class SSFEEvaluator(OptunaEvaluator): 

    def __init__(
        self,
        filepath: str,
        loader_func: Callable, 
        model_factory: Callable, 
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        *,
        random_state: int = 0, 
        n_runs: int = 2, 
        max_samples: int | None = None, 
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.filepath         = filepath
        self.loader           = loader_func
        self.factory          = model_factory
        self.param_space_fn   = param_space
        self.random_state     = random_state
        self.n_runs           = max(1, int(n_runs))
        self.max_samples      = max_samples
        self.compute_strategy = compute_strategy
        self.cache_           = None

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any], trial: optuna.Trial | None = None) -> float:
        X, collate_fn = self.load_data_once() 
        scores        = []

        for run_idx in range(self.n_runs): 
            X_run = self.subset_if_needed(X, seed=self.random_state + run_idx)
            model = self.build_model(
                params=params,
                collate_fn=collate_fn,
                run_seed=self.random_state + run_idx 
            )

            try: 
                model.fit(X_run)
                score = self.score(model)
            finally: 
                del model 
                _cleanup_cuda()

            scores.append(float(score))

            if trial is not None: 
                trial.report(float(np.mean(scores)), step=run_idx)
                if trial.should_prune(): 
                    raise optuna.TrialPruned() 

        return float(np.mean(scores))

    def load_data_once(self): 
        if self.cache_ is None: 
            raw = self.loader(self.filepath)
            self.cache_ = self.unwrap_data(raw)
        return self.cache_ 

    def unwrap_data(self, raw): 
        # spatial/image loader 
        if isinstance(raw, dict) and "dataset" in raw: 
            return raw["dataset"], raw.get("collate_fn")
        # tabular loader 
        if isinstance(raw, dict) and "features" in raw: 
            return np.asarray(raw["features"], dtype=np.float32), raw.get("collate_fn")
        # direct dataset
        if isinstance(raw, np.ndarray): 
            return np.asarray(raw, dtype=np.float32), None 
        return raw, None 

    def subset_if_needed(self, X, seed: int): 
        if self.max_samples is None: 
            return X 
        n = len(X)
        if n <= self.max_samples: 
            return X 

        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=self.max_samples, replace=False))

        if isinstance(X, np.ndarray): 
            return X[idx]
        return Subset(X, idx.tolist())

    def build_model(
        self,
        *,
        params: Dict[str, Any],
        collate_fn,
        run_seed: int
    ): 
        kwargs = dict(params)
        kwargs.setdefault("random_state", run_seed)
        kwargs.setdefault("compute_strategy", self.compute_strategy)
        if collate_fn is not None: 
            kwargs.setdefault("collate_fn", collate_fn)

        model = self.factory(**kwargs)
        if callable(model) and not hasattr(model, "fit"): 
            model = model() 

        if not hasattr(model, "fit"): 
            raise AttributeError("SSFE model must implement fit()")
        if not hasattr(model, "validate"): 
            raise AttributeError("SSFE model must implement validate()")
        return model 

    @staticmethod
    def score(model) -> float: 
        if not hasattr(model, "best_val_score_"): 
            raise AttributeError("model must retain best validation score")
        return float(model.best_val_score_)


class _MappedZipLoader: 
    '''
    Credit Codex 5.3 High: Converts expert loaders into a zipped loader accessable as a mapping 
    '''

    def __init__(
        self,
        loaders: Mapping[str, DataLoader]
    ): 
        if not loaders: 
            raise ValueError("at least one expert loader is required.")
        self.expert_ids_ = list(loaders.keys())
        self.loaders_    = {k: loaders[k] for k in self.expert_ids_}

    def __iter__(self): 
        iters = [iter(self.loaders_[k]) for k in self.expert_ids_]
        for batches in zip(*iters): 
            yield {k: b for k, b in zip(self.expert_ids_, batches)}
    
    def __len__(self): 
        return min(len(self.loaders_[k]) for k in self.expert_ids_)


class MultiviewSSFEEEvaluator(OptunaEvaluator): 
    '''
    Evaluator for multiview SSFE optimization. 
    '''

    def __init__(
        self,
        filepath: str,
        loader_func: Callable,
        model_factory: Callable,
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        *,
        random_state: int = 0,
        eval_fraction: float = 0.2,
        n_runs: int = 1,
        batch_size: int = 32,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
    ):
        self.filepath           = filepath
        self.loader             = loader_func
        self.factory            = model_factory
        self.param_space_fn     = param_space
        self.random_state       = random_state
        self.eval_fraction      = eval_fraction
        self.n_runs             = max(1, int(n_runs))
        self.batch_size         = batch_size
        self.compute_strategy   = compute_strategy
        self.cache_             = None

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def load_data_once(self):
        if self.cache_ is None:
            self.cache_ = self.loader(self.filepath)
        return self.cache_

    def evaluate(self, params: Dict[str, Any], trial: optuna.Trial | None = None) -> float:
        data = self.load_data_once()
        n = int(np.asarray(data["sample_ids"]).shape[0])
        if n < 4:
            raise ValueError("need at least 4 aligned samples for train/val split")

        scores: list[float] = []
        for run_idx in range(self.n_runs):
            seed = self.random_state + run_idx
            train_idx, val_idx = self.holdout_split(n=n, seed=seed)
            model = self.build_model(params=params, run_seed=seed)

            try:
                train_expert, val_expert = self.build_expert_loaders(
                    data=data, train_idx=train_idx, val_idx=val_idx, params=params
                )
                cache_expert = self.build_cache_loaders(data=data, params=params)
                self.initialize_experts(model, cache_expert)
                model.fit(_MappedZipLoader(train_expert), _MappedZipLoader(val_expert))
                score = float(getattr(model, "best_val_score_", np.inf))
                if not np.isfinite(score):
                    return float("inf")
                scores.append(score)
                if trial is not None:
                    trial.report(float(np.mean(scores)), step=run_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            finally:
                del model
                _cleanup_cuda()

        return float(np.mean(scores))

    def build_cache_loaders(
        self,
        *,
        data: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Dict[str, DataLoader]:
        admin = data["admin"]
        viirs = data["viirs"]

        x = np.asarray(admin["features"], dtype=np.float32)
        w = np.asarray(admin["wide_cond"], dtype=np.float32)
        node_ids = np.asarray(admin.get("node_ids", np.arange(x.shape[0])), dtype=np.int64)
        if x.shape[0] != w.shape[0]:
            raise ValueError("admin features/wide_cond row mismatch")

        viirs_ds = viirs["dataset"]
        if len(viirs_ds) != x.shape[0]:
            raise ValueError("aligned admin/viirs row mismatch")

        admin_ds = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(node_ids),
            torch.from_numpy(w),
        )
        viirs_collate = viirs.get("collate_fn")

        bs = int(params.get("batch_size", self.batch_size))
        pin = str(self.compute_strategy.device).startswith("cuda")
        common = dict(
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
            drop_last=False,
        )

        return {
            "admin": DataLoader(admin_ds, **common),
            "viirs": DataLoader(viirs_ds, collate_fn=viirs_collate, **common),
        }

    def build_model(self, *, params: Dict[str, Any], run_seed: int):
        kwargs = dict(params)
        kwargs.setdefault("random_state", run_seed)
        try:
            model = self.factory(compute_strategy=self.compute_strategy, **kwargs)
        except TypeError:
            model = self.factory(**kwargs)
        if callable(model) and not hasattr(model, "fit"):
            model = model()
        if not hasattr(model, "fit"):
            raise AttributeError("multiview model must implement fit(train_loader, val_loader)")
        return model

    def holdout_split(self, *, n: int, seed: int) -> tuple[NDArray, NDArray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(n, dtype=np.int64)
        rng.shuffle(idx)
        n_val = int(round(n * self.eval_fraction))
        n_val = min(max(1, n_val), n - 1)
        return idx[n_val:], idx[:n_val]

    def build_expert_loaders(
        self,
        *,
        data: Dict[str, Any],
        train_idx: NDArray,
        val_idx: NDArray,
        params: Dict[str, Any],
    ) -> tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
        admin = data["admin"]
        viirs = data["viirs"]

        x = np.asarray(admin["features"], dtype=np.float32)
        w = np.asarray(admin["wide_cond"], dtype=np.float32)
        node_ids = np.asarray(admin.get("node_ids", np.arange(x.shape[0])), dtype=np.int64)
        if x.shape[0] != w.shape[0]:
            raise ValueError("admin features/wide_cond row mismatch")

        viirs_ds = viirs["dataset"]
        if len(viirs_ds) != x.shape[0]:
            raise ValueError("aligned admin/viirs row mismatch")

        admin_ds = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(node_ids),
            torch.from_numpy(w),
        )
        viirs_collate = viirs.get("collate_fn")

        bs = int(params.get("batch_size", self.batch_size))
        pin = str(self.compute_strategy.device).startswith("cuda")
        common = dict(
            batch_size=bs, 
            shuffle=False, 
            num_workers=0, 
            pin_memory=pin, 
            drop_last=False
        )

        train_expert = {
            "admin": DataLoader(Subset(admin_ds, train_idx.tolist()), **common),
            "viirs": DataLoader(
                Subset(viirs_ds, train_idx.tolist()),
                collate_fn=viirs_collate,
                **common,
            ),
        }
        val_expert = {
            "admin": DataLoader(Subset(admin_ds, val_idx.tolist()), **common),
            "viirs": DataLoader(
                Subset(viirs_ds, val_idx.tolist()),
                collate_fn=viirs_collate,
                **common,
            ),
        }
        return train_expert, val_expert

    @staticmethod
    def initialize_experts(model, train_expert: Dict[str, DataLoader]):
        if not hasattr(model, "experts"):
            raise AttributeError("multiview model must expose .experts mapping")
        for eid, expert in model.experts.items():
            loader = train_expert.get(eid)
            if loader is None:
                raise KeyError(f"missing train loader for expert='{eid}'")
            expert.init_fit(loader, state_loader=loader)

# ---------------------------------------------------------
# Evaluator for Standard Models (xgb, rf, logistic, svm, etc) 
# ---------------------------------------------------------

def _nested_standard_worker(
    *,
    name: str,
    fold_idx: int,
    filepath: str,
    loader_func: Callable,
    model_factory: Callable,
    param_space: Callable,
    task: TaskSpec,
    train_idx: NDArray,
    test_idx: NDArray,
    inner_config: CVConfig,
    outer_config: CVConfig,
    base_cfg: EngineConfig,
    nested_cfg: NestedCVConfig,
    feature_transform_factory=None,
    scale_X: bool = True,
    scale_y: bool | None = None,
    metric: str | None = None,
    param_transform=None
) -> float:
    evaluator = StandardEvaluator(
        filepath=filepath,
        loader_func=loader_func,
        base_factory_func=model_factory,
        param_space=param_space,
        task=task,
        config=inner_config,
        feature_transform_factory=feature_transform_factory,
        outer_config=outer_config,
        scale_X=scale_X,
        scale_y=scale_y,
        metric=metric,
        param_transform=param_transform
    )

    inner_eval = evaluator.inner_evaluator(train_idx)
    inner_cfg  = EngineConfig(
        n_trials=nested_cfg.inner_n_trials,
        direction=base_cfg.direction,
        sampler_type=nested_cfg.inner_sampler_type or base_cfg.sampler_type,
        random_state=base_cfg.random_state,
        early_stopping_rounds=nested_cfg.inner_early_stopping_rounds,
        early_stopping_delta=nested_cfg.inner_early_stopping_delta,
        mp_enabled=False
    )

    best_params, _, _ = run_optimization(f"{name}_fold{fold_idx}", inner_eval, inner_cfg)
    score = evaluator.outer_score(best_params, train_idx, test_idx)
    return score, best_params 


class StandardEvaluator(OptunaEvaluator): 

    def __init__(
        self, 
        filepath: str, 
        loader_func: Callable, 
        base_factory_func: Callable, 
        param_space: Callable[[optuna.Trial], Dict[str, Any]], 
        task: TaskSpec, 
        config: CVConfig,
        feature_transform_factory=None,
        *,
        outer_config: CVConfig | None = None, 
        scale_X: bool = True, 
        scale_y: bool | None = None, 
        metric: str | None = None, 
        param_transform: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None 
    ): 
        self.filepath           = filepath 
        self.loader             = loader_func 
        self.factory            = base_factory_func 
        self.param_space_fn     = param_space 
        self.task               = task 
        self.config             = config
        self.config.verbose     = False
        self.feature_transforms = feature_transform_factory
        self.outer_config       = outer_config
        self.scale_X            = scale_X 
        self.scale_y            = scale_y 
        self.metric             = metric 
        self.param_transform    = param_transform
        self._data_cache        = None 

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: 
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:

        model = self.factory(**params)

        cv = CrossValidator(
            filepath=self.filepath,
            loader=self.loader, 
            task=self.task, 
            scale_y=False, 
            feature_transform_factory=self.feature_transforms
        )

        results = cv.run(
            models={"opt_candidate": model},
            config=self.config,
        )

        if "error" in results.columns and results["error"].notna().any(): 
            err_msgs = results.loc[results["error"].notna(), "error"].unique().tolist() 
            raise RuntimeError(f"cross-val fold errors: {err_msgs}")

        summary = cv.summarize(results)
        for m in self.task.metrics: 
            key = f"{m}_mean"
            if key in summary.columns and not np.isnan(summary.iloc[0][key]): 
                return summary.iloc[0][key]

        for metric in ["f1_macro_mean", "accuracy_mean", "r2_mean"]: 
            if metric in summary.columns: 
                return summary.iloc[0][metric]
        
        raise ValueError("no suitable metric found in summary results ")

    def load_data(self):
        if self._data_cache is None: 
            data   = self.loader(self.filepath)
            X      = data["features"]
            y      = data["labels"] 
            coords = data.get("coords")
            coords = None if coords is None else np.asarray(coords, dtype=np.float32)
            if self.task.task_type == "classification" and y.ndim == 2 and y.shape[1]: 
                y = y.ravel() 
            self._data_cache = (data, X, y, coords)
        return self._data_cache

    def outer_splits(self): 
        if self.outer_config is None: 
            raise ValueError("outer_config must be specified for nested cv")

        _, X, y, _ = self.load_data() 
        y_for_split   = y if y.ndim == 1 else y[:, 0]
        splitter      = self.outer_config.get_splitter(self.task)
        return splitter.split(X, y_for_split)

    def inner_evaluator(self, train_idx: NDArray):
        data, X, y, coords = self.load_data()
        coords_train       = None if coords is None else coords[train_idx]

        def inner_loader(_): 
            return {
                "features": X[train_idx],
                "labels": y[train_idx],
                "coords": coords_train,
                "feature_names": data.get("feature_names"),
                "sample_ids": data.get("sample_ids")
            }

        return StandardEvaluator(
            filepath="virtual",
            loader_func=inner_loader,
            base_factory_func=self.factory,
            param_space=self.param_space_fn,
            task=self.task,
            config=self.config,
            feature_transform_factory=self.feature_transforms,
            outer_config=self.outer_config,
            scale_X=self.scale_X,
            scale_y=self.scale_y,
            metric=self.metric,
            param_transform=self.param_transform
        )

    def outer_score(self, params, train_idx, test_idx): 
        _, X, y, coords = self.load_data()

        params = dict(params)
        if self.param_transform is not None:
            params = self.param_transform(params)

        model = self.factory(**params)
        if callable(model) and not hasattr(model, "fit"):
            model = model()

        scale_y = self.scale_y
        if scale_y is None:
            scale_y = False if self.task.task_type == "classification" else True

        model = ScaledEstimator(model, scale_X=self.scale_X, scale_y=scale_y)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        coords_train = None if coords is None else coords[train_idx]
        coords_test = None if coords is None else coords[test_idx]

        model.fit(X_train, y_train, coords_train)

        y_pred = model.predict(X_test, coords_test)
        y_prob = predict_proba_if_any(model, X_test, coords_test)

        labels = np.sort(np.unique(np.concatenate([y_train, y_test])))
        y_idx_train = np.searchsorted(labels, y_train)
        counts = np.bincount(y_idx_train, minlength=labels.size)
        class_weights = counts.max() / np.clip(counts, 1, None)

        metrics = self.task.compute_metrics(
            y_test, y_pred, y_prob,
            class_labels=labels,
            class_weights=class_weights
        )
        return float(select_metric_value(metrics, self.task, self.metric))

    def build_nested_worker_specs(self, name, nested_cfg, base_cfg):
        specs = []
        splits = list(self.outer_splits())
        device_ids = nested_cfg.devices or base_cfg.devices or [None]

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            device_id = device_ids[fold_idx % len(device_ids)]
            specs.append(
                WorkerSpec(
                    fn=_nested_standard_worker,
                    kwargs={
                        "name": name,
                        "fold_idx": fold_idx,
                        "filepath": self.filepath,
                        "loader_func": self.loader,
                        "model_factory": self.factory,
                        "param_space": self.param_space_fn,
                        "task": self.task,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "inner_config": self.config,
                        "outer_config": self.outer_config,
                        "base_cfg": base_cfg,
                        "nested_cfg": nested_cfg,
                        "feature_transform_factory": self.feature_transforms,
                        "scale_X": self.scale_X,
                        "scale_y": self.scale_y,
                        "metric": self.metric,
                        "param_transform": self.param_transform,
                    },
                    device_id=device_id
                )
            )
        return specs

# ---------------------------------------------------------
# Spatial Helpers 
# ---------------------------------------------------------

def _cleanup_cuda(): 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect() 

def _resolve_compute_strategy(base: ComputeStrategy, device_id: int | None): 
    if device_id is None or base.device != "cuda": 
        return base 
    return replace(base, gpu_id=int(device_id), device=f"cuda:{int(device_id)}")

def _make_spatial_loader(
    dataset, 
    collate_fn, 
    batch_size: int, 
    compute_strategy: ComputeStrategy,
    shuffle: bool 
): 
    pin = compute_strategy.device == "cuda" 
    if compute_strategy.n_jobs == -1: 
        num_workers = 8 
    else: 
        num_workers = compute_strategy.n_jobs

    base       = getattr(dataset, "dataset", dataset)
    is_packed  = hasattr(base, "is_packed") and base.is_packed  

    worker_override = getattr(base, "prefetch_workers", None)
    if worker_override is not None: 
        num_workers = int(worker_override)

    batch_size = 1 if is_packed else batch_size 

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
        collate_fn=collate_fn,
        persistent_workers=False, 
        prefetch_factor=prefetch_factor  
    )

def _iter_spatial_splits(
    data,
    dataset,
    labels,
    task,
    config: CVConfig
): 
    y      = labels 
    y_bucket = np.floor(np.clip(y, 0.0, None)).astype(np.int64)
    idx    = np.arange(len(y))
    splits = config.get_splitter(task).split(idx, y_bucket)
    return splits, None


def _spatial_eval_fold(
    *,
    dataset,
    labels,
    collate_fn,
    train_idx,
    test_idx,
    params: Dict[str, Any],
    model_factory: Callable,
    batch_size: int, 
    compute_strategy: ComputeStrategy,
):
    train_ds    = Subset(dataset, train_idx)
    test_ds     = Subset(dataset, test_idx)

    model = model_factory(collate_fn=collate_fn, **params)

    val_loader = _make_spatial_loader(
        test_ds, collate_fn, batch_size, compute_strategy, shuffle=False)

    model.fit(train_ds, labels[train_idx])

    if hasattr(model, "best_val_score_"): 
        return float(model.best_val_score_)

    probs  = model.predict_proba(val_loader)
    y_true = []
    for batch in val_loader: 
        yb = batch[1]
        y_true.append(yb.cpu().numpy() if hasattr(yb, "cpu") else np.asarray(yb))

    class_weights = None 
    if hasattr(model, "class_counts_") and model.class_counts_ is not None: 
        counts = np.asarray(model.class_counts_) 
        if counts.size == len(model.classes_): 
            class_weights = counts.max() / np.clip(counts, 1, None)

    rps = ranked_probability_score(
        y_true, 
        probs, 
        class_labels=model.classes_, 
        normalize=True, 
        class_weights=class_weights
    )

    del model 
    _cleanup_cuda()

    return float(rps)

# ---------------------------------------------------------
# Spatial SFE Evaluator 
# ---------------------------------------------------------

def _split_device_groups(devices: list[int], n_groups: int) -> list[list[int]]: 
    if not devices: 
        return [[] for _ in range(n_groups)]

    chunk  = max(1, len(devices) // n_groups) 
    groups = [] 
    for i in range(n_groups):
        start = i * chunk 
        end   = start + chunk 
        group = devices[start:end]
        if not group: 
            group = [devices[-1]]
        groups.append(group)

    leftover = devices[chunk * n_groups:]
    if leftover: 
        groups[-1].extend(leftover)
    return groups 

def _spatial_eval_worker(
    *,
    filepath: str, 
    loader_func: Callable, 
    model_factory: Callable, 
    params: Dict[str, Any],
    train_idx: NDArray, 
    test_idx: NDArray, 
    batch_size: int, 
    compute_strategy: ComputeStrategy,
    device_id: int | None = None 
) -> float: 

    data       = loader_func(filepath)
    dataset    = data["dataset"]
    labels     = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
    collate_fn = data.get("collate_fn")

    # Clean up inputs to fold evaluation 
    compute_strategy = _resolve_compute_strategy(compute_strategy, device_id)

    return _spatial_eval_fold(
        dataset=dataset,
        labels=labels,
        collate_fn=collate_fn,
        train_idx=train_idx,
        test_idx=test_idx,
        params=params,
        model_factory=model_factory,
        batch_size=batch_size,
        compute_strategy=compute_strategy,
    )

class SpatialEvaluator(OptunaEvaluator): 

    def __init__(
        self,
        filepath,
        loader_func,
        model_factory,
        param_space,
        task,
        config,
        batch_size: int = 1, 
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False), 
        drop_last: bool = False 
    ): 
        self.filepath         = filepath 
        self.loader           = loader_func 
        self.factory          = model_factory
        self.param_space_fn   = param_space
        self.task             = task 
        self.config           = config 
        self.config.verbose   = False
        
        self.data             = self.loader(filepath)
        self.dataset          = self.data["dataset"]
        self.labels           = np.asarray(self.data["labels"], dtype=np.float32).reshape(-1)
        self.coords           = self.data.get("coords")
        self.collate_fn       = self.data.get("collate_fn") 
        self.batch_size       = batch_size 
        self.compute_strategy = compute_strategy
        self.drop_last        = drop_last 

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any], trial: optuna.Trial | None = None) -> float:

        visible_devices = self.compute_strategy.visible_devices() or [0]
        DevicePool.get_instance(visible_devices)

        scores = []

        splits, _ = _iter_spatial_splits(
            self.data, self.dataset, self.labels, self.task, self.config)

        try: 
            with DevicePool.get_instance().claim() as device_id: 
                thread_strategy = _resolve_compute_strategy(self.compute_strategy, device_id) 

                for fold_idx, (train_idx, test_idx) in enumerate(splits): 
                    loss = _spatial_eval_fold(
                        dataset=self.dataset,
                        labels=self.labels,
                        collate_fn=self.collate_fn,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        params=params,
                        model_factory=self.factory,
                        batch_size=self.batch_size,
                        compute_strategy=thread_strategy,
                    )
                    scores.append(loss)

                    if trial is not None: 
                        trial.report(float(np.mean(scores)), step=fold_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                pass 
        finally: 
            if hasattr(self.dataset, "close"): 
                self.dataset.close() 

        return float(np.mean(scores))

    def build_worker_specs(self, params: Dict[str, Any], devices=None): 
        specs = []

        splits_iter, _ = _iter_spatial_splits(
            self.data, self.dataset, self.labels, self.task, self.config)
        splits = list(splits_iter)

        device_ids = devices or [None]

        for fold_idx, (train_idx, test_idx) in enumerate(splits): 
            device_id = device_ids[fold_idx % len(device_ids)]
            specs.append(
                WorkerSpec(
                    fn=_spatial_eval_worker,
                    kwargs={
                        "filepath": self.filepath,
                        "loader_func": self.loader,
                        "model_factory": self.factory,
                        "params": params,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "batch_size": self.batch_size,
                        "compute_strategy": self.compute_strategy,
                        "device_id": device_id
                    },
                    device_id=device_id
                )
            )

        return specs 

    def reduce_worker_results(self, results): 
        return float(np.mean(results))

# ---------------------------------------------------------
# Correct-and-smooth evaluator
# ---------------------------------------------------------

class CorrectAndSmoothEvaluator(OptunaEvaluator):
    def __init__(
        self,
        P: NDArray,
        W_by_name: Any,
        y_train: NDArray,
        train_mask: NDArray,
        test_mask: NDArray,
        class_labels: NDArray,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.P = P
        self.W_by_name = W_by_name
        self.y_train = y_train
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.y_true = y_train[test_mask]
        self.class_labels = class_labels
        self.compute_strategy = compute_strategy

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "adjacency": trial.suggest_categorical(
                "adjacency", list(self.W_by_name.keys())
            ),
            "correction_layers": trial.suggest_int("correction_layers", 1, 25),
            "correction_alpha": trial.suggest_float("correction_alpha", 0.0, 1.0),
            "smoothing_layers": trial.suggest_int("smoothing_layers", 1, 25),
            "smoothing_alpha": trial.suggest_float("smoothing_alpha", 0.0, 1.0),
            "autoscale": trial.suggest_categorical("autoscale", [True, False]),
        }

    def evaluate(self, params: Dict[str, Any]) -> float:
        adj_name = params.pop("adjacency")
        W = self.W_by_name[adj_name]

        cs = CorrectAndSmooth(
            class_labels=self.class_labels,
            correction=(params.pop("correction_layers"), params.pop("correction_alpha")),
            smoothing=(params.pop("smoothing_layers"), params.pop("smoothing_alpha")),
            autoscale=params.pop("autoscale"),
            compute_strategy=self.compute_strategy
        )

        P_cs = cs(self.P, self.y_train, self.train_mask, W)
        pred_labels = cs.predict(P_cs)
        return accuracy_score(self.y_true, pred_labels[self.test_mask])

# ---------------------------------------------------------
# Metric learning evaluator
# ---------------------------------------------------------

def _apply_train_test_transforms(transforms, X, train_mask):
    X_train = X[train_mask]
    X_test = X[~train_mask]
    for t in transforms:
        if hasattr(t, "fit_transform"):
            X_train = t.fit_transform(X_train)
        else:
            t.fit(X_train)
            X_train = t.transform(X_train)
        X_test = t.transform(X_test)
    X_full = np.zeros((X.shape[0], X_train.shape[1]), dtype=X_train.dtype)
    X_full[train_mask] = X_train
    X_full[~train_mask] = X_test
    return X_full


class MetricCASEvaluator(OptunaEvaluator):
    def __init__(
        self,
        filepath: str,
        base_factory_func: Callable,
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        *,
        dataset_loaders: dict,
        proba_path: str,
        proba_model_name: str | None = None,
        random_state: int = 0,
        train_size: float = 0.3,
        passthrough_adj_fn=None,
        feature_transform_factory=None,
        adjacency_factory=None
    ):
        self.filepath = filepath
        self.factory = base_factory_func
        self.param_space_fn = param_space
        self.dataset_loaders = dataset_loaders
        self.proba_path = proba_path
        self.proba_model_name = proba_model_name
        self.random_state = random_state
        self.train_size = train_size
        self.passthrough_adj_fn = passthrough_adj_fn
        self.feature_transform_factory = feature_transform_factory

        if adjacency_factory is None:
            adjacency_factory = make_queen_adjacency_factory()
        self.adjacency_factory = adjacency_factory

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = self.param_space_fn(trial)
        params["dataset"] = trial.suggest_categorical(
            "dataset",
            list(self.dataset_loaders.keys())
        )
        return params

    def evaluate(self, params: Dict[str, Any]) -> float:
        dataset_key = params.pop("dataset", None)
        loader = self.dataset_loaders.get(dataset_key)
        if loader is None:
            raise ValueError("no loader available for evaluator")

        data = loader(self.filepath)
        X = data["features"]
        y = np.asarray(data["labels"]).reshape(-1)
        fips = np.asarray(data["sample_ids"], dtype="U5")

        oof = load_oof_predictions(self.proba_path)
        oof_fips = np.asarray(oof["fips_codes"]).astype("U5")
        common = [f for f in fips if f in set(oof_fips)]
        if not common:
            raise ValueError("no common fips between dataset and oof probs")

        if len(common) != len(fips):
            idx = align_on_fips(common, fips)
            X = X[idx]
            y = y[idx]
            fips = fips[idx]

        adj = None
        if self.passthrough_adj_fn is not None:
            adj = self.passthrough_adj_fn(list(fips))
        elif self.adjacency_factory is not None:
            if not callable(self.adjacency_factory):
                raise TypeError("adjacency_factory must be callable")
            adj = self.adjacency_factory(list(fips))

        P, class_labels = load_probs_for_fips(
            fips,
            self.proba_path
        )

        train_mask = make_train_mask(
            y,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=True
        )
        test_mask = ~train_mask

        if self.feature_transform_factory is not None:
            feature_names = data.get("feature_names")
            factory = self.feature_transform_factory(feature_names)
            transforms = factory() if callable(factory) else factory
            if transforms:
                X = _apply_train_test_transforms(transforms, X, train_mask)

        if adj is None:
            raise ValueError("EdgeLearner requires adjacency_factory to supply priori adj")

        model = self.factory(**params)
        model.fit(X, y, adj, train_mask=train_mask)
        adj = model(X, adj)

        cs = CorrectAndSmoothEvaluator(
            P=P,
            W_by_name={"metric": adj},
            y_train=y,
            train_mask=train_mask,
            test_mask=test_mask,
            class_labels=class_labels
        )

        cfg = EngineConfig(
            n_trials=150,
            sampler_type="multivariate-tpe",
            early_stopping_rounds=40,
            early_stopping_delta=1e-4,
            random_state=self.random_state
        )

        _, best_value, _ = run_optimization("CorrectAndSmooth_inner", cs, cfg)
        return best_value

# ---------------------------------------------------------
# Evaluator for Tabular based MLP models using hybrid ordinal loss 
# or derivatives of hybrid ordinal loss (such as mixed loss)
# ---------------------------------------------------------

class TabularEvaluator(OptunaEvaluator):

    '''
    Primary evaluator for MLP models specific for tabular data 
    '''


    def __init__(
        self,
        X,
        y,
        param_space,
        model_factory: Callable, 
        random_state: int = 0,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.param_space_fn   = param_space
        self.model_factory    = model_factory 
        self.random_state     = random_state
        self.compute_strategy = compute_strategy

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=params.get("eval_fraction", 0.15),
            random_state=self.random_state
        )
        
        y_bucket = np.floor(np.clip(self.y, 0.0, None)).astype(np.int64)
        train_idx, val_idx = next(splitter.split(self.X, y_bucket))

        scaler = StandardScaler() 
        X_tr   = scaler.fit_transform(self.X[train_idx])
        X_val  = scaler.transform(self.X[val_idx])

        model  = self.model_factory(
            in_dim=self.X.shape[1],
            **params,
            random_state=self.random_state,
            compute_strategy=self.compute_strategy
        )

        model.fit(X_tr, self.y[train_idx])
        return model.loss(X_val, self.y[val_idx])
