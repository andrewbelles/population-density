#!/usr/bin/env python3 
# 
# evaluators.py  Andrew Belles  Jan 24th, 2026 
# 
# Implementation of all optimization evaluators 
# 
# 

import gc, optuna, torch 

import numpy as np 

from abc                       import ABC, abstractmethod 

from dataclasses               import replace  

from typing                    import Any, Callable, Dict 

from numpy.typing              import NDArray

from sklearn.metrics           import accuracy_score, cohen_kappa_score  

from sklearn.model_selection   import StratifiedShuffleSplit, StratifiedGroupKFold 

from torch.utils.data          import DataLoader, Subset 

from optimization.engine       import (
    run_optimization, 
    NestedCVConfig,
    EngineConfig, 
    WorkerSpec
)

from analysis.cross_validation import (
    CLASSIFICATION,
    CrossValidator,
    CVConfig,
    TaskSpec,
    ScaledEstimator
)

from models.estimators         import EmbeddingProjector 

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

        metrics = self.task.compute_metrics(y_test, y_pred, y_prob)
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
        persistent_workers=(num_workers > 0), 
        prefetch_factor=prefetch_factor  
    )

def _iter_spatial_splits(
    data,
    dataset,
    labels,
    task,
    config: CVConfig
): 
    is_packed = getattr(dataset, "is_packed", False)

    if is_packed: 
        y        = data["sample_labels"]
        groups   = data["sample_groups"]
        splitter = StratifiedGroupKFold(
            n_splits=config.n_splits,
            shuffle=True, 
            random_state=config.random_state
        )
        splits   = splitter.split(np.zeros_like(y), y, groups)
        return splits, groups, True 

    y      = labels 
    idx    = np.arange(len(y))
    splits = config.get_splitter(task).split(idx, y)
    return splits, None, False


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
    groups=None,
    is_packed: bool,
    trial=None, 
    fold_idx=0 
):
    if is_packed:
        if groups is None: 
            raise ValueError("if is_packed then groups is not None at runtime")
        train_packs = np.unique(groups[train_idx])
        test_packs  = np.unique(groups[test_idx])
        train_ds    = Subset(dataset, train_packs)
        test_ds     = Subset(dataset, test_packs)
    else: 
        train_ds    = Subset(dataset, train_idx)
        test_ds     = Subset(dataset, test_idx)

    def pruning_callback(epoch, metrics): 
        if trial is None: 
            return 

        current_score = metrics.get("val_loss")
        if current_score is not None: 
            trial.report(-1.0 * current_score, step=epoch)
            if trial.should_prune(): 
                raise optuna.TrialPruned()

    model = model_factory(collate_fn=collate_fn, **params)

    val_loader   = _make_spatial_loader(
        test_ds, collate_fn, batch_size, compute_strategy, shuffle=False)

    if is_packed: 
        train_loader = _make_spatial_loader(
            train_ds, collate_fn, batch_size, compute_strategy, shuffle=True)
        model.fit(train_loader, y=None, val_loader=val_loader, callbacks=[pruning_callback])
        val_loss     = model.loss(val_loader)

    else: 
        model.fit(train_ds, labels[train_idx], callbacks=[pruning_callback])
        val_loss     = model.loss(test_ds)

    y_true_list = []
    y_pred_list = [] 

    model.eval() 
    with torch.no_grad(): 
        for batch in val_loader: 
            if isinstance(batch, (tuple, list)): 
                xb, mb, rois, yb = batch[0], batch[1], batch[2], batch[3] 
            else: 
                yb = batch["labels"]

            preds = model.predict(batch) 

            if hasattr(yb, "cpu"): 
                y_true = yb.cpu().numpy() 
            else: 
                y_true = np.asarray(yb) 

            if hasattr(preds, "cpu"): 
                y_pred = preds.cpu().numpy() 
            else: 
                y_pred = np.asarray(preds) 

            y_true_list.append(y_true.flatten()) 
            y_pred_list.append(y_pred.flatten())

    y_true = np.concatenate(y_true_list) 
    y_pred = np.concatenate(y_pred_list) 

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic") 

    del model 
    _cleanup_cuda()

    return float(qwk)

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
    labels     = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    collate_fn = data.get("collate_fn")

    # Clean up inputs to fold evaluation 
    compute_strategy = _resolve_compute_strategy(compute_strategy, device_id)
    is_packed        = getattr(dataset, "is_packed", False)
    groups           = data["sample_groups"] if is_packed else None 

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
        groups=groups,
        is_packed=is_packed
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
        self.labels           = np.asarray(self.data["labels"], dtype=np.int64).reshape(-1)
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

        splits, groups, is_packed = _iter_spatial_splits(
            self.data, self.dataset, self.labels, self.task, self.config)

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
                    groups=groups,
                    is_packed=is_packed,
                    trial=trial,
                    fold_idx=fold_idx
                )
                scores.append(loss)

                if trial is not None: 
                    trial.report(float(np.mean(scores)), step=fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned() 

        return float(np.mean(scores))

    def build_worker_specs(self, params: Dict[str, Any], devices=None): 
        specs = []

        splits_iter, _, _ = _iter_spatial_splits(
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
# XGB ordinal evaluator
# ---------------------------------------------------------

class XGBOrdinalEvaluator(OptunaEvaluator):
    def __init__(
        self,
        filepath,
        loader_func,
        model_factory,
        param_space,
        config,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.filepath = filepath
        self.loader = loader_func
        self.factory = model_factory
        self.param_space_fn = param_space
        self.config = config
        self.strategy = compute_strategy

        data = self.loader(filepath)
        self.X = np.asarray(data["features"], dtype=np.float32)
        self.y = np.asarray(data["labels"], dtype=np.int64).reshape(-1)

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:
        splitter = self.config.get_splitter(CLASSIFICATION)
        scores = []

        for train_idx, test_idx in splitter.split(self.X, self.y):
            model = self.factory(compute_strategy=self.strategy, **params)
            if callable(model) and not hasattr(model, "fit"):
                model = model()
            if not hasattr(model, "fit"):
                raise TypeError("model_factory must return a model with .fit()")

            model.fit(self.X[train_idx], self.y[train_idx])
            scores.append(float(model.loss(self.X[test_idx], self.y[test_idx])))

        return float(np.mean(scores))

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
# Projector evaluator
# ---------------------------------------------------------

class ProjectorEvaluator(OptunaEvaluator):
    def __init__(
        self,
        X,
        y,
        param_space,
        random_state: int = 0,
        compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.param_space_fn = param_space
        self.random_state = random_state
        self.compute_strategy = compute_strategy

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=params.get("eval_fraction", 0.15),
            random_state=self.random_state
        )
        train_idx, val_idx = next(splitter.split(self.X, self.y))

        proj = EmbeddingProjector(
            in_dim=self.X.shape[1],
            **params,
            random_state=self.random_state,
            device=self.compute_strategy.device
        )
        proj.fit(self.X[train_idx], self.y[train_idx])
        preds = proj.predict(self.X[val_idx])
        qwk   = cohen_kappa_score(self.y[val_idx], preds, weights="quadratic")
        return float(qwk)
