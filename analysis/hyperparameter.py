#!/usr/bin/env python3 
# 
# hyperparameter.py  Andrew Belles  Dec 26th, 2025 
# 
# A generalized interface for Bayesian Optimization using Optuna. 
# Aims to offer a general contract for different Models to leverage
# 

from numpy.typing import NDArray

import optuna, yaml 
from optuna.samplers import CmaEsSampler, TPESampler

from pathlib import Path 
import numpy as np 
from abc import ABC, abstractmethod 
from typing import Any, Callable, Dict, Literal 

from preprocessing.loaders import (
    _align_on_fips,
    load_oof_predictions
)

from analysis.cross_validation import (
    CrossValidator, 
    CVConfig,
    ScaledEstimator, 
    TaskSpec,
)

from models.post_processing import (
    CorrectAndSmooth,
    make_train_mask, 
    normalized_proba 
)

from analysis.graph_metrics import (
    MetricAnalyzer,
)

from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Generalize Interface/Contract for Optimizer 
# ---------------------------------------------------------

class OptunaEvaluator(ABC): 
    
    @abstractmethod 
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: 
        pass 

    @abstractmethod 
    def evaluate(self, params: Dict[str, Any]) -> float: 
        pass 


def run_optimization(
    name: str, 
    evaluator: OptunaEvaluator, 
    n_trials: int = 50, 
    direction: Literal["maximize", "minimize"] = "maximize", 
    config_path: str | None = None, 
    config_key: str | None = None, 
    early_stopping_rounds: int | None = None, 
    early_stopping_delta: float = 0.0, 
    random_state: int = 0,
    sampler_type: str = "tpe"
): 

    if sampler_type == "cmaes": 
        sampler = CmaEsSampler(seed=random_state)
    elif sampler_type == "multivariate-tpe": 
        sampler = TPESampler(multivariate=True, seed=random_state)
    else: 
        sampler = TPESampler(multivariate=False, seed=random_state)

    callbacks = []
    if early_stopping_rounds is not None: 
        if early_stopping_rounds < 1: 
            raise ValueError("early_stopping_rounds must be >= 1")
        best_value = None 
        best_trial = None 

        def stagnation_callback(study, trial): 
            nonlocal best_value, best_trial 
            if trial.state != optuna.trial.TrialState.COMPLETE: 
                return 
            if best_value is None: 
                best_value = study.best_value 
                best_trial = trial.number 
                return 
        
            if study.direction == optuna.study.StudyDirection.MAXIMIZE: 
                improved = study.best_value > best_value + early_stopping_delta 
            else: 
                improved = study.best_value < best_value - early_stopping_delta

            if improved: 
                best_value = study.best_value 
                best_trial = trial.number 
                return 

            if (best_trial is not None and 
                trial.number - best_trial >= early_stopping_rounds):
                print(f"> Early stopping after {early_stopping_rounds} "
                       "stagnant trials.")
                study.stop() 

        callbacks.append(stagnation_callback)

    study = optuna.create_study(
        study_name=name, 
        direction=direction, 
        sampler=sampler
    )


    def objective(trial): 

        params = evaluator.suggest_params(trial)

        try: 
            score = evaluator.evaluate(params) 

        except Exception as e: 
            print(f"trial failure: {e}")
            return float("-inf") if direction == "maximize" else float("inf")

        return score 

    print(f"OPTIMIZATION: Starting {name} ({n_trials} trials)")
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)

    if config_path is not None: 
        key = config_key or name 
        _save_model_config(config_path, key, study.best_params)
        print(f"> Saved config: {config_path} ({key})")

    print("> Optimization Results:")
    print(f"Best Value: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    return study.best_params, study.best_value

# ---------------------------------------------------------
# Standard BaseModel Optimizer 
# ---------------------------------------------------------

class StandardEvaluator(OptunaEvaluator): 

    def __init__(
        self, 
        filepath: str, 
        loader_func: Callable, 
        base_factory_func: Callable, 
        param_space: Callable[[optuna.Trial], Dict[str, Any]], 
        task: TaskSpec, 
        config: CVConfig
    ): 
        self.filepath       = filepath 
        self.loader         = loader_func 
        self.factory        = base_factory_func 
        self.param_space_fn = param_space 
        self.task           = task 
        self.config         = config
        self.config.verbose = False

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: 
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:

        model = self.factory(**params)

        cv = CrossValidator(
            filepath=self.filepath,
            loader=self.loader, 
            task=self.task, 
            scale_y=False, 
        )

        results = cv.run(
            models={"opt_candidate": model},
            config=self.config,
        )

        summary = cv.summarize(results)

        for metric in ["f1_macro_mean", "accuracy_mean", "r2_mean"]: 
            if metric in summary.columns: 
                return summary.iloc[0][metric]
        
        raise ValueError("no suitable metric found in summary results ")

# ---------------------------------------------------------
# Correct and Smooth Model 
# ---------------------------------------------------------

class CorrectAndSmoothEvaluator(OptunaEvaluator): 

    def __init__(
        self,
        P: NDArray, 
        W_by_name: Any, 
        y_train: NDArray, 
        train_mask: NDArray, 
        test_mask: NDArray, 
        class_labels: NDArray
    ): 

        self.P            = P 
        self.W_by_name    = W_by_name 
        self.y_train      = y_train 
        self.train_mask   = train_mask
        self.test_mask    = test_mask 
        self.y_true       = y_train[test_mask]
        self.class_labels = class_labels

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "adjacency": trial.suggest_categorical(
                "adjacency", list(self.W_by_name.keys())),
            "correction_alpha": trial.suggest_float("correction_alpha", 0.0, 1.0), 
            "smoothing_alpha": trial.suggest_float("smoothing_alpha", 0.0, 1.0), 
            "correction_max_iter": trial.suggest_int("correction_max_iter", 1, 25), 
            "smoothing_max_iter": trial.suggest_int("smoothing_max_iter", 1, 25), 
            "autoscale": trial.suggest_categorical("autoscale", [True, False])
        }

    def evaluate(self, params: Dict[str, Any]) -> float:
        adj_name = params.pop("adjacency")
        W = self.W_by_name[adj_name]

        cs = CorrectAndSmooth(
            class_labels=self.class_labels, 
            **params 
        )

        P_cs        = cs.fit(self.P, self.y_train, W, self.train_mask)

        P_norm      = normalized_proba(P_cs, self.test_mask)
        pred_idx    = np.argmax(P_norm, axis=1)
        pred_labels = self.class_labels[pred_idx]

        return accuracy_score(self.y_true, pred_labels)

# ---------------------------------------------------------
# Metric Learning Evaluator  
# ---------------------------------------------------------

class MetricCASEvaluator(OptunaEvaluator): 
    '''
    Targets downstream optimization of accuracy 
    '''

    def __init__(
        self, 
        filepath: str, 
        # loader_func: Callable, 
        base_factory_func: Callable,
        param_space: Callable[[optuna.Trial], Dict[str, Any]],
        *,
        dataset_loaders: dict, 
        proba_path: str, 
        proba_model_name: str | None = None, 
        random_state: int = 0, 
        train_size: float = 0.3
    ): 
        self.filepath         = filepath 
        self.factory          = base_factory_func 
        self.param_space_fn   = param_space 
        self.dataset_loaders  = dataset_loaders
        self.proba_path       = proba_path 
        self.proba_model_name = proba_model_name 
        self.random_state     = random_state
        self.train_size       = train_size 

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = self.param_space_fn(trial)
        params["dataset"] = trial.suggest_categorical(
            "dataset",
            list(self.dataset_loaders.keys())
        )
        return params 

    def evaluate(self, params: Dict[str, Any]) -> float:
        dataset_key = params.pop("dataset", None)
        loader      = self.dataset_loaders.get(dataset_key)
        if loader is None: 
            raise ValueError("no loader available for evaluator")

        data = loader(self.filepath)
        X    = data["features"]
        y    = np.asarray(data["labels"]).reshape(-1)
        fips = np.asarray(data["sample_ids"], dtype="U5")

        max_components = max(1, min(128, X.shape[1]))
        max_neighbors  = max(1, min(100, X.shape[0] - 1))

        if "n_components_frac" in params: 
            frac = params.pop("n_components_frac")
            params["n_components"] = 1 + int(round(frac * (max_components - 1)))
        if "n_neighbors_frac" in params: 
            frac = params.pop("n_neighbors_frac")
            params["n_neighbors"] = 1 + int(round(frac * (max_neighbors - 1)))

        oof = load_oof_predictions(self.proba_path)
        oof_fips = np.asarray(oof["fips_codes"]).astype("U5")
        common = [f for f in fips if f in set(oof_fips)]
        if not common: 
            raise ValueError("no common fips between dataset and oof probs")

        if len(common) != len(fips): 
            idx = _align_on_fips(common, fips)
            X = X[idx] 
            y = y[idx]
            fips = fips[idx]

        P, class_labels = _load_probs_for_fips(
            self.proba_path,
            fips,
            model_name=self.proba_model_name
        )

        train_mask = make_train_mask(
            y,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=True
        )
        test_mask  = ~train_mask

        model = self.factory(**params)
        model.fit(X, y, train_mask=train_mask)

        adj = model.get_graph(X)


        # Downstream target is to maximize correct and smooth accuracy
        cs = CorrectAndSmoothEvaluator(
            P=P,
            W_by_name={"metric": adj}, 
            y_train=y,
            train_mask=train_mask,
            test_mask=test_mask,
            class_labels=class_labels
        )

        _, best_value = run_optimization(
            name="CorrectAndSmooth_inner",
            evaluator=cs, 
            n_trials=100,
            early_stopping_rounds=40,
            early_stopping_delta=1e-4,
            sampler_type="multivariate-tpe", 
            random_state=self.random_state
        )

        return best_value 
  
# ---------------------------------------------------------
# Definitions of Parameter Space  
# ---------------------------------------------------------

'''
all sample spaces have a high emphasis on more aggressive regularization
'''

def define_xgb_space(trial): 
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000), 
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 

        "max_depth": trial.suggest_int("max_depth", 3, 6), 
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),

        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9), 
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-1, 1e2, log=True), 
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 1e2, log=True),

        "gamma": trial.suggest_float("gamma", 1e-1, 1e1, log=True), 

        "tree_method": "hist", 
        "n_jobs": -1 
    }

def define_rf_space(trial): 
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000), 
        "max_depth": trial.suggest_int("max_depth", 4, 15), 

        "min_samples_split": trial.suggest_int("min_samples_split", 10, 60), 
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100), 
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]), 

        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),

        "n_jobs": -1 
    }

def define_svm_space(trial): 

    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])

    params = {
        "kernel": kernel, 
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "probability": True 
    }

    if kernel == "linear": 
        pass 
    elif kernel == "rbf": 
        gamma_type = trial.suggest_categorical("gamma_mode", ["auto_scale", "custom"])
        if gamma_type == "custom": 
            params["gamma"] = trial.suggest_float("gamma_custom", 1e-4, 1e1, log=True)
        else: 
            params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto"])

    params["shrinking"] = trial.suggest_categorical("shrinking", [True, False])
    return params

def define_logistic_space(trial): 

    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True), 
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]), 
    }

def define_idml_space(trial): 
    return {
        "n_neighbors_frac": trial.suggest_float("n_neighbors_frac", 0.0, 1.0), 
        "n_components_frac": trial.suggest_float("n_components_frac", 0.0, 1.0),
        "confidence_threshold": trial.suggest_float("confidence_threshold", 0.60, 0.99), 
        "label_spreading_alpha": trial.suggest_float("label_spreading_alpha", 0.1, 0.9),
        "max_iter": trial.suggest_int("max_iter", 1, 10), 
        "n_jobs": -1 
    }

# ---------------------------------------------------------
# Helper functions  
# ---------------------------------------------------------

def _load_yaml_config(path: Path) -> Dict[str, Any]: 
    if not path.exists(): 
        return {}
    with path.open("r", encoding="utf-8") as handle: 
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict): 
        raise ValueError(f"config file {path} must contain a mapping at root")
    return data 

def _save_yaml_config(path: Path, data: Dict[str, Any]): 
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: 
        yaml.safe_dump(data, handle, sort_keys=False)

def _save_model_config(path: str, model_key: str, params: Dict[str, Any]): 
    config_path = Path(path) 
    data = _load_yaml_config(config_path)

    models = data.get("models")
    if models is None: 
        models = {}
    elif not isinstance(models, dict): 
        raise ValueError(f"config file {config_path} has a non-mapping 'models' entry")

    models[model_key] = params 
    data["models"] = models 
    _save_yaml_config(config_path, data)

def _load_probs_for_fips(
    proba_path: str, 
    fips_order, 
    model_name=None,
    agg="mean"
): 
    oof  = load_oof_predictions(proba_path)
    fips = np.asarray(oof["fips_codes"]).astype("U5")
    idx  = _align_on_fips(fips_order, fips)

    probs = np.asarray(oof["probs"], dtype=np.float64)
    if probs.ndim != 3: 
        raise ValueError(f"expected probs shape (n, m, c), got {probs.shape}")

    if model_name is not None: 
        names = np.asarray(oof["model_names"]).reshape(-1).tolist() 
        if model_name not in names: 
            raise ValueError(f"model_name '{model_name}' not in {names}")
        m_idx = names.index(model_name)
        P = probs[:, m_idx, :]
    else: 
        if probs.shape[1] == 1: 
            P = probs[:, 0, :]
        elif agg == "mean": 
            P = probs.mean(axis=1)
        else: 
            raise ValueError("multiple models present, set model_name or agg='mean'")

    P = P[idx]

    class_labels = np.array(oof["class_labels"]).reshape(-1)
    if class_labels.size == 0:
        class_labels = np.arange(P.shape[1], dtype=np.float64)

    return P, class_labels 

# ---------------------------------------------------------
# Nested Cross Validation 
# ---------------------------------------------------------

'''
Aims to mitigate selection bias by wrapping optimization call in 
a nested cross validation loop. 
'''

def _select_metric_value(
    metrics: Dict[str, float], 
    task: TaskSpec, 
    metric: str | None
) -> float: 

    if metric is not None: 
        if metric not in metrics or np.isnan(metrics[metric]): 
            raise ValueError(f"metric {metric} missing or NaN")
        return metrics[metric]

    if task.task_type == "classification": 
        for key in ("f1_macro", "accuracy", "roc_auc"): 
            if key in metrics and not np.isnan(metrics[key]): 
                return metrics[key]
    else: 
        if "r2" in metrics and not np.isnan(metrics["r2"]): 
            return metrics["r2"]

    raise ValueError("no suitable metric found")

def _predict_proba_if_any(model, X_test, coords_test): 
    if not hasattr(model, "predict_proba"): 
        return None 
    proba = model.predict_proba(X_test, coords_test)
    if proba is None: 
        return None 

    proba = np.asarray(proba, np.float64)
    if proba.ndim == 2 and proba.shape[1] == 2: 
        return proba[:, 1]
    return proba 

def run_nested_cv(
    *,
    name: str, 
    filepath: str, 
    loader_func: Callable, 
    model_factory: Callable, 
    param_space: Callable[[optuna.Trial], Dict[str, Any]], 
    task: TaskSpec, 
    outer_config: CVConfig, 
    inner_config: CVConfig, 
    n_trials: int = 100, 
    direction: Literal["maximize", "minimize"] = "maximize", 
    random_state: int = 0, 
    metric: str | None = None, 
    scale_X: bool = True, 
    scale_y: bool | None = None, 
    param_transform: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None, 
    early_stopping_rounds: int | None = None, 
    early_stopping_delta: float = 0.0,
    sampler_type: str = "multivariate-tpe"
): 

    data   = loader_func(filepath)
    X      = data["features"]
    y      = data["labels"].ravel() 
    coords = data.get("coords")
    coords = None if coords is None else np.asarray(coords, dtype=np.float64)

    if task.task_type == "classification" and y.ndim == 2 and y.shape[1] == 1: 
        y = y.ravel() 

    if scale_y is None: 
        scale_y = False if task.task_type == "classification" else True 

    outer_splitter = outer_config.get_splitter(task)
    y_for_split = y if y.ndim == 1 else y[:, 0]

    fold_scores: list[float] = []
    fold_params: list[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_splitter.split(X, y_for_split)
    ): 
        print(f"> Nested fold {fold_idx+1}/{outer_config.n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        coords_train = None if coords is None else coords[train_idx]
        coords_test  = None if coords is None else coords[test_idx]

        def inner_loader(_): 
            return {
                "features": X[train_idx], 
                "labels": y[train_idx], 
                "coords": coords_train, 
                "feature_names": data.get("feature_names"), 
                "sample_ids": data.get("sample_ids")
            }

        evaluator = StandardEvaluator(
            filepath="virtual_path", 
            loader_func=inner_loader, 
            base_factory_func=model_factory, 
            param_space=param_space, 
            task=task,
            config=inner_config
        )

        best_params, _ = run_optimization(
            name=f"{name}_fold{fold_idx}",
            evaluator=evaluator,
            n_trials=n_trials,
            direction=direction, 
            random_state=random_state, 
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_delta=early_stopping_delta,
            sampler_type=sampler_type
        )
        
        if param_transform is not None: 
            best_params = param_transform(best_params)

                
        # Smoke test to get best_value from best_params  

        model = model_factory(**best_params)
        if callable(model) and not hasattr(model, "fit"): 
            model = model() 

        model = ScaledEstimator(model, scale_X=scale_X, scale_y=scale_y)
        model.fit(X_train, y_train, coords_train)

        y_pred = model.predict(X_test, coords_test)
        y_prob = _predict_proba_if_any(model, X_test, coords_test)

        metrics = task.compute_metrics(y_test, y_pred, y_prob)
        score   = _select_metric_value(metrics, task, metric)

        fold_scores.append(score)
        fold_params.append(best_params)

    mean_score   = float(np.mean(fold_scores))
    if direction == "maximize": 
        best_idx = int(np.argmax(fold_scores))
    else: 
        best_idx = int(np.argmin(fold_scores))
    best_params  = fold_params[best_idx]
    
    return mean_score, best_params, fold_scores, fold_params 
