#!/usr/bin/env python3 
# 
# hyperparameter.py  Andrew Belles  Dec 26th, 2025 
# 
# A generalized interface for Bayesian Optimization using Optuna. 
# Aims to offer a general contract for different Models to leverage
# 

from numpy.typing import NDArray

import optuna, itertools 
from optuna.samplers import CmaEsSampler, TPESampler

import numpy as np 
from abc import ABC, abstractmethod 
from typing import Any, Callable, Dict, Literal 

from preprocessing.loaders import (
    load_oof_predictions
)

from analysis.cross_validation import (
    CrossValidator, 
    CVConfig,
    ScaledEstimator, 
    TaskSpec,
)

from models.graph.processing import (
    CorrectAndSmooth,
)

from models.graph.construction import (
    make_mobility_adjacency_factory,
    make_queen_adjacency_factory,
    normalize_adjacency
)

from sklearn.metrics import accuracy_score

from utils.helpers import (
    make_train_mask,
    align_on_fips 
)

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
        config: CVConfig,
        feature_transform_factory=None 
    ): 
        self.filepath       = filepath 
        self.loader         = loader_func 
        self.factory        = base_factory_func 
        self.param_space_fn = param_space 
        self.task           = task 
        self.config         = config
        self.config.verbose = False
        self.feature_transform_factory = feature_transform_factory

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: 
        return self.param_space_fn(trial)

    def evaluate(self, params: Dict[str, Any]) -> float:

        model = self.factory(**params)

        cv = CrossValidator(
            filepath=self.filepath,
            loader=self.loader, 
            task=self.task, 
            scale_y=False, 
            feature_transform_factory=self.feature_transform_factory
        )

        results = cv.run(
            models={"opt_candidate": model},
            config=self.config,
        )

        if "error" in results.columns and results["error"].notna().any(): 
            err_msgs = results.loc[results["error"].notna(), "error"].unique().tolist() 
            raise RuntimeError(f"cross-val fold errors: {err_msgs}")

        summary = cv.summarize(results)

        for metric in ["f1_macro_mean", "accuracy_mean", "r2_mean"]: 
            if metric in summary.columns: 
                return summary.iloc[0][metric]
        
        raise ValueError("no suitable metric found in summary results ")

class PipelineEvaluator(OptunaEvaluator): 
    '''
    Optimization of each step of classifier pipeline 
    '''
    pass 

class CNNEvaluator(OptunaEvaluator): 

    def __init__(
        self, 
        filepath, 
        loader_func,
        model_factory, 
        param_space, 
        task,
        config
    ): 
        self.filepath       = filepath 
        self.loader         = loader_func 
        self.factory        = model_factory 
        self.param_space_fn = param_space 
        self.task           = task 
        self.config         = config 
        self.config.verbose = False 

        data   = self.loader(filepath)
        self.X = np.asarray(data["features"], dtype=np.float32)
        self.y = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
        self.coords = data.get("coords")

    def suggest_params(self, trial): 
        return self.param_space_fn(trial)

    def evaluate(self, params): 
        splitter    = self.config.get_splitter(self.task)
        y_for_split = self.y 
        scores      = []

        for train_idx, test_idx in splitter.split(self.X, y_for_split): 
            model = self.factory(**params)
            model.fit(self.X[train_idx], self.y[train_idx])

            y_prob = model.predict_proba(self.X[test_idx])
            y_pred = np.argmax(y_prob, axis=1)

            metrics = self.task.compute_metrics(self.y[test_idx], y_pred, y_prob)
            score   = metrics.get("f1_macro", np.nan)
            if np.isnan(score): 
                score = metrics.get("accuracy", np.nan)
            scores.append(score)

        return float(np.mean(scores))

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
            autoscale=params.pop("autoscale")
        )

        P_cs = cs(self.P, self.y_train, self.train_mask, W)
        pred_labels = cs.predict(P_cs)
        return accuracy_score(self.y_true, pred_labels[self.test_mask])

# ---------------------------------------------------------
# Metric Learning Evaluator  
# ---------------------------------------------------------

def _apply_train_test_transforms(transforms, X, train_mask):
    X_train = X[train_mask]
    X_test  = X[~train_mask]
    for t in transforms: 
        if hasattr(t, "fit_transform"): 
            X_train = t.fit_transform(X_train)
        else: 
            t.fit(X_train)
            X_train = t.transform(X_train)
        X_test = t.transform(X_test)
    X_full = np.zeros((X.shape[0], X_train.shape[1]), dtype=X_train.dtype)
    X_full[train_mask]  = X_train
    X_full[~train_mask] = X_test 
    return X_full


class MetricCASEvaluator(OptunaEvaluator): 
    '''
    Targets downstream optimization of accuracy 
    '''

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
        self.filepath         = filepath 
        self.factory          = base_factory_func 
        self.param_space_fn   = param_space 
        self.dataset_loaders  = dataset_loaders
        self.proba_path       = proba_path 
        self.proba_model_name = proba_model_name 
        self.random_state     = random_state
        self.train_size       = train_size 
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
        loader      = self.dataset_loaders.get(dataset_key)
        if loader is None: 
            raise ValueError("no loader available for evaluator")

        data = loader(self.filepath)
        X    = data["features"]
        y    = np.asarray(data["labels"]).reshape(-1)
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

        if self.feature_transform_factory is not None: 
            feature_names = data.get('feature_names')
            factory = self.feature_transform_factory(feature_names)
            transforms = factory() if callable(factory) else factory
            if transforms: 
                X = _apply_train_test_transforms(transforms, X, train_mask)

        if adj is None: 
            raise ValueError("EdgeLearner requires adjacency_factory to supply priori adj")

        model = self.factory(**params)
        model.fit(X, y, adj, train_mask=train_mask)
        adj = model(X, adj)

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
            n_trials=150,
            early_stopping_rounds=40,
            early_stopping_delta=1e-4,
            sampler_type="multivariate-tpe", 
            random_state=self.random_state
        )

        return best_value 
  
# ---------------------------------------------------------
# Graph Construction Optimization   
# ---------------------------------------------------------

class MobilityEvaluator(OptunaEvaluator): 

    def __init__(
        self,
        *,
        mobility_path: str, 
        proba_path: str, 
        k_min: int = 5, 
        k_neighbors: int | None = None, 
        proba_model_name: str | None = None, 
        train_size: float = 0.3, 
        random_state: int = 0 
    ): 
        self.mobility_path    = mobility_path 
        self.proba_path       = proba_path 
        self.k_min            = k_min 
        self.proba_model_name = proba_model_name 
        self.random_state     = random_state 
        self.k_neighbors      = k_neighbors

        oof = load_oof_predictions(proba_path)
        self.fips = np.asarray(oof["fips_codes"]).astype("U5")
        self.y    = np.asarray(oof["labels"]).reshape(-1)

        self.P, self.class_labels = _load_probs_for_fips(
            proba_path, self.fips, model_name=proba_model_name
        )

        self.train_mask = make_train_mask(
            self.y,
            train_size=train_size,
            random_state=random_state,
            stratify=True 
        )
        self.test_mask = ~self.train_mask 

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        if self.k_neighbors is not None: 
            return {"k_neighbors": self.k_neighbors}
        return {
            "k_neighbors": trial.suggest_int("k_neighbors", self.k_min, 100)
        }


    def evaluate(self, params: Dict[str, Any]) -> float:
        k = int(params["k_neighbors"])

        adj_factory = make_mobility_adjacency_factory(
            self.mobility_path,
            self.proba_path,
            k_neighbors=k
        )
        adj = adj_factory(list(self.fips))
        W   = normalize_adjacency(adj)

        cs  = CorrectAndSmoothEvaluator(
            P=self.P,
            W_by_name={"mobility": W}, 
            y_train=self.y,
            train_mask=self.train_mask,
            test_mask=self.test_mask,
            class_labels=self.class_labels
        )

        _, best_value = run_optimization(
            name="CorrectAndSmooth_inner",
            evaluator=cs, 
            n_trials=150,
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

        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), 
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1e2, log=True), 
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e2, log=True),

        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        # "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),

        "focal_gamma": trial.suggest_float("focal_gamma", 0.0, 5.0),

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

        "class_weight": trial.suggest_categorical("class_weight", 
                                                  [None, "balanced", "balanced_subsample"]),

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

def define_mobility_space(trial): 
    return {
        "k_neighbors": trial.suggest_int("k_neighbors", 5, 100)
    }

def make_layer_choices(
    sizes=(32, 64, 128, 256),
    min_layers=1,
    max_layers=3
): 
    choices = {}
    for L in range(min_layers, max_layers+1): 
        for combo in itertools.product(sizes, repeat=L): 
            if any(combo[i] < combo[i + 1] for i in range(len(combo) - 1)): 
                continue 
            key = "-".join(str(x) for x in combo)
            choices[key] = combo 
    return choices 

def define_gate_space(trial):
    layer_choices = make_layer_choices()
    key = trial.suggest_categorical("hidden_dims", list(layer_choices.keys()))
    return {
        "hidden_dims": layer_choices[key], 
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True), 
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), 
        "epochs": trial.suggest_int("epochs", 1000, 6000), 
        "batch_size": trial.suggest_categorical("batch_size", [4096, 8192, 16384]), 
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "residual": trial.suggest_categorical("residual", [True, False]), 
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False])
    }

def define_cnn_space(trial): 
    conv_choices = {
        "32-64-128-256": (32, 64, 128, 256),
        "32-64-128": (32, 64, 128),
    }
    key = trial.suggest_categorical("conv_channels", list(conv_choices.keys())) 
    return {
        "conv_channels": conv_choices[key],
        "fc_dim": trial.suggest_categorical("fc_dim", [256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "lr": trial.suggest_float("lr", 1e-5, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        # hardcoded 
        "kernel_size": 3,
        "pool_size": 2, 
        "use_bn": True,
        "epochs": 150, 
        "early_stopping_rounds": 20, 
        "eval_fraction": 0.15
    }

# --------------------------------------------------------- 
# Static Helpers 
# ---------------------------------------------------------

def _load_probs_for_fips(
    proba_path: str, 
    fips_order, 
    model_name=None,
    agg="mean"
): 
    oof  = load_oof_predictions(proba_path)
    fips = np.asarray(oof["fips_codes"]).astype("U5")
    idx  = align_on_fips(fips_order, fips)

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
