#!/usr/bin/env python3 
# 
# cross_validation.py  Andrew Belles  Dec 14th, 2025 
# 
# Interface which models can register with for intelligent folding of test data 
# and flexible number of repeats to gain understanding about models' efficacy and stability 
# 

import numpy as np 
import pandas as pd

from dataclasses import dataclass, field 
from typing import Callable, Literal, Mapping 
from numpy.typing import NDArray 

from sklearn.model_selection import (
    KFold, 
    RepeatedKFold, 
    RepeatedStratifiedKFold, 
    StratifiedKFold, 
    BaseCrossValidator
)

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, clone 
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    log_loss 
)

from support.helpers import (
    ModelFactory,
) 

from preprocessing.loaders import DatasetLoader

from scipy.io import savemat

# ---------------------------------------------------------
# Task Specification 
# --------------------------------------------------------- 

TaskType = Literal["regression", "classification"]

@dataclass(frozen=True)
class TaskSpec: 

    task_type: TaskType 
    metrics: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self): 
        defaults = {
            "regression": ("r2", "rmse"), 
            "classification": ("accuracy", "f1_macro", "roc_auc")
        }
        if not self.metrics: 
            object.__setattr__(self, "metrics", defaults[self.task_type])

    def compute_metrics(
        self, 
        y_true: NDArray, 
        y_pred: NDArray, 
        y_prob: NDArray | None = None
    ) -> dict[str, float]: 


        y_true = np.asarray(y_true).ravel()
        n_classes = len(np.unique(y_true))
        results = {}

        if self.task_type == "regression": 
            for m in self.metrics: 
                if m == "r2":
                    results["r2"] = float(r2_score(y_true, y_pred))
                elif m == "rmse": 
                    results["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        else: 
            for m in self.metrics: 
                if m == "accuracy": 
                    results["accuracy"] = float(accuracy_score(y_true, y_pred))
                elif m == "f1": 
                    avg = "binary" if n_classes == 2 else "macro"
                    results["f1"] = float(f1_score(y_true, y_pred, average=avg))
                elif m == "f1_macro": 
                    results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
                elif m == "roc_auc" and y_prob is not None: 
                    try: 
                        y_prob_arr = np.asarray(y_prob)
                        if (y_prob_arr.ndim == 1 or 
                            (y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 1)):
                            results["roc_auc"] = float(roc_auc_score(y_true, y_prob_arr.ravel()))
                        else: 
                            results["roc_auc"] = float(roc_auc_score(
                                y_true, 
                                y_prob_arr,
                                multi_class="ovr", 
                                average="macro"
                            )) 

                    except ValueError:
                        results["roc_auc"] = np.nan 
                elif m == "log_loss" and y_prob is not None: 
                    results["log_loss"] = float(log_loss(y_true, y_prob))

        return results 

    def compute_baseline(
        self, 
        y_train: NDArray, 
        y_test: NDArray
    ) -> dict[str, float]: 

        if self.task_type == "regression": 
            y_base = np.full_like(y_test, np.mean(y_train), dtype=np.float64)
            return self.compute_metrics(y_test, y_base)
        else: 
            unique, counts = np.unique(y_train, return_counts=True)
            majority = unique[np.argmax(counts)]
            y_base = np.full_like(y_test, majority)
            return self.compute_metrics(y_test, y_base)

REGRESSION     = TaskSpec("regression")
CLASSIFICATION = TaskSpec("classification") 

# ---------------------------------------------------------
# Cross-Validator Configuration 
# ---------------------------------------------------------

@dataclass(frozen=True)
class CVConfig: 

    '''Configuration for cross-validation'''

    n_splits: int = 5 
    n_repeats: int = 1 
    test_size: float = 0.2  # only used for shuffle splits 
    stratify: bool = False  # only meaningful for classification  
    shuffle: bool = True 
    random_state: int = 0

    def get_splitter(self, task: TaskSpec) -> BaseCrossValidator: 

        if self.n_repeats > 1: 
            # Classification task 
            if task.task_type == "classification" and self.stratify: 
                return RepeatedStratifiedKFold(
                    n_splits=self.n_splits, 
                    n_repeats=self.n_repeats, 
                    random_state=self.random_state
                )

            # Regression task 
            return RepeatedKFold(
                n_splits=self.n_splits, 
                n_repeats=self.n_repeats, 
                random_state=self.random_state 
            )
        
        # No repeats 
        if task.task_type == "classification" and self.stratify: 
            return StratifiedKFold(
                n_splits=self.n_splits, 
                shuffle=self.shuffle, 
                random_state=self.random_state if self.shuffle else None 
            )
    
        return KFold(
            n_splits=self.n_splits, 
            shuffle=self.shuffle, 
            random_state=self.random_state if self.shuffle else None 
        )

# ---------------------------------------------------------
# Model Wrapper 
# ---------------------------------------------------------

class ScaledEstimator(BaseEstimator): 

    '''Wraps any model with per-fold standarization'''

    def __init__(self, estimator: BaseEstimator, scale_X: bool = True, scale_y: bool = True): 

        self.estimator = estimator 
        self.scale_X   = scale_X 
        self.scale_y   = scale_y 
        self._X_scaler: StandardScaler | None = None 
        self._y_scaler: StandardScaler | None = None 

    def fit(self, X, y, coords=None): 

        X = np.asarray(X, dtype=np.float64) 
        y = np.asarray(y, dtype=np.float64)
         
        if y.ndim == 2 and y.shape[1] == 1: 
            y = y.ravel() 

        if self.scale_X: 
            self._X_scaler = StandardScaler() 
            X = self._X_scaler.fit_transform(X)


        if self.scale_y and y.ndim <= 2: 
            self._y_scaler = StandardScaler() 
            y_shape = y.shape 
            y = self._y_scaler.fit_transform(y.reshape(-1, 1) if y.ndim ==1 else y)
            if len(y_shape) == 1: 
                y = y.ravel() 


        try:
            self.estimator.fit(X, y, coords=coords)
        except TypeError: 
            self.estimator.fit(X, y) 

        return self 

    def predict(self, X, coords=None): 
        X = np.asarray(X, dtype=np.float64)
        if self.scale_X and self._X_scaler is not None: 
            X = self._X_scaler.transform(X)
        
        try: 
            y_pred = self.estimator.predict(X, coords=coords)
        except TypeError: 
            y_pred = self.estimator.predict(X)

        if self.scale_y and self._y_scaler is not None: 
            y_pred = np.asarray(y_pred, dtype=np.float64)
            shape  = y_pred.shape 
            y_pred = self._y_scaler.inverse_transform(
                y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred 
            )
            if len(shape) == 1: 
                y_pred = y_pred.ravel() 

        return y_pred 

    def predict_proba(self, X, coords=None): 

        X = np.asarray(X, dtype=np.float64)
        if self.scale_X and self._X_scaler is not None: 
            X = self._X_scaler.transform(X)

        if hasattr(self.estimator, "predict_proba"): 
            try: 
                return self.estimator.predict_proba(X, coords=coords) 
            except TypeError: 
                return self.estimator.predict_proba(X) 
        raise AttributeError("Estimator does not support predict_proba")

# ---------------------------------------------------------
# Cross Validator  
# ---------------------------------------------------------

class CrossValidator: 

    '''
    Sklearn-based cross-validation harness. 

    Supports both regression and classification via TaskSpec. 
    '''

    def __init__(
        self, 
        *, 
        filepath: str, 
        loader: DatasetLoader, 
        task: TaskSpec = REGRESSION, 
        scale_X: bool = True, 
        scale_y: bool = True,
    ): 
        self.filepath = filepath 
        self.task     = task 
        self.scale_X  = scale_X 
        self.scale_y  = False if task.task_type == "classification" else scale_y 

        data   = loader(filepath)
    
        self.X = np.asarray(data["features"], dtype=np.float64) 
        self.y = np.asarray(data["labels"], dtype=np.float64)
        self.coords = np.asarray(data["coords"], dtype=np.float64)

        if task.task_type == "classification" and self.y.ndim == 2 and self.y.shape[1] == 1:
            self.y = self.y.ravel() 

        self.predictions_ : pd.DataFrame | None = None  
        self.oof_: dict | None = None 
        self.models_: dict[str, list[BaseEstimator]] = {} 
        self.sample_ids = np.asarray(
            data["sample_ids"], dtype="U5"
        ) if "sample_ids" in data else np.asarray(np.arange(self.X.shape[0]), dtype=np.int64) - 1
        
    def run(
        self, 
        *,
        models: Mapping[str, ModelFactory], 
        config: CVConfig, 
        label_transforms: Mapping[str, tuple[Callable, Callable | None]] | None = None, 
        oof: bool = False, 
        collect: bool = False, 
        splits=None
    ) -> pd.DataFrame: 

        splitter    = config.get_splitter(self.task)
        y_for_split = self._split_target()
        splits_iter = self._resolve_splits(splitter, y_for_split, splits)

        results   = []
        pred_rows = []
        model_store = {name: [] for name in models}

        oof_sums, oof_counts, oof_classes = (None, None, None)
        if oof: 
            oof_sums, oof_counts, oof_classes = self._init_oof(models, y_for_split)

        for model_name, make_model in models.items(): 
            print(f"> {model_name} now folding...")
            for fold_i, (train_idx, test_idx) in enumerate(splits_iter): 
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                coords_train, coords_test = self.coords[train_idx], self.coords[test_idx]

                base_model = make_model()
                model      = self._build_model(base_model)
                transform, inverse = self._get_transform(label_transforms, model_name)

                try: 
                    y_pred = self._fit_predict(
                        model,
                        X_train,
                        y_train, 
                        coords_train,
                        X_test, 
                        coords_test, 
                        transform 
                    )
                    y_pred_eval = inverse(y_pred) if inverse is not None else y_pred 

                    proba  = self._predict_proba(model, X_test, coords_test)
                    y_prob = self._align_proba(proba, model, oof_classes)
                    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2: 
                        y_prob_eval = y_prob[:, 1]
                    else: 
                        y_prob_eval = y_prob 

                    results.append(self._metrics_row(
                        model_name,
                        fold_i, 
                        train_idx, test_idx, 
                        y_train, y_test, 
                        y_pred_eval, y_prob_eval 
                    ))

                    if collect: 
                        self._collect_rows(
                            pred_rows, 
                            model_name, 
                            fold_i, 
                            test_idx,  
                            y_test, 
                            y_pred_eval
                        )

                    if oof: 
                        self._update_oof(
                            oof_sums, 
                            oof_counts, 
                            model_name, 
                            test_idx, 
                            y_pred_eval,
                            y_prob
                        )

                except Exception as e: 
                    results.append(self._error_row(model_name, fold_i, train_idx, test_idx, e))

                model_store[model_name].append(model)

        self.models_ = model_store

        if collect and pred_rows: 
            self.predictions_ = pd.DataFrame(pred_rows)

        if oof: 
            self.oof_ = {"sums": oof_sums, "counts": oof_counts, "classes": oof_classes}

        return pd.DataFrame(results )

    # -----------------------------------------------------
    # Run Helper functions 
    # -----------------------------------------------------

    def _split_target(self) -> NDArray: 
        return self.y if self.y.ndim ==1 else self.y[:, 0]

    def _resolve_splits(self, splitter, y_for_split, splits): 
        if splits is not None: 
            self.splits_ = list(splits)
            return self.splits_ 
        self.splits_ = list(splitter.split(self.X, y_for_split))
        return self.splits_

    def _build_model(self, base_model): 
        return ScaledEstimator(
            clone(base_model) if hasattr(base_model, "get_params") else base_model, 
            scale_X=self.scale_X, 
            scale_y=self.scale_y 
        )

    def _get_transform(self, label_transforms, model_name): 
        if label_transforms is None: 
            return None, None 
        transform, inverse = label_transforms.get(model_name, (None, None))
        if transform is not None and self.task.task_type != "regression":
            raise ValueError("label_transforms only supported for regression")
        return transform, inverse 

    def _fit_predict(
        self, 
        model, 
        X_train, 
        y_train, 
        coords_train, 
        X_test, 
        coords_test, 
        transform
    ): 

        y_train_fit = transform(y_train) if transform else y_train 
        model.fit(X_train, y_train_fit, coords_train)
        return model.predict(X_test, coords_test) 

    def _predict_proba(self, model, X_test, coords_test): 
        try: 
            return np.asarray(model.predict_proba(X_test, coords_test))
        except (AttributeError, TypeError, IndexError): 
            return None 

    def _align_proba(self, proba, model, oof_classes): 
        if proba is None: 
            return None 
        if proba.ndim ==1: 
            proba = np.column_stack([1.0 - proba, proba])

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "estimator"): 
            classes = getattr(model.estimator, "classes_", None)
        
        if classes is not None and oof_classes is not None: 
            aligned   = np.zeros((proba.shape[0], len(oof_classes)), dtype=np.float64)
            class_map = {c: i for i, c in enumerate(classes)}
            for j, c in enumerate(oof_classes): 
                if c in class_map: 
                    aligned[:, j] = proba[:, class_map[c]]
            return aligned 
        return proba 

    def _init_oof(self, models, y_for_split): 
        if self.task.task_type == "classification": 
            oof_classes = np.unique(y_for_split)
            n_classes   = len(oof_classes)
        else: 
            oof_classes, n_classes = None, 1 

        oof_counts = {}
        oof_sums   = {}
        for model_name in models: 
            oof_sums[model_name]  = np.zeros((self.n_samples, n_classes), dtype=np.float64) 
            oof_counts[model_name] = np.zeros((self.n_samples, 1), dtype=np.int64)
        return oof_sums, oof_counts, oof_classes

    def _metrics_row(
        self,
        model_name,
        fold_i, 
        train_idx, 
        test_idx, 
        y_train, 
        y_test, 
        y_pred_eval,
        y_prob
    ): 

        metrics  = self.task.compute_metrics(y_test, y_pred_eval, y_prob)
        baseline = self.task.compute_baseline(y_train, y_test)

        row = {
            "model": model_name, 
            "fold": fold_i, 
            "n_train": len(train_idx),
            "n_test": len(test_idx)
        }

        for k, v in metrics.items(): 
            row[k] = v
            row[f"baseline_{k}"] = baseline.get(k, np.nan)
            
        if "r2" in metrics: 
            row["win_r2"]     = float(metrics["r2"] > baseline.get("r2", -np.inf))
            row["win_r2_gt0"] = float(metrics["r2"] > 0)
        
        if "rmse" in metrics: 
            row["win_rmse"] = float(metrics["rmse"] < baseline.get("rmse", np.inf))

        if "accuracy" in metrics: 
            row["win_accuracy"] = float(metrics["accuracy"] > baseline.get("accuracy", 0))
        
        return row 

    def _collect_rows(
        self, 
        pred_rows, 
        model_name, 
        fold_i, 
        test_idx, 
        y_test, 
        y_pred
    ):
        y_test_flat = y_test.ravel() if y_test.ndim > 1 else y_test 
        y_pred_flat = np.asarray(y_pred).ravel() if hasattr(y_pred, "ravel") else y_pred 
        for i, idx in enumerate(test_idx): 
            pred_rows.append({
                "model": model_name, 
                "fold": fold_i, 
                "idx": int(idx), 
                "y_true": float(y_test_flat[i]), 
                "y_pred": float(y_pred_flat[i]), 
                "residual": float(y_test_flat[i] - y_pred_flat[i])
            })

    def _update_oof(self, oof_sums, oof_counts, model_name, test_idx, y_pred_eval, y_prob): 
        if self.task.task_type == "classification": 
            if y_prob is None: 
                raise ValueError(f"{model_name} missing predict_proba for OOF stacking")
            oof_sums[model_name][test_idx]   += y_prob
            oof_counts[model_name][test_idx] += 1 
        else: 
            y_pred = np.asarray(y_pred_eval).reshape(-1, 1)
            oof_sums[model_name][test_idx]   += y_pred 
            oof_counts[model_name][test_idx] += 1

    def _error_row(self, model_name, fold_i, train_idx, test_idx, e): 
        row = {
            "model": model_name, 
            "fold": fold_i, 
            "n_train": len(train_idx), 
            "n_test": len(test_idx), 
            "error": str(e)
        }
        for m in self.task.metrics:
            row[m] = np.nan 
            row[f"baseline_{m}"] = np.nan 
        return row 

    # -----------------------------------------------------
    # Summary functions 
    # -----------------------------------------------------

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame: 

        metric_cols = [c for c in results_df.columns if c in self.task.metrics]
        agg_dict = {m: ["mean", "std", "min", "max"] for m in metric_cols}
        agg_dict["fold"] = ["count"] 

        summary = results_df.groupby("model").agg(agg_dict)
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index() 

        n_folds = results_df["fold"].nunique()
        for m in metric_cols: 
            if f"{m}_std" in summary.columns: 
                ci_half = 1.96 * summary[f"{m}_std"] / np.sqrt(n_folds)
                summary[f"{m}_ci_lower"] = summary[f"{m}_mean"] - ci_half 
                summary[f"{m}_ci_upper"] = summary[f"{m}_mean"] + ci_half 

        return summary 

    def format_summary(self, summary_df: pd.DataFrame): 

        metrics = [m for m in self.task.metrics if f"{m}_mean" in summary_df.columns]

        header_parts = [f"{'Model':<20}"]
        for m in metrics: 
            header_parts.append(f"{m.upper():<30}")
        print(" ".join(header_parts))
        print("=" * (20 + 30 * len(metrics)))

        for _, row in summary_df.iterrows():
            parts = [f"{row['model']:<20}"]
            for m in metrics: 
                mean  = row.get(f"{m}_mean", np.nan)
                ci_lo = row.get(f"{m}_ci_lower", np.nan) 
                ci_hi = row.get(f"{m}_ci_upper", np.nan) 
                parts.append(f"{mean:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]".ljust(30))
            print(" ".join(parts))

    def save_oof(self, out_path: str, model_order: list[str] | None = None): 
        if self.oof_ is None: 
            raise ValueError("OOF not collected. run with oof=True")

        sums    = self.oof_["sums"]
        counts  = self.oof_["counts"]
        classes = self.oof_.get("classes")

        if model_order is None: 
            model_order = list(sums.keys())

        probs = {}
        for m in model_order: 
            denom = np.clip(counts[m], 1, None)
            probs[m] = sums[m] / denom 

        probs_stack = np.stack([probs[m] for m in model_order], axis=1)

        if classes is None: 
            preds_stack   = np.stack([probs[m].reshape(-1) for m in model_order], axis=1)
            feature_names = np.array([f"{m}_pred" for m in model_order], dtype="U64")
            class_labels  = np.array([], dtype=np.int64)
        else: 
            class_labels  = np.array(classes)
            preds_stack   = np.stack([class_labels[np.argmax(probs[m], axis=1)] for m in model_order], 
                                     axis=1)
            feature_names = np.array([f"{m}_p{c}" for m in model_order for c in class_labels], 
                                     dtype="U64") 

        features = np.hstack([probs[m] for m in model_order])

        mat = {
            "features": features, 
            "probs": probs_stack, 
            "preds": preds_stack, 
            "labels": self.y.reshape(-1, 1) if self.y.ndim == 1 else self.y, 
            "feature_names": feature_names, 
            "fips_codes": self.sample_ids, 
            "model_names": np.array(model_order, dtype="U64"), 
            "class_labels": np.array(classes) if classes is not None else np.array([]), 
            "n_samples": features.shape[0], 
            "n_models": len(model_order), 
            "n_classes": int(probs_stack.shape[2]) if probs_stack.ndim == 3 else 1 
        }
        savemat(out_path, mat)

    def get_model(self, model_name: str, fold: int = -1) -> BaseEstimator:
        return self._get_models(model_name)[fold]

    def _get_models(self, model_name: str) -> list[BaseEstimator]:
        if self.models_ is None: 
            raise ValueError("models not collected")
        if model_name not in self.models_: 
            raise KeyError(f"model {model_name} not found")
        return self.models_[model_name]

    @property
    def n_samples(self) -> int: 
        return self.X.shape[0]
