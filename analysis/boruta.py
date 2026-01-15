#!/usr/bin/env python3 
# 
# boruta.py  Andrew Belles  Jan 13th, 2026 
# 
# Noise filteration method by ranking features abilities to perform over random permutation 
# 
# 

import numpy as np 

from numpy.typing import NDArray
import pandas as pd 

from dataclasses import dataclass 

from typing import Callable, Sequence 

from scipy.stats             import binomtest 
from sklearn.ensemble        import RandomForestClassifier 
from sklearn.inspection      import permutation_importance 
from sklearn.model_selection import StratifiedShuffleSplit

from preprocessing.loaders import DatasetDict 
from utils.feature_matrix import (
    coerce_supervised,
    drop_nan_rows,
    apply_feature_subset
)

@dataclass 
class BorutaConfig: 
    n_iter: int = 50 
    n_estimators: int = 600 
    max_depth: int = 7 
    max_features: str | int | float | None = "sqrt"
    min_samples_leaf: int = 1 
    random_state: int = 0 
    alpha: float = 0.05 
    perm_repeats: int = 5 
    shadow_qunatile: float = 0.95 
    test_size: float = 0.25 
    scoring: str | None = "balanced_accuracy"

    def to_dict(self) -> dict[str, object]: 
        return {
            "n_iter": int(self.n_iter),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "max_features": self.max_features,
            "min_samples_leaf": int(self.min_samples_leaf),
            "random_state": int(self.random_state),
            "alpha": float(self.alpha),
            "perm_repeats": int(self.perm_repeats),
            "shadow_qunatile": float(self.shadow_qunatile),
            "test_size": float(self.test_size),
            "scoring": self.scoring
        }

@dataclass 
class BorutaResult: 
    summary: pd.DataFrame 
    feature_names: NDArray[np.str_]
    sample_ids: NDArray[np.str_]
    hits: NDArray[np.int64]
    importance_mean: NDArray[np.float64]
    importance_std: NDArray[np.float64]
    shadow_max: NDArray[np.float64]
    meta: dict[str, object]

    def to_dict(self) -> dict[str, object]: 
        return {
            "summary": self.summary.to_numpy(),
            "summary_index": self.summary.index.to_numpy(dtype="U128"),
            "summary_columns": self.summary.columns.to_numpy(dtype="U128"),
            "feature_names": np.asarray(self.feature_names, dtype="U128"),
            "hits": self.hits.astype(np.int64),
            "importance_mean": self.importance_mean.astype(np.float64),
            "importance_std": self.importance_std.astype(np.float64),
            "shadow_max": self.shadow_max.astype(np.float64),
            "meta": dict(self.meta)
        }

    def save_csv(self, path: str): 
        self.summary.to_csv(path, index=True)

class BorutaProbe: 
    def __init__(
        self,
        config: BorutaConfig,
        *,
        verbose: bool = False 
    ): 
        self.config  = config 
        self.verbose = verbose 

    def compute(
        self,
        filepath: str, 
        loader: Callable[[str], DatasetDict],
        feature_subset: Sequence[int | str] | None = None 
    ) -> BorutaResult: 
        data   = loader(filepath)
        matrix = coerce_supervised(data)
        matrix = drop_nan_rows(matrix)
        matrix = apply_feature_subset(matrix, feature_subset)

        X = matrix.X 
        y = matrix.y 
        n_samples, n_features = X.shape 

        if n_features == 0: 
            raise ValueError("no features to analyze")
        if n_samples == 0: 
            raise ValueError("no samples to analyze")

        rng  = np.random.default_rng(self.config.random_state)
        hits = np.zeros(n_features, dtype=np.int64)

        importance_history = np.zeros((self.config.n_iter, n_features), dtype=np.float64)
        shadow_max         = np.zeros(self.config.n_iter, dtype=np.float64)

        splitter           = StratifiedShuffleSplit(
            n_splits=self.config.n_iter,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        for t, (train_idx, test_idx) in enumerate(splitter.split(X, y)): 
            if self.verbose: 
                print(f"[{t + 1}/{self.config.n_iter}] boruta iteration")

            shadow = self._make_shadow(X, rng)
            X_aug  = np.hstack([X, shadow])

            rf     = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                max_features=self.config.max_features,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=-1,
                random_state=int(rng.integers(0, 2**31 - 1))
            )
            rf.fit(X_aug[train_idx], y[train_idx])
            importances   = rf.feature_importances_ 
            real_imp      = importances[:n_features]
            shadow_imp    = importances[n_features:]

            thresh        = (float(np.quantile(shadow_imp, self.config.shadow_qunatile)) 
                             if shadow_imp.size else 0.0)
            shadow_max[t] = thresh 

            hits += (real_imp > thresh).astype(np.int64)
            importance_history[t] = real_imp 

        hit_rate = hits / float(self.config.n_iter)
        importance_mean = importance_history.mean(axis=0)
        importance_std  = importance_history.std(axis=0)

        p_greater = np.zeros(n_features, dtype=np.float64)
        p_less    = np.zeros(n_features, dtype=np.float64)
        status    = []

        for i, h in enumerate(hits): 
            p_greater[i] = binomtest(h, self.config.n_iter, 0.5, alternative="greater").pvalue
            p_less[i]    = binomtest(h, self.config.n_iter, 0.5, alternative="less").pvalue 

            if p_greater[i] < self.config.alpha: 
                status.append("confirmed")
            elif p_less[i] < self.config.alpha: 
                status.append("rejected")
            else: 
                status.append("tentative")

        summary = pd.DataFrame({
            "hits": hits, 
            "hit_rate": hit_rate,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
            "p_greater": p_greater,
            "p_less": p_less, 
            "status": status
        }, index=matrix.feature_names)

        meta = {
            "method": "boruta",
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "shadow_max_mean": float(np.mean(shadow_max)),
            "shadow_max_std": float(np.std(shadow_max))
        }
        meta.update(self.config.to_dict())

        return BorutaResult(
            summary=summary, 
            feature_names=matrix.feature_names,
            sample_ids=matrix.sample_ids,
            hits=hits,
            importance_mean=importance_mean,
            importance_std=importance_std,
            shadow_max=shadow_max,
            meta=meta
        )

    @staticmethod 
    def _make_shadow(X: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]: 
        shadow = np.empty_like(X)
        for j in range(X.shape[1]): 
            shadow[:, j] = rng.permutation(X[:, j])
        return shadow 


def select_features(
    result: BorutaResult,
    *,
    keep_tentative: bool = True 
) -> list[str]: 
    keep  = {"confirmed", "tentative"} if keep_tentative else {"confirmed"}
    return [str(name) for name, status in zip(result.feature_names, result.summary["status"])
            if status in keep]

def filter_dataset(
    data: DatasetDict,
    result: BorutaResult,
    *,
    keep_tentative: bool = True 
) -> DatasetDict: 
    keep_names    = set(select_features(result, keep_tentative=keep_tentative))
    feature_names = np.asarray(data.get("feature_names"))
    if feature_names is None or len(feature_names) == 0: 
        raise ValueError("dataset missing feature_names for filtering")

    cols = [i for i, n in enumerate(feature_names) if str(n) in keep_names]
    if not cols: 
        raise ValueError("no features kept after Boruta filtering")

    X   = np.asarray(data["features"])[:, cols]
    out                  = dict(data)
    out["features"]      = X 
    out["feature_names"] = feature_names[cols]
    return out # Into? 
