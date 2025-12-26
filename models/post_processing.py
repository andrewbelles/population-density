#!/usr/bin/env python3 
# 
# post_processing.py  Andrew Belles  Dec 25th, 2025 
# 
# Modules that aim to post process probabilities output by Classifiers  
# 

import numpy as np 
import pandas as pd 

from typing import Callable

from numpy.typing import NDArray

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

def make_train_mask(y, train_size=0.3, random_state=0, stratify=True): 

    y = np.asarray(y).reshape(-1)
    if not (0.0 < train_size < 1.0): 
        raise ValueError("train_size must be in (0, 1)")
    rng = np.random.default_rng(random_state)
    n = y.shape[0]
    mask = np.zeros(n, dtype=bool)

    if stratify: 
        classes = np.unique(y)
        chosen  = []
        for c in classes: 
            idx     = np.where(y == c)[0]
            n_train = int(round(train_size * idx.size))
            if idx.size > 1: 
                n_train = max(1, min(n_train, idx.size - 1))
            else: 
                n_train = 1 
            chosen.append(rng.choice(idx, size=n_train, replace=False))
        chosen = np.unique(np.concatenate(chosen))
    else: 
        n_train = int(round(train_size * n))
        chosen  = rng.choice(np.arange(n), size=n_train, replace=False)

    mask[chosen] = True 
    return mask 


def normalized_proba(P, mask, eps=1e-12): 
    P    = np.asarray(P, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    X = P[mask].copy() 
    if X.size == 0: 
        return X 

    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    X = np.divide(X, np.clip(s, eps, None))

    zero_rows = (s <= eps).ravel() 
    if np.any(zero_rows): 
        X[zero_rows] = 1.0 / X.shape[1]

    return X 


class CorrectAndSmooth: 

    def __init__(
        self, 
        *, 
        correction_alpha: float = 0.01,
        smoothing_alpha: float = 0.15, 
        correction_max_iter: int = 3, 
        smoothing_max_iter: int = 8, 
        tol: float = 1e-6, 
        norm: Callable = lambda x: np.linalg.norm(x), 
        relative: bool = True, 
        autoscale: bool = True, 
        class_labels: NDArray | None = None 
    ): 

        self.correction_alpha    = correction_alpha 
        self.smoothing_alpha     = smoothing_alpha
        self.correction_max_iter = correction_max_iter 
        self.smoothing_max_iter  = smoothing_max_iter 
        self.tol                 = tol 
        self.relative            = relative 
        self.autoscale           = autoscale 
        self.class_labels        = class_labels

        self._norm_tuple = (None, None)

        self._norm_fn = norm

    def evaluate(
        self, 
        y_true, 
        *, 
        mask=None, 
        average="macro", 
        print_report: bool = False 
    ): 

        if self._P_smooth is None: 
            raise ValueError("model must be fit before evaluation")

        y = np.asarray(y_true).reshape(-1)
        if mask is not None: 
            mask = np.asarray(mask, dtype=bool)
            y = y[mask]
            P = normalized_proba(self._P_smooth, mask)
        else: 
            P = normalized_proba(
                self._P_smooth, 
                np.ones(self._P_smooth.shape[0], dtype=bool)
            ) 

        if P.ndim == 2 and P.shape[1] > 1: 
            y_pred = np.argmax(P, axis=1)
            if self.class_labels is not None: 
                y_pred = np.asarray(self.class_labels)[y_pred]
        else: 
            y_pred = (P.ravel() >= 0.5).astype(int)

        acc = float(accuracy_score(y, y_pred))
        f1  = float(f1_score(y, y_pred, average=average))
        
        if P.ndim == 2 and P.shape[1] > 1: 
            roc = float(roc_auc_score(
                y, 
                P, 
                multi_class="ovr", 
                average=average
            ))
        else: 
            roc = float(roc_auc_score(y, P.ravel()))

        df = pd.DataFrame([{
            "accuracy": acc, 
            "f1_macro": f1, 
            "roc_auc": roc 
        }]) 

        if print_report:
            print(f"Acc={acc:.4f} | f1_macro={f1:.4f} | roc_auc={roc:.4f}")

        return df  

    def predict(self): 
        if self._P_smooth is None: 
            raise ValueError("call fit first")
        P = normalized_proba(
            self._P_smooth,
            np.ones(self._P_smooth.shape[0], dtype=bool)
        )

        if P.shape[1] == 1: 
            return P.ravel() 

        idx = np.argmax(P, axis=1)
        if self.class_labels is not None: 
            return np.asarray(self.class_labels)[idx]
        return idx 

    def predict_proba(self): 
        if self._P_smooth is None: 
            raise ValueError("call fit first")
        return self._P_smooth
    
    def fit_predict(self, P, y_train, W, train_mask=None):
        self._fit_helper(P, y_train, train_mask, W) 
        return self.predict()

    def fit_predict_from_builder(self, builder: Callable, *args, train_mask=None, **kwargs):
        P, _, W, y_train = builder(*args, **kwargs)
        self._fit_helper(P, y_train, train_mask, W) 
        return self.predict()

    def fit(self, P, y_train, W, train_mask=None):
        return self._fit_helper(P, y_train, train_mask, W)

    def fit_from_builder(self, builder: Callable, *args, train_mask=None, **kwargs): 
        P, _, W, y_train = builder(*args, **kwargs) 
        return self._fit_helper(P, y_train, train_mask, W)

    def _fit_helper(self, P, y_train, train_mask, W): 
        P = self._as_2d(P)
        train_mask = self._as_mask(train_mask, P.shape[0])

        Y = self._prepare_targets(y_train, P, train_mask)

        E = self._compute_residuals(P, Y, train_mask)
        E_prop = self._propagate(
            E,
            W,
            alpha=self.correction_alpha, 
            clamp_idx=train_mask, 
            clamp_values=E[train_mask], 
            max_iter=self.correction_max_iter 
        )

        if self.autoscale: 
            E_prop = self._autoscale_residuals(E_prop, train_mask)

        P_corr = P + E_prop 

        P_smooth = self._propagate(
            P_corr, 
            W, 
            alpha=self.smoothing_alpha, 
            clamp_idx=train_mask, 
            clamp_values=Y[train_mask], 
            max_iter=self.smoothing_max_iter
        )

        self._P_smooth = P_smooth
        return P_smooth 

    def _infer_train_mask(self, y_train): 
        y = np.asarray(y_train)
        if y.ndim == 1: 
            if np.issubdtype(y.dtype, np.floating): 
                return ~np.isnan(y)
            raise ValueError("train_mask required when y_train has no NaNs")
        if y.ndim == 2: 
            if np.issubdtype(y.dtype, np.floating): 
                return ~np.isnan(y).any(axis=1)
            raise ValueError("train_mask required when y_train has no NaNs")
        raise ValueError("y_train must be 1d or 2d")
    
    def _propagate(self, Y0, W, *, alpha, clamp_idx, clamp_values, max_iter): 

        Y = np.asarray(Y0, dtype=np.float64) 
        clamp_values = np.asarray(clamp_values, dtype=np.float64)

        for _ in range(max_iter): 
            Y_next = (1.0 - alpha) * Y + alpha * (W @ Y)
            Y_next[clamp_idx] = clamp_values 

            if self._has_converged(Y_next, Y): 
                return Y_next 

            Y = Y_next 

        return Y 

    def _has_converged(self, Y_next, Y_prev): 
        _, norm_prev = self._norm_tuple

        diff = self._norm_fn(Y_next - Y_prev)
        self._norm_tuple = (norm_prev, diff)

        if norm_prev is not None and self.relative: 
            denom = max(norm_prev, self.tol**2)
            return diff <= self.tol * denom 
        return diff <= self.tol 

    def _autoscale_residuals(self, E_prop, train_mask): 
        train_rows = E_prop[train_mask]
        test_rows  = E_prop[~train_mask]

        if train_rows.size == 0 or test_rows.size == 0: 
            return E_prop 

        train_mean = np.mean(np.abs(train_rows).sum(axis=1))
        test_mean  = np.mean(np.abs(test_rows).sum(axis=1))

        if test_mean > self.tol: 
            scale = train_mean / test_mean 
            scale = min(scale, 10.0)
            E_prop[~train_mask] *= scale 

        return E_prop

    def _as_2d(self, P): 
        P = np.asarray(P, dtype=np.float64) 
        if P.ndim == 1: 
            return P.reshape(-1, 1)
        if P.ndim == 2: 
            return P 
        raise ValueError(f"P must be 1d or 2d, got shape {P.shape}")

    def _as_mask(self, mask, n): 
        m = np.asarray(mask)
        if m.shape[0] != n: 
            raise ValueError("train_mask length mismatch")
        return m.astype(bool)

    def _prepare_targets(self, y_train, P, train_mask): 
        y = np.asarray(y_train)

        if P.shape[1] == 1: 
            y_vec = y.reshape(-1, 1) if y.ndim == 1 else y 
            if y_vec.shape[0] != P.shape[0]: 
                raise ValueError("y_train length must match P rows")
            return y_vec.astype(np.float64)

        if y.ndim == 2 and y.shape[1] == P.shape[1]: 
            if y.shape[0] != P.shape[0]: 
                raise ValueError("y_train length must match P rows")
            return y.astype(np.float64)

        if y.ndim != 1: 
            raise ValueError("y_train must be 1d labels or 2d one-hot")

        if y.shape[0] != P.shape[0]: 
            raise ValueError("y_train length must match P rows")

        if self.class_labels is None: 
            class_labels = np.arange(P.shape[1])
        else: 
            class_labels = np.asarray(self.class_labels)

        Y = np.zeros_like(P, dtype=np.float64)
        for i, c in enumerate(class_labels): 
            Y[train_mask, i] = (y[train_mask] == c).astype(np.float64)
        return Y 

    def _compute_residuals(self, P, Y, train_mask): 
        E = np.zeros_like(P, dtype=np.float64)
        E[train_mask] = Y[train_mask] - P[train_mask]
        return E 
