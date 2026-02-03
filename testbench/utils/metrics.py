#!/usr/bin/env python3 
# 
# metrics.py  Andrew Belles  Jan 7th, 2026 
# 
# Computational helpers for returning metrics and scores from evaluations using the 
# resulting dataframes from cross validators 
# 

import numpy as np 

import pandas as pd 

from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from analysis.cross_validation import (
    TaskSpec,
    brier_multiclass,
    ece,
    as_label_indices,
    ranked_probability_score
) 
from sklearn.metrics import log_loss, cohen_kappa_score

from sklearn.decomposition       import PCA 

from sklearn.cross_decomposition import CCA 

from sklearn.feature_selection   import mutual_info_regression

OPT_TASK = TaskSpec("classification", ("qwk",))

def _softmax_rows(probs: NDArray) -> NDArray: 
    probs   = np.asarray(probs, dtype=np.float64)
    row_max = probs.max(axis=1, keepdims=True)
    exp     = np.exp(probs - row_max)
    denom  = exp.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0 
    return exp / denom 

def best_score(metrics): 
    for key in ("f1_macro", "accuracy", "roc_auc"): 
        val = metrics.get(key, np.nan)
        if not np.isnan(val): 
            return float(val)
    return float("-inf")

def score_from_summary(summary): 
    row = summary.iloc[0]
    for key in ("f1_macro_mean", "accuracy_mean", "roc_auc_mean", "r2_mean"): 
        if key in row and not np.isnan(row[key]): 
            return float(row[key])
    return float("nan")

def metrics_from_summary(summary): 
    row = summary.iloc[0]
    return {
        "accuracy": float(row["accuracy_mean"]) if "accuracy_mean" in row else np.nan, 
        "f1_macro": float(row["f1_macro_mean"]) if "f1_macro_mean" in row else np.nan,
        "roc_auc": float(row["roc_auc_mean"]) if "roc_auc_mean" in row else np.nan,
        "ece": float(row["ece_mean"]) if "ece_mean" in row else np.nan,
        "qwk": float(row["qwk_mean"]) if "qwk_mean" in row else np.nan,
        "rps": float(row["rps_mean"]) if "rps_mean" in row else np.nan 
    }


def metrics_from_probs(y_true, probs, class_labels):
    y_true = np.asarray(y_true).reshape(-1)
    probs  = np.asarray(probs, dtype=np.float64)

    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        p = probs.reshape(-1)
        probs = np.column_stack([1.0 - p, p])

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2d (n, c), got {probs.shape}")

    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, row_sum, out=np.zeros_like(probs), where=row_sum != 0)

    K = probs.shape[1]

    if class_labels is not None and np.asarray(class_labels).size == K:
        labels = np.asarray(class_labels)
        idx_map = {v: i for i, v in enumerate(labels)}
        y_idx = np.array([idx_map[v] for v in y_true], dtype=np.int64)
    else:
        labels = np.arange(K)
        y_idx = y_true.astype(np.int64)

    preds = np.argmax(probs, axis=1)

    counts = np.bincount(y_idx, minlength=K)
    weights = counts.max() / np.clip(counts, 1, None)

    if K == 2:
        roc = roc_auc_score(y_idx, probs[:, 1])
    else:
        try:
            roc = roc_auc_score(
                y_idx,
                probs,
                multi_class="ovr",
                average="macro",
                labels=np.arange(K)
            )
        except ValueError:
            roc = np.nan

    return {
        "accuracy": float(accuracy_score(y_idx, preds)),
        "f1_macro": float(f1_score(y_idx, preds, average="macro")),
        "roc_auc": float(roc),
        "ece": ece(probs, y_idx),
        "qwk": float(cohen_kappa_score(y_idx, preds, weights="quadratic")),
        "rps": ranked_probability_score(
            y_true, 
            probs, 
            class_labels=labels, 
            normalize=True,
            class_weights=weights
        ),
    }

def rank_by_label(results, labels): 
    by_label = {label: [] for label in labels}
    for item in results: 
        by_label[item["label"]].append(item)
    for _, items in by_label.items(): 
        items.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
    return by_label

def summarize_boruta(path: str) -> pd.DataFrame: 
    df       = pd.read_csv(path)
    required = {"status", "group", "feature"}
    missing  = required - set(df.columns)
    if missing: 
        raise ValueError(f"boruta summary missing columns: {sorted(missing)}")
    return df 

def linear_cka(X, Y): 
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    xy = Xc.T @ Yc 
    xx = Xc.T @ Xc 
    yy = Yc.T @ Yc 
    num = (xy * xy).sum() 
    den = np.sqrt((xx * xx).sum() * (yy * yy).sum()) + 1e-12 
    return float(num / den)

def cca_score(X, Y, n_components=3):
    n = X.shape[0]
    k = min(n_components, X.shape[1], Y.shape[1], n - 1) 
    if k <= 0: 
        return float("nan")
    cca = CCA(n_components=k, max_iter=1000)
    Xc, Yc = cca.fit_transform(X, Y)
    corrs = [np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(k)]
    return float(np.nanmean(np.abs(corrs)))

def block_mi(X, Y, n_components=1, random_state=0): 
    if X.shape[1] > n_components:
        Xr = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    else: 
        Xr = X 

    if Y.shape[1] > n_components:
        Yr = PCA(n_components=n_components, random_state=random_state).fit_transform(Y)
    else: 
        Yr = Y 

    mi_xy = mutual_info_regression(Xr, Yr[:, 0], random_state=random_state)
    mi_yx = mutual_info_regression(Yr, Xr[:, 0], random_state=random_state)
    return float(0.5 * (np.mean(mi_xy) + np.mean(mi_yx)))

def distance_correlation(X, Y): 
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    a = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    b = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)

    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean() 
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean() 

    dcov  = (A * B).mean() 
    dvarx = (A * A).mean() 
    dvary = (B * B).mean() 

    denom = np.sqrt(dvarx * dvary) + 1e-12 
    return float(dcov / denom)

def moran_i(x: NDArray, W) -> float:

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size 

    if n == 0 or W.nnz == 0: 
        return float("nan")
    x = x - x.mean() 
    denom = np.dot(x, x)
    if denom == 0.0: 
        return 0.0 

    S0 = W.sum()
    num = x @ (W @ x) 
    return float((n / S0) * (num / denom))
