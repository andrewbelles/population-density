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
    as_label_indices
) 
from sklearn.metrics import log_loss, cohen_kappa_score

OPT_TASK = TaskSpec("classification", ("f1_macro",))

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
        "log_loss": float(row["log_loss_mean"]) if "log_loss_mean" in row else np.nan,
        "brier": float(row["brier_mean"]) if "brier_mean" in row else np.nan,
        "ece": float(row["ece_mean"]) if "ece_mean" in row else np.nan,
        "qwk": float(row["qwk_mean"]) if "qwk_mean" in row else np.nan,
        "ord_mae": float(row["ord_mae_mean"]) if "ord_mae_mean" in row else np.nan,
    }

def metrics_from_probs(y_true, probs, class_labels): 
    y_true = np.asarray(y_true).reshape(-1)
    probs  = np.asarray(probs, dtype=np.float64)

    if probs.ndim == 1 or probs.shape[1] == 1: 
        p     = probs.reshape(-1) 
        preds = ( p >= 0.5 ).astype(int)
        roc   = roc_auc_score(y_true, p) if np.unique(y_true).size > 1 else float("nan")
    else: 
        if not np.allclose(probs.sum(axis=1), 1.0, rtol=1e-3, atol=1e-3):
            probs = _softmax_rows(probs)

        preds  = np.argmax(probs, axis=1)
        labels = class_labels if class_labels.size else np.unique(y_true)
        roc    = roc_auc_score(
            y_true,
            probs,
            multi_class="ovr",
            average="macro",
            labels=labels
        )

    y_idx, labels = as_label_indices(y_true)
    p_idx = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "roc_auc": float(roc),
        "log_loss": float(log_loss(y_idx, probs, labels=np.arange(labels.size))) 
        if probs.ndim > 1 else float(log_loss(y_true, probs)),
        "brier": brier_multiclass(probs, y_idx, labels.size) if probs.ndim > 1 
        else float(np.mean((probs - y_true) ** 2)),
        "ece": ece(probs, y_idx),
        "qwk": float(cohen_kappa_score(y_idx, p_idx, weights="quadratic")),
        "ord_mae": float(np.mean(np.abs(y_idx - p_idx))),
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
