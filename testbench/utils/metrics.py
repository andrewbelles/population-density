#!/usr/bin/env python3 
# 
# metrics.py  Andrew Belles  Jan 7th, 2026 
# 
# Computational helpers for returning metrics and scores from evaluations using the 
# resulting dataframes from cross validators 
# 

import numpy as np 

from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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

def majority_vote(pred_matrix): 
    out = np.empty(pred_matrix.shape[0], dtype=pred_matrix.dtype)
    for i, row in enumerate(pred_matrix):
        vals, counts = np.unique(row, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out 

def metrics_from_summary(summary): 
    row = summary.iloc[0]
    return {
        "accuracy": float(row["accuracy_mean"]) if "accuracy_mean" in row else np.nan, 
        "f1_macro": float(row["f1_macro_mean"]) if "f1_macro_mean" in row else np.nan,
        "roc_auc": float(row["roc_auc_mean"]) if "roc_auc_mean" in row else np.nan
    }

def metrics_from_preds(y_true, y_pred, class_labels=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro"))

    labels = (np.asarray(class_labels).reshape(-1) if class_labels is not None and 
        len(class_labels) else np.unique(y_true))
    labels = np.array(sorted(labels.tolist()))
    scores = np.zeros((y_pred.shape[0], labels.size), dtype=np.float64)
    for i, lbl in enumerate(labels): 
        scores[y_pred == lbl, i] = 1.0 
    roc = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=labels))

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": roc}

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

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "roc_auc": float(roc)
    }

def rank_by_label(results, labels): 
    by_label = {label: [] for label in labels}
    for item in results: 
        by_label[item["label"]].append(item)
    for _, items in by_label.items(): 
        items.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
    return by_label

def summarize_vif(vif_df): 
    values = vif_df.to_numpy() 
    finite = values[np.isfinite(values)]
    if finite.size == 0: 
        return "no finite vif values"

    return (
        f"median={np.median(finite):.3f}"
        f"p95={np.percentile(finite, 95):.3f}"
        f"max={finite.max():.3f}"
    )
