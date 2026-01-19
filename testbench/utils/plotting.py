#!/usr/bin/env python3 
# 
# plotting.py  Andrew Belles  Jan 7th, 2026 
# 
# Plot helper functions for testbench visualizations 
# 
# 

import re 

import numpy as np 

from inspect import signature, Parameter 

from pathlib import Path

import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix 

GROUP_ORDER = ("VIIRS", "TIGER", "NLCD", "CROSS", "PASSTHROUGH", "OOF")

palette = [
    "#0B1F3B",  # navy
    "#145A6A",  # deep teal
    "#1F7A3F",  # forest green
    "#B8860B",  # dark goldenrod
    "#A64B00",  # burnt orange
    "#7A1E1E",  # deep red
]

def get_labels(class_labels, n_classes: int): 
    labels = np.asarray(class_labels).reshape(-1)
    if labels.size == 0: 
        return np.arange(n_classes, dtype=int)
    return labels 

def get_pred_labels(P, class_labels): 
    P = np.asarray(P)
    if P.ndim == 1 or P.shape[1] == 1: 
        labels = get_labels(class_labels, 2)
        return np.where(P.reshape(-1) >= 0.5, labels[1], labels[0])
    labels = get_labels(class_labels, P.shape[1])
    return labels[np.argmax(P, axis=1)]

def get_label_indices(y, labels): 
    idx = {lbl: i for i, lbl in enumerate(labels)}
    return np.array([idx[v] for v in y], dtype=int)

def pick_variant(data): 
    for key in ("base", "passthrough"): 
        if key in data: 
            return key, data[key]
    return next(iter(data.items()))

def save_or_show(figs, out_dir: str | None): 
    if out_dir: 
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            if not fig.get_constrained_layout(): 
                fig.tight_layout() 
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)
    else: 
        for fig in figs.values(): 
            fig.tight_layout() 
        plt.show()

def call_plot(fn, data, **kwargs): 
    params = signature(fn).parameters 
    if any(p.kind == Parameter.VAR_KEYWORD for p in params.values()):
        return fn(data, **kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params} 
    return fn(data, **filtered)

def apply_metric_ylim(ax, values): 
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: 
        return 
    if np.max(np.abs(vals)) > 1.0: 
        return 
    if np.any(vals < 0): 
        ax.set_ylim(-1.0, 1.0)
    else: 
        ax.set_ylim(0.0, 1.0)

def confusion_panel(ax, y_true, P, labels, title, display_labels=None): 

    cm      = confusion_matrix(y_true, get_pred_labels(P, labels), labels=labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1 
    cm_pct  = cm / row_sum

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(cm.shape[0]): 
        for j in range(cm.shape[1]): 
            ax.text(
                j, i,
                f"{cm[i, j]}\n{cm_pct[i, j]*100:.1f}%",
                ha="center", va="center", fontsize=8
            )

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    if display_labels is not None: 
        ax.set_xticklabels(display_labels)
        ax.set_yticklabels(display_labels)
    return im 

def confidence_hist(ax, y_true, P, labels, title): 
    P = np.asarray(P)
    if P.ndim > 1: 
        row_sum = P.sum(axis=1, keepdims=True)
        if not np.allclose(row_sum, 1.0, rtol=1e-3, atol=1e-3): 
            row_sum[row_sum == 0] = 1.0 
            P = P / row_sum 
        conf = P.max(axis=1)
    else: 
        conf = np.clip(P.reshape(-1), 0.0, 1.0)

    y_pred  = get_pred_labels(P, labels)
    correct = (y_pred == y_true)

    ax.hist(conf[correct], bins=25, alpha=0.7, label="correct")
    ax.hist(conf[~correct], bins=25, alpha=0.7, label="incorrect")
    ax.set_title(title)
    ax.set_xlabel("max probability")
    ax.set_ylabel("count")
    ax.legend()
