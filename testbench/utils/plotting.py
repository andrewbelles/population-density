#!/usr/bin/env python3 
# 
# plotting.py  Andrew Belles  Jan 7th, 2026 
# 
# Plot helper functions for testbench visualizations 
# 
# 

import numpy as np 

from inspect import signature, Parameter 

from pathlib import Path

import matplotlib.pyplot as plt 

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
