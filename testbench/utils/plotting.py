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

def get_test_mask(meta): 
    train_mask = meta.get("train_mask")
    return (~np.asarray(train_mask) if train_mask is not None else slice(None))

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

def pairwise_payload(vdata): 
    result = vdata["pairwise"]["result"]
    vif    = np.asarray(result["vif"], dtype=np.float64)
    names  = np.asarray(result["vif_index"], dtype="U128")
    groups = result.get("feature_groups")
    return vif, names, groups 

def full_payload(vdata): 
    result = vdata["full"]["result"]
    vif    = np.asarray(result["vif"], dtype=np.float64)
    names  = np.asarray(result["vif_index"], dtype="U128")
    return vif, names

def short_group(name: str) -> str: 
    n = str(name).upper() 
    return {
        "VIIRS": "VIIRS",
        "TIGER": "TIGER",
        "NLCD": "NLCD",
        "COORDS": "COORDS",
        "CROSS": "CROSS",
        "PASSTHROUGH": "PASS",
        "OOF": "OOF"
    }.get(n, n[:6])

def short_label(name: str) -> str: 
    s = str(name)
    if "::" in s: 
        prefix, rest = s.split("::", 1)
    elif "__" in s: 
        prefix, rest = s.split("__", 1)
    else: 
        return s 
    p = {
        "VIIRS": "V",
        "TIGER": "T",
        "NLCD": "N",
        "COORDS": "C",
        "CROSS": "X",
        "PASSTHROUGH": "P",
        "OOF": "O"
    }.get(prefix.upper(), prefix[:1].upper())
    rest = rest.replace("__", "/")
    return f"{p}:{rest}"

def ordered_groups(groups): 
    uniq    = list(dict.fromkeys(groups))
    pref    = {k: i for i, k in enumerate(GROUP_ORDER)}
    known   = [g for g in uniq if str(g).upper() in pref]
    known.sort(key=lambda g: pref[str(g).upper()])
    unknown = [g for g in uniq if str(g).upper() not in pref]
    return known + unknown 

def reorder_by_group(vif, names, groups): 
    if groups is None: 
        return vif, names, groups, None 
    order = ordered_groups(groups)
    idx   = []
    for g in order: 
        idx.extend(np.where(groups == g)[0].tolist())
    idx = np.asarray(idx, dtype=int)
    return vif[np.ix_(idx, idx)], names[idx], groups[idx], order 

def group_spans(groups): 
    spans = []
    if groups is None or len(groups) == 0:
        return spans 
    start = 0 
    for i in range(1, len(groups) + 1): 
        if i == len(groups) or groups[i] != groups[start]: 
            spans.append((start, i, groups[start]))
            start = i 
    return spans 

def set_group_ticks(ax, spans, axis="x"): 
    centers = [(s + e - 1) / 2 for s, e, _ in spans]
    labels  = [short_group(g) for _, _, g in spans]
    if axis == "x": 
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else: 
        ax.set_yticks(centers)
        ax.set_yticklabels(labels)

def pair_label(a: str, b: str) -> str: 
    return f"{clean_feature(a)} - {clean_feature(b)}"

def format_feature(name: str, max_len: int = 34) -> str: 
    s = str(name).replace("::", ":").replace("__", "/")
    if len(s) <= max_len: 
        return s 
    return s[: max_len - 3] + "..."

def format_pair(a: str, b: str, max_len: int = 28) -> str: 
    return f"{format_feature(a, max_len)}\n{format_feature(b, max_len)}"

def variant_items(data): 
    if isinstance(data, dict) and ("base" in data or "passthrough" in data): 
        for key in ("base", "passthrough"): 
            if key in data: 
                yield key, data[key]
    else: 
        yield "base", data 
