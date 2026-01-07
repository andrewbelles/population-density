#!/usr/env/python3 
# 
# plots.py  Andrew Belles  Jan 7th, 2026 
# 
# Plotting functionality that uses existing testbench modules 
# to pull information required for informative visuals 
# 

import argparse, io 
import matplotlib.pyplot as plt 

import numpy as np 

from dataclasses import dataclass 
from typing import Callable, Mapping

from sklearn.metrics import confusion_matrix 

import testbench.adjacency  as adjacency 
import testbench.downstream as downstream
import testbench.stacking   as stacking 

from testbench.utils.oof   import load_probs_labels_fips 
from testbench.utils.graph import coords_for_fips 
from testbench.utils.paths import MOBILITY_PATH

from testbench.utils.plotting import (
    get_labels,
    get_pred_labels,
    get_label_indices, 
    pick_variant, 
    save_or_show,
    call_plot 
)
from utils.helpers import project_path

# ---------------------------------------------------------
# Testbench Data Fetchers 
# ---------------------------------------------------------

def build_stacking_data(*, cross: str = "off", **_):

    buf = io.StringIO() 

    def _run(passthrough: bool): 
        return {
            "stacking": stacking.test_stacking(buf, passthrough)["metadata"],
            "cs": stacking.test_cs(buf, passthrough)["metadata"]
        }

    if cross == "both":
        return {"base": _run(False), "passthrough": _run(True)}
    return {("passthrough" if cross == "on" else "base"): _run(cross == "on")}

def build_adjacency_data(*, metric_keys=None, **_): 
    
    buf = io.StringIO()

    P, y, fips, _ = load_probs_labels_fips()
    coords        = coords_for_fips(MOBILITY_PATH, fips)

    graphs = [
        adjacency.test_queen_metrics(P, y, fips, coords, buf),
        adjacency.test_mobility_metrics(P, y, fips, coords, buf),
        adjacency.test_knn_metrics(P, y, fips, coords, buf)
    ]

    learned = []
    if metric_keys:
        learned = adjacency.test_learned_metrics(metric_keys, buf)

    return {"graphs": graphs, "learned": learned}

def build_downstream_data(*, cross: str = "off", **_): 

    buf = io.StringIO() 

    def _run(passthrough: bool): 
        return {
            "stacking": downstream.test_metric(buf, passthrough),
            "cs": downstream.test_cs(buf, passthrough)
        }

    if cross == "both":
        return {"base": _run(False), "passthrough": _run(True)}
    return {("passthrough" if cross == "on" else "base"): _run(cross == "on")}

# --------------------------------------------------------- 
# Plot Calls 
# ---------------------------------------------------------

'''
Stacking Plots 
'''

def plot_confusion(data): 
    _, data = pick_variant(data)
    meta    = data["stacking"]

    y_true  = np.asarray(meta["labels"]).reshape(-1)
    P       = meta["probs"]
    labels  = get_labels(meta["class_labels"], P.shape[1] if P.ndim > 1 else 2)
    y_pred  = get_pred_labels(P, labels)

    con_mat = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots() 
    im = ax.imshow(con_mat, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (Stacking)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return fig 

def plot_class_distance(data, log_hist: bool = False): 
    _, data = pick_variant(data)
    figs = {}
    for key, meta in (("stacking", data["stacking"]), ("cs", data["cs"])): 

        y_true  = np.asarray(meta["labels"]).reshape(-1)
        P       = meta["probs"]
        labels  = get_labels(meta["class_labels"], P.shape[1] if P.ndim > 1 else 2)
        y_pred  = get_pred_labels(P, labels)
        y_idx   = get_label_indices(y_true, labels)
        p_idx   = get_label_indices(y_pred, labels)
        dist    = p_idx - y_idx 

        fig, ax = plt.subplots() 
        ax.hist(dist, bins=25, alpha=0.8)
        ax.set_title(f"Signed Class Distance ({key})")
        ax.set_xlabel("pred - true")
        ax.set_ylabel("count")
        if log_hist: 
            ax.set_yscale("log")
        figs[f"stacking_class_distance_{key}"] = fig 

    return figs 

def plot_confidence_correctness(data): 
    _, data = pick_variant(data)
    meta    = data["stacking"]

    y_true  = np.asarray(meta["labels"]).reshape(-1)
    P       = meta["probs"]
    labels  = get_labels(meta["class_labels"], P.shape[1] if P.ndim > 1 else 2)
    y_pred  = get_pred_labels(P, labels)

    conf    = P.max(axis=1) if P.ndim > 1 else P.reshape(-1)
    correct = (y_pred == y_true)

    fig, ax = plt.subplots() 
    ax.hist(conf[correct], bins=25, alpha=0.7, label="correct")
    ax.hist(conf[~correct], bins=25, alpha=0.7, label="incorrect")
    ax.set_title("Confidence vs. Correctness")
    ax.set_xlabel("max probability")
    ax.set_ylabel("count")
    ax.legend()
    return fig 

'''
Adjacency Plots 
'''

def plot_graph_metric_bars(data, metric_key="avg_degree"): 
    items   = data["graphs"] + data.get("learned", [])
    names   = [d["name"] for d in items]
    values  = [d["metrics"].get(metric_key, np.nan) for d in items]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title(f"Graph Metric: {metric_key}")
    ax.set_xlabel("adjacency")
    ax.set_ylabel(metric_key)
    return fig 

def plot_degree_distribution(data, log_hist: bool = False): 
    items   = data["graphs"] + data.get("learned", [])
    fig, ax = plt.subplots() 

    for d in items:
        adj = d["adj"]
        deg = np.asarray(adj.sum(axis=1)).reshape(-1) 
        ax.hist(deg, bins=30, alpha=0.5, label=d["name"])

    ax.set_title("Degree Distribution")
    ax.set_xlabel("degree")
    ax.set_ylabel("count")
    if log_hist: 
        ax.set_xscale("log")
    ax.legend() 
    return fig 

def plot_edge_weight_distribution(data, log_hist: bool = False): 
    items   = data["graphs"] + data.get("learned", [])
    fig, ax = plt.subplots() 

    for d in items:
        w = d["adj"].data
        if w.size == 0: 
            continue 
        ax.hist(w, bins=30, alpha=0.5, label=d["name"])

    ax.set_title("Edge Weight Distribution")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    if log_hist: 
        ax.set_xscale("log")
    ax.legend() 
    return fig 

'''
Downstream Tests 
'''

def plot_edge_keep_ratio(data): 
    names   = []
    ratios  = []
    for variant, split in data.items(): 
        m = split["metric"]["metadata"]
        names.append(variant)
        ratios.append(m["edge_keep_ratio"])

    fig, ax = plt.subplots()
    ax.bar(names, ratios)
    ax.set_title("Edge Keep Ratio")
    ax.set_xlabel("variant")
    ax.set_ylabel("kept / base") 
    return fig 

def plot_edge_weight_hist(data): 
    fig, ax = plt.subplots() 
    for variant, split in data.items(): 
        w = split["metric"]["metadata"]["edge_weights"]
        if w.size == 0: 
            continue 
        ax.hist(w, bins=30, alpha=0.5, label=variant)

    ax.set_title("Learned Edge Weights")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.legend()
    return fig 

def plot_cs_distance(data, log_hist: bool = False): 
    fig, ax = plt.subplots() 
    for variant, split in data.items(): 
        m      = split["cs"]["metadata"]
        labels = get_labels(m["class_labels"], m["probs"].shape[1])
        y_true = np.asarray(m["labels"]).reshape(-1)

        for key, P in (("base", m["probs"]), ("cs", m["probs_corr"])): 
            y_pred = get_pred_labels(P, labels)
            y_idx  = get_label_indices(y_true, labels)
            p_idx  = get_label_indices(y_pred, labels)
            dist   = p_idx - y_idx 
            ax.hist(dist, bins=25, alpha=0.5, label=f"{variant}-{key}")

    ax.set_title("Signed Class Distance (Downstream C+S)")
    ax.set_xlabel("pred - true")
    ax.set_ylabel("count")
    if log_hist: 
        ax.set_yscale("log")
    ax.legend() 
    return fig 

@dataclass(frozen=True)
class PlotGroup: 
    name: str 
    build: Callable[..., dict]
    plots: Mapping[str, Callable[..., object]]


class Plotter: 
    
    def __init__(
        self, 
        group: PlotGroup, 
        *, 
        cross: str, 
        out_dir: str | None, 
        log_hist: bool = False, 
        **kwargs
    ): 
        self.group    = group 
        self.cross    = cross 
        self.out_dir  = out_dir 
        self.log_hist = log_hist  
        self.kwargs   = kwargs 

    def run(self, selected=None):
        data  = self.group.build(cross=self.cross, **self.kwargs)
        plots = self.group.plots if not selected else {k: self.group.plots[k] for k in selected}
        
        figs = {}
        for name, fn in plots.items(): 
            out = call_plot(fn, data, log_hist=self.log_hist)
            if isinstance(out, dict): 
                figs.update(out)
            else: 
                figs[name] = out 

        save_or_show(figs, self.out_dir)


# ---------------------------------------------------------
# Test Groups Definition 
# ---------------------------------------------------------

PLOT_GROUPS = {
    "stacking": PlotGroup(
        name="stacking",
        build=build_stacking_data,
        plots={
            "confusion": plot_confusion,
            "class_distance": plot_class_distance,
            "confidence": plot_confidence_correctness
        }
    ),
    "adjacency": PlotGroup(
        name="adjacency",
        build=build_adjacency_data,
        plots={
            "metric_corrective_edge_ratio": lambda d: plot_graph_metric_bars(
                d, "corrective_edge_ratio"
            ),
            "metric_recoverable_error_rate": lambda d: plot_graph_metric_bars(
                d, "recoverable_error_rate"
            ),
            "metric_distance_weighted_rer": lambda d: plot_graph_metric_bars(
                d, "distance_weighted_rer"
            ),
            "metric_locality_ratio": lambda d: plot_graph_metric_bars(
                d, "locality_ratio"
            ),
            "metric_avg_degree": lambda d: plot_graph_metric_bars(d, "avg_degree"),
            "metric_smoothness_gap": lambda d: plot_graph_metric_bars(d, "smoothness_gap"),
            "degree_dist": plot_degree_distribution,
            "weight_dist": plot_edge_weight_distribution,
        },
    ),
    "downstream": PlotGroup(
        name="downstream",
        build=build_downstream_data,
        plots={
            "edge_keep": plot_edge_keep_ratio,
            "edge_weights": plot_edge_weight_hist,
            "cs_distance": plot_cs_distance,
        },
    ),
}

# ---------------------------------------------------------
# Main Entry 
# --------------------------------------------------------- 

OUT_DIR = project_path("testbench", "images") 

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--group", choices=PLOT_GROUPS.keys(), required=True)
    parser.add_argument("--plots", nargs="*", default=None)
    parser.add_argument("--cross", choices=["off", "on", "both"], default="both")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--metric-keys", nargs="*", default=None)
    parser.add_argument("--log-hist", action="store_true")
    args = parser.parse_args()

    group   = PLOT_GROUPS[args.group]
    plotter = Plotter(
        group,
        cross=args.cross,
        out_dir=args.out,
        log_hist=args.log_hist, 
        metric_keys=args.metric_keys 
    )
    plotter.run()


if __name__ == "__main__": 
    main()
