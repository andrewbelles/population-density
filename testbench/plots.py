#!/usr/bin/env python3 
# 
# plots.py  Andrew Belles  Jan 7th, 2026 
# 
# Plotting functionality that uses existing testbench modules 
# to pull information required for informative visuals 
# 

import argparse, io

from pathlib import Path 

import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm 

from matplotlib.lines  import Line2D

import numpy as np 

import geopandas 

from shapely import line_interpolate_point
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass 
from typing import Callable, Mapping

from preprocessing.loaders import load_oof_predictions

import testbench.adjacency    as adjacency 

import testbench.stacking     as stacking 

import testbench.eval         as eval 

from testbench.utils.oof   import load_probs_labels_fips 

from testbench.utils.graph import coords_for_fips 

from testbench.utils.paths import (
    MOBILITY_PATH,
    SHAPEFILE,
    check_paths_exist 
) 

from testbench.utils.plotting import (
    apply_metric_ylim,
    get_labels,
    get_pred_labels,
    get_label_indices,
    pick_variant,
    save_or_show,
    call_plot,
    confusion_panel,
    confidence_hist,
    palette 
)

from utils.helpers import (
    project_path,
    _mat_str_vector,
    _mat_scalar
)

from scipy.io import loadmat 


# Silence-able logging 
def _log(msg, quiet=False): 
    if not quiet: 
        print(msg)

def _require_expert_probs(data): 
    items = data.get("expert_probs") or []
    if not items: 
        raise ValueError("no expert_probs. run with --cached-experts and --datasets")
    return items 

# ---------------------------------------------------------
# Testbench Data Fetchers 
# ---------------------------------------------------------

def build_stacking_data(
    *, 
    cross: str = "off", 
    datasets: list[str] | None = None, 
    plots=None,
    cached_experts: bool = False, 
    **_
):

    need_experts  = (plots is None) or ("expert_confusion" in plots)
    need_stacking = (plots is None) or any(
        p in {"confusion", "class_distance", "confidence", "map_confidence"}
        for p in plots 
    ) 

    out = {}

    if need_experts: 
        expert_data  = stacking.test_expert_oof(
            datasets=datasets,
            cached=cached_experts 
        )
        out["experts"] = expert_data["experts"]

    if not need_stacking: 
        return out 

    def _run(passthrough: bool): 
        return {
            "stacking": stacking.test_stacking(passthrough)["metadata"],
            "cs": stacking.test_cs_opt(passthrough)["metadata"]
        }

    if cross == "both":
        out.update({"base": _run(False), "passthrough": _run(True)})
    else:
        out.update({("passthrough" if cross == "on" else "base"): _run(cross == "on")})
    return out 

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

def build_embedding_data(
    *, 
    embedding_dir: str | None = None, 
    embedding_paths=None,
    datasets: list[str] | None = None, 
    cached_experts: bool = False, 
    **_
): 
    if embedding_paths is None: 
        base  = embedding_dir or project_path("testbench", "local")
        paths = sorted(Path(base).glob("*.mat")) 
    else: 
        paths = [Path(p) for p in embedding_paths]

    entries = []
    for p in paths: 
        mat = loadmat(p)
        if "coords" not in mat or "labels" not in mat: 
            continue 

        coords = np.asarray(mat["coords"], dtype=np.float64)
        labels = np.asarray(mat["labels"]).reshape(-1).astype(np.int64)

        name   = mat.get("name")
        if name is None: 
            name = p.stem 
        else: 
            name = _mat_str_vector(name)[0]

        entries.append({
            "name": name, 
            "coords": coords,
            "labels": labels,
            "n_neighbors": int(_mat_scalar(mat.get("n_neighbors"))),
            "min_dist": float(_mat_scalar(mat.get("min_dist")))
        })

    expert_probs = []
    if cached_experts:
        if not datasets: 
            raise ValueError("cached_experts requires --datasets")
        expert_data = stacking.test_expert_oof(datasets=datasets, cached=True)
        paths = expert_data["experts"]
        check_paths_exist(list(paths.values()), "cached expert prob files")

        for name, path in paths.items(): 
            oof = load_oof_predictions(path) 
            probs = np.asarray(oof["probs"], dtype=np.float64)
            if probs.ndim == 3: 
                P = probs[:, 0, :] if probs.shape[1] == 1 else probs.mean(axis=1)
            else: 
                P = probs 
            expert_probs.append({
                "name": name, 
                "probs": P, 
                "labels": np.asarray(oof["labels"]).reshape(-1), 
                "class_labels": np.asarray(oof.get("class_labels", []))
            })

    return {"embeddings": entries, "expert_probs": expert_probs}

def build_sweep_data(
    *,
    dataset: str = "SAIPE", 
    **_ 
): 
    out = eval.test_scalar_ens_sweep(
        _buf=None, 
        dataset=dataset 
    ) 
    return out 

# --------------------------------------------------------- 
# Stacking  
# ---------------------------------------------------------

def plot_confusion(data): 
    _, data   = pick_variant(data)
    meta_b    = data["stacking"]
    meta_cs   = data["cs"]
    y_true    = np.asarray(meta_b["labels"]).reshape(-1)
    labels    = get_labels(meta_b["class_labels"], meta_b["probs"].shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True, constrained_layout=True)
    im = confusion_panel(axes[0], y_true, meta_b["probs"], labels, "Base")
    confusion_panel(axes[1], y_true, meta_cs["probs_corr"], labels, "C+S")

    fig.colorbar(im, ax=axes.ravel().tolist())
    fig.suptitle("Confusion Matrices (Row %)")
    return fig 

def plot_expert_confusion(data): 
    experts   = data.get("experts")
    if experts is None: 
        _, variant = pick_variant(data)
        experts    = variant.get("experts")
    if experts is None: 
        raise ValueError("missing experts. build_stacking_data must include test_expert_oof")

    names     = list(experts.keys())
    paths     = [experts[n] for n in names]
    fig, axes = plt.subplots(
        1, len(paths),
        figsize=(4 * len(paths), 4),
        sharey=True,
        constrained_layout=True
    )
    if len(paths) == 1: 
        axes = [axes]

    im = None 
    for ax, name, path in zip(axes, names, paths): 
        oof    = load_oof_predictions(path)
        probs  = np.asarray(oof["probs"], dtype=np.float64)
        if probs.ndim == 3: 
            P  = probs[:, 0, :] if probs.shape[1] == 1 else probs.mean(axis=1)
        else: 
            P  = probs 
        y_true = np.asarray(oof["labels"]).reshape(-1)
        class_labels = np.asarray(oof.get("class_labels", []))
        labels = get_labels(class_labels, P.shape[1] if P.ndim > 1 else 2)

        im = confusion_panel(ax, y_true, P, labels, title=name)

    if im is not None: 
        fig.colorbar(im, ax=axes)
    fig.suptitle("Expert Confusion Matrices (Row %)")
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
    _, data    = pick_variant(data)
    meta_b     = data["stacking"]
    meta_cs    = data["cs"]

    y_true     = np.asarray(meta_b["labels"]).reshape(-1)
    labels     = get_labels(meta_b["class_labels"], meta_b["probs"].shape[1])

    train_mask = meta_cs.get("train_mask")
    if train_mask is not None: 
        test_mask = ~np.asarray(train_mask)
        y_true    = y_true[test_mask]
        P_base    = meta_b["probs"][test_mask]
        P_cs      = meta_cs["probs_corr"][test_mask]
    else:
        P_base    = meta_b["probs"]
        P_cs      = meta_cs["probs_corr"]

    fig, axes  = plt.subplots(1, 2, figsize=(10,7), sharey=True)
    confidence_hist(axes[0], y_true, P_base, labels, "Base")
    confidence_hist(axes[1], y_true, P_cs, labels, "C+S")

    fig.suptitle("Confidence vs. Correctness (Test Samples)")
    return fig 

def plot_map_predictions(data): 
    EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}

    _, data   = pick_variant(data)
    meta_b    = data["stacking"]
    meta_cs   = data["cs"]

    fips      = np.asarray(meta_b["fips"]).reshape(-1)
    y_true    = np.asarray(meta_b["labels"]).reshape(-1)
    labels    = get_labels(meta_b["class_labels"], meta_b["probs"].shape[1]) 
    gdf       = geopandas.read_file(SHAPEFILE)
    gdf["GEOID"] = gdf["GEOID"].astype("U5")
    gdf       = gdf[~gdf["STATEFP"].isin(EXCLUDE_STATEFP)]

    classes   = np.unique(y_true)
    label_map = {c: l for c, l in zip(classes, labels)}
    y_true    = np.array([label_map[v] for v in y_true], dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), constrained_layout=True)
    def _plot(ax, values, title, show_legend=False): 
        mapping = {f: int(v) for f, v in zip(fips, values)}
        view    = gdf.assign(pred=gdf["GEOID"].map(mapping))
        view.plot(column="pred", categorical=True, legend=show_legend,
                  missing_kwds={"color": "lightgrey"}, ax=ax)
        ax.set_title(title)
        ax.axis("off")

    _plot(axes[0], get_pred_labels(meta_b["probs"], labels), "Base")
    _plot(axes[1], get_pred_labels(meta_cs["probs_corr"], labels), "C+S")
    _plot(axes[2], y_true, "True", show_legend=True)

    return fig 

# --------------------------------------------------------- 
# Adjacency  
# ---------------------------------------------------------

def plot_graph_metric_bars(data, metric_key="avg_degree"): 
    items   = data["graphs"] + data.get("learned", [])
    names   = [d["name"] for d in items]
    values  = [d["metrics"].get(metric_key, np.nan) for d in items]
    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title(f"Graph Metric: {metric_key}")
    ax.set_xlabel("adjacency")
    ax.set_ylabel(metric_key)

    apply_metric_ylim(ax, values)
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

# --------------------------------------------------------- 
# Embeddings Group  
# ---------------------------------------------------------

def plot_embedding_umap(data): 
    entries = data.get("embeddings", [])
    if not entries: 
        raise ValueError("no embedding PCA data to plot")

    all_coords = np.vstack([e["coords"][:, :2] for e in entries])
    mins = all_coords.min(axis=0)
    maxs = all_coords.max(axis=0)
    pad  = 0.05 * (maxs- mins + 1e-9)
    mins -= pad 
    maxs += pad 

    n = len(entries)
    fig = plt.figure(figsize=(5.2 * n, 4.5), constrained_layout=True)

    for i, item in enumerate(entries, start=1): 
        coords = np.asarray(item["coords"][:, :2])
        labels = np.asarray(item["labels"]).reshape(-1)
        uniq   = np.unique(labels)
        idx    = np.searchsorted(uniq, labels)

        cmap   = ListedColormap(palette) 
        norm   = BoundaryNorm(np.arange(len(palette) + 1) - 0.5, len(palette))

        ax     = fig.add_subplot(1, n, i)
        sc     = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=idx, cmap=cmap, norm=norm,
            alpha=0.7, linewidths=0
        )

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.grid(False)

        title = item["name"].strip()
        nn = item.get("n_neighbors")
        md = float(item.get("min_dist"))
        if nn is not None and md is not None:
            title += f" (UMAP nn={nn}, min_dist={md:.2f})"
        ax.set_title(title)

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

        if i == n: 
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(np.arange(len(uniq)))
            cbar.set_ticklabels([str(u) for u in uniq])
            cbar.set_label("Class")

    return fig 

def plot_qwk_penalty(data): 
    items = _require_expert_probs(data)
    n     = len(items)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True, constrained_layout=True)
    if n == 1: 
        axes = [axes]
    im = None 
    for ax, item in zip(axes, items): 
        P = item["probs"]
        y_true = item["labels"] 
        labels = get_labels(item["class_labels"], P.shape[1] if P.ndim > 1 else 2)
        y_pred = get_pred_labels(P, labels)
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        k      = len(labels)
        i, j   = np.indices((k, k))
        weights = (i - j)**2 
        penalty = cm * weights 
        im = ax.imshow(penalty, cmap="Reds")
        for r in range(k): 
            for c in range(k): 
                if cm[r, c] > 0: 
                    ax.text(c, r, f"{cm[r,c]}", ha="center", va="center", fontsize=8)

        ax.set_title(item["name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(k))
        ax.set_yticks(np.arange(k))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    if im is not None: 
        fig.colorbar(im, ax=axes)
    fig.suptitle("QWK Penalty Contribution")
    return fig 

def plot_ordinal_distribution(data): 
    items = _require_expert_probs(data)
    n     = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, constrained_layout=True)
    if n == 1: 
        axes = [axes]

    for ax, item in zip(axes, items): 
        P = item["probs"]
        y_true = item["labels"] 
        labels = get_labels(item["class_labels"], P.shape[1] if P.ndim > 1 else 2)

        k = len(labels)
        idx = np.arange(k, dtype=np.float64)

        if P.ndim == 1: 
            preds = P.reshape(-1)
        else: 
            preds = (P * idx).sum(axis=1)

        y_idx = get_label_indices(y_true, labels)

        dist_per_class = []
        positions      = []
        for i in range(k): 
            vals = preds[y_idx == i]
            if vals.size > 0: 
                dist_per_class.append(vals)
                positions.append(i)

        ax.plot([0, k-1], [0, k-1], color="green", linestyle="--", alpha=0.5, label="Ideal")

        box = ax.boxplot(
            dist_per_class,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5}
        )

        for patch in box["boxes"]: 
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.6)
        
        ax.set_title(item["name"])
        ax.set_xlabel("True Class")
        ax.set_ylabel("Prediced Score")
        
        ax.set_xticks(np.arange(k))
        ax.set_xticklabels(labels)

        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(-0.5, k - 0.5)

    fig.suptitle("Ordinal Prediction Distribution")
    return fig 

# ---------------------------------------------------------
# Evaluation Plots 
# ---------------------------------------------------------

def plot_ens_beta_sweep(data): 
    betas    = np.asarray(data["betas"], dtype=np.float64)
    qwk_2013 = np.asarray(data["qwk_2013"], dtype=np.float64)
    qwk_2023 = np.asarray(data["qwk_2023"], dtype=np.float64)
    dataset  = data.get("dataset", "Unknown")

    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(betas, qwk_2013, color="blue", linewidth=0.8, alpha=0.7, label="Optimized QWK (2013)")
    ax.plot(betas, qwk_2023, color="green", linewidth=1.0, label="Generalized QWK (2023)")
    ax.set_xlabel(r"ENS $\beta$")
    ax.set_ylabel("QWK")
    ax.set_title(r"ENS $\beta$ " + f"({dataset})")
    ax.grid(True, alpha=0.25)
    ax.legend() 
    return fig 

# ---------------------------------------------------------
# Plot interface 
# ---------------------------------------------------------

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
        quiet: bool = False, 
        **kwargs
    ): 
        self.group    = group 
        self.cross    = cross 
        self.out_dir  = out_dir 
        self.log_hist = log_hist
        self.quiet    = quiet 
        self.kwargs   = kwargs 

    def run(self, selected=None):
        _log(f"[{self.group.name}] build data", quiet=self.quiet)
        data  = self.group.build(cross=self.cross, plots=selected, **self.kwargs)
        plots = self.group.plots if not selected else {k: self.group.plots[k] for k in selected}
        _log(f"[{self.group.name}] plots: {len(plots)}", quiet=self.quiet)

        figs = {}
        for name, fn in plots.items(): 
            _log(f"[{self.group.name}] plot {name}", quiet=self.quiet)
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
            "expert_confusion": plot_expert_confusion,
            "class_distance": plot_class_distance,
            "confidence": plot_confidence_correctness,
            "map_confidence": plot_map_predictions
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
    "embeddings": PlotGroup(
        name="embeddings",
        build=build_embedding_data,
        plots={
            "manifold_umap": plot_embedding_umap,
            "qwk_penalty": plot_qwk_penalty,
            "ordinal_distribution": plot_ordinal_distribution
        }
    ), 
    "sweep": PlotGroup(
        name="sweep",
        build=build_sweep_data,
        plots={
            "ens_beta_qwk": plot_ens_beta_sweep
        }
    )
}

# ---------------------------------------------------------
# Main Entry 
# --------------------------------------------------------- 

OUT_DIR = project_path("testbench", "images") 

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--group", choices=PLOT_GROUPS.keys(), default="all")
    parser.add_argument("--plots", nargs="*", default=None)
    parser.add_argument("--cross", choices=["off", "on", "both"], default="both")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--metric-keys", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--log-hist", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--embedding-paths", nargs="*", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--cached-experts", action="store_true")
    args = parser.parse_args()

    plot_kwargs = dict(
        cross=args.cross, 
        out_dir=args.out,
        log_hist=args.log_hist, 
        quiet=args.quiet, 
        datasets=args.datasets,
        dataset=args.dataset, 
        metric_keys=args.metric_keys,
        embedding_paths=args.embedding_paths,
        cached_experts=args.cached_experts
    )

    if args.group == "all": 
        for group in PLOT_GROUPS.values(): 
            plotter = Plotter(
                group,
                **plot_kwargs 
            )
            plotter.run(selected=args.plots)
    else: 
        group   = PLOT_GROUPS[args.group]
        plotter = Plotter(
            group,
            **plot_kwargs 
        )
        plotter.run(selected=args.plots)


if __name__ == "__main__": 
    main()
