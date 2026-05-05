#
# visualizations.py  Andrew Belles  May 4th, 2026
#
# Plotting helpers for synthetic graph-spectral experiments.
#

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "population_estimation_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic.common import SyntheticBagDataset, ensure_dir


def _save(fig: plt.Figure, path: str | Path) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pad_axis(ax: plt.Axes, x: np.ndarray, y: np.ndarray, pad_frac: float = 0.08) -> None:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xpad = max((xmax - xmin) * float(pad_frac), 0.25)
    ypad = max((ymax - ymin) * float(pad_frac), 0.25)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")


def plot_latent_signal(dataset: SyntheticBagDataset, path: str | Path) -> None:
    h = np.asarray(dataset.h, dtype=np.float64)
    residual = np.asarray(dataset.residual, dtype=np.float64)
    signal_frac = np.asarray([np.mean(mask) for mask in dataset.signal_masks], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), constrained_layout=True)
    sc0 = axes[0].scatter(
        h[:, 0],
        h[:, 1],
        c=residual,
        s=26,
        cmap="coolwarm",
        alpha=0.88,
        linewidths=0,
    )
    axes[0].set_title("Latent residual surface", fontsize=12)
    axes[0].set_xlabel("latent h1")
    axes[0].set_ylabel("latent h2")
    _pad_axis(axes[0], h[:, 0], h[:, 1])
    cb0 = fig.colorbar(sc0, ax=axes[0], label="residual", shrink=0.86)
    cb0.ax.tick_params(labelsize=8)
    sc1 = axes[1].scatter(
        h[:, 0],
        h[:, 1],
        c=signal_frac,
        s=26,
        cmap="viridis",
        alpha=0.88,
        linewidths=0,
    )
    axes[1].set_title("Rare signal concentration", fontsize=12)
    axes[1].set_xlabel("latent h1")
    axes[1].set_ylabel("latent h2")
    _pad_axis(axes[1], h[:, 0], h[:, 1])
    cb1 = fig.colorbar(sc1, ax=axes[1], label="signal fraction", shrink=0.86)
    cb1.ax.tick_params(labelsize=8)
    _save(fig, path)


def plot_graph_edge_map(
    dataset: SyntheticBagDataset,
    weights: np.ndarray,
    path: str | Path,
    *,
    title: str,
    max_edges: int = 900,
) -> None:
    h = np.asarray(dataset.h, dtype=np.float64)
    residual = np.asarray(dataset.residual, dtype=np.float64)
    w = 0.5 * (np.asarray(weights, dtype=np.float64) + np.asarray(weights, dtype=np.float64).T)
    np.fill_diagonal(w, 0.0)
    rows, cols = np.triu_indices(w.shape[0], k=1)
    vals = w[rows, cols]
    mask = np.isfinite(vals) & (vals > 0)
    rows = rows[mask]
    cols = cols[mask]
    vals = vals[mask]
    if vals.size > int(max_edges):
        keep = np.argpartition(vals, kth=vals.size - int(max_edges))[-int(max_edges) :]
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]
    if vals.size:
        denom = max(float(np.max(vals)), 1e-12)
        order = np.argsort(vals)
        rows = rows[order]
        cols = cols[order]
        vals = vals[order] / denom
    fig, ax = plt.subplots(figsize=(7.8, 7.2), constrained_layout=True)
    for i, j, value in zip(rows, cols, vals, strict=False):
        ax.plot(
            [h[i, 0], h[j, 0]],
            [h[i, 1], h[j, 1]],
            color="#5f6470",
            alpha=float(0.035 + 0.16 * value),
            linewidth=float(0.20 + 0.95 * value),
            zorder=1,
        )
    sc = ax.scatter(h[:, 0], h[:, 1], c=residual, s=24, cmap="coolwarm", linewidths=0, alpha=0.92, zorder=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("latent h1")
    ax.set_ylabel("latent h2")
    _pad_axis(ax, h[:, 0], h[:, 1])
    cb = fig.colorbar(sc, ax=ax, label="residual", shrink=0.88)
    cb.ax.tick_params(labelsize=8)
    _save(fig, path)


def plot_trial_method_error(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    path: str | Path,
    *,
    top_n: int = 12,
    title: str = "Residual error by synthetic trial",
) -> None:
    if fold_metrics.empty or summary.empty:
        return
    if "method_label" not in summary.columns:
        raise ValueError("summary must include method_label")
    label_by_method = dict(zip(summary["method"], summary["method_label"], strict=False))
    ranked = [str(m) for m in summary["method"].tolist() if str(m) != "baseline"]
    methods = ranked[: max(1, int(top_n) - 1)]
    if "baseline" in set(fold_metrics["method"].astype(str)):
        methods = ["baseline", *methods]
    methods = list(dict.fromkeys(methods))
    data = fold_metrics.loc[fold_metrics["method"].astype(str).isin(methods)].copy()
    if data.empty:
        return
    trial_error = (
        data.groupby(["trial", "method"], as_index=False)
        .agg(
            corrected_residual_rmse=("corrected_residual_rmse", "mean"),
            corrected_residual_rmse_std=("corrected_residual_rmse", "std"),
            n_folds=("fold", "nunique"),
        )
        .assign(method_label=lambda df: df["method"].map(label_by_method).fillna(df["method"]))
    )
    trial_error["corrected_residual_rmse_se"] = (
        trial_error["corrected_residual_rmse_std"].fillna(0.0) / np.sqrt(np.clip(trial_error["n_folds"].to_numpy(dtype=np.float64), 1.0, None))
    )
    order = (
        trial_error.groupby("method_label", as_index=False)["corrected_residual_rmse"]
        .mean()
        .sort_values("corrected_residual_rmse", ascending=True)["method_label"]
        .tolist()
    )
    y_pos = {name: idx for idx, name in enumerate(order)}
    fig_h = max(5.8, 0.42 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, fig_h), constrained_layout=True)
    for idx, label in enumerate(order):
        label_rows = trial_error.loc[trial_error["method_label"] == label].copy()
        vals = label_rows["corrected_residual_rmse"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        y = np.full(vals.shape, float(y_pos[label]), dtype=np.float64)
        if vals.size > 1:
            jitter = np.linspace(-0.12, 0.12, vals.size)
            y = y + jitter
        color = "#2f5d8c" if label != "Baseline only" else "#6f7278"
        xerr = label_rows["corrected_residual_rmse_se"].to_numpy(dtype=np.float64)
        ax.errorbar(
            vals,
            y,
            xerr=xerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            alpha=0.35,
            zorder=2,
        )
        ax.scatter(vals, y, s=34, color=color, alpha=0.78, linewidths=0, zorder=3)
        ax.plot(
            [float(np.min(vals)), float(np.max(vals))],
            [float(y_pos[label]), float(y_pos[label])],
            color=color,
            alpha=0.24,
            linewidth=2.2,
            zorder=2,
        )
        ax.scatter(float(np.mean(vals)), float(y_pos[label]), s=78, color=color, edgecolor="white", linewidth=0.9, zorder=4)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean corrected residual RMSE, with fold SE")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="#d8dadd", linewidth=0.8, alpha=0.75)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin = float(np.nanmin(trial_error["corrected_residual_rmse"]))
    xmax = float(np.nanmax(trial_error["corrected_residual_rmse"]))
    pad = max((xmax - xmin) * 0.08, 0.01)
    ax.set_xlim(xmin - pad, xmax + pad)
    _save(fig, path)


def plot_mem_graph_error(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    path: str | Path,
    *,
    top_n: int = 12,
    title: str = "MEM residual error by graph basis",
) -> None:
    if fold_metrics.empty or summary.empty:
        return
    if "method_label" not in summary.columns:
        raise ValueError("summary must include method_label")
    mem_summary = summary.loc[
        (summary["method"].astype(str).str.startswith("mem_")) & (summary["feature_kind"].astype(str) == "mem")
    ].copy()
    if mem_summary.empty:
        return
    mem_summary = mem_summary.sort_values("corrected_residual_rmse_mean", ascending=True)
    methods = mem_summary["method"].astype(str).head(int(max(1, top_n))).tolist()
    label_by_method = dict(zip(mem_summary["method"], mem_summary["method_label"], strict=False))
    data = fold_metrics.loc[fold_metrics["method"].astype(str).isin(methods)].copy()
    if data.empty:
        return
    trial_error = (
        data.groupby(["trial", "method"], as_index=False)
        .agg(
            corrected_residual_rmse=("corrected_residual_rmse", "mean"),
            corrected_residual_rmse_std=("corrected_residual_rmse", "std"),
            n_folds=("fold", "nunique"),
        )
        .assign(method_label=lambda df: df["method"].map(label_by_method).fillna(df["method"]))
    )
    trial_error["corrected_residual_rmse_se"] = (
        trial_error["corrected_residual_rmse_std"].fillna(0.0) / np.sqrt(np.clip(trial_error["n_folds"].to_numpy(dtype=np.float64), 1.0, None))
    )
    order = (
        trial_error.groupby("method_label", as_index=False)["corrected_residual_rmse"]
        .mean()
        .sort_values("corrected_residual_rmse", ascending=True)["method_label"]
        .tolist()
    )
    y_pos = {name: idx for idx, name in enumerate(order)}
    fig_h = max(5.4, 0.48 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(10.8, fig_h), constrained_layout=True)
    for label in order:
        label_rows = trial_error.loc[trial_error["method_label"] == label].copy()
        vals = label_rows["corrected_residual_rmse"].to_numpy(dtype=np.float64)
        if vals.size == 0:
            continue
        y = np.full(vals.shape, float(y_pos[label]), dtype=np.float64)
        if vals.size > 1:
            y = y + np.linspace(-0.12, 0.12, vals.size)
        color = "#486b3f" if "Random" not in label else "#7a7d84"
        xerr = label_rows["corrected_residual_rmse_se"].to_numpy(dtype=np.float64)
        ax.errorbar(
            vals,
            y,
            xerr=xerr,
            fmt="none",
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            alpha=0.35,
            zorder=2,
        )
        ax.scatter(vals, y, s=38, color=color, alpha=0.80, linewidths=0, zorder=3)
        ax.scatter(float(np.mean(vals)), float(y_pos[label]), s=86, color=color, edgecolor="white", linewidth=0.9, zorder=4)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean corrected residual RMSE, with fold SE")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="x", color="#d8dadd", linewidth=0.8, alpha=0.75)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmin = float(np.nanmin(trial_error["corrected_residual_rmse"]))
    xmax = float(np.nanmax(trial_error["corrected_residual_rmse"]))
    pad = max((xmax - xmin) * 0.08, 0.01)
    ax.set_xlim(xmin - pad, xmax + pad)
    _save(fig, path)
