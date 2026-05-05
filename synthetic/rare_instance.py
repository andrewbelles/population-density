#
# rare_instance.py  Andrew Belles  May 4th, 2026
#
# Synthetic rare-instance bags for graph-spectral residual correction.
#

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml

from synthetic.common import (
    SyntheticBagDataset,
    attach_graph_names,
    build_graph_block,
    build_pooling_table,
    ensure_dir,
    evaluate_feature_sets,
    graph_diagnostics,
    learned_subsampled_barlow_gsl_graph,
    random_knn_graph,
    rbf_knn_graph,
    repeated_kfold_splits,
    repeated_shuffle_splits,
    sigmoid,
    standardize_columns,
    summarize_fold_metrics,
    unit_vector,
    write_frame,
)

LOGGER = logging.getLogger("synthetic.rare_instance")

TYPE_BACKGROUND = 0
TYPE_POSITIVE = 1
TYPE_NEGATIVE = 2
TYPE_DISTRACTOR = 3
DEFAULT_CONFIG_PATH = Path("configs/synthetic/rare_instance.yaml")


DEFAULTS: dict[str, Any] = {
    "output_dir": "synthetic/artifacts/rare_instance",
    "image_dir": "synthetic/images/rare_instance",
    "seed": 7,
    "n_trials": 1,
    "trial_seed_stride": 100000,
    "n_bags": 500,
    "instance_dim": 16,
    "latent_dim": 2,
    "bag_poisson_lambda": 32.0,
    "bag_size_offset": 16,
    "min_bag_size": 16,
    "max_bag_size": 72,
    "residual_direction": "1,-1",
    "baseline_direction": "2,1",
    "sine_amplitude": 0.25,
    "sine_frequency": 2.0,
    "gamma_residual": 0.35,
    "residual_noise": 0.20,
    "baseline_noise": 0.25,
    "p_min": 0.02,
    "p_max": 0.08,
    "p_distractor": 0.05,
    "gamma_instance": 1.0,
    "background_state_scale": 0.25,
    "instance_noise_scale": 1.0,
    "pool_strategies": "mean,max,meanmax,top_frac,oracle_signal",
    "top_frac": 0.10,
    "learned_graph_enabled": True,
    "learned_graph_pool": "meanmax",
    "learned_graph_hidden_dim": 64,
    "learned_graph_latent_dim": 12,
    "learned_graph_projector_hidden_dim": 64,
    "learned_graph_projector_dim": 32,
    "learned_graph_keep_rate": 0.75,
    "learned_graph_epochs": 350,
    "learned_graph_lr": 0.001,
    "learned_graph_weight_decay": 0.0001,
    "learned_graph_activation": "gelu",
    "learned_graph_barlow_lambda": 0.005,
    "learned_graph_device": "auto",
    "knn_k": 12,
    "mem_k": 16,
    "splitter": "kfold",
    "n_folds": 5,
    "n_repeats": 5,
    "train_frac": 0.70,
    "alpha_min_log10": -4.0,
    "alpha_max_log10": 4.0,
    "alpha_count": 17,
    "edge_plot_graphs": "oracle_latent,meanmax_knn,learned_gsl,random",
    "edge_plot_max_edges": 900,
    "error_plot_top_n": 12,
    "mem_graph_plot_top_n": 12,
    "print_top_n": 16,
    "no_plots": False,
    "log_level": "INFO",
}


def _split_csv(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(x).strip() for x in str(value).split(",") if str(x).strip()]


def _parse_vector(value: object, *, expected_len: int, name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        arr = np.asarray([float(p) for p in parts], dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != int(expected_len):
        raise ValueError(f"{name} must have length={int(expected_len)}, got {arr.shape[0]}")
    return unit_vector(arr)


def _read_yaml(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: config root must be a mapping")
    return raw


def _flatten_config(raw: dict[str, object]) -> dict[str, object]:
    known = set(DEFAULTS)
    out: dict[str, object] = {}

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            key_s = str(key)
            if isinstance(value, dict):
                visit(value)
            elif key_s in known:
                out[key_s] = value

    visit(raw)
    for list_key in ("pool_strategies", "edge_plot_graphs", "residual_direction", "baseline_direction"):
        if list_key in out and isinstance(out[list_key], (list, tuple)):
            out[list_key] = ",".join(str(x) for x in out[list_key])
    return out


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


POOL_LABELS = {
    "mean": "Mean pooled",
    "max": "Max pooled",
    "meanmax": "Mean-max pooled",
    "top_frac": "Top-q pooled",
    "oracle_signal": "Oracle signal pooled",
}


GRAPH_LABELS = {
    "oracle_latent": "Oracle latent graph",
    "random": "Random graph",
    "learned_gsl": "Learned GSL graph",
}


def pool_label(name: object) -> str:
    key = str(name).strip()
    return POOL_LABELS.get(key, key.replace("_", " ").title())


def graph_label(name: object, args: SimpleNamespace) -> str:
    key = str(name).strip()
    if key == "learned_gsl":
        return f"Learned GSL graph ({pool_label(args.learned_graph_pool)})"
    if key.endswith("_knn"):
        return f"{pool_label(key.removesuffix('_knn'))} kNN graph"
    return GRAPH_LABELS.get(key, key.replace("_", " ").title())


def method_label(method: object, args: SimpleNamespace) -> str:
    key = str(method).strip()
    if key == "baseline":
        return "Baseline only"
    if key == "raw_latent_h_oracle":
        return "Oracle latent features"
    if key == "raw_mem_latent_h_oracle":
        return "Oracle latent features + MEM"
    if key == "mem_learned_gsl":
        return f"Learned GSL MEM ({pool_label(args.learned_graph_pool)})"
    if key.startswith("raw_mem_"):
        return f"{pool_label(key.removeprefix('raw_mem_'))} + MEM"
    if key.startswith("raw_"):
        return pool_label(key.removeprefix("raw_"))
    if key.startswith("mem_"):
        return f"{graph_label(key.removeprefix('mem_'), args)} MEM"
    return key.replace("_", " ").title()


def add_display_columns(summary: pd.DataFrame, graph_summary: pd.DataFrame, args: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_summary = summary.copy()
    out_graph = graph_summary.copy()
    if "method" in out_summary.columns:
        out_summary["method_label"] = [method_label(m, args) for m in out_summary["method"]]
    if "graph" in out_summary.columns:
        out_summary["graph_label"] = [graph_label(g, args) if pd.notna(g) else "" for g in out_summary["graph"]]
    if "graph" in out_graph.columns:
        out_graph["graph_label"] = [graph_label(g, args) for g in out_graph["graph"]]
    return out_summary, out_graph


def _table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}")


def log_pretty_tables(summary: pd.DataFrame, graph_summary: pd.DataFrame, learned: pd.DataFrame, args: SimpleNamespace) -> None:
    method_cols = [
        "method_label",
        "feature_kind",
        "feature_dim",
        "relative_rmse_improvement_pct_mean",
        "residual_r2_vs_zero_mean",
        "residual_corr_pearson_mean",
    ]
    method_table = (
        summary.loc[:, [c for c in method_cols if c in summary.columns]]
        .head(int(args.print_top_n))
        .rename(
            columns={
                "method_label": "method",
                "feature_kind": "type",
                "feature_dim": "dim",
                "relative_rmse_improvement_pct_mean": "rmse_improve_pct",
                "residual_r2_vs_zero_mean": "resid_r2",
                "residual_corr_pearson_mean": "resid_corr",
            }
        )
    )
    LOGGER.info("method summary\n%s", _table(method_table))

    graph_cols = [
        "graph_label",
        "graph_oracle_weight_spearman",
        "residual_moran_score",
        "mem_residual_projection_r2",
        "mem_k",
    ]
    graph_table = graph_summary.loc[:, [c for c in graph_cols if c in graph_summary.columns]].rename(
        columns={
            "graph_label": "graph",
            "graph_oracle_weight_spearman": "oracle_spearman",
            "residual_moran_score": "resid_moran",
            "mem_residual_projection_r2": "mem_resid_r2",
        }
    )
    LOGGER.info("graph diagnostics\n%s", _table(graph_table))

    if not learned.empty:
        learned_table = learned.copy()
        learned_table["graph"] = [graph_label(g, args) for g in learned_table["graph"]]
        learned_table["source_representation"] = [pool_label(v) for v in learned_table["source_representation"]]
        learned_table = learned_table.rename(columns={"source_representation": "source", "embedding_dim": "dim"})
        cols = ["trial", "graph", "source", "objective", "dim", "keep_rate", "barlow_lambda", "train_loss"]
        LOGGER.info("learned graph\n%s", _table(learned_table.loc[:, [c for c in cols if c in learned_table.columns]]))


def resolve_args(cli: argparse.Namespace) -> SimpleNamespace:
    raw = _read_yaml(cli.config)
    values = dict(DEFAULTS)
    values.update(_flatten_config(raw))
    for key, value in vars(cli).items():
        if key == "config" or value is None:
            continue
        values[key] = value
    for key in ("pool_strategies", "edge_plot_graphs", "residual_direction", "baseline_direction"):
        if isinstance(values.get(key), (list, tuple)):
            values[key] = ",".join(str(x) for x in values[key])
    values["learned_graph_enabled"] = _coerce_bool(values["learned_graph_enabled"])
    values["no_plots"] = _coerce_bool(values["no_plots"])
    values["config"] = str(cli.config) if cli.config is not None else None
    return SimpleNamespace(**values)


def generate_dataset(
    *,
    n_bags: int,
    instance_dim: int,
    latent_dim: int,
    seed: int,
    bag_poisson_lambda: float,
    bag_size_offset: int,
    gamma_residual: float,
    residual_noise: float,
    baseline_noise: float,
    gamma_instance: float,
    residual_direction: object,
    baseline_direction: object,
    sine_amplitude: float,
    sine_frequency: float,
    p_min: float,
    p_max: float,
    p_distractor: float,
    background_state_scale: float,
    instance_noise_scale: float,
    min_bag_size: int,
    max_bag_size: int,
) -> SyntheticBagDataset:
    rng = np.random.default_rng(int(seed))
    bag_sizes = rng.poisson(float(bag_poisson_lambda), size=int(n_bags)) + int(bag_size_offset)
    bag_sizes = np.clip(bag_sizes, int(min_bag_size), int(max_bag_size)).astype(np.int64)

    h = rng.normal(size=(int(n_bags), int(latent_dim))).astype(np.float64)
    w = _parse_vector(residual_direction, expected_len=int(latent_dim), name="residual_direction")
    theta = np.asarray(h @ w, dtype=np.float64)
    f = theta + float(sine_amplitude) * np.sin(float(sine_frequency) * theta)
    g = (f - float(np.mean(f))) / max(float(np.std(f)), 1e-12)
    residual = float(gamma_residual) * g + rng.normal(scale=float(residual_noise), size=int(n_bags))

    u = _parse_vector(baseline_direction, expected_len=int(latent_dim), name="baseline_direction")
    baseline = np.asarray(h @ u + rng.normal(scale=float(baseline_noise), size=int(n_bags)), dtype=np.float64)
    target = baseline + residual

    v_pos = unit_vector(rng.normal(size=int(instance_dim)))
    v_neg = unit_vector(rng.normal(size=int(instance_dim)))
    v_dist = unit_vector(rng.normal(size=int(instance_dim)))
    a = rng.normal(size=(int(instance_dim), int(latent_dim))).astype(np.float64)

    bags: list[np.ndarray] = []
    instance_types: list[np.ndarray] = []
    signal_masks: list[np.ndarray] = []
    for i, m_i in enumerate(bag_sizes):
        p_dep = float(p_min) + (float(p_max) - float(p_min)) * float(sigmoid(abs(theta[i])))
        p_pos = p_dep * float(sigmoid(theta[i]))
        p_neg = p_dep * float(sigmoid(-theta[i]))
        p_dist = float(p_distractor)
        p_bg = max(0.0, 1.0 - (p_pos + p_neg + p_dist))
        probs = np.asarray([p_bg, p_pos, p_neg, p_dist], dtype=np.float64)
        probs = probs / float(np.sum(probs))
        types = rng.choice(
            np.asarray([TYPE_BACKGROUND, TYPE_POSITIVE, TYPE_NEGATIVE, TYPE_DISTRACTOR], dtype=np.int64),
            size=int(m_i),
            p=probs,
        )
        base = float(background_state_scale) * (a @ h[i])
        eps = rng.normal(scale=float(instance_noise_scale), size=(int(m_i), int(instance_dim))).astype(np.float64)
        x = base[None, :] + eps
        pos_mask = types == TYPE_POSITIVE
        neg_mask = types == TYPE_NEGATIVE
        dist_mask = types == TYPE_DISTRACTOR
        if bool(pos_mask.any()):
            x[pos_mask] += float(gamma_instance) * abs(float(theta[i])) * v_pos[None, :]
        if bool(neg_mask.any()):
            x[neg_mask] += float(gamma_instance) * abs(float(theta[i])) * v_neg[None, :]
        if bool(dist_mask.any()):
            zeta = np.abs(rng.normal(size=int(dist_mask.sum()))).reshape(-1, 1)
            x[dist_mask] += float(gamma_instance) * zeta * v_dist[None, :]
        bags.append(np.asarray(x, dtype=np.float64))
        instance_types.append(types)
        signal_masks.append(np.logical_or(pos_mask, neg_mask))

    return SyntheticBagDataset(
        bags=bags,
        instance_types=instance_types,
        signal_masks=signal_masks,
        h=h,
        theta=theta,
        residual=np.asarray(residual, dtype=np.float64),
        baseline=baseline,
        target=target,
        bag_sizes=bag_sizes,
        metadata={
            "seed": int(seed),
            "n_bags": int(n_bags),
            "instance_dim": int(instance_dim),
            "latent_dim": int(latent_dim),
            "bag_poisson_lambda": float(bag_poisson_lambda),
            "bag_size_offset": int(bag_size_offset),
            "gamma_residual": float(gamma_residual),
            "residual_noise": float(residual_noise),
            "baseline_noise": float(baseline_noise),
            "gamma_instance": float(gamma_instance),
            "sine_amplitude": float(sine_amplitude),
            "sine_frequency": float(sine_frequency),
            "p_min": float(p_min),
            "p_max": float(p_max),
            "p_distractor": float(p_distractor),
            "background_state_scale": float(background_state_scale),
            "instance_noise_scale": float(instance_noise_scale),
        },
    )


def node_table(dataset: SyntheticBagDataset, *, trial: int) -> pd.DataFrame:
    signal_count = np.asarray([int(mask.sum()) for mask in dataset.signal_masks], dtype=np.int64)
    dist_count = np.asarray([int(np.sum(types == TYPE_DISTRACTOR)) for types in dataset.instance_types], dtype=np.int64)
    rows = {
        "trial": int(trial),
        "node": np.arange(dataset.h.shape[0], dtype=np.int64),
        "theta": dataset.theta,
        "baseline": dataset.baseline,
        "residual": dataset.residual,
        "target": dataset.target,
        "bag_size": dataset.bag_sizes,
        "signal_count": signal_count,
        "signal_frac": signal_count / np.clip(dataset.bag_sizes, 1, None),
        "distractor_count": dist_count,
        "distractor_frac": dist_count / np.clip(dataset.bag_sizes, 1, None),
    }
    for idx in range(dataset.h.shape[1]):
        rows[f"h{idx + 1}"] = dataset.h[:, idx]
    return pd.DataFrame(rows)


def build_representations(dataset: SyntheticBagDataset, *, top_frac: float, strategies: object) -> dict[str, np.ndarray]:
    return build_pooling_table(dataset, strategies=_split_csv(strategies), top_frac=float(top_frac))


def build_graphs(
    representations: dict[str, np.ndarray],
    dataset: SyntheticBagDataset,
    *,
    args: SimpleNamespace,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(int(seed) + 5000)
    weights: dict[str, np.ndarray] = {}
    learned_rows: list[dict[str, object]] = []
    for name, z in representations.items():
        weights[f"{name}_knn"] = rbf_knn_graph(z, k=int(args.knn_k), bandwidth_k=int(args.knn_k), symmetrize=True)
    if bool(args.learned_graph_enabled):
        pool_name = str(args.learned_graph_pool)
        if pool_name not in representations:
            raise ValueError(f"learned_graph_pool={pool_name!r} not found in pool_strategies")
        learned = learned_subsampled_barlow_gsl_graph(
            dataset,
            source_representation=pool_name,
            top_frac=float(args.top_frac),
            graph_k=int(args.knn_k),
            latent_dim=int(args.learned_graph_latent_dim),
            hidden_dim=int(args.learned_graph_hidden_dim),
            projector_hidden_dim=int(args.learned_graph_projector_hidden_dim),
            projector_dim=int(args.learned_graph_projector_dim),
            keep_rate=float(args.learned_graph_keep_rate),
            epochs=int(args.learned_graph_epochs),
            learning_rate=float(args.learned_graph_lr),
            weight_decay=float(args.learned_graph_weight_decay),
            activation=str(args.learned_graph_activation),
            barlow_lambda=float(args.learned_graph_barlow_lambda),
            device=str(args.learned_graph_device),
            seed=int(seed) + 7000,
        )
        weights["learned_gsl"] = learned.weights
        learned_rows.append(
            {
                "graph": "learned_gsl",
                "source_representation": learned.source_representation,
                "embedding_dim": int(learned.embedding.shape[1]),
                "objective": "subsampled_barlow",
                "keep_rate": float(args.learned_graph_keep_rate),
                "barlow_lambda": float(args.learned_graph_barlow_lambda),
                "train_loss": float(learned.train_loss),
            }
        )
    weights["oracle_latent"] = rbf_knn_graph(standardize_columns(dataset.h), k=int(args.knn_k), bandwidth_k=int(args.knn_k), symmetrize=True)
    weights["random"] = random_knn_graph(dataset.h.shape[0], k=int(args.knn_k), rng=rng)
    graphs = {
        name: build_graph_block(name, w, mem_k=int(args.mem_k), row_topk=max(int(args.knn_k) * 4, int(args.mem_k) * 4, 32))
        for name, w in weights.items()
    }
    return weights, graphs, pd.DataFrame(learned_rows)


def build_feature_sets(
    representations: dict[str, np.ndarray],
    graphs: dict[str, object],
    dataset: SyntheticBagDataset,
) -> dict[str, np.ndarray | None]:
    features: dict[str, np.ndarray | None] = {"baseline": None}
    features["raw_latent_h_oracle"] = standardize_columns(dataset.h)
    for rep_name, z in representations.items():
        features[f"raw_{rep_name}"] = z
    for graph_name, block in graphs.items():
        features[f"mem_{graph_name}"] = np.asarray(block.basis, dtype=np.float64)
    for rep_name, z in representations.items():
        graph_name = f"{rep_name}_knn"
        basis = np.asarray(graphs[graph_name].basis, dtype=np.float64)
        features[f"raw_mem_{rep_name}"] = np.column_stack([z, basis])
    features["raw_mem_latent_h_oracle"] = np.column_stack(
        [standardize_columns(dataset.h), np.asarray(graphs["oracle_latent"].basis, dtype=np.float64)]
    )
    return features


def build_splits(args: SimpleNamespace, *, n: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = str(args.splitter).strip().lower()
    if splitter == "kfold":
        return repeated_kfold_splits(n, n_trials=int(args.n_repeats), n_folds=int(args.n_folds), seed=int(seed) + 10000)
    if splitter == "shuffle":
        return repeated_shuffle_splits(n, n_repeats=int(args.n_repeats), train_frac=float(args.train_frac), seed=int(seed) + 10000)
    raise ValueError(f"unsupported splitter={args.splitter!r}")


def remove_stale_outputs(output_dir: Path, image_dir: Path) -> None:
    # Older development runs wrote plots/artifacts that are now represented by stdout tables.
    stale = [
        output_dir / "convex_weights.csv",
        output_dir / "convex_weights.parquet",
        image_dir / "convex_weights.png",
        image_dir / "graph_edges_learned_convex.png",
        image_dir / "method_performance.png",
        image_dir / "pooling_summary.png",
        image_dir / "graph_diagnostics.png",
    ]
    for path in stale:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def run_one_trial(args: SimpleNamespace, *, trial: int) -> dict[str, object]:
    seed = int(args.seed) + int(trial) * int(args.trial_seed_stride)
    dataset = generate_dataset(
        n_bags=int(args.n_bags),
        instance_dim=int(args.instance_dim),
        latent_dim=int(args.latent_dim),
        seed=seed,
        bag_poisson_lambda=float(args.bag_poisson_lambda),
        bag_size_offset=int(args.bag_size_offset),
        gamma_residual=float(args.gamma_residual),
        residual_noise=float(args.residual_noise),
        baseline_noise=float(args.baseline_noise),
        gamma_instance=float(args.gamma_instance),
        residual_direction=args.residual_direction,
        baseline_direction=args.baseline_direction,
        sine_amplitude=float(args.sine_amplitude),
        sine_frequency=float(args.sine_frequency),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        p_distractor=float(args.p_distractor),
        background_state_scale=float(args.background_state_scale),
        instance_noise_scale=float(args.instance_noise_scale),
        min_bag_size=int(args.min_bag_size),
        max_bag_size=int(args.max_bag_size),
    )
    representations = build_representations(dataset, top_frac=float(args.top_frac), strategies=args.pool_strategies)
    weights, graphs, learned_df = build_graphs(representations, dataset, args=args, seed=seed)
    features = build_feature_sets(representations, graphs, dataset)
    splits = build_splits(args, n=dataset.h.shape[0], seed=seed)
    alpha_grid = np.logspace(float(args.alpha_min_log10), float(args.alpha_max_log10), int(args.alpha_count))
    fold_metrics, _summary = evaluate_feature_sets(
        features=features,
        residual=dataset.residual,
        baseline=dataset.baseline,
        target=dataset.target,
        splits=splits,
        alpha_grid=alpha_grid,
        seed=seed + 20000,
    )
    fold_metrics.insert(0, "trial", int(trial))
    fold_metrics["eval_id"] = fold_metrics["trial"].astype(str) + ":" + fold_metrics["fold"].astype(str)
    graph_summary = graph_diagnostics(graphs, oracle_weights=weights["oracle_latent"], residual=dataset.residual)
    graph_summary.insert(0, "trial", int(trial))
    if not learned_df.empty:
        learned_df.insert(0, "trial", int(trial))
    return {
        "dataset": dataset,
        "weights": weights,
        "fold_metrics": fold_metrics,
        "graph_summary": graph_summary,
        "nodes": node_table(dataset, trial=int(trial)),
        "learned": learned_df,
    }


def run_experiment(args: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trials = [run_one_trial(args, trial=t) for t in range(int(args.n_trials))]
    fold_metrics = pd.concat([pd.DataFrame(t["fold_metrics"]) for t in trials], axis=0, ignore_index=True)
    graph_summary = pd.concat([pd.DataFrame(t["graph_summary"]) for t in trials], axis=0, ignore_index=True)
    nodes = pd.concat([pd.DataFrame(t["nodes"]) for t in trials], axis=0, ignore_index=True)
    learned_diagnostics = pd.concat([pd.DataFrame(t["learned"]) for t in trials], axis=0, ignore_index=True)
    summary = summarize_fold_metrics(fold_metrics)
    summary = attach_graph_names(summary).sort_values("relative_rmse_improvement_pct_mean", ascending=False).reset_index(drop=True)
    graph_summary_agg = (
        graph_summary.groupby("graph", as_index=False)
        .agg(
            graph_oracle_weight_spearman=("graph_oracle_weight_spearman", "mean"),
            residual_moran_score=("residual_moran_score", "mean"),
            mem_residual_projection_r2=("mem_residual_projection_r2", "mean"),
            mem_k=("mem_k", "median"),
            positive_eigenvalue_mean=("positive_eigenvalue_mean", "mean"),
            n_trials=("trial", "nunique"),
        )
        .sort_values("graph_oracle_weight_spearman", ascending=False)
        .reset_index(drop=True)
    )
    summary = summary.merge(graph_summary_agg, on="graph", how="left")
    summary, graph_summary_agg = add_display_columns(summary, graph_summary_agg, args)

    output_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(args.image_dir)
    remove_stale_outputs(output_dir, image_dir)
    write_frame(nodes, output_dir / "nodes.csv")
    write_frame(fold_metrics, output_dir / "fold_metrics.csv")
    write_frame(summary, output_dir / "summary.csv")
    write_frame(graph_summary, output_dir / "graph_diagnostics.by_trial.csv")
    write_frame(graph_summary_agg, output_dir / "graph_diagnostics.csv")
    if not learned_diagnostics.empty:
        write_frame(learned_diagnostics, output_dir / "learned_graph_diagnostics.csv")
    write_frame(nodes, output_dir / "nodes.parquet")
    write_frame(fold_metrics, output_dir / "fold_metrics.parquet")
    write_frame(summary, output_dir / "summary.parquet")
    write_frame(graph_summary_agg, output_dir / "graph_diagnostics.parquet")
    with (output_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True)

    if not bool(args.no_plots):
        from synthetic.visualizations import (
            plot_graph_edge_map,
            plot_latent_signal,
            plot_mem_graph_error,
            plot_trial_method_error,
        )

        first = trials[0]
        dataset = first["dataset"]
        weights = first["weights"]
        plot_latent_signal(dataset, image_dir / "latent_signal.png")
        plot_trial_method_error(
            fold_metrics,
            summary,
            image_dir / "trial_method_error.png",
            top_n=int(args.error_plot_top_n),
        )
        plot_mem_graph_error(
            fold_metrics,
            summary,
            image_dir / "mem_graph_error.png",
            top_n=int(args.mem_graph_plot_top_n),
        )
        for graph_name in _split_csv(args.edge_plot_graphs):
            if graph_name not in weights:
                LOGGER.warning("skipping missing edge plot graph=%s", graph_name)
                continue
            safe_name = graph_name.replace("/", "_").replace(" ", "_")
            plot_graph_edge_map(
                dataset,
                weights[graph_name],
                image_dir / f"graph_edges_{safe_name}.png",
                title=graph_label(graph_name, args),
                max_edges=int(args.edge_plot_max_edges),
            )

    log_pretty_tables(summary, graph_summary_agg, learned_diagnostics, args)
    LOGGER.info("wrote synthetic rare-instance artifacts to %s", output_dir.resolve())
    return summary, fold_metrics, graph_summary_agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the rare-instance synthetic graph-spectral residual experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir")
    parser.add_argument("--image-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--trial-seed-stride", type=int)
    parser.add_argument("--n-bags", type=int)
    parser.add_argument("--instance-dim", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--bag-poisson-lambda", type=float)
    parser.add_argument("--bag-size-offset", type=int)
    parser.add_argument("--min-bag-size", type=int)
    parser.add_argument("--max-bag-size", type=int)
    parser.add_argument("--residual-direction")
    parser.add_argument("--baseline-direction")
    parser.add_argument("--sine-amplitude", type=float)
    parser.add_argument("--sine-frequency", type=float)
    parser.add_argument("--gamma-residual", type=float)
    parser.add_argument("--residual-noise", type=float)
    parser.add_argument("--baseline-noise", type=float)
    parser.add_argument("--p-min", type=float)
    parser.add_argument("--p-max", type=float)
    parser.add_argument("--p-distractor", type=float)
    parser.add_argument("--gamma-instance", type=float)
    parser.add_argument("--background-state-scale", type=float)
    parser.add_argument("--instance-noise-scale", type=float)
    parser.add_argument("--pool-strategies")
    parser.add_argument("--top-frac", type=float)
    parser.add_argument("--learned-graph-enabled", action=argparse.BooleanOptionalAction)
    parser.add_argument("--learned-graph-pool")
    parser.add_argument("--learned-graph-hidden-dim", type=int)
    parser.add_argument("--learned-graph-latent-dim", type=int)
    parser.add_argument("--learned-graph-projector-hidden-dim", type=int)
    parser.add_argument("--learned-graph-projector-dim", type=int)
    parser.add_argument("--learned-graph-keep-rate", type=float)
    parser.add_argument("--learned-graph-epochs", type=int)
    parser.add_argument("--learned-graph-lr", type=float)
    parser.add_argument("--learned-graph-weight-decay", type=float)
    parser.add_argument("--learned-graph-activation")
    parser.add_argument("--learned-graph-barlow-lambda", type=float)
    parser.add_argument("--learned-graph-device")
    parser.add_argument("--knn-k", type=int)
    parser.add_argument("--mem-k", type=int)
    parser.add_argument("--splitter")
    parser.add_argument("--n-folds", type=int)
    parser.add_argument("--n-repeats", type=int)
    parser.add_argument("--train-frac", type=float)
    parser.add_argument("--alpha-min-log10", type=float)
    parser.add_argument("--alpha-max-log10", type=float)
    parser.add_argument("--alpha-count", type=int)
    parser.add_argument("--edge-plot-graphs")
    parser.add_argument("--edge-plot-max-edges", type=int)
    parser.add_argument("--error-plot-top-n", type=int)
    parser.add_argument("--mem-graph-plot-top-n", type=int)
    parser.add_argument("--print-top-n", type=int)
    parser.add_argument("--no-plots", action=argparse.BooleanOptionalAction)
    parser.add_argument("--log-level")
    return parser


def main() -> None:
    parser = build_parser()
    args = resolve_args(parser.parse_args())
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s %(name)s] %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    run_experiment(args)


if __name__ == "__main__":
    main()
