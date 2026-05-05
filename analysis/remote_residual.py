#!/usr/bin/env python3
#
# remote_residual.py  Andrew Belles  May 3rd, 2026
#
# Diagnostics for whether remote-sensing MEM explains residual structure left
# after administrative embeddings.
#

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.common import write_frame
from graph.config import load_config as load_graph_config
from graph.topology import run as run_graph_topology
from nowcast.common import (
    BlockRows,
    build_state_group_splits,
    compute_topology_leakage_proxy,
    fit_predict,
    load_modality_block,
    load_pep_year,
    load_topology_edges,
    load_topology_rows,
    mape_pop_pct,
)
from nowcast.config import NowcastConfig, load_config as load_nowcast_config


LOGGER = logging.getLogger("analysis.remote_residual")


@dataclass(slots=True)
class GraphRunConfig:
    key: str
    graph_best_trial_json: Path


@dataclass(slots=True)
class RemoteResidualConfig:
    nowcast_config_path: Path
    graph_config_path: Path
    output_root: Path
    linear_best_trial_json: Path | None
    materialize_graphs: bool
    strict_year: int
    direct_modality: str
    model_key: str
    hard_case_quantile: float
    graphs: dict[str, GraphRunConfig]


@dataclass(slots=True)
class GraphMemBlock:
    key: str
    graph_tag_base: str
    graph_tag: str
    graph_kind: str
    mem_top_k: int
    fips: np.ndarray
    x: np.ndarray
    edges: pd.DataFrame


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = _as_path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = _as_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: config must be a mapping")
    return dict(raw)


def _optional_path(repo_root: Path, value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _resolve_path(repo_root, text)


def load_remote_residual_config(path: str | Path) -> RemoteResidualConfig:
    cfg_path = _as_path(path)
    raw = _read_yaml(cfg_path)
    repo_root = cfg_path.resolve().parent.parent.parent
    graphs_raw = dict(raw.get("graphs", {}))
    if not graphs_raw:
        raise ValueError("remote residual config requires graphs")
    graphs: dict[str, GraphRunConfig] = {}
    for key, value in graphs_raw.items():
        item = dict(value)
        graph_key = str(key).strip().lower()
        graphs[graph_key] = GraphRunConfig(
            key=graph_key,
            graph_best_trial_json=_resolve_path(repo_root, item["graph_best_trial_json"]),
        )
    return RemoteResidualConfig(
        nowcast_config_path=_resolve_path(repo_root, raw.get("nowcast_config_path", "configs/nowcast/nowcast.yaml")),
        graph_config_path=_resolve_path(repo_root, raw.get("graph_config_path", "configs/graph/topology.yaml")),
        output_root=_resolve_path(repo_root, raw.get("output_root", "analysis/artifacts/remote_residual")),
        linear_best_trial_json=_optional_path(repo_root, raw.get("linear_best_trial_json")),
        materialize_graphs=bool(raw.get("materialize_graphs", True)),
        strict_year=int(raw.get("strict_year", 2020)),
        direct_modality=str(raw.get("direct_modality", "admin")).strip().lower(),
        model_key=str(raw.get("model_key", "huber")).strip().lower(),
        hard_case_quantile=float(raw.get("hard_case_quantile", 0.90)),
        graphs=graphs,
    )


def aligned_rows(
    *,
    truth: pd.DataFrame,
    direct: BlockRows,
    mems: dict[str, GraphMemBlock],
) -> dict[str, Any]:
    truth = truth.copy()
    truth["fips"] = truth["fips"].astype(str).str.strip().str.zfill(5)
    direct_idx = {str(f): i for i, f in enumerate(np.asarray(direct.fips, dtype="U5").tolist())}
    mem_idx = {
        key: {str(f): i for i, f in enumerate(np.asarray(mem.fips, dtype="U5").tolist())}
        for key, mem in mems.items()
    }
    common = set(truth["fips"].astype(str).tolist()).intersection(direct_idx)
    for idx in mem_idx.values():
        common &= set(idx)
    sample_ids = np.asarray([str(f) for f in truth["fips"].astype(str).tolist() if str(f) in common], dtype="U5")
    if sample_ids.size <= 1:
        raise ValueError("remote residual diagnostic has too few shared counties")
    truth_idx = {str(f): i for i, f in enumerate(truth["fips"].astype(str).tolist())}
    t_rows = np.asarray([truth_idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
    d_rows = np.asarray([direct_idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
    mem_rows = {
        key: np.asarray([idx[str(f)] for f in sample_ids.tolist()], dtype=np.int64)
        for key, idx in mem_idx.items()
    }
    return {
        "sample_ids": sample_ids,
        "y_log": np.asarray(truth["y_log"].to_numpy()[t_rows], dtype=np.float64),
        "y_level": np.asarray(truth["y_level"].to_numpy()[t_rows], dtype=np.float64),
        "pep_log": np.asarray(truth["pep_log"].to_numpy()[t_rows], dtype=np.float64),
        "pep_level": np.asarray(truth["pep_population"].to_numpy()[t_rows], dtype=np.float64),
        "direct": np.asarray(direct.x[d_rows], dtype=np.float64),
        "mem": {key: np.asarray(mems[key].x[mem_rows[key]], dtype=np.float64) for key in mems},
    }


def feature_sets(direct: np.ndarray, mems: dict[str, np.ndarray]) -> dict[str, tuple[str, np.ndarray]]:
    out: dict[str, tuple[str, np.ndarray]] = {"admin_direct": ("none", np.asarray(direct, dtype=np.float64))}
    if "admin" in mems:
        out["admin_direct_admin_mem"] = ("admin", np.concatenate([direct, mems["admin"]], axis=1))
    if "remote" in mems:
        out["admin_direct_remote_mem"] = ("remote", np.concatenate([direct, mems["remote"]], axis=1))
    if "full" in mems:
        out["admin_direct_full_mem"] = ("full", np.concatenate([direct, mems["full"]], axis=1))
    return out


def finite_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[keep]
    yv = yv[keep]
    if xv.size <= 1 or float(np.std(xv)) <= 0.0 or float(np.std(yv)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def adjusted_fold_mape(pep_mape: float, model_mape: float, leakage_proxy: float) -> tuple[float, float]:
    rel = (float(pep_mape) - float(model_mape)) / max(float(pep_mape), 1e-9)
    adjusted_rel = rel - max(rel, 0.0) * float(leakage_proxy)
    return float(float(pep_mape) * (1.0 - adjusted_rel)), float(adjusted_rel * 100.0)


def run_joint_incremental(
    *,
    sample_ids: np.ndarray,
    y_log: np.ndarray,
    y_level: np.ndarray,
    pep_log: np.ndarray,
    pep_level: np.ndarray,
    sets: dict[str, tuple[str, np.ndarray]],
    mems: dict[str, GraphMemBlock],
    nowcast_cfg: NowcastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = nowcast_cfg.downstream.model_cfg()
    splits = build_state_group_splits(
        sample_ids,
        n_splits=int(nowcast_cfg.evaluation.n_splits),
        strategy=str(nowcast_cfg.evaluation.fold_strategy),
        region_level=str(nowcast_cfg.evaluation.fold_region_level),
    )
    true_resid = np.asarray(y_log, dtype=np.float64) - np.asarray(pep_log, dtype=np.float64)
    pred_by_model = {name: np.full(sample_ids.shape[0], np.nan, dtype=np.float64) for name in sets}
    fold_rows: list[dict[str, object]] = []
    for split in splits:
        tr = np.asarray(split.train_idx, dtype=np.int64)
        te = np.asarray(split.test_idx, dtype=np.int64)
        pep_mape = mape_pop_pct(y_log[te], pep_log[te])
        for model_name, (graph_key, x) in sets.items():
            x_arr = np.asarray(x, dtype=np.float64)
            train_mask = np.isfinite(x_arr[tr]).all(axis=1) & np.isfinite(true_resid[tr])
            test_mask = np.isfinite(x_arr[te]).all(axis=1) & np.isfinite(y_log[te]) & np.isfinite(pep_log[te])
            tr_sub = tr[train_mask]
            te_sub = te[test_mask]
            if tr_sub.size <= 1 or te_sub.size <= 0:
                continue
            pred_corr, _sigma, feat_dim = fit_predict(
                model_cfg=model_cfg,
                Xtr=x_arr[tr_sub],
                ytr=true_resid[tr_sub],
                Xte=x_arr[te_sub],
                seed=int(nowcast_cfg.evaluation.seed) + int(split.fold_id),
            )
            pred = np.asarray(pep_log[te_sub], dtype=np.float64) + np.asarray(pred_corr, dtype=np.float64)
            pred_by_model[model_name][te_sub] = pred
            leakage = 0.0
            if graph_key in mems:
                leakage = compute_topology_leakage_proxy(
                    edges=mems[graph_key].edges,
                    sample_ids=sample_ids,
                    test_idx=te_sub,
                    mode=str(nowcast_cfg.analysis.leakage_proxy_mode),
                )
            mape = mape_pop_pct(y_log[te_sub], pred)
            adj_mape, adj_rel = adjusted_fold_mape(pep_mape, mape, leakage)
            fold_rows.append(
                {
                    "diagnostic": "incremental",
                    "model": model_name,
                    "fold": int(split.fold_id),
                    "n_train": int(tr_sub.size),
                    "n_test": int(te_sub.size),
                    "feature_dim": int(feat_dim),
                    "graph_key": str(graph_key),
                    "topology_leakage_proxy": float(leakage),
                    "pep_mape_pop_pct": float(pep_mape),
                    "mape_pop_pct": float(mape),
                    "adjusted_mape_pop_pct": float(adj_mape),
                    "adjusted_relative_improvement_pct": float(adj_rel),
                    "heldout_states": ",".join(split.heldout_states),
                    "heldout_regions": ",".join(split.heldout_regions),
                }
            )
    abs_rows: list[dict[str, object]] = []
    for model_name, pred in pred_by_model.items():
        for i, fips in enumerate(sample_ids.tolist()):
            if not np.isfinite(pred[i]):
                continue
            abs_rows.append(
                {
                    "diagnostic": "incremental",
                    "model": model_name,
                    "fips": str(fips),
                    "state": str(fips)[:2],
                    "y_log": float(y_log[i]),
                    "y_level": float(y_level[i]),
                    "pep_log": float(pep_log[i]),
                    "pep_level": float(pep_level[i]),
                    "pred_log": float(pred[i]),
                    "pred_level": float(np.exp(pred[i])),
                    "true_resid_log": float(true_resid[i]),
                    "pred_correction_log": float(pred[i] - pep_log[i]),
                    "ape_pop_pct": float(abs(np.exp(pred[i]) - y_level[i]) / max(abs(y_level[i]), 1e-9) * 100.0),
                    "pep_ape_pop_pct": float(abs(pep_level[i] - y_level[i]) / max(abs(y_level[i]), 1e-9) * 100.0),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(abs_rows)


def run_orthogonal_residual(
    *,
    sample_ids: np.ndarray,
    y_log: np.ndarray,
    y_level: np.ndarray,
    pep_log: np.ndarray,
    pep_level: np.ndarray,
    direct: np.ndarray,
    mems: dict[str, np.ndarray],
    mem_blocks: dict[str, GraphMemBlock],
    nowcast_cfg: NowcastConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = nowcast_cfg.downstream.model_cfg()
    splits = build_state_group_splits(
        sample_ids,
        n_splits=int(nowcast_cfg.evaluation.n_splits),
        strategy=str(nowcast_cfg.evaluation.fold_strategy),
        region_level=str(nowcast_cfg.evaluation.fold_region_level),
    )
    true_resid = np.asarray(y_log, dtype=np.float64) - np.asarray(pep_log, dtype=np.float64)
    candidates = [key for key in ("admin", "remote", "full") if key in mems]
    pred_by_model = {f"admin_residual_{key}_mem": np.full(sample_ids.shape[0], np.nan, dtype=np.float64) for key in candidates}
    pred_by_model["admin_direct"] = np.full(sample_ids.shape[0], np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    for split in splits:
        tr = np.asarray(split.train_idx, dtype=np.int64)
        te = np.asarray(split.test_idx, dtype=np.int64)
        pep_mape = mape_pop_pct(y_log[te], pep_log[te])
        train_mask = np.isfinite(direct[tr]).all(axis=1) & np.isfinite(true_resid[tr])
        test_mask = np.isfinite(direct[te]).all(axis=1) & np.isfinite(y_log[te]) & np.isfinite(pep_log[te])
        tr_sub = tr[train_mask]
        te_sub = te[test_mask]
        if tr_sub.size <= 1 or te_sub.size <= 0:
            continue
        admin_corr_te, _sigma_admin, admin_dim = fit_predict(
            model_cfg=model_cfg,
            Xtr=direct[tr_sub],
            ytr=true_resid[tr_sub],
            Xte=direct[te_sub],
            seed=int(nowcast_cfg.evaluation.seed) + 5000 + int(split.fold_id),
        )
        admin_corr_tr, _sigma_tr, _ = fit_predict(
            model_cfg=model_cfg,
            Xtr=direct[tr_sub],
            ytr=true_resid[tr_sub],
            Xte=direct[tr_sub],
            seed=int(nowcast_cfg.evaluation.seed) + 6000 + int(split.fold_id),
        )
        admin_pred = np.asarray(pep_log[te_sub], dtype=np.float64) + np.asarray(admin_corr_te, dtype=np.float64)
        pred_by_model["admin_direct"][te_sub] = admin_pred
        admin_mape = mape_pop_pct(y_log[te_sub], admin_pred)
        admin_adj_mape, admin_adj_rel = adjusted_fold_mape(pep_mape, admin_mape, 0.0)
        fold_rows.append(
            {
                "diagnostic": "orthogonal",
                "model": "admin_direct",
                "fold": int(split.fold_id),
                "n_train": int(tr_sub.size),
                "n_test": int(te_sub.size),
                "feature_dim": int(admin_dim),
                "graph_key": "none",
                "topology_leakage_proxy": 0.0,
                "pep_mape_pop_pct": float(pep_mape),
                "mape_pop_pct": float(admin_mape),
                "adjusted_mape_pop_pct": float(admin_adj_mape),
                "adjusted_relative_improvement_pct": float(admin_adj_rel),
                "heldout_states": ",".join(split.heldout_states),
                "heldout_regions": ",".join(split.heldout_regions),
            }
        )
        leftover_tr = np.asarray(true_resid[tr_sub], dtype=np.float64) - np.asarray(admin_corr_tr, dtype=np.float64)
        for key in candidates:
            x_mem = np.asarray(mems[key], dtype=np.float64)
            mem_train_mask = np.isfinite(x_mem[tr_sub]).all(axis=1) & np.isfinite(leftover_tr)
            mem_test_mask = np.isfinite(x_mem[te_sub]).all(axis=1)
            tr_mem = tr_sub[mem_train_mask]
            te_mem = te_sub[mem_test_mask]
            if tr_mem.size <= 1 or te_mem.size <= 0:
                continue
            leftover_target = np.asarray(true_resid[tr_mem], dtype=np.float64)
            admin_corr_tr_mem, _sigma_admin_mem, _ = fit_predict(
                model_cfg=model_cfg,
                Xtr=direct[tr_sub],
                ytr=true_resid[tr_sub],
                Xte=direct[tr_mem],
                seed=int(nowcast_cfg.evaluation.seed) + 7000 + int(split.fold_id),
            )
            leftover_target = leftover_target - np.asarray(admin_corr_tr_mem, dtype=np.float64)
            extra_te, _sigma_extra, mem_dim = fit_predict(
                model_cfg=model_cfg,
                Xtr=x_mem[tr_mem],
                ytr=leftover_target,
                Xte=x_mem[te_mem],
                seed=int(nowcast_cfg.evaluation.seed) + 8000 + int(split.fold_id),
            )
            # Align admin correction for the MEM-filtered test subset.
            admin_corr_for_mem, _sigma_admin_te, _ = fit_predict(
                model_cfg=model_cfg,
                Xtr=direct[tr_sub],
                ytr=true_resid[tr_sub],
                Xte=direct[te_mem],
                seed=int(nowcast_cfg.evaluation.seed) + 9000 + int(split.fold_id),
            )
            pred = np.asarray(pep_log[te_mem], dtype=np.float64) + np.asarray(admin_corr_for_mem, dtype=np.float64) + np.asarray(extra_te, dtype=np.float64)
            model_name = f"admin_residual_{key}_mem"
            pred_by_model[model_name][te_mem] = pred
            leakage = compute_topology_leakage_proxy(
                edges=mem_blocks[key].edges,
                sample_ids=sample_ids,
                test_idx=te_mem,
                mode=str(nowcast_cfg.analysis.leakage_proxy_mode),
            )
            mape = mape_pop_pct(y_log[te_mem], pred)
            adj_mape, adj_rel = adjusted_fold_mape(pep_mape, mape, leakage)
            fold_rows.append(
                {
                    "diagnostic": "orthogonal",
                    "model": model_name,
                    "fold": int(split.fold_id),
                    "n_train": int(tr_mem.size),
                    "n_test": int(te_mem.size),
                    "feature_dim": int(mem_dim),
                    "graph_key": str(key),
                    "topology_leakage_proxy": float(leakage),
                    "pep_mape_pop_pct": float(pep_mape),
                    "mape_pop_pct": float(mape),
                    "adjusted_mape_pop_pct": float(adj_mape),
                    "adjusted_relative_improvement_pct": float(adj_rel),
                    "heldout_states": ",".join(split.heldout_states),
                    "heldout_regions": ",".join(split.heldout_regions),
                }
            )
    abs_rows: list[dict[str, object]] = []
    for model_name, pred in pred_by_model.items():
        for i, fips in enumerate(sample_ids.tolist()):
            if not np.isfinite(pred[i]):
                continue
            abs_rows.append(
                {
                    "diagnostic": "orthogonal",
                    "model": model_name,
                    "fips": str(fips),
                    "state": str(fips)[:2],
                    "y_log": float(y_log[i]),
                    "y_level": float(y_level[i]),
                    "pep_log": float(pep_log[i]),
                    "pep_level": float(pep_level[i]),
                    "pred_log": float(pred[i]),
                    "pred_level": float(np.exp(pred[i])),
                    "true_resid_log": float(true_resid[i]),
                    "pred_correction_log": float(pred[i] - pep_log[i]),
                    "ape_pop_pct": float(abs(np.exp(pred[i]) - y_level[i]) / max(abs(y_level[i]), 1e-9) * 100.0),
                    "pep_ape_pop_pct": float(abs(pep_level[i] - y_level[i]) / max(abs(y_level[i]), 1e-9) * 100.0),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(abs_rows)


def build_summary(fold_df: pd.DataFrame, abs_df: pd.DataFrame, *, hard_case_quantile: float) -> pd.DataFrame:
    if fold_df.empty or abs_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    hard_cutoff = float(np.nanquantile(np.asarray(abs_df["pep_ape_pop_pct"], dtype=np.float64), float(hard_case_quantile)))
    for (diagnostic, model), folds in fold_df.groupby(["diagnostic", "model"], sort=True):
        part = abs_df.loc[(abs_df["diagnostic"] == diagnostic) & (abs_df["model"] == model)].copy()
        hard = part.loc[np.asarray(part["pep_ape_pop_pct"], dtype=np.float64) >= hard_cutoff].copy()
        baseline = float(np.nanmean(np.asarray(folds["pep_mape_pop_pct"], dtype=np.float64)))
        treatment = float(np.nanmean(np.asarray(folds["mape_pop_pct"], dtype=np.float64)))
        adjusted = float(np.nanmean(np.asarray(folds["adjusted_mape_pop_pct"], dtype=np.float64)))
        hard_base = float(np.nanmean(np.asarray(hard["pep_ape_pop_pct"], dtype=np.float64))) if not hard.empty else float("nan")
        hard_treat = float(np.nanmean(np.asarray(hard["ape_pop_pct"], dtype=np.float64))) if not hard.empty else float("nan")
        rows.append(
            {
                "diagnostic": str(diagnostic),
                "model": str(model),
                "n_counties": int(part["fips"].nunique()),
                "n_hard_case_counties": int(hard["fips"].nunique()),
                "baseline_mape_pct": baseline,
                "treatment_mape_pct": treatment,
                "adjusted_mape_pct": adjusted,
                "adjusted_relative_improvement_pct": float((baseline - adjusted) / max(baseline, 1e-9) * 100.0),
                "hard_case_cutoff_pep_ape_pct": hard_cutoff,
                "hard_case_treatment_mape_pct": hard_treat,
                "hard_case_relative_improvement_pct": float((hard_base - hard_treat) / max(hard_base, 1e-9) * 100.0) if np.isfinite(hard_base) else float("nan"),
                "residual_corr_pearson": finite_pearson(np.asarray(part["true_resid_log"], dtype=np.float64), np.asarray(part["pred_correction_log"], dtype=np.float64)),
                "mean_topology_leakage_proxy": float(np.nanmean(np.asarray(folds["topology_leakage_proxy"], dtype=np.float64))),
            }
        )
    return pd.DataFrame(rows).sort_values(["diagnostic", "adjusted_mape_pct", "model"]).reset_index(drop=True)


def load_mem_block_for_graph(cfg: RemoteResidualConfig, graph_key: str, nowcast_base: NowcastConfig) -> GraphMemBlock:
    graph_run = cfg.graphs[graph_key]
    graph_cfg = load_graph_config(cfg.graph_config_path, best_trial_json=graph_run.graph_best_trial_json)
    if cfg.materialize_graphs:
        run_graph_topology(graph_cfg, skip_existing=True, family_end_year=int(cfg.strict_year), source_year=int(cfg.strict_year))
    nowcast_cfg = load_nowcast_config(
        cfg.nowcast_config_path,
        graph_best_trial_json=graph_run.graph_best_trial_json,
    )
    rows = load_topology_rows(
        basis_parquet=nowcast_cfg.paths.topology_basis_parquet,
        runs_parquet=nowcast_cfg.paths.topology_runs_parquet,
        graph_tag_base=nowcast_cfg.graph.graph_tag_base,
        graph_kind=nowcast_cfg.graph.graph_kind,
        family_end_year=int(cfg.strict_year),
        source_year=int(cfg.strict_year),
        top_k=int(nowcast_cfg.graph.mem_top_k),
    )
    edges = load_topology_edges(
        edges_parquet=nowcast_cfg.paths.topology_edges_parquet,
        graph_tag_name=rows.graph_tag,
        graph_kind=rows.graph_kind,
        source_year=int(cfg.strict_year),
    )
    return GraphMemBlock(
        key=str(graph_key),
        graph_tag_base=str(nowcast_cfg.graph.graph_tag_base),
        graph_tag=str(rows.graph_tag),
        graph_kind=str(rows.graph_kind),
        mem_top_k=int(nowcast_cfg.graph.mem_top_k),
        fips=np.asarray(rows.fips, dtype="U5"),
        x=np.asarray(rows.x, dtype=np.float64),
        edges=edges,
    )


def run(cfg: RemoteResidualConfig) -> None:
    nowcast_cfg = load_nowcast_config(
        cfg.nowcast_config_path,
        linear_best_trial_json=cfg.linear_best_trial_json,
    )
    if int(nowcast_cfg.evaluation.strict_year) != int(cfg.strict_year):
        LOGGER.warning("config strict_year=%d differs from nowcast strict_year=%d; using %d", int(cfg.strict_year), int(nowcast_cfg.evaluation.strict_year), int(cfg.strict_year))
    truth = load_pep_year(nowcast_cfg.paths.pep_parquet, year=int(cfg.strict_year))
    truth = truth.loc[np.isfinite(np.asarray(truth["y_log"], dtype=np.float64)) & np.isfinite(np.asarray(truth["pep_log"], dtype=np.float64))].copy()
    direct_cfg = nowcast_cfg.block_cfg(cfg.direct_modality)
    direct = load_modality_block(
        direct_cfg,
        family_end_year=int(cfg.strict_year),
        source_year=int(cfg.strict_year),
        pool_mode=str(nowcast_cfg.evaluation.tile_pool_mode),
    )
    mem_blocks = {key: load_mem_block_for_graph(cfg, key, nowcast_cfg) for key in cfg.graphs}
    aligned = aligned_rows(truth=truth, direct=direct, mems=mem_blocks)
    sets = feature_sets(np.asarray(aligned["direct"], dtype=np.float64), {k: np.asarray(v, dtype=np.float64) for k, v in aligned["mem"].items()})
    fold_inc, abs_inc = run_joint_incremental(
        sample_ids=np.asarray(aligned["sample_ids"], dtype="U5"),
        y_log=np.asarray(aligned["y_log"], dtype=np.float64),
        y_level=np.asarray(aligned["y_level"], dtype=np.float64),
        pep_log=np.asarray(aligned["pep_log"], dtype=np.float64),
        pep_level=np.asarray(aligned["pep_level"], dtype=np.float64),
        sets=sets,
        mems=mem_blocks,
        nowcast_cfg=nowcast_cfg,
    )
    fold_ortho, abs_ortho = run_orthogonal_residual(
        sample_ids=np.asarray(aligned["sample_ids"], dtype="U5"),
        y_log=np.asarray(aligned["y_log"], dtype=np.float64),
        y_level=np.asarray(aligned["y_level"], dtype=np.float64),
        pep_log=np.asarray(aligned["pep_log"], dtype=np.float64),
        pep_level=np.asarray(aligned["pep_level"], dtype=np.float64),
        direct=np.asarray(aligned["direct"], dtype=np.float64),
        mems={k: np.asarray(v, dtype=np.float64) for k, v in aligned["mem"].items()},
        mem_blocks=mem_blocks,
        nowcast_cfg=nowcast_cfg,
    )
    fold_df = pd.concat([fold_inc, fold_ortho], axis=0, ignore_index=True)
    abs_df = pd.concat([abs_inc, abs_ortho], axis=0, ignore_index=True)
    summary = build_summary(fold_df, abs_df, hard_case_quantile=float(cfg.hard_case_quantile))

    out = Path(cfg.output_root)
    out.mkdir(parents=True, exist_ok=True)
    write_frame(summary, out / "remote_residual_summary.parquet")
    write_frame(fold_df, out / "remote_residual_fold_metrics.parquet")
    write_frame(abs_df, out / "remote_residual_abs_errors.parquet")
    meta = {
        "strict_year": int(cfg.strict_year),
        "direct_modality": str(cfg.direct_modality),
        "model_key": str(nowcast_cfg.downstream.selected),
        "linear_best_trial_json": "" if cfg.linear_best_trial_json is None else str(cfg.linear_best_trial_json),
        "n_shared_counties": int(np.asarray(aligned["sample_ids"]).shape[0]),
        "graphs": {
            key: {
                "graph_tag_base": block.graph_tag_base,
                "graph_tag": block.graph_tag,
                "mem_top_k": int(block.mem_top_k),
            }
            for key, block in mem_blocks.items()
        },
    }
    with open(out / "remote_residual_summary.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    LOGGER.info("remote residual summary\n%s", summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    LOGGER.info("wrote remote residual artifacts to %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose remote MEM value after administrative residual correction.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/remote_residual.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run(load_remote_residual_config(args.config))


if __name__ == "__main__":
    main()
