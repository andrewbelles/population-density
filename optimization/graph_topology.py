#!/usr/bin/env python3
#
# graph_topology.py  Andrew Belles  Apr 7th, 2026
#
# Optuna tuner for graph topology ablations scored by strict downstream admin+MEM evaluation.
#

import argparse
import copy
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml

from graph.config import GraphConfig as StageGraphConfig
from graph.config import ModalityConfig as GraphModalityConfig
from graph.config import TopologyConfig, load_config as load_graph_config
from graph.topology import graph_tag, train_graph_slice
from nowcast.common import (
    TopologyRows,
    align_rows,
    apply_block_pca,
    build_state_group_splits,
    fit_predict,
    load_modality_block,
    load_pep_year,
    mape_pop_pct,
)
from nowcast.config import NowcastConfig, load_config as load_nowcast_config
from optimization.common import (
    GL2Config,
    GL2StudyStopper,
    StudyConfig,
    UPSweepConfig,
    best_completed_trial,
    centered_unit_interval_score,
    create_study,
    normalize_weights,
    scalarize_values,
    setup_logging,
    signed_tanh_score,
    suggest_from_space,
    trial_payload,
    up_s_should_stop,
    write_json,
)


LOGGER = logging.getLogger("optimization.graph_topology")
DEAD_GRAPH_FIELDS = {"pool_mode", "attention_hidden_dim", "attention_dropout", "netvlad_clusters", "remote_gating"}
FUSION_LOGIT_PREFIX = "fusion_logit."


@dataclass(slots=True)
class SliceConfig:
    family_end_year: int
    source_year: int


@dataclass(slots=True)
class ObjectiveConfig:
    model_key: str
    direct_modality: str
    weight_adjusted_global_delta: float
    weight_adjusted_hard_case_delta: float
    weight_low_top_k: float
    hard_case_quantile: float
    adjusted_global_scale_pct: float
    adjusted_hard_case_scale_pct: float
    center_low_top_k: bool

    @property
    def weights(self) -> list[float]:
        return normalize_weights([self.weight_adjusted_global_delta, self.weight_adjusted_hard_case_delta, self.weight_low_top_k])

    @property
    def predictive_weights(self) -> list[float]:
        return normalize_weights([self.weight_adjusted_global_delta, self.weight_adjusted_hard_case_delta])

    def normalize_adjusted_global(self, value: float) -> float:
        return signed_tanh_score(float(value), scale=float(self.adjusted_global_scale_pct))

    def normalize_adjusted_hard_case(self, value: float) -> float:
        return signed_tanh_score(float(value), scale=float(self.adjusted_hard_case_scale_pct))

    def normalize_low_top_k(self, value: float) -> float:
        if bool(self.center_low_top_k):
            return centered_unit_interval_score(float(value))
        return float(np.clip(float(value), 0.0, 1.0))


@dataclass(slots=True)
class GroupConfig:
    signal_modalities: list[str]
    admin_modality: str
    include_full_signal: bool
    include_signal_only: bool
    include_remove_one: bool
    include_admin_only: bool


@dataclass(slots=True)
class GraphTuneConfig:
    graph_config_path: Path
    nowcast_config_path: Path
    output_root: Path
    slice_cfg: SliceConfig
    study: StudyConfig
    gl2: GL2Config
    up2: UPSweepConfig
    objective: ObjectiveConfig
    groups: GroupConfig
    graph_overrides: dict[str, Any]
    search_space: dict[str, dict[str, Any]]


@dataclass(slots=True)
class AblationGroup:
    name: str
    modalities: list[str]
    graph_tag_base: str
    group_kinds: list[str]


@dataclass(slots=True)
class DownstreamScore:
    objective_scalar: float
    adjusted_global_relative_improvement_pct: float
    adjusted_hard_case_relative_improvement_pct: float
    low_top_k_score: float
    normalized_adjusted_global_score: float
    normalized_adjusted_hard_case_score: float
    normalized_low_top_k_score: float
    selected_mem_top_k: int
    baseline_mape_pop_pct: float
    treatment_mape_pop_pct: float
    adjusted_mape_pop_pct: float
    baseline_hard_case_mape_pop_pct: float
    treatment_hard_case_mape_pop_pct: float
    adjusted_hard_case_mape_pop_pct: float
    n_eval_counties: int
    n_hard_case_counties: int
    k_sweep_history: list[dict[str, Any]]


@dataclass(slots=True)
class PreparedDownstreamInputs:
    sample_ids: np.ndarray
    y_log: np.ndarray
    y_level: np.ndarray
    pep_log: np.ndarray
    direct_x: np.ndarray
    direct_mask: np.ndarray
    mem_x_full: np.ndarray
    mem_mask: np.ndarray
    weight_aligned: np.ndarray
    splits: list[Any]
    max_k: int
    direct_modality: str


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def load_tune_config(path: str | Path) -> GraphTuneConfig:
    cfg_path = _as_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    slice_raw = dict(_require(raw, "slice"))
    study_raw = dict(_require(raw, "study"))
    gl2_raw = dict(raw.get("gl2", {}))
    up2_raw = dict(raw.get("up2", {}))
    objective_raw = dict(_require(raw, "objective"))
    groups_raw = dict(_require(raw, "groups"))
    graph_overrides = dict(raw.get("graph_overrides", {}))
    search_space = {str(k): dict(v) for k, v in dict(raw.get("search_space", {})).items()}
    graph_field_names = set(StageGraphConfig.__dataclass_fields__.keys())
    bad_search = sorted(DEAD_GRAPH_FIELDS.intersection(search_space))
    if bad_search:
        raise ValueError(f"graph topology tuner does not permit dead search dimensions: {bad_search}")
    bad_overrides = sorted(DEAD_GRAPH_FIELDS.intersection(graph_overrides))
    if bad_overrides:
        raise ValueError(f"graph topology tuner does not permit dead graph overrides: {bad_overrides}")
    unknown_search = sorted(
        name
        for name in set(search_space)
        if (not str(name).startswith(FUSION_LOGIT_PREFIX)) and name not in graph_field_names
    )
    if unknown_search:
        raise ValueError(f"graph topology tuner search space contains unknown graph fields: {unknown_search}")
    unknown_overrides = sorted(
        name
        for name in set(graph_overrides)
        if (not str(name).startswith(FUSION_LOGIT_PREFIX)) and name not in graph_field_names
    )
    if unknown_overrides:
        raise ValueError(f"graph topology tuner overrides contain unknown graph fields: {unknown_overrides}")
    return GraphTuneConfig(
        graph_config_path=_as_path(_require(raw, "graph_config_path")),
        nowcast_config_path=_as_path(_require(raw, "nowcast_config_path")),
        output_root=_as_path(raw.get("output_root", "optimization/runs")),
        slice_cfg=SliceConfig(
            family_end_year=int(_require(slice_raw, "family_end_year")),
            source_year=int(_require(slice_raw, "source_year")),
        ),
        study=StudyConfig(
            study_name=str(study_raw.get("study_name", "graph_topology_ablation")).strip(),
            direction=str(study_raw.get("direction", "maximize")).strip().lower(),
            n_trials=int(study_raw.get("n_trials", 40)),
            timeout_sec=int(study_raw.get("timeout_sec", 0)),
            sampler_seed=int(study_raw.get("sampler_seed", 0)),
            n_startup_trials=int(study_raw.get("n_startup_trials", 8)),
            gc_after_trial=bool(study_raw.get("gc_after_trial", True)),
        ),
        gl2=GL2Config(
            enabled=bool(gl2_raw.get("enabled", True)),
            min_trials=int(gl2_raw.get("min_trials", 12)),
            patience=int(gl2_raw.get("patience", 10)),
            max_generalization_loss_pct=float(gl2_raw.get("max_generalization_loss_pct", 0.5)),
            min_relative_improvement_pct=float(gl2_raw.get("min_relative_improvement_pct", 0.0)),
        ),
        up2=UPSweepConfig(
            enabled=bool(up2_raw.get("enabled", True)),
            successive_worsening_strips=int(up2_raw.get("successive_worsening_strips", 2)),
            strip_length=int(up2_raw.get("strip_length", 5)),
        ),
        objective=ObjectiveConfig(
            model_key=str(objective_raw.get("model_key", "huber")).strip().lower(),
            direct_modality=str(objective_raw.get("direct_modality", "admin")).strip().lower(),
            weight_adjusted_global_delta=float(objective_raw.get("weight_adjusted_global_delta", 0.70)),
            weight_adjusted_hard_case_delta=float(objective_raw.get("weight_adjusted_hard_case_delta", 0.20)),
            weight_low_top_k=float(objective_raw.get("weight_low_top_k", 0.10)),
            hard_case_quantile=float(objective_raw.get("hard_case_quantile", 0.90)),
            adjusted_global_scale_pct=float(objective_raw.get("adjusted_global_scale_pct", 5.0)),
            adjusted_hard_case_scale_pct=float(objective_raw.get("adjusted_hard_case_scale_pct", 5.0)),
            center_low_top_k=bool(objective_raw.get("center_low_top_k", True)),
        ),
        groups=GroupConfig(
            signal_modalities=[str(x).strip().lower() for x in list(groups_raw.get("signal_modalities", ["admin", "viirs", "s5p"]))],
            admin_modality=str(groups_raw.get("admin_modality", "admin")).strip().lower(),
            include_full_signal=bool(groups_raw.get("include_full_signal", True)),
            include_signal_only=bool(groups_raw.get("include_signal_only", True)),
            include_remove_one=bool(groups_raw.get("include_remove_one", True)),
            include_admin_only=bool(groups_raw.get("include_admin_only", False)),
        ),
        graph_overrides=graph_overrides,
        search_space=search_space,
    )


def graph_modality_from_nowcast(mod_cfg: Any) -> GraphModalityConfig:
    return GraphModalityConfig(
        enabled=bool(mod_cfg.enabled),
        name=str(mod_cfg.name).strip().lower(),
        kind=str(mod_cfg.kind).strip().lower(),
        input_parquet=Path(mod_cfg.input_parquet),
        family_tag_base=str(mod_cfg.family_tag_base),
        bag_keep_rate=1.0,
    )


def modality_registry(base_graph_cfg: TopologyConfig, nowcast_cfg: NowcastConfig, tune_cfg: GraphTuneConfig) -> dict[str, GraphModalityConfig]:
    registry: dict[str, GraphModalityConfig] = {str(k): copy.deepcopy(v) for k, v in base_graph_cfg.blocks.items()}
    admin_key = str(tune_cfg.groups.admin_modality)
    if admin_key not in registry and admin_key in nowcast_cfg.blocks:
        registry[admin_key] = graph_modality_from_nowcast(nowcast_cfg.block_cfg(admin_key))
    for key in tune_cfg.groups.signal_modalities:
        if key in registry:
            continue
        if key in nowcast_cfg.blocks:
            registry[key] = graph_modality_from_nowcast(nowcast_cfg.block_cfg(key))
    return registry


def build_ablation_groups(base_graph_cfg: TopologyConfig, registry: dict[str, GraphModalityConfig], tune_cfg: GraphTuneConfig) -> list[AblationGroup]:
    signal_modalities = [m for m in tune_cfg.groups.signal_modalities if m in registry and bool(registry[m].enabled)]
    if not signal_modalities:
        raise ValueError("no enabled signal modalities were available for graph tuning")
    groups: list[AblationGroup] = []
    seen: dict[tuple[str, ...], int] = {}

    def add_group(name: str, modalities: list[str], *, group_kind: str) -> None:
        mods = [str(m).strip().lower() for m in modalities if str(m).strip().lower() in registry]
        if not mods:
            return
        key = tuple(mods)
        if key in seen:
            groups[seen[key]].group_kinds.append(str(group_kind))
            return
        tag_base = f"gsl_meanmax_consensus_{'_'.join(mods)}"
        groups.append(AblationGroup(name=str(name), modalities=mods, graph_tag_base=tag_base, group_kinds=[str(group_kind)]))
        seen[key] = int(len(groups) - 1)

    if bool(tune_cfg.groups.include_full_signal):
        add_group("full_signal", list(signal_modalities), group_kind="full_signal")
    if bool(tune_cfg.groups.include_signal_only):
        for modality in signal_modalities:
            add_group(f"signal_only_{modality}", [str(modality)], group_kind="signal_only")
    if bool(tune_cfg.groups.include_remove_one) and len(signal_modalities) > 1:
        for modality in signal_modalities:
            mods = [m for m in signal_modalities if m != modality]
            add_group(f"remove_{modality}", mods, group_kind="remove_one")
    admin_key = str(tune_cfg.groups.admin_modality)
    if bool(tune_cfg.groups.include_admin_only) and admin_key in registry and bool(registry[admin_key].enabled):
        # Optional explicit alias for the single-admin signal case. When admin is
        # already part of signal_modalities, this merges into signal_only_admin.
        add_group("admin_only", [admin_key], group_kind="admin_only")
    return groups


def apply_graph_params(base_graph: StageGraphConfig, *, overrides: dict[str, Any], params: dict[str, Any], graph_tag_base: str) -> StageGraphConfig:
    data = {
        field: getattr(base_graph, field)
        for field in base_graph.__dataclass_fields__.keys()
    }
    override_split = split_param_mapping(params=dict(overrides))
    param_split = split_param_mapping(params=dict(params))
    data.update({str(k): v for k, v in dict(override_split["graph_params"]).items()})
    data.update({str(k): v for k, v in dict(param_split["graph_params"]).items()})
    fusion_logits = {str(k): float(v) for k, v in dict(data.get("fusion_logits", getattr(base_graph, "fusion_logits", {}))).items()}
    fusion_logits.update({str(k): float(v) for k, v in dict(override_split["fusion_logits"]).items()})
    fusion_logits.update({str(k): float(v) for k, v in dict(param_split["fusion_logits"]).items()})
    data["fusion_logits"] = fusion_logits
    data["graph_tag_base"] = str(graph_tag_base)
    return StageGraphConfig(**data)


def build_trial_topology_config(
    base_graph_cfg: TopologyConfig,
    registry: dict[str, GraphModalityConfig],
    *,
    group: AblationGroup,
    graph_cfg: StageGraphConfig,
) -> TopologyConfig:
    blocks = {str(k): copy.deepcopy(v) for k, v in registry.items()}
    return TopologyConfig(
        years=base_graph_cfg.years,
        modalities=list(group.modalities),
        paths=base_graph_cfg.paths,
        graph=graph_cfg,
        blocks=blocks,
    )


def split_param_mapping(
    *,
    params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    graph_params: dict[str, Any] = {}
    fusion_logits: dict[str, float] = {}
    for raw_name, value in dict(params).items():
        name = str(raw_name)
        if name in DEAD_GRAPH_FIELDS:
            continue
        if str(name).startswith(FUSION_LOGIT_PREFIX):
            component = str(name).split(".", 1)[1].strip().lower()
            if component:
                fusion_logits[str(component)] = float(value)
            continue
        graph_params[name] = value
    return {"graph_params": graph_params, "fusion_logits": fusion_logits}


def split_trial_params(
    *,
    trial: optuna.trial.Trial,
    tune_cfg: GraphTuneConfig,
    group: AblationGroup,
) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for raw_name, spec in tune_cfg.search_space.items():
        name = str(raw_name)
        if str(name).startswith(FUSION_LOGIT_PREFIX):
            component = str(name).split(".", 1)[1].strip().lower()
            if component != "consensus" and component not in set(group.modalities):
                continue
        sampled[name] = suggest_from_space(trial, name, dict(spec))
    split = split_param_mapping(params=sampled)
    out = dict(split["graph_params"])
    for component, value in dict(split["fusion_logits"]).items():
        out[f"{FUSION_LOGIT_PREFIX}{component}"] = float(value)
    return out


def compute_topology_leakage_proxy_matrix(
    *,
    weights: np.ndarray,
    sample_ids: np.ndarray,
    test_idx: np.ndarray,
    mode: str,
) -> float:
    w = np.asarray(weights, dtype=np.float64)
    ids = np.asarray(sample_ids, dtype="U5")
    n = int(ids.shape[0])
    if w.ndim != 2 or w.shape[0] != n or w.shape[1] != n:
        raise ValueError("weights must be square and aligned to sample_ids")
    te_idx = np.asarray(test_idx, dtype=np.int64)
    if te_idx.size <= 0:
        return 0.0
    test_mask = np.zeros(n, dtype=bool)
    test_mask[te_idx] = True
    train_mask = ~test_mask
    if int(np.count_nonzero(test_mask)) <= 0 or int(np.count_nonzero(train_mask)) <= 0:
        return 0.0
    mode_name = str(mode).strip().lower()
    outbound = np.asarray(w[np.ix_(test_mask, np.ones(n, dtype=bool))], dtype=np.float64)
    outbound_sum = np.asarray(np.sum(outbound, axis=1), dtype=np.float64)
    outbound_cross = np.asarray(np.sum(w[np.ix_(test_mask, train_mask)], axis=1), dtype=np.float64)
    outbound_ratio = np.divide(outbound_cross, outbound_sum, out=np.full(outbound_sum.shape, np.nan, dtype=np.float64), where=outbound_sum > 0.0)
    outbound_proxy = float(np.nanmean(outbound_ratio)) if outbound_ratio.size > 0 else 0.0
    if mode_name == "outbound":
        return float(np.clip(outbound_proxy, 0.0, 1.0))
    inbound_sum = np.asarray(np.sum(w[np.ix_(np.ones(n, dtype=bool), test_mask)], axis=0), dtype=np.float64)
    inbound_cross = np.asarray(np.sum(w[np.ix_(train_mask, test_mask)], axis=0), dtype=np.float64)
    inbound_ratio = np.divide(inbound_cross, inbound_sum, out=np.full(inbound_sum.shape, np.nan, dtype=np.float64), where=inbound_sum > 0.0)
    inbound_proxy = float(np.nanmean(inbound_ratio)) if inbound_ratio.size > 0 else 0.0
    if mode_name == "inbound":
        return float(np.clip(inbound_proxy, 0.0, 1.0))
    return float(np.clip(0.5 * (outbound_proxy + inbound_proxy), 0.0, 1.0))


def relative_improvement_pct(*, baseline_error: float, treatment_error: float) -> float:
    base = float(baseline_error)
    if (not np.isfinite(base)) or base <= 1e-9:
        return float("nan")
    treat = float(treatment_error)
    return float((base - treat) / base * 100.0)


def adjusted_relative_improvement_pct(*, relative_improvement_pct_value: float, leakage_proxy: float) -> float:
    rel = float(relative_improvement_pct_value)
    proxy = float(np.clip(leakage_proxy, 0.0, 1.0))
    return float(rel - max(rel, 0.0) * proxy)


def low_top_k_score(*, selected_k: int, max_k: int) -> float:
    k = int(max(1, selected_k))
    k_max = int(max(1, max_k))
    if k_max <= 1:
        return 1.0
    return float(1.0 - (float(k - 1) / float(k_max - 1)))


def align_weight_matrix_to_sample_ids(*, artifact_fips: np.ndarray, weights: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    art_ids = np.asarray(artifact_fips, dtype="U5")
    samp_ids = np.asarray(sample_ids, dtype="U5")
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != art_ids.shape[0] or w.shape[1] != art_ids.shape[0]:
        raise ValueError("artifact weights must be square and aligned to artifact_fips")
    n = int(samp_ids.shape[0])
    out = np.zeros((n, n), dtype=np.float64)
    art_index = {str(fid): i for i, fid in enumerate(art_ids.tolist())}
    rows: list[int] = []
    cols: list[int] = []
    for i, fid in enumerate(samp_ids.tolist()):
        j = art_index.get(str(fid))
        if j is None:
            continue
        rows.append(int(i))
        cols.append(int(j))
    if rows:
        rr = np.asarray(rows, dtype=np.int64)
        cc = np.asarray(cols, dtype=np.int64)
        out[np.ix_(rr, rr)] = np.asarray(w[np.ix_(cc, cc)], dtype=np.float64)
    return out


def prepare_downstream_candidate(
    *,
    nowcast_cfg: NowcastConfig,
    family_end_year: int,
    source_year: int,
    artifact: Any,
    objective_cfg: ObjectiveConfig,
) -> PreparedDownstreamInputs:
    truth_pep = load_pep_year(nowcast_cfg.paths.pep_parquet, year=int(nowcast_cfg.evaluation.strict_year))
    truth_pep = truth_pep.loc[
        np.isfinite(np.asarray(truth_pep["y_log"], dtype=np.float64))
        & np.isfinite(np.asarray(truth_pep["pep_log"], dtype=np.float64))
    ].copy()
    if truth_pep.empty:
        raise ValueError("strict evaluation produced no truth rows")

    direct_modality = str(objective_cfg.direct_modality)
    direct_cfg = nowcast_cfg.block_cfg(str(direct_modality))
    direct_block = load_modality_block(
        direct_cfg,
        family_end_year=int(family_end_year),
        source_year=int(source_year),
        pool_mode=str(nowcast_cfg.evaluation.tile_pool_mode),
    )
    mem_block = TopologyRows(
        fips=np.asarray(artifact.fips, dtype="U5"),
        x=np.asarray(artifact.evecs_learn, dtype=np.float64),
        graph_tag=f"candidate_y{int(family_end_year)}",
        graph_kind="learned",
        graph_loss=float(artifact.graph_loss),
        graph_counties=int(np.asarray(artifact.fips, dtype="U5").shape[0]),
    )
    aligned = align_rows(
        truth_pep=truth_pep,
        direct_blocks={str(direct_modality): direct_block},
        mem_block=mem_block,
    )
    sample_ids = np.asarray(aligned["sample_ids"], dtype="U5")
    y_log = np.asarray(aligned["y_log"], dtype=np.float64)
    y_level = np.asarray(aligned["y_level"], dtype=np.float64)
    pep_log = np.asarray(aligned["pep_log"], dtype=np.float64)
    direct_x = np.asarray(aligned[str(direct_modality)], dtype=np.float64)
    direct_mask = np.asarray(aligned[f"{direct_modality}_mask"], dtype=bool)
    mem_x_full = np.asarray(aligned["mem"], dtype=np.float64)
    mem_mask = np.asarray(aligned["mem_mask"], dtype=bool)
    weight_aligned = align_weight_matrix_to_sample_ids(
        artifact_fips=np.asarray(artifact.fips, dtype="U5"),
        weights=np.asarray(artifact.w_learn, dtype=np.float64),
        sample_ids=sample_ids,
    )
    model_cfg = nowcast_cfg.downstream.model_cfg(str(objective_cfg.model_key))
    splits = build_state_group_splits(
        sample_ids,
        n_splits=int(nowcast_cfg.evaluation.n_splits),
        strategy=str(nowcast_cfg.evaluation.fold_strategy),
        region_level=str(nowcast_cfg.evaluation.fold_region_level),
    )
    max_k = int(max(1, min(int(mem_x_full.shape[1]), int(np.asarray(artifact.evecs_learn, dtype=np.float64).shape[1]))))
    return PreparedDownstreamInputs(
        sample_ids=np.asarray(sample_ids, dtype="U5"),
        y_log=np.asarray(y_log, dtype=np.float64),
        y_level=np.asarray(y_level, dtype=np.float64),
        pep_log=np.asarray(pep_log, dtype=np.float64),
        direct_x=np.asarray(direct_x, dtype=np.float64),
        direct_mask=np.asarray(direct_mask, dtype=bool),
        mem_x_full=np.asarray(mem_x_full, dtype=np.float64),
        mem_mask=np.asarray(mem_mask, dtype=bool),
        weight_aligned=np.asarray(weight_aligned, dtype=np.float64),
        splits=list(splits),
        max_k=int(max_k),
        direct_modality=str(direct_modality),
    )


def score_downstream_candidate_fixed_k(
    *,
    prepared: PreparedDownstreamInputs,
    nowcast_cfg: NowcastConfig,
    objective_cfg: ObjectiveConfig,
    model_cfg: Any,
    mem_k: int,
    leakage_proxy_mode: str,
) -> DownstreamScore:
    mem_k_eff = int(max(1, min(int(mem_k), int(prepared.max_k))))
    pred = np.full(prepared.sample_ids.shape[0], np.nan, dtype=np.float64)
    fold_global_baseline: list[float] = []
    fold_global_treatment: list[float] = []
    fold_global_adjusted: list[float] = []
    fold_hard_baseline: list[float] = []
    fold_hard_treatment: list[float] = []
    fold_hard_adjusted: list[float] = []
    n_hard_total = 0

    for split in prepared.splits:
        tr_idx = np.asarray(split.train_idx, dtype=np.int64)
        te_idx = np.asarray(split.test_idx, dtype=np.int64)
        train_mask = np.asarray(prepared.direct_mask[tr_idx], dtype=bool) & np.asarray(prepared.mem_mask[tr_idx], dtype=bool)
        test_mask = np.asarray(prepared.direct_mask[te_idx], dtype=bool) & np.asarray(prepared.mem_mask[te_idx], dtype=bool)
        tr_sub = np.asarray(tr_idx[train_mask], dtype=np.int64)
        te_sub = np.asarray(te_idx[test_mask], dtype=np.int64)
        if tr_sub.shape[0] <= 1 or te_sub.shape[0] <= 0:
            continue
        Xemb_tr, Xemb_te, _ = apply_block_pca(
            blocks_tr={str(prepared.direct_modality): np.asarray(prepared.direct_x[tr_sub], dtype=np.float64)},
            blocks_te={str(prepared.direct_modality): np.asarray(prepared.direct_x[te_sub], dtype=np.float64)},
            reduce=bool(nowcast_cfg.evaluation.model_pca_reduce),
            dim=int(nowcast_cfg.evaluation.model_pca_dim),
            mode=str(nowcast_cfg.evaluation.model_pca_mode),
        )
        Xmem_tr = np.asarray(prepared.mem_x_full[tr_sub, :mem_k_eff], dtype=np.float64)
        Xmem_te = np.asarray(prepared.mem_x_full[te_sub, :mem_k_eff], dtype=np.float64)
        Ftr = np.concatenate([Xemb_tr, Xmem_tr], axis=1)
        Fte = np.concatenate([Xemb_te, Xmem_te], axis=1)
        ytr = np.asarray(prepared.y_log[tr_sub], dtype=np.float64)
        pep_tr = np.asarray(prepared.pep_log[tr_sub], dtype=np.float64)
        pep_te = np.asarray(prepared.pep_log[te_sub], dtype=np.float64)
        y_level_te = np.asarray(prepared.y_level[te_sub], dtype=np.float64)
        resid_tr = np.asarray(ytr - pep_tr, dtype=np.float64)
        train_fit_mask = np.isfinite(Ftr).all(axis=1) & np.isfinite(resid_tr)
        test_fit_mask = np.isfinite(Fte).all(axis=1) & np.isfinite(pep_te)
        tr_fit = np.asarray(np.flatnonzero(train_fit_mask), dtype=np.int64)
        te_fit = np.asarray(np.flatnonzero(test_fit_mask), dtype=np.int64)
        if tr_fit.shape[0] <= 1 or te_fit.shape[0] <= 0:
            continue
        mu_raw, _sig, _feat_dim = fit_predict(
            model_cfg=model_cfg,
            Xtr=np.asarray(Ftr[tr_fit], dtype=np.float64),
            ytr=np.asarray(resid_tr[tr_fit], dtype=np.float64),
            Xte=np.asarray(Fte[te_fit], dtype=np.float64),
            seed=int(nowcast_cfg.evaluation.seed) + 1000 + int(split.fold_id),
        )
        te_pred_idx = np.asarray(te_sub[te_fit], dtype=np.int64)
        mu = np.asarray(pep_te[te_fit], dtype=np.float64) + np.asarray(mu_raw, dtype=np.float64)
        pred[te_pred_idx] = mu

        yte_fit = np.asarray(prepared.y_log[te_pred_idx], dtype=np.float64)
        pep_te_fit = np.asarray(prepared.pep_log[te_pred_idx], dtype=np.float64)
        y_level_te_fit = np.asarray(prepared.y_level[te_pred_idx], dtype=np.float64)
        leakage_proxy = compute_topology_leakage_proxy_matrix(
            weights=np.asarray(prepared.weight_aligned, dtype=np.float64),
            sample_ids=prepared.sample_ids,
            test_idx=te_idx,
            mode=str(leakage_proxy_mode),
        )
        pep_mape = mape_pop_pct(yte_fit, pep_te_fit)
        treat_mape = mape_pop_pct(yte_fit, mu)
        rel_improve = relative_improvement_pct(baseline_error=pep_mape, treatment_error=treat_mape)
        adj_rel_improve = adjusted_relative_improvement_pct(relative_improvement_pct_value=rel_improve, leakage_proxy=leakage_proxy)
        fold_global_baseline.append(float(pep_mape))
        fold_global_treatment.append(float(treat_mape))
        fold_global_adjusted.append(float(pep_mape * (1.0 - adj_rel_improve / 100.0)))

        baseline_ape = np.abs(np.exp(pep_te_fit) - y_level_te_fit) / np.clip(np.abs(y_level_te_fit), 1e-9, None) * 100.0
        if baseline_ape.size > 0:
            cut = float(np.quantile(np.asarray(baseline_ape, dtype=np.float64), float(objective_cfg.hard_case_quantile)))
            hard_mask = np.asarray(baseline_ape >= cut, dtype=bool)
            if int(np.count_nonzero(hard_mask)) > 0:
                pep_hard = mape_pop_pct(np.asarray(yte_fit[hard_mask], dtype=np.float64), np.asarray(pep_te_fit[hard_mask], dtype=np.float64))
                treat_hard = mape_pop_pct(np.asarray(yte_fit[hard_mask], dtype=np.float64), np.asarray(mu[hard_mask], dtype=np.float64))
                rel_hard = relative_improvement_pct(baseline_error=pep_hard, treatment_error=treat_hard)
                adj_rel_hard = adjusted_relative_improvement_pct(relative_improvement_pct_value=rel_hard, leakage_proxy=leakage_proxy)
                fold_hard_baseline.append(float(pep_hard))
                fold_hard_treatment.append(float(treat_hard))
                fold_hard_adjusted.append(float(pep_hard * (1.0 - adj_rel_hard / 100.0)))
                n_hard_total += int(np.count_nonzero(hard_mask))

    valid = np.isfinite(pred) & np.isfinite(prepared.y_log) & np.isfinite(prepared.pep_log)
    if int(np.count_nonzero(valid)) <= 1 or len(fold_global_baseline) <= 0:
        raise RuntimeError("fixed-k downstream evaluation produced no valid folds")
    baseline_mape = float(np.mean(np.asarray(fold_global_baseline, dtype=np.float64)))
    treatment_mape = float(np.mean(np.asarray(fold_global_treatment, dtype=np.float64)))
    adjusted_mape = float(np.mean(np.asarray(fold_global_adjusted, dtype=np.float64)))
    adjusted_global_rel = relative_improvement_pct(baseline_error=baseline_mape, treatment_error=adjusted_mape)

    if len(fold_hard_baseline) > 0:
        baseline_hard = float(np.mean(np.asarray(fold_hard_baseline, dtype=np.float64)))
        treatment_hard = float(np.mean(np.asarray(fold_hard_treatment, dtype=np.float64)))
        adjusted_hard = float(np.mean(np.asarray(fold_hard_adjusted, dtype=np.float64)))
        adjusted_hard_rel = relative_improvement_pct(baseline_error=baseline_hard, treatment_error=adjusted_hard)
    else:
        baseline_hard = float("nan")
        treatment_hard = float("nan")
        adjusted_hard = float("nan")
        adjusted_hard_rel = float("nan")

    low_k = low_top_k_score(selected_k=int(mem_k_eff), max_k=int(prepared.max_k))
    norm_global = objective_cfg.normalize_adjusted_global(float(adjusted_global_rel))
    norm_hard = objective_cfg.normalize_adjusted_hard_case(float(adjusted_hard_rel))
    norm_low_k = objective_cfg.normalize_low_top_k(float(low_k))
    scalar = scalarize_values([float(norm_global), float(norm_hard), float(norm_low_k)], objective_cfg.weights)
    return DownstreamScore(
        objective_scalar=float(scalar),
        adjusted_global_relative_improvement_pct=float(adjusted_global_rel),
        adjusted_hard_case_relative_improvement_pct=float(adjusted_hard_rel),
        low_top_k_score=float(low_k),
        normalized_adjusted_global_score=float(norm_global),
        normalized_adjusted_hard_case_score=float(norm_hard),
        normalized_low_top_k_score=float(norm_low_k),
        selected_mem_top_k=int(mem_k_eff),
        baseline_mape_pop_pct=float(baseline_mape),
        treatment_mape_pop_pct=float(treatment_mape),
        adjusted_mape_pop_pct=float(adjusted_mape),
        baseline_hard_case_mape_pop_pct=float(baseline_hard),
        treatment_hard_case_mape_pop_pct=float(treatment_hard),
        adjusted_hard_case_mape_pop_pct=float(adjusted_hard),
        n_eval_counties=int(np.count_nonzero(valid)),
        n_hard_case_counties=int(n_hard_total),
        k_sweep_history=[],
    )


def evaluate_downstream_candidate(
    *,
    nowcast_cfg: NowcastConfig,
    family_end_year: int,
    source_year: int,
    artifact: Any,
    objective_cfg: ObjectiveConfig,
    leakage_proxy_mode: str,
    up2_cfg: UPSweepConfig,
) -> DownstreamScore:
    prepared = prepare_downstream_candidate(
        nowcast_cfg=nowcast_cfg,
        family_end_year=family_end_year,
        source_year=source_year,
        artifact=artifact,
        objective_cfg=objective_cfg,
    )
    model_cfg = nowcast_cfg.downstream.model_cfg(str(objective_cfg.model_key))
    k_history: list[dict[str, Any]] = []
    strip_scores: list[float] = []
    best_score: DownstreamScore | None = None

    for mem_k in range(1, int(prepared.max_k) + 1):
        score = score_downstream_candidate_fixed_k(
            prepared=prepared,
            nowcast_cfg=nowcast_cfg,
            objective_cfg=objective_cfg,
            model_cfg=model_cfg,
            mem_k=int(mem_k),
            leakage_proxy_mode=str(leakage_proxy_mode),
        )
        predictive_scalar = scalarize_values(
            [float(score.normalized_adjusted_global_score), float(score.normalized_adjusted_hard_case_score)],
            objective_cfg.predictive_weights,
        )
        point = {
            "mem_top_k": int(score.selected_mem_top_k),
            "adjusted_global_relative_improvement_pct": float(score.adjusted_global_relative_improvement_pct),
            "adjusted_hard_case_relative_improvement_pct": float(score.adjusted_hard_case_relative_improvement_pct),
            "low_top_k_score": float(score.low_top_k_score),
            "normalized_adjusted_global_score": float(score.normalized_adjusted_global_score),
            "normalized_adjusted_hard_case_score": float(score.normalized_adjusted_hard_case_score),
            "normalized_low_top_k_score": float(score.normalized_low_top_k_score),
            "predictive_scalar": float(predictive_scalar),
            "scalar_objective": float(score.objective_scalar),
            "baseline_mape_pop_pct": float(score.baseline_mape_pop_pct),
            "treatment_mape_pop_pct": float(score.treatment_mape_pop_pct),
            "adjusted_mape_pop_pct": float(score.adjusted_mape_pop_pct),
            "baseline_hard_case_mape_pop_pct": float(score.baseline_hard_case_mape_pop_pct),
            "treatment_hard_case_mape_pop_pct": float(score.treatment_hard_case_mape_pop_pct),
            "adjusted_hard_case_mape_pop_pct": float(score.adjusted_hard_case_mape_pop_pct),
            "n_eval_counties": int(score.n_eval_counties),
            "n_hard_case_counties": int(score.n_hard_case_counties),
        }
        k_history.append(point)
        score = replace(score, k_sweep_history=list(k_history))
        if best_score is None or float(score.objective_scalar) > float(best_score.objective_scalar):
            best_score = score

        if bool(up2_cfg.enabled) and int(up2_cfg.strip_length) > 0 and int(mem_k) % int(up2_cfg.strip_length) == 0:
            strip_scores.append(float(predictive_scalar))
            if up_s_should_stop(strip_scores, successive_worsening_strips=int(up2_cfg.successive_worsening_strips)):
                break

    if best_score is None:
        raise RuntimeError("candidate graph produced no valid MEM-rank downstream scores")
    return best_score


def group_run_path(root: Path, *, graph_tag_name: str, source_year: int) -> Path:
    return Path(root) / f"{graph_tag_name}_source_{int(source_year)}.json"


def optimize_group(
    *,
    tune_cfg: GraphTuneConfig,
    base_graph_cfg: TopologyConfig,
    nowcast_cfg: NowcastConfig,
    registry: dict[str, GraphModalityConfig],
    group: AblationGroup,
    skip_existing: bool,
) -> Path:
    family_end_year = int(tune_cfg.slice_cfg.family_end_year)
    source_year = int(tune_cfg.slice_cfg.source_year)
    run_path = group_run_path(
        tune_cfg.output_root,
        graph_tag_name=str(graph_tag(str(group.graph_tag_base), family_end_year)),
        source_year=source_year,
    )
    if bool(skip_existing) and run_path.exists():
        LOGGER.info("skip existing group=%s path=%s", str(group.name), run_path)
        return run_path

    study_cfg = replace(tune_cfg.study, study_name=f"{tune_cfg.study.study_name}:{group.name}")
    study = create_study(study_cfg)
    stopper = GL2StudyStopper(tune_cfg.gl2)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_params = split_trial_params(trial=trial, tune_cfg=tune_cfg, group=group)
        graph_cfg = apply_graph_params(
            base_graph_cfg.graph,
            overrides=tune_cfg.graph_overrides,
            params=trial_params,
            graph_tag_base=str(group.graph_tag_base),
        )
        trial_topology_cfg = build_trial_topology_config(
            base_graph_cfg,
            registry,
            group=group,
            graph_cfg=graph_cfg,
        )
        artifact = train_graph_slice(
            trial_topology_cfg,
            family_end_year=family_end_year,
            source_year=source_year,
        )
        score = evaluate_downstream_candidate(
            nowcast_cfg=nowcast_cfg,
            family_end_year=family_end_year,
            source_year=source_year,
            artifact=artifact,
            objective_cfg=tune_cfg.objective,
            leakage_proxy_mode=str(nowcast_cfg.analysis.leakage_proxy_mode),
            up2_cfg=tune_cfg.up2,
        )
        scalar = float(score.objective_scalar)
        trial.set_user_attr("scalar_objective", float(scalar))
        trial.set_user_attr("graph_loss", float(artifact.graph_loss))
        trial.set_user_attr("graph_counties", int(np.asarray(artifact.fips, dtype="U5").shape[0]))
        trial.set_user_attr("selected_mem_top_k", int(score.selected_mem_top_k))
        trial.set_user_attr("adjusted_global_relative_improvement_pct", float(score.adjusted_global_relative_improvement_pct))
        trial.set_user_attr("adjusted_hard_case_relative_improvement_pct", float(score.adjusted_hard_case_relative_improvement_pct))
        trial.set_user_attr("low_top_k_score", float(score.low_top_k_score))
        trial.set_user_attr("normalized_adjusted_global_score", float(score.normalized_adjusted_global_score))
        trial.set_user_attr("normalized_adjusted_hard_case_score", float(score.normalized_adjusted_hard_case_score))
        trial.set_user_attr("normalized_low_top_k_score", float(score.normalized_low_top_k_score))
        trial.set_user_attr("baseline_mape_pop_pct", float(score.baseline_mape_pop_pct))
        trial.set_user_attr("treatment_mape_pop_pct", float(score.treatment_mape_pop_pct))
        trial.set_user_attr("adjusted_mape_pop_pct", float(score.adjusted_mape_pop_pct))
        trial.set_user_attr("baseline_hard_case_mape_pop_pct", float(score.baseline_hard_case_mape_pop_pct))
        trial.set_user_attr("treatment_hard_case_mape_pop_pct", float(score.treatment_hard_case_mape_pop_pct))
        trial.set_user_attr("adjusted_hard_case_mape_pop_pct", float(score.adjusted_hard_case_mape_pop_pct))
        trial.set_user_attr("n_eval_counties", int(score.n_eval_counties))
        trial.set_user_attr("n_hard_case_counties", int(score.n_hard_case_counties))
        trial.set_user_attr("k_sweep_history", list(score.k_sweep_history))
        LOGGER.info(
            "[optuna graph] group=%s trial=%d modalities=%s adj_global=%.4f adj_hard=%.4f low_k=%.4f norm_global=%.4f norm_hard=%.4f norm_k=%.4f k=%d scalar=%.4f graph_loss=%.6f",
            str(group.name),
            int(trial.number),
            ",".join(group.modalities),
            float(score.adjusted_global_relative_improvement_pct),
            float(score.adjusted_hard_case_relative_improvement_pct),
            float(score.low_top_k_score),
            float(score.normalized_adjusted_global_score),
            float(score.normalized_adjusted_hard_case_score),
            float(score.normalized_low_top_k_score),
            int(score.selected_mem_top_k),
            float(scalar),
            float(artifact.graph_loss),
        )
        return float(scalar)

    study.optimize(
        objective,
        n_trials=int(study_cfg.n_trials),
        timeout=None if int(study_cfg.timeout_sec) <= 0 else int(study_cfg.timeout_sec),
        callbacks=[stopper],
        gc_after_trial=bool(study_cfg.gc_after_trial),
        show_progress_bar=False,
    )
    best_trial = best_completed_trial(study)
    best_value = None if best_trial.value is None else float(best_trial.value)
    payload = {
        "group_name": str(group.name),
        "group_kinds": list(group.group_kinds),
        "modalities": list(group.modalities),
        "graph_tag_base": str(group.graph_tag_base),
        "graph_tag": str(graph_tag(str(group.graph_tag_base), family_end_year)),
        "family_end_year": int(family_end_year),
        "source_year": int(source_year),
        "objective": {
            "model_key": str(tune_cfg.objective.model_key),
            "direct_modality": str(tune_cfg.objective.direct_modality),
            "hard_case_quantile": float(tune_cfg.objective.hard_case_quantile),
            "weights": {
                "adjusted_global_delta": float(tune_cfg.objective.weights[0]),
                "adjusted_hard_case_delta": float(tune_cfg.objective.weights[1]),
                "low_top_k": float(tune_cfg.objective.weights[2]),
            },
            "scaling": {
                "transform": "signed_tanh",
                "adjusted_global_scale_pct": float(tune_cfg.objective.adjusted_global_scale_pct),
                "adjusted_hard_case_scale_pct": float(tune_cfg.objective.adjusted_hard_case_scale_pct),
                "center_low_top_k": bool(tune_cfg.objective.center_low_top_k),
            },
        },
        "up2_mem_rank_sweep": {
            "enabled": bool(tune_cfg.up2.enabled),
            "successive_worsening_strips": int(tune_cfg.up2.successive_worsening_strips),
            "strip_length": int(tune_cfg.up2.strip_length),
        },
        "study": {
            "study_name": str(study.study_name),
            "direction": str(getattr(study.direction, "name", str(study.direction))).strip().lower(),
            "n_trials_requested": int(study_cfg.n_trials),
            "n_trials_completed": int(len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])),
            "n_trials_total": int(len(study.trials)),
        },
        "best_trial": {
            "number": int(best_trial.number),
            "value": best_value,
            "scalar_objective": float(best_trial.user_attrs.get("scalar_objective", best_value if best_value is not None else float("-inf"))),
            "params": dict(best_trial.params),
            "user_attrs": {str(k): v for k, v in best_trial.user_attrs.items()},
        },
        "graph_overrides": dict(tune_cfg.graph_overrides),
        "search_space": {str(k): dict(v) for k, v in tune_cfg.search_space.items()},
        "completed_trials": [
            trial_payload(t)
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ],
    }
    write_json(run_path, payload)
    LOGGER.info(
        "[optuna graph] wrote group=%s best_scalar=%.4f adj_global=%.4f adj_hard=%.4f low_k=%.4f path=%s",
        str(group.name),
        float(payload["best_trial"]["scalar_objective"]),
        float(best_trial.user_attrs.get("adjusted_global_relative_improvement_pct", float("nan"))),
        float(best_trial.user_attrs.get("adjusted_hard_case_relative_improvement_pct", float("nan"))),
        float(best_trial.user_attrs.get("low_top_k_score", float("nan"))),
        run_path,
    )
    return run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune graph topology ablations against strict admin+MEM downstream performance.")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization/config.graph_topology.yaml"))
    parser.add_argument("--group", type=str, default="", help="optional single ablation group name")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(str(args.log_level))
    tune_cfg = load_tune_config(args.config)
    base_graph_cfg = load_graph_config(tune_cfg.graph_config_path)
    nowcast_cfg = load_nowcast_config(tune_cfg.nowcast_config_path)
    registry = modality_registry(base_graph_cfg, nowcast_cfg, tune_cfg)
    groups = build_ablation_groups(base_graph_cfg, registry, tune_cfg)
    if str(args.group).strip():
        want = str(args.group).strip().lower()
        groups = [g for g in groups if str(g.name).strip().lower() == want]
        if not groups:
            raise ValueError(f"unknown ablation group={want!r}")
    for group in groups:
        LOGGER.info(
            "[optuna graph] group=%s modalities=%s kinds=%s family=%d source=%d",
            str(group.name),
            ",".join(group.modalities),
            ",".join(group.group_kinds),
            int(tune_cfg.slice_cfg.family_end_year),
            int(tune_cfg.slice_cfg.source_year),
        )
        optimize_group(
            tune_cfg=tune_cfg,
            base_graph_cfg=base_graph_cfg,
            nowcast_cfg=nowcast_cfg,
            registry=registry,
            group=group,
            skip_existing=bool(args.skip_existing),
        )


if __name__ == "__main__":
    main()
