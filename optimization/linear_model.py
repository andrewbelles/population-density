#!/usr/bin/env python3
#
# linear_model.py  Andrew Belles  Apr 8th, 2026
#
# Optuna tuner for strict downstream linear-model parameters on top of saved graph ablation runs.
#

import argparse
import copy
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml

from graph.config import ModalityConfig as GraphModalityConfig
from graph.config import TopologyConfig, load_config as load_graph_config
from graph.topology import train_graph_slice
from nowcast.config import DownstreamModelConfig, NowcastConfig, load_config as load_nowcast_config
from optimization.common import (
    GL2Config,
    GL2StudyStopper,
    StudyConfig,
    best_completed_trial,
    create_study,
    setup_logging,
    suggest_from_space,
    trial_payload,
    write_json,
)
from optimization.graph_topology import (
    AblationGroup,
    ObjectiveConfig,
    PreparedDownstreamInputs,
    apply_graph_params,
    build_trial_topology_config,
    graph_modality_from_nowcast,
    prepare_downstream_candidate,
    score_downstream_candidate_fixed_k,
    split_param_mapping,
)


LOGGER = logging.getLogger("optimization.linear_model")


@dataclass(slots=True)
class LinearModelTuneConfig:
    graph_config_path: Path
    nowcast_config_path: Path
    graph_runs_root: Path
    output_root: Path
    study: StudyConfig
    gl2: GL2Config
    objective: ObjectiveConfig
    search_space: dict[str, dict[str, Any]]


@dataclass(slots=True)
class SavedGraphRun:
    path: Path
    group: AblationGroup
    graph_tag: str
    family_end_year: int
    source_year: int
    graph_overrides: dict[str, Any]
    graph_params: dict[str, Any]
    selected_mem_top_k: int
    payload: dict[str, Any]


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def load_tune_config(path: str | Path) -> LinearModelTuneConfig:
    cfg_path = _as_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    study_raw = dict(_require(raw, "study"))
    gl2_raw = dict(raw.get("gl2", {}))
    objective_raw = dict(_require(raw, "objective"))
    search_space = {str(k): dict(v) for k, v in dict(raw.get("search_space", {})).items()}
    dead_search = sorted(set(search_space) - {"huber_alpha", "huber_epsilon", "huber_asymmetry", "huber_tol", "huber_max_iter"})
    if dead_search:
        raise ValueError(f"unsupported linear-model search dimensions: {dead_search}")
    return LinearModelTuneConfig(
        graph_config_path=_as_path(_require(raw, "graph_config_path")),
        nowcast_config_path=_as_path(_require(raw, "nowcast_config_path")),
        graph_runs_root=_as_path(raw.get("graph_runs_root", "optimization/runs")),
        output_root=_as_path(raw.get("output_root", "optimization/runs")),
        study=StudyConfig(
            study_name=str(study_raw.get("study_name", "linear_model_ablation")).strip(),
            direction=str(study_raw.get("direction", "maximize")).strip().lower(),
            n_trials=int(study_raw.get("n_trials", 30)),
            timeout_sec=int(study_raw.get("timeout_sec", 0)),
            sampler_seed=int(study_raw.get("sampler_seed", 0)),
            n_startup_trials=int(study_raw.get("n_startup_trials", 8)),
            gc_after_trial=bool(study_raw.get("gc_after_trial", True)),
        ),
        gl2=GL2Config(
            enabled=bool(gl2_raw.get("enabled", False)),
            min_trials=int(gl2_raw.get("min_trials", 12)),
            patience=int(gl2_raw.get("patience", 8)),
            max_generalization_loss_pct=float(gl2_raw.get("max_generalization_loss_pct", 0.5)),
            min_relative_improvement_pct=float(gl2_raw.get("min_relative_improvement_pct", 0.0)),
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
        search_space=search_space,
    )


def build_registry(base_graph_cfg: TopologyConfig, nowcast_cfg: NowcastConfig) -> dict[str, GraphModalityConfig]:
    registry: dict[str, GraphModalityConfig] = {str(k): copy.deepcopy(v) for k, v in base_graph_cfg.blocks.items()}
    for key, mod_cfg in nowcast_cfg.blocks.items():
        if key in registry:
            continue
        registry[str(key)] = graph_modality_from_nowcast(mod_cfg)
    return registry


def linear_model_run_path(root: Path, *, graph_tag_name: str, source_year: int) -> Path:
    return Path(root) / f"linear_model__{graph_tag_name}_source_{int(source_year)}.json"


def is_graph_run_payload(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload, dict)
        and "group_name" in payload
        and "graph_tag_base" in payload
        and "modalities" in payload
        and isinstance(payload.get("best_trial"), dict)
        and isinstance(payload["best_trial"].get("params"), dict)
    )


def load_saved_graph_runs(
    *,
    root: Path,
    registry: dict[str, GraphModalityConfig],
    group_filter: str | None = None,
    graph_tag_filter: str | None = None,
) -> list[SavedGraphRun]:
    out: list[SavedGraphRun] = []
    want_group = None if not group_filter else str(group_filter).strip().lower()
    want_tag = None if not graph_tag_filter else str(graph_tag_filter).strip().lower()
    for path in sorted(Path(root).glob("*.json")):
        if str(path.name).startswith("linear_model__"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not is_graph_run_payload(payload):
            continue
        group_name = str(payload["group_name"]).strip()
        graph_tag_name = str(payload.get("graph_tag", "")).strip()
        if want_group is not None and group_name.lower() != want_group:
            continue
        if want_tag is not None and graph_tag_name.lower() != want_tag:
            continue
        group = AblationGroup(
            name=group_name,
            modalities=[str(x).strip().lower() for x in list(payload.get("modalities", []))],
            graph_tag_base=str(payload["graph_tag_base"]).strip(),
            group_kinds=[str(x) for x in list(payload.get("group_kinds", []))],
        )
        param_split = split_param_mapping(params={str(k): v for k, v in dict(payload["best_trial"]["params"]).items()})
        graph_params = {
            **dict(param_split["graph_params"]),
            **{f"fusion_logit.{str(k)}": float(v) for k, v in dict(param_split["fusion_logits"]).items()},
        }
        selected_mem_top_k = int(payload["best_trial"].get("user_attrs", {}).get("selected_mem_top_k", 0))
        if selected_mem_top_k <= 0:
            raise ValueError(f"{path}: missing positive selected_mem_top_k in best_trial.user_attrs")
        out.append(
            SavedGraphRun(
                path=path,
                group=group,
                graph_tag=graph_tag_name,
                family_end_year=int(payload["family_end_year"]),
                source_year=int(payload["source_year"]),
                graph_overrides={str(k): v for k, v in dict(payload.get("graph_overrides", {})).items()},
                graph_params=graph_params,
                selected_mem_top_k=int(selected_mem_top_k),
                payload=payload,
            )
        )
    return out


def rebuild_best_artifact(
    *,
    saved_run: SavedGraphRun,
    base_graph_cfg: TopologyConfig,
    registry: dict[str, GraphModalityConfig],
) -> Any:
    graph_cfg = apply_graph_params(
        base_graph_cfg.graph,
        overrides=saved_run.graph_overrides,
        params=saved_run.graph_params,
        graph_tag_base=str(saved_run.group.graph_tag_base),
    )
    trial_topology_cfg = build_trial_topology_config(
        base_graph_cfg,
        registry,
        group=saved_run.group,
        graph_cfg=graph_cfg,
    )
    return train_graph_slice(
        trial_topology_cfg,
        family_end_year=int(saved_run.family_end_year),
        source_year=int(saved_run.source_year),
    )


def build_model_cfg(base_model_cfg: DownstreamModelConfig, params: dict[str, Any]) -> DownstreamModelConfig:
    allowed = {"huber_alpha", "huber_epsilon", "huber_asymmetry", "huber_tol", "huber_max_iter"}
    bad = sorted(set(params) - allowed)
    if bad:
        raise ValueError(f"unsupported model parameter overrides: {bad}")
    return replace(base_model_cfg, **params)


def optimize_saved_run(
    *,
    tune_cfg: LinearModelTuneConfig,
    saved_run: SavedGraphRun,
    base_graph_cfg: TopologyConfig,
    nowcast_cfg: NowcastConfig,
    registry: dict[str, GraphModalityConfig],
    skip_existing: bool,
) -> Path:
    run_path = linear_model_run_path(tune_cfg.output_root, graph_tag_name=str(saved_run.graph_tag), source_year=int(saved_run.source_year))
    if bool(skip_existing) and run_path.exists():
        LOGGER.info("skip existing graph_tag=%s source=%d path=%s", str(saved_run.graph_tag), int(saved_run.source_year), run_path)
        return run_path

    LOGGER.info(
        "[optuna linear] rebuild graph group=%s graph_tag=%s source=%d modalities=%s",
        str(saved_run.group.name),
        str(saved_run.graph_tag),
        int(saved_run.source_year),
        ",".join(saved_run.group.modalities),
    )
    artifact = rebuild_best_artifact(saved_run=saved_run, base_graph_cfg=base_graph_cfg, registry=registry)
    prepared = prepare_downstream_candidate(
        nowcast_cfg=nowcast_cfg,
        family_end_year=int(saved_run.family_end_year),
        source_year=int(saved_run.source_year),
        artifact=artifact,
        objective_cfg=tune_cfg.objective,
    )
    mem_k_fixed = int(max(1, min(int(saved_run.selected_mem_top_k), int(prepared.max_k))))
    base_model_cfg = nowcast_cfg.downstream.model_cfg(str(tune_cfg.objective.model_key))

    study_cfg = replace(tune_cfg.study, study_name=f"{tune_cfg.study.study_name}:{saved_run.group.name}")
    study = create_study(study_cfg)
    stopper = GL2StudyStopper(tune_cfg.gl2)

    def objective(trial: optuna.trial.Trial) -> float:
        sampled = {str(name): suggest_from_space(trial, str(name), dict(spec)) for name, spec in tune_cfg.search_space.items()}
        model_cfg = build_model_cfg(base_model_cfg, sampled)
        score = score_downstream_candidate_fixed_k(
            prepared=prepared,
            nowcast_cfg=nowcast_cfg,
            objective_cfg=tune_cfg.objective,
            model_cfg=model_cfg,
            mem_k=int(mem_k_fixed),
            leakage_proxy_mode=str(nowcast_cfg.analysis.leakage_proxy_mode),
        )
        scalar = float(score.objective_scalar)
        trial.set_user_attr("scalar_objective", float(scalar))
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
        LOGGER.info(
            "[optuna linear] group=%s graph_tag=%s trial=%d adj_global=%.4f adj_hard=%.4f low_k=%.4f asym=%.4f scalar=%.4f",
            str(saved_run.group.name),
            str(saved_run.graph_tag),
            int(trial.number),
            float(score.adjusted_global_relative_improvement_pct),
            float(score.adjusted_hard_case_relative_improvement_pct),
            float(score.low_top_k_score),
            float(getattr(model_cfg, "huber_asymmetry", 0.0)),
            float(scalar),
        )
        return scalar

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
        "graph_run_path": str(saved_run.path),
        "group_name": str(saved_run.group.name),
        "group_kinds": list(saved_run.group.group_kinds),
        "modalities": list(saved_run.group.modalities),
        "graph_tag_base": str(saved_run.group.graph_tag_base),
        "graph_tag": str(saved_run.graph_tag),
        "family_end_year": int(saved_run.family_end_year),
        "source_year": int(saved_run.source_year),
        "fixed_mem_top_k": int(mem_k_fixed),
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
        "search_space": {str(k): dict(v) for k, v in tune_cfg.search_space.items()},
        "completed_trials": [
            trial_payload(t)
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ],
    }
    write_json(run_path, payload)
    LOGGER.info(
        "[optuna linear] wrote graph_tag=%s best_scalar=%.4f adj_global=%.4f adj_hard=%.4f path=%s",
        str(saved_run.graph_tag),
        float(payload["best_trial"]["scalar_objective"]),
        float(best_trial.user_attrs.get("adjusted_global_relative_improvement_pct", float("nan"))),
        float(best_trial.user_attrs.get("adjusted_hard_case_relative_improvement_pct", float("nan"))),
        run_path,
    )
    return run_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune downstream linear-model parameters on top of saved graph ablation runs.")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization/linear_model.yaml"))
    parser.add_argument("--group", type=str, default="", help="optional single saved graph group name")
    parser.add_argument("--graph-tag", type=str, default="", help="optional single saved graph tag")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(str(args.log_level))
    tune_cfg = load_tune_config(args.config)
    base_graph_cfg = load_graph_config(tune_cfg.graph_config_path)
    nowcast_cfg = load_nowcast_config(tune_cfg.nowcast_config_path)
    registry = build_registry(base_graph_cfg, nowcast_cfg)
    saved_runs = load_saved_graph_runs(
        root=tune_cfg.graph_runs_root,
        registry=registry,
        group_filter=None if not str(args.group).strip() else str(args.group).strip(),
        graph_tag_filter=None if not str(args.graph_tag).strip() else str(args.graph_tag).strip(),
    )
    if not saved_runs:
        raise ValueError("no saved graph optimization runs matched the requested filters")
    for saved_run in saved_runs:
        optimize_saved_run(
            tune_cfg=tune_cfg,
            saved_run=saved_run,
            base_graph_cfg=base_graph_cfg,
            nowcast_cfg=nowcast_cfg,
            registry=registry,
            skip_existing=bool(args.skip_existing),
        )


if __name__ == "__main__":
    main()
