#!/usr/bin/env python3
#
# config.py  Andrew Belles  Mar 27th, 2026
#
# YAML contract loader for the graph topology stage.
#

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RESERVED_MODALITY_KEYS = {"years", "modalities", "paths", "graph"}


@dataclass(slots=True)
class YearRange:
    start: int
    end: int

    @property
    def values(self) -> list[int]:
        if int(self.end) < int(self.start):
            raise ValueError(f"invalid year range: start={self.start} end={self.end}")
        return list(range(int(self.start), int(self.end) + 1))


@dataclass(slots=True)
class PathsConfig:
    geo_coords_path: Path
    run_root: Path
    runs_parquet: Path
    basis_parquet: Path
    edges_parquet: Path


@dataclass(slots=True)
class ModalityConfig:
    enabled: bool
    name: str
    kind: str
    input_parquet: Path
    family_tag_base: str
    bag_keep_rate: float


@dataclass(slots=True)
class GraphConfig:
    graph_tag_base: str
    graph_objective: str
    fusion_logits: dict[str, float]
    mem_top_k: int
    block_pca_dim: int
    hidden_dim: int
    joint_dim: int
    consensus_dim: int
    dropout: float
    temperature: float
    tau_graph: float
    w_pull: float
    beta_geo: float
    support_k: int
    final_row_topk: int
    knn_k: int
    knn_bandwidth_k: int
    epochs: int
    lr: float
    weight_decay: float
    geo_gamma: float
    projector_hidden_dim: int
    projector_dim: int
    barlow_lambda: float
    consensus_ssl_weight: float
    consensus_alignment_weight: float
    complementary_orthogonality_weight: float
    dense_noise_std: float
    spatial_negative_mining: bool
    geo_residual_graph: bool
    mutual_knn: bool
    degree_penalty: bool
    degree_penalty_weight: float
    device: str
    seed: int
    write_knn_reference: bool


@dataclass(slots=True)
class TopologyConfig:
    years: YearRange
    modalities: list[str]
    paths: PathsConfig
    graph: GraphConfig
    blocks: dict[str, ModalityConfig]

    @property
    def anchor_year(self) -> int:
        return int(self.years.start)

    def block_cfg(self, modality: str) -> ModalityConfig:
        key = str(modality).strip().lower()
        if key not in self.blocks:
            raise KeyError(f"unknown topology modality={modality!r}; known={sorted(self.blocks)}")
        return self.blocks[key]


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _require(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise KeyError(f"missing required config key: {key}")
    return section[key]


def _parse_paths(section: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        geo_coords_path=_as_path(_require(section, "geo_coords_path")),
        run_root=_as_path(_require(section, "run_root")),
        runs_parquet=_as_path(_require(section, "runs_parquet")),
        basis_parquet=_as_path(_require(section, "basis_parquet")),
        edges_parquet=_as_path(_require(section, "edges_parquet")),
    )


def _parse_modality(section: dict[str, Any], name: str) -> ModalityConfig:
    kind = str(section.get("kind", "bag" if str(name).strip().lower() != "admin" else "dense")).strip().lower()
    if kind not in {"dense", "bag"}:
        raise ValueError(f"unsupported topology modality kind={kind!r} for {name}")
    return ModalityConfig(
        enabled=bool(section.get("enabled", True)),
        name=str(name).strip().lower(),
        kind=str(kind),
        input_parquet=_as_path(_require(section, "input_parquet")),
        family_tag_base=str(_require(section, "family_tag_base")),
        bag_keep_rate=float(section.get("bag_keep_rate", 1.0)),
    )


def _parse_graph(section: dict[str, Any]) -> GraphConfig:
    joint_dim = int(section.get("joint_dim", 40))
    return GraphConfig(
        graph_tag_base=str(section.get("graph_tag_base", "gsl_meanmax_consensus")),
        graph_objective=str(section.get("graph_objective", "barlow")),
        fusion_logits={str(k).strip().lower(): float(v) for k, v in dict(section.get("fusion_logits", {})).items()},
        mem_top_k=int(section.get("mem_top_k", 11)),
        block_pca_dim=int(section.get("block_pca_dim", 0)),
        hidden_dim=int(section.get("hidden_dim", 144)),
        joint_dim=joint_dim,
        consensus_dim=int(section.get("consensus_dim", joint_dim)),
        dropout=float(section.get("dropout", 0.02)),
        temperature=float(section.get("temperature", 0.12)),
        tau_graph=float(section.get("tau_graph", 1.0)),
        w_pull=float(section.get("w_pull", 0.0)),
        beta_geo=float(section.get("beta_geo", 0.2)),
        support_k=int(section.get("support_k", 28)),
        final_row_topk=int(section.get("final_row_topk", 10)),
        knn_k=int(section.get("knn_k", 6)),
        knn_bandwidth_k=int(section.get("knn_bandwidth_k", section.get("knn_k", 6))),
        epochs=int(section.get("epochs", 120)),
        lr=float(section.get("lr", 1e-3)),
        weight_decay=float(section.get("weight_decay", 1e-5)),
        geo_gamma=float(section.get("geo_gamma", 1.0)),
        projector_hidden_dim=int(section.get("projector_hidden_dim", 128)),
        projector_dim=int(section.get("projector_dim", 64)),
        barlow_lambda=float(section.get("barlow_lambda", 5e-3)),
        consensus_ssl_weight=float(section.get("consensus_ssl_weight", 1.0)),
        consensus_alignment_weight=float(section.get("consensus_alignment_weight", 0.10)),
        complementary_orthogonality_weight=float(section.get("complementary_orthogonality_weight", 0.05)),
        dense_noise_std=float(section.get("dense_noise_std", 0.05)),
        spatial_negative_mining=bool(section.get("spatial_negative_mining", False)),
        geo_residual_graph=bool(section.get("geo_residual_graph", False)),
        mutual_knn=bool(section.get("mutual_knn", False)),
        degree_penalty=bool(section.get("degree_penalty", False)),
        degree_penalty_weight=float(section.get("degree_penalty_weight", 0.05)),
        device=str(section.get("device", "cuda")),
        seed=int(section.get("seed", 0)),
        write_knn_reference=bool(section.get("write_knn_reference", True)),
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = _as_path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    return dict(raw)


def _apply_best_trial_overlay(raw: dict[str, Any], *, best_trial_json: str | Path) -> dict[str, Any]:
    run_path = _as_path(best_trial_json)
    with open(run_path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"optimization run JSON must be a mapping: {run_path}")
    best_trial = dict(payload.get("best_trial") or {})
    params = dict(best_trial.get("params") or {})
    graph_overrides = dict(payload.get("graph_overrides") or {})
    raw_out = dict(raw)

    modalities = [str(x).strip().lower() for x in list(payload.get("modalities") or []) if str(x).strip()]
    if modalities:
        raw_out["modalities"] = modalities
        missing = [m for m in modalities if m not in raw_out]
        if missing:
            raise KeyError(f"{run_path}: base graph config is missing modality section(s): {missing}")

    graph_section = dict(raw_out.get("graph") or {})
    if "graph_tag_base" in payload and str(payload["graph_tag_base"]).strip():
        graph_section["graph_tag_base"] = str(payload["graph_tag_base"]).strip()

    fusion_logits = {
        str(k).strip().lower(): float(v)
        for k, v in dict(graph_section.get("fusion_logits", {})).items()
        if str(k).strip()
    }

    for source in (graph_overrides, params):
        for raw_key, value in dict(source).items():
            key = str(raw_key).strip()
            if not key:
                continue
            if key.startswith("fusion_logit."):
                component = str(key).split(".", 1)[1].strip().lower()
                if component:
                    fusion_logits[component] = float(value)
                continue
            graph_section[key] = value

    if fusion_logits:
        graph_section["fusion_logits"] = fusion_logits
    raw_out["graph"] = graph_section
    return raw_out


def _parse_topology(raw: dict[str, Any]) -> TopologyConfig:
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")

    years_raw = dict(_require(raw, "years"))
    years = YearRange(start=int(_require(years_raw, "start")), end=int(_require(years_raw, "end")))
    modalities = [str(x).strip().lower() for x in list(_require(raw, "modalities"))]
    if not modalities:
        raise ValueError("modalities must not be empty")

    blocks: dict[str, ModalityConfig] = {}
    for key in modalities:
        if key in RESERVED_MODALITY_KEYS:
            raise ValueError(f"invalid topology modality name={key!r}")
        if key not in raw:
            raise KeyError(f"config missing topology modality section: {key}")
        blocks[key] = _parse_modality(dict(raw[key]), key)

    return TopologyConfig(
        years=years,
        modalities=modalities,
        paths=_parse_paths(dict(_require(raw, "paths"))),
        graph=_parse_graph(dict(_require(raw, "graph"))),
        blocks=blocks,
    )


def load_config(path: str | Path, *, best_trial_json: str | Path | None = None) -> TopologyConfig:
    raw = _read_yaml(path)
    if best_trial_json is not None:
        raw = _apply_best_trial_overlay(raw, best_trial_json=best_trial_json)
    return _parse_topology(raw)
