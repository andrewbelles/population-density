#!/usr/bin/env python3
#
# visualizations.py  Andrew Belles  Apr 8th, 2026
#
# Simple graph-stage visualizations for CONUS county topology artifacts.
#

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import ListedColormap, hsv_to_rgb
from matplotlib.lines import Line2D
from pyarrow import parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from graph.config import TopologyConfig, load_config as load_graph_config
from graph.topology import checkpoint_path, graph_tag, to_lat_lon


LOGGER = logging.getLogger("graph.visualizations")
CONUS_EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}


@dataclass(slots=True)
class PlotSelection:
    hypersphere_alignment: bool
    mem_clusters: bool
    dominant_modality: bool


@dataclass(slots=True)
class GraphPlotsConfig:
    graph_config_path: Path
    county_shapefile: Path
    output_dir: Path
    graph_tag_base: str | None
    family_end_year: int
    source_year: int
    mem_basis_top_k: int
    mem_cluster_k: int
    plots: PlotSelection


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def load_plots_config(path: str | Path) -> GraphPlotsConfig:
    cfg_path = _as_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    plots_raw = dict(raw.get("plots", {}))
    graph_raw = dict(raw.get("graph", {}))
    return GraphPlotsConfig(
        graph_config_path=_as_path(raw.get("graph_config_path", "configs/graph/topology.yaml")),
        county_shapefile=_as_path(raw.get("county_shapefile", "ingestion/metadata/geography/county_shapefile/tl_2020_us_county.shp")),
        output_dir=_as_path(raw.get("output_dir", "graph/images")),
        graph_tag_base=None if graph_raw.get("graph_tag_base") in {None, ""} else str(graph_raw.get("graph_tag_base")).strip(),
        family_end_year=int(graph_raw.get("family_end_year", 2020)),
        source_year=int(graph_raw.get("source_year", 2020)),
        mem_basis_top_k=int(raw.get("mem_basis_top_k", 8)),
        mem_cluster_k=int(raw.get("mem_cluster_k", 8)),
        plots=PlotSelection(
            hypersphere_alignment=bool(plots_raw.get("hypersphere_alignment", True)),
            mem_clusters=bool(plots_raw.get("mem_clusters", True)),
            dominant_modality=bool(plots_raw.get("dominant_modality", True)),
        ),
    )


def resolved_graph_config(cfg: GraphPlotsConfig) -> TopologyConfig:
    base = load_graph_config(cfg.graph_config_path)
    if cfg.graph_tag_base is None:
        return base
    return replace(base, graph=replace(base.graph, graph_tag_base=str(cfg.graph_tag_base)))


def load_graph_checkpoint(config: TopologyConfig, *, family_end_year: int, source_year: int) -> dict[str, Any]:
    path = checkpoint_path(config, family_end_year=int(family_end_year), source_year=int(source_year))
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"fips", "coords", "component_embeddings", "component_pre_degree"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path}: checkpoint missing required visualization fields: {missing}")
    return payload


def load_conus_counties(county_shapefile: Path, *, fips_filter: np.ndarray | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(county_shapefile)
    if "GEOID" not in gdf.columns:
        raise ValueError(f"{county_shapefile}: missing GEOID")
    gdf["fips"] = gdf["GEOID"].astype(str).str.zfill(5)
    if "STATEFP" in gdf.columns:
        gdf = gdf.loc[~gdf["STATEFP"].astype(str).isin(CONUS_EXCLUDE_STATEFP)].copy()
    if fips_filter is not None:
        keep = set(np.asarray(fips_filter, dtype="U5").tolist())
        gdf = gdf.loc[gdf["fips"].isin(keep)].copy()
    if gdf.empty:
        raise ValueError(f"{county_shapefile}: no CONUS counties matched requested FIPS")
    if gdf.crs is None:
        raise ValueError(f"{county_shapefile}: missing CRS")
    return gdf.to_crs("EPSG:5070")


def geo_rgb_from_coords(coords: np.ndarray) -> np.ndarray:
    lat, lon = to_lat_lon(np.asarray(coords, dtype=np.float64))
    lon_norm = (lon - float(np.min(lon))) / max(float(np.max(lon) - np.min(lon)), 1e-9)
    lat_norm = (lat - float(np.min(lat))) / max(float(np.max(lat) - np.min(lat)), 1e-9)
    hsv = np.stack([lon_norm, np.full_like(lon_norm, 0.75), 0.35 + 0.60 * lat_norm], axis=1)
    return hsv_to_rgb(hsv)


def short_family_label(graph_tag_base: str | None) -> str:
    raw = "" if graph_tag_base is None else str(graph_tag_base).strip()
    if not raw:
        return "Graph"
    prefixes = (
        "gsl_meanmax_consensus_",
        "gsl_meanmax_late_",
        "gsl_gem_",
    )
    label = raw
    for prefix in prefixes:
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if not label:
        return "Graph"
    parts = [part.upper() if part.lower() == "s5p" else part.capitalize() for part in str(label).split("_") if str(part).strip()]
    return " + ".join(parts) if parts else "Graph"


def percentile_rank_scores(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(int(arr.size), dtype=np.float64)
    if int(arr.size) == 1:
        return np.ones_like(arr, dtype=np.float64)
    return ranks / float(int(arr.size) - 1)


def project_to_unit_sphere(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("project_to_unit_sphere expects [n,d]")
    n_comp = int(min(3, arr.shape[1]))
    proj = PCA(n_components=n_comp, random_state=0).fit_transform(arr)
    if n_comp < 3:
        proj = np.pad(proj, ((0, 0), (0, 3 - n_comp)))
    denom = np.clip(np.linalg.norm(proj, axis=1, keepdims=True), 1e-9, None)
    return proj / denom


def save_hypersphere_alignment_plot(
    *,
    checkpoint: dict[str, Any],
    output_path: Path,
    family_label: str,
    source_year: int,
) -> None:
    component_embeddings = {str(k): np.asarray(v, dtype=np.float64) for k, v in dict(checkpoint["component_embeddings"]).items()}
    coords = np.asarray(checkpoint["coords"], dtype=np.float64)
    colors = geo_rgb_from_coords(coords)
    order = ["consensus", *sorted([name for name in component_embeddings.keys() if name != "consensus"])]
    fig = plt.figure(figsize=(4.5 * len(order), 4.6))
    for idx, name in enumerate(order, start=1):
        ax = fig.add_subplot(1, len(order), idx, projection="3d")
        pts = project_to_unit_sphere(component_embeddings[name])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=4, alpha=0.85, linewidths=0.0)
        ax.set_title("Consensus" if name == "consensus" else str(name).upper(), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        ax.grid(False)
    fig.suptitle(f"Hypersphere Structure: {family_label} ({int(source_year)})", fontsize=13)
    fig.text(0.5, 0.02, "Color follows geography", ha="center", va="center", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_mem_basis_frame(config: TopologyConfig, *, family_end_year: int, source_year: int, top_k: int) -> pd.DataFrame:
    tag = graph_tag(str(config.graph.graph_tag_base), int(family_end_year))
    table = pq.read_table(
        str(config.paths.basis_parquet),
        columns=["fips", "basis_index", "basis_value"],
        filters=[
            ("graph_tag", "=", str(tag)),
            ("graph_kind", "=", "learned"),
            ("source_year", "=", int(source_year)),
        ],
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"{config.paths.basis_parquet}: no learned basis rows for graph_tag={tag} source_year={int(source_year)}")
    frame["fips"] = frame["fips"].astype(str).str.zfill(5)
    frame["basis_index"] = pd.to_numeric(frame["basis_index"], errors="raise").astype(np.int64)
    frame = frame.loc[frame["basis_index"] < int(max(1, top_k))].copy()
    if frame.empty:
        raise ValueError("requested MEM top-k produced no basis rows")
    return frame


def save_mem_cluster_map(
    *,
    config: TopologyConfig,
    county_shapefile: Path,
    family_end_year: int,
    source_year: int,
    top_k: int,
    n_clusters: int,
    output_path: Path,
) -> None:
    basis = load_mem_basis_frame(config, family_end_year=int(family_end_year), source_year=int(source_year), top_k=int(top_k))
    wide = (
        basis.pivot(index="fips", columns="basis_index", values="basis_value")
        .fillna(0.0)
        .sort_index()
    )
    x = np.asarray(wide.to_numpy(), dtype=np.float64)
    k_eff = int(max(2, min(int(n_clusters), int(x.shape[0]))))
    labels = KMeans(n_clusters=k_eff, n_init=20, random_state=0).fit_predict(x)
    cluster_df = pd.DataFrame({"fips": wide.index.astype(str), "cluster": labels.astype(int)})
    counties = load_conus_counties(county_shapefile, fips_filter=np.asarray(cluster_df["fips"], dtype="U5"))
    gdf = counties.merge(cluster_df, on="fips", how="inner")
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    cmap = plt.get_cmap("tab20", k_eff)
    gdf.plot(
        ax=ax,
        column="cluster",
        categorical=True,
        cmap=cmap,
        linewidth=0.0,
        legend=False,
    )
    counties.boundary.plot(ax=ax, linewidth=0.05, color="#202020", alpha=0.15)
    ax.set_title(f"MEM Cluster Regions: {short_family_label(str(config.graph.graph_tag_base))} ({int(source_year)})", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_dominant_modality_map(
    *,
    checkpoint: dict[str, Any],
    county_shapefile: Path,
    output_path: Path,
    family_label: str,
    source_year: int,
) -> None:
    fips = np.asarray(checkpoint["fips"], dtype="U5")
    component_pre_degree = {str(k): np.asarray(v, dtype=np.float64).reshape(-1) for k, v in dict(checkpoint["component_pre_degree"]).items()}
    fusion_weights = dict(checkpoint.get("fusion_weights", {}))
    modality_names = [name for name in sorted(component_pre_degree.keys()) if name != "consensus"]
    if not modality_names:
        raise ValueError("dominant modality map requires non-consensus component degrees")
    scores = np.column_stack([
        float(fusion_weights.get(str(name), 0.0)) * percentile_rank_scores(np.asarray(component_pre_degree[name], dtype=np.float64))
        for name in modality_names
    ])
    winner_idx = np.asarray(np.argmax(scores, axis=1), dtype=np.int64)
    winners = np.asarray([modality_names[int(i)] for i in winner_idx], dtype=object)
    winners_df = pd.DataFrame({"fips": fips.astype(str), "dominant_modality": winners})
    counties = load_conus_counties(county_shapefile, fips_filter=fips)
    gdf = counties.merge(winners_df, on="fips", how="inner")
    unique_mods = list(dict.fromkeys([str(x) for x in gdf["dominant_modality"].tolist()]))
    palette = {
        "admin": "#2f5fb3",
        "viirs": "#d8a310",
        "s5p": "#2c8a57",
    }
    colors = [palette.get(str(name), "#7f7f7f") for name in unique_mods]
    cmap = ListedColormap(colors)
    code_map = {name: idx for idx, name in enumerate(unique_mods)}
    gdf["dominant_code"] = gdf["dominant_modality"].map(code_map).astype(int)
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    gdf.plot(ax=ax, column="dominant_code", categorical=True, cmap=cmap, linewidth=0.0, legend=False)
    counties.boundary.plot(ax=ax, linewidth=0.05, color="#202020", alpha=0.15)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=palette.get(name, "#7f7f7f"), markeredgecolor="none", label=str(name).upper())
        for name in unique_mods
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, title="Dominant")
    ax.set_title(f"Dominant Local Graph Source: {family_label} ({int(source_year)})", fontsize=13)
    ax.set_axis_off()
    fig.text(0.5, 0.02, "Based on weighted local relative strength, not raw global scale", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(cfg: GraphPlotsConfig) -> None:
    topology_cfg = resolved_graph_config(cfg)
    tag = graph_tag(str(topology_cfg.graph.graph_tag_base), int(cfg.family_end_year))
    family_label = short_family_label(str(topology_cfg.graph.graph_tag_base))
    checkpoint = load_graph_checkpoint(topology_cfg, family_end_year=int(cfg.family_end_year), source_year=int(cfg.source_year))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.plots.hypersphere_alignment):
        out = output_dir / f"{tag}_source_{int(cfg.source_year)}__hypersphere.png"
        save_hypersphere_alignment_plot(checkpoint=checkpoint, output_path=out, family_label=str(family_label), source_year=int(cfg.source_year))
        LOGGER.info("wrote %s", out)
    if bool(cfg.plots.mem_clusters):
        out = output_dir / f"{tag}_source_{int(cfg.source_year)}__mem_clusters.png"
        save_mem_cluster_map(
            config=topology_cfg,
            county_shapefile=cfg.county_shapefile,
            family_end_year=int(cfg.family_end_year),
            source_year=int(cfg.source_year),
            top_k=int(cfg.mem_basis_top_k),
            n_clusters=int(cfg.mem_cluster_k),
            output_path=out,
        )
        LOGGER.info("wrote %s", out)
    if bool(cfg.plots.dominant_modality):
        out = output_dir / f"{tag}_source_{int(cfg.source_year)}__dominant_modality.png"
        save_dominant_modality_map(
            checkpoint=checkpoint,
            county_shapefile=cfg.county_shapefile,
            output_path=out,
            family_label=str(family_label),
            source_year=int(cfg.source_year),
        )
        LOGGER.info("wrote %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render graph topology visualizations.")
    parser.add_argument("--config", type=Path, default=Path("configs/graph/graph_plots.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(str(args.log_level))
    cfg = load_plots_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
