#!/usr/bin/env python3
#
# usps.py  Andrew Belles  Mar 27th, 2026
#
# USPS raw staging plus exact county scalar aggregation for the admin pathway.
#

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.io import loadmat

from ingestion.common import ensure_dir, materialize_source, parquet_has_rows, stage_copy, write_parquet
from ingestion.config import IngestConfig


AREA_CRS = "EPSG:5070"
LOGGER = logging.getLogger("ingestion.usps")
USPS_FEATURE_COLS = [
    "usps_flux_rate",
    "usps_comm_ratio",
    "usps_inst_ratio",
    "usps_coverage_ratio",
    "usps_log_density_land",
    "usps_address_hhi",
    "usps_log_density_iqr",
    "usps_total_res",
    "usps_residency_velocity",
    "usps_b2r_ratio",
    "usps_vacancy_aging_ratio",
    "usps_dormancy_index",
]


def _sum_if_exists(group: pd.DataFrame, cols: list[str]) -> float:
    val = 0.0
    used = False
    for col in cols:
        if col in group.columns:
            used = True
            val += float(pd.to_numeric(group[col], errors="coerce").fillna(0.0).sum())
    return float(val if used else np.nan)


def _mat_str_vector(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr)
    if vals.ndim == 0:
        vals = vals.reshape(1)
    if vals.ndim > 1:
        vals = vals.reshape(-1)
    return np.asarray([str(v).strip() for v in vals.tolist()], dtype="U")


def _stage_raw_inputs(config: IngestConfig) -> list[Path]:
    raw_dir = ensure_dir(config.paths.raw_root / config.usps.raw_subdir)
    staged: list[Path] = []
    for path in sorted(config.usps.source_dir.glob(config.usps.zip_glob)):
        staged.append(stage_copy(path, raw_dir / path.name))
    for year in config.years.values:
        gpkg_name = config.usps.gpkg_template.format(year=int(year))
        gpkg_path = config.usps.source_dir / gpkg_name
        if gpkg_path.exists():
            staged.append(stage_copy(gpkg_path, raw_dir / gpkg_path.name))
    LOGGER.debug("staged usps raw files=%d", len(staged))
    return staged


def _load_county_area_map(county_shapefile: Path) -> dict[str, float]:
    counties = gpd.read_file(county_shapefile)
    if "GEOID" not in counties.columns or "ALAND" not in counties.columns:
        raise ValueError(f"{county_shapefile}: missing GEOID or ALAND")
    out: dict[str, float] = {}
    for _, row in counties.iterrows():
        out[str(row["GEOID"]).strip().zfill(5)] = float(row["ALAND"]) / 1e6
    return out


def _aggregate_one_year(gpkg_path: Path, county_shapefile: Path, *, year: int) -> pd.DataFrame:
    LOGGER.debug("aggregate usps year=%d source=%s", int(year), gpkg_path)
    gdf = gpd.read_file(gpkg_path)
    gdf.columns = [str(c).lower() for c in gdf.columns]

    required = ["geoid", "total_addresses", "total_business", "total_other", "flux_rate"]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"{gpkg_path}: missing USPS columns {missing}")

    gdf["geoid"] = gdf["geoid"].astype(str).str.zfill(11)
    gdf["fips"] = gdf["geoid"].str[:5]
    if gdf.crs is None or str(gdf.crs) != AREA_CRS:
        gdf = gdf.to_crs(AREA_CRS)
    gdf["tract_area_sqkm"] = gdf.geometry.area / 1e6
    gdf = gdf.loc[pd.to_numeric(gdf["tract_area_sqkm"], errors="coerce") > 1e-6].copy()

    county_area_map = _load_county_area_map(county_shapefile)

    def agg(group: pd.DataFrame) -> pd.Series:
        fips = str(group.name).strip().zfill(5)
        s_addr = float(pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0).sum())
        s_bus = float(pd.to_numeric(group["total_business"], errors="coerce").fillna(0.0).sum())
        s_oth = float(pd.to_numeric(group["total_other"], errors="coerce").fillna(0.0).sum())
        if "total_res" in group.columns:
            s_res = float(pd.to_numeric(group["total_res"], errors="coerce").fillna(0.0).sum())
        else:
            s_res = float(max(s_addr - s_bus - s_oth, 0.0))

        s_covered_area = float(pd.to_numeric(group["tract_area_sqkm"], errors="coerce").fillna(0.0).sum())
        s_land_area = float(county_area_map.get(fips, s_covered_area))
        s_land_area = max(s_land_area, 1e-6)
        s_covered_area = max(s_covered_area, 1e-6)
        s_addr_safe = max(s_addr, 1.0)
        out: dict[str, float] = {}

        if s_addr > 0.0:
            out["usps_flux_rate"] = float(
                np.average(
                    pd.to_numeric(group["flux_rate"], errors="coerce").fillna(0.0),
                    weights=pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0),
                )
            )
            out["usps_comm_ratio"] = float(s_bus / s_addr_safe)
            out["usps_inst_ratio"] = float(s_oth / s_addr_safe)
        else:
            out["usps_flux_rate"] = 0.0
            out["usps_comm_ratio"] = 0.0
            out["usps_inst_ratio"] = 0.0

        out["usps_coverage_ratio"] = float(min(s_covered_area / s_land_area, 1.05))
        out["usps_log_density_land"] = float(np.log1p(s_addr_safe / s_land_area))
        shares = pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0) / s_addr_safe
        out["usps_address_hhi"] = float(np.square(shares).sum())

        local_dens = pd.to_numeric(group["total_addresses"], errors="coerce").fillna(0.0) / np.maximum(
            pd.to_numeric(group["tract_area_sqkm"], errors="coerce").fillna(np.nan),
            1e-6,
        )
        local_log = np.log1p(local_dens.to_numpy(dtype=np.float64))
        if local_log.shape[0] > 1:
            q75, q25 = np.percentile(local_log, [75, 25])
            out["usps_log_density_iqr"] = float(q75 - q25)
        else:
            out["usps_log_density_iqr"] = 0.0

        s_res_safe = max(s_res, 1.0)
        out["usps_total_res"] = float(s_res)
        out["usps_residency_velocity"] = 0.0
        out["usps_b2r_ratio"] = float(s_bus / s_res_safe)

        s_vac_short_res = _sum_if_exists(group, ["vac_short_res", "vac_3_res"])
        s_vac_long_res = _sum_if_exists(group, ["vac_long_res", "vac_3_6_r", "vac_6_12r", "vac_12_24r", "vac_24_36r", "vac_36_res"])
        if (not np.isfinite(s_vac_short_res)) and ("res_vac" in group.columns):
            s_vac_short_res = float(pd.to_numeric(group["res_vac"], errors="coerce").fillna(0.0).sum())
        if not np.isfinite(s_vac_short_res):
            s_vac_short_res = 0.0
        if not np.isfinite(s_vac_long_res):
            if "res_vac" in group.columns:
                s_res_vac = float(pd.to_numeric(group["res_vac"], errors="coerce").fillna(0.0).sum())
                s_vac_long_res = max(s_res_vac - s_vac_short_res, 0.0)
            else:
                s_vac_long_res = 0.0
        out["usps_vacancy_aging_ratio"] = float(s_vac_short_res / max(s_vac_long_res, 1.0))

        s_nostat_res = _sum_if_exists(group, ["nostat_res"])
        if not np.isfinite(s_nostat_res):
            s_nostat_res = 0.0
        out["usps_dormancy_index"] = float(s_nostat_res / s_res_safe)
        return pd.Series(out)

    df = gdf.groupby("fips", sort=False).apply(agg, include_groups=False).reset_index()
    df["year"] = int(year)
    df = df.loc[:, ["fips", "year", *USPS_FEATURE_COLS]].copy()
    LOGGER.debug("aggregated usps year=%d rows=%d", int(year), int(df.shape[0]))
    return df.sort_values(["year", "fips"]).reset_index(drop=True)


def _load_prev_total_res_by_fips(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    mat = loadmat(str(path))
    required = {"features", "feature_names", "fips_codes"}
    if not required.issubset(mat):
        return {}
    x = np.asarray(mat["features"], dtype=np.float64)
    names = _mat_str_vector(mat["feature_names"])
    fips = np.asarray([str(v).strip().zfill(5) for v in _mat_str_vector(mat["fips_codes"]).tolist()], dtype="U5")
    if x.ndim != 2 or x.shape[0] != fips.shape[0]:
        return {}
    name_to_idx = {str(name): i for i, name in enumerate(names.tolist())}
    idx = None
    for cand in ("usps_total_res", "total_res"):
        if cand in name_to_idx:
            idx = int(name_to_idx[cand])
            break
    if idx is None:
        return {}
    vals = np.asarray(x[:, idx], dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for fips_code, value in zip(fips.tolist(), vals.tolist()):
        if np.isfinite(value):
            out[str(fips_code)] = float(value)
    return out


def _apply_residency_velocity(panel: pd.DataFrame, config: IngestConfig) -> pd.DataFrame:
    out = panel.sort_values(["year", "fips"]).reset_index(drop=True).copy()
    out["usps_residency_velocity"] = 0.0
    by_year = {int(year): frame.copy() for year, frame in out.groupby("year", sort=True)}
    years_sorted = sorted(by_year)
    prev_template = config.usps.prev_scalar_mat_template
    prev_maps_from_panel: dict[int, dict[str, float]] = {}
    for prev_year in years_sorted:
        part = by_year[prev_year]
        prev_maps_from_panel[int(prev_year)] = {
            str(f): float(v)
            for f, v in zip(
                np.asarray(part["fips"], dtype="U8").tolist(),
                pd.to_numeric(part["usps_total_res"], errors="coerce").fillna(0.0).tolist(),
            )
        }
    for year in years_sorted:
        prev_map: dict[str, float] = {}
        prev_year = int(year) - 1
        if prev_template:
            prev_path = Path(str(prev_template).format(year=int(prev_year))).expanduser()
            try:
                prev_path = materialize_source(
                    prev_path,
                    [prev_path, Path(f"data/datasets/usps_scalar_{int(prev_year)}.mat")],
                )
                prev_map = _load_prev_total_res_by_fips(prev_path)
            except FileNotFoundError:
                prev_map = {}
        if not prev_map:
            prev_map = prev_maps_from_panel.get(int(prev_year), {})
        if not prev_map:
            continue
        mask = np.asarray(out["year"], dtype=np.int64) == int(year)
        cur = pd.to_numeric(out.loc[mask, "usps_total_res"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        prev = np.asarray([prev_map.get(str(f), np.nan) for f in out.loc[mask, "fips"].astype(str).tolist()], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = (cur - prev) / np.maximum(prev, 1.0)
        vel[~np.isfinite(vel)] = 0.0
        out.loc[mask, "usps_residency_velocity"] = vel
    return out


def run(config: IngestConfig, *, skip_existing: bool = False) -> tuple[list[Path], Path | None]:
    if not config.usps.enabled:
        return [], None
    if bool(skip_existing) and parquet_has_rows(config.usps.table_path):
        LOGGER.debug("skip existing usps table=%s", config.usps.table_path)
        raw_dir = config.paths.raw_root / config.usps.raw_subdir
        staged = sorted(p for p in raw_dir.iterdir() if p.is_file()) if raw_dir.exists() else []
        return staged, config.usps.table_path

    staged = _stage_raw_inputs(config)
    frames: list[pd.DataFrame] = []
    for year in config.years.values:
        gpkg_path = config.paths.raw_root / config.usps.raw_subdir / config.usps.gpkg_template.format(year=int(year))
        if gpkg_path.exists():
            frames.append(_aggregate_one_year(gpkg_path, config.paths.county_shapefile, year=int(year)))
    if not frames:
        LOGGER.debug("no usps yearly frames produced")
        return staged, None
    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged = _apply_residency_velocity(merged, config)
    write_parquet(merged, config.usps.table_path)
    LOGGER.debug("usps merged rows=%d out=%s", int(merged.shape[0]), config.usps.table_path)
    return staged, config.usps.table_path
