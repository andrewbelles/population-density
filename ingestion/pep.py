#!/usr/bin/env python3
#
# pep.py  Andrew Belles  Mar 27th, 2026
#
# County-year PEP component assembly for 2020-2024.
#

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ingestion.common import load_counties, materialize_source, parquet_has_rows, write_parquet
from ingestion.config import IngestConfig


LOGGER = logging.getLogger("ingestion.pep")


def _county_only(df: pd.DataFrame) -> pd.DataFrame:
    state = df["STATE"].astype(str).str.strip().str.zfill(2)
    county = df["COUNTY"].astype(str).str.strip().str.zfill(3)
    out = df.copy()
    out["fips"] = state + county
    out = out[(out["fips"] != "00000") & (~out["fips"].str.endswith("000"))].copy()
    return out.drop_duplicates(subset=["fips"], keep="first")


def _legacy_pep_candidates(year: int) -> list[Path]:
    if int(year) == 2020:
        return [
            Path("data/census/co-est2020-alldata.csv"),
            Path("data/pep/co-est2020-alldata.csv"),
            Path("data/census/co-est2023-alldata.csv"),
            Path("data/census/co-est2024-alldata.csv"),
        ]
    if int(year) <= 2023:
        return [Path("data/census/co-est2023-alldata.csv"), Path("data/census/co-est2024-alldata.csv")]
    return [Path("data/census/co-est2024-alldata.csv")]


def _pick_source(config: IngestConfig, year: int) -> Path:
    if year == 2020:
        primary = config.pep.census_2020_csv
        fallbacks = [config.pep.census_2023_csv, config.pep.census_2024_csv, *_legacy_pep_candidates(year)]
        return materialize_source(primary, [primary, *fallbacks])
    if year <= 2023:
        primary = config.pep.census_2023_csv
        return materialize_source(primary, [primary, config.pep.census_2024_csv, *_legacy_pep_candidates(year)])
    primary = config.pep.census_2024_csv
    return materialize_source(primary, [primary, *_legacy_pep_candidates(year)])


def _canon_fips_from_mat(arr: np.ndarray) -> np.ndarray:
    vals = np.asarray(arr)
    if vals.ndim == 0:
        vals = vals.reshape(1)
    if vals.ndim > 1:
        vals = vals.reshape(-1)
    return np.asarray([str(x).strip() for x in vals.tolist()], dtype="U8")


def _load_2020_supervision(config: IngestConfig) -> pd.DataFrame:
    wide_path = config.pep.wide_2020_mat
    pep_path = config.pep.pep_2020_mat
    if wide_path is None or pep_path is None:
        return pd.DataFrame(columns=["fips", "year"])
    wide_path = materialize_source(wide_path, [wide_path, Path("data/datasets/wide_scalar_2020.mat")])
    pep_path = materialize_source(pep_path, [pep_path, Path("data/datasets/pep_arithmetic_2020.mat")])

    wide = loadmat(wide_path)
    pep = loadmat(pep_path)
    anchor_key = "pop_t_hat_with_resid" if str(config.pep.pep_2020_anchor).strip().lower() == "with_resid" else "pop_t_hat_no_resid"
    if anchor_key not in pep:
        raise KeyError(f"{pep_path} missing required key: {anchor_key}")

    wide_fips = _canon_fips_from_mat(wide["fips_codes"])
    y_log = np.asarray(wide["labels"], dtype=np.float64).reshape(-1)
    y_level = np.asarray(wide["labels_level"], dtype=np.float64).reshape(-1)
    y_prev = np.asarray(wide["labels_prev"], dtype=np.float64).reshape(-1)
    y_delta = np.asarray(wide["labels_delta"], dtype=np.float64).reshape(-1)
    pep_fips = _canon_fips_from_mat(pep["fips_codes"])
    pep_level = np.asarray(pep[anchor_key], dtype=np.float64).reshape(-1)
    pep_log = np.log(np.clip(pep_level, 1e-9, None))

    wide_idx = {str(f): i for i, f in enumerate(wide_fips.tolist())}
    pep_idx = {str(f): i for i, f in enumerate(pep_fips.tolist())}
    common = [f for f in wide_fips.tolist() if f in pep_idx]
    rows: list[dict[str, float | int | str]] = []
    for fips in common:
        iw = wide_idx[str(fips)]
        ip = pep_idx[str(fips)]
        rows.append(
            {
                "fips": str(fips).zfill(5),
                "year": 2020,
                "label": float(y_log[iw]),
                "label_level": float(y_level[iw]),
                "label_prev": float(y_prev[iw]),
                "label_delta": float(y_delta[iw]),
                "target_correction_log": float(y_log[iw] - pep_log[ip]),
                "target_correction_level": float(y_level[iw] - pep_level[ip]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _extract_year_frame(config: IngestConfig, counties: pd.DataFrame, *, year: int) -> pd.DataFrame:
    source = _pick_source(config, year)
    LOGGER.debug("year=%d pep source=%s", int(year), source)
    df = _county_only(pd.read_csv(source, encoding="latin-1", dtype=str))

    pop_col = f"POPESTIMATE{int(year)}"
    prev_col = f"POPESTIMATE{int(year) - 1}" if int(year) > 2020 else "ESTIMATESBASE2020"
    births_col = f"BIRTHS{int(year)}"
    deaths_col = f"DEATHS{int(year)}"
    dom_col = f"DOMESTICMIG{int(year)}"
    intl_col = f"INTERNATIONALMIG{int(year)}"
    resid_col = f"RESIDUAL{int(year)}"

    out = pd.DataFrame(
        {
            "fips": df["fips"].astype(str).str.zfill(5),
            "year": int(year),
            "pep_population": pd.to_numeric(df.get(pop_col), errors="coerce"),
            "pep_population_prev": pd.to_numeric(df.get(prev_col), errors="coerce"),
            "pep_births": pd.to_numeric(df.get(births_col), errors="coerce"),
            "pep_deaths": pd.to_numeric(df.get(deaths_col), errors="coerce"),
            "pep_domestic_migration": pd.to_numeric(df.get(dom_col), errors="coerce"),
            "pep_international_migration": pd.to_numeric(df.get(intl_col), errors="coerce"),
            "pep_residual": pd.to_numeric(df.get(resid_col), errors="coerce"),
        }
    )
    out["pep_net_migration"] = out[["pep_domestic_migration", "pep_international_migration"]].sum(axis=1, min_count=1)
    out = counties.merge(out, on="fips", how="right")
    return out


def run(config: IngestConfig, *, skip_existing: bool = False) -> Path | None:
    if not config.pep.enabled:
        return None
    if bool(skip_existing) and parquet_has_rows(config.pep.table_path):
        LOGGER.debug("skip existing pep table=%s", config.pep.table_path)
        return config.pep.table_path
    counties = load_counties(config.paths.county_shapefile).loc[:, ["fips", "county_name", "state_abbr"]].copy()
    frames = [_extract_year_frame(config, counties, year=year) for year in config.years.values]
    out = pd.concat(frames, axis=0, ignore_index=True)
    sup_2020 = _load_2020_supervision(config)
    if not sup_2020.empty:
        LOGGER.debug("merged strict 2020 supervision rows=%d", int(sup_2020.shape[0]))
        out = out.merge(sup_2020, on=["fips", "year"], how="left")
    write_parquet(out, config.pep.table_path)
    LOGGER.debug("pep panel rows=%d out=%s", int(out.shape[0]), config.pep.table_path)
    return config.pep.table_path
