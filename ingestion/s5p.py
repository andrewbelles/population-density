#!/usr/bin/env python3
#
# s5p.py  Andrew Belles  Mar 27th, 2026
#
# Sentinel-5P NO2 raw staging and county tensor extraction.
#

from __future__ import annotations

import logging
from pathlib import Path

from ingestion.common import parquet_has_rows
from ingestion.config import IngestConfig
from ingestion.raster import build_county_tensor_parquet, discover_source, stage_raster


LOGGER = logging.getLogger("ingestion.s5p")


def run(config: IngestConfig, *, skip_existing: bool = False) -> list[Path]:
    if not config.s5p.enabled:
        return []

    outputs = [config.paths.dataset_root / config.s5p.tensor_subdir / f"s5p_county_tensors_{year}.parquet" for year in config.years.values]
    if bool(skip_existing) and outputs and all(parquet_has_rows(path) for path in outputs):
        LOGGER.debug("skip existing s5p tensors=%s", [str(p) for p in outputs])
        return outputs

    outputs = []
    for year in config.years.values:
        source = discover_source(config.s5p.source_dir, config.s5p.source_globs, year=year)
        LOGGER.debug("year=%d source=%s", int(year), source)
        staged = stage_raster(
            source,
            raw_root=config.paths.raw_root,
            subdir=config.s5p.raw_subdir,
            preserve_name=bool(config.s5p.preserve_name),
            stage_compressed=bool(config.s5p.stage_compressed),
            target_name=None,
        )
        out_path = config.paths.dataset_root / config.s5p.tensor_subdir / f"s5p_county_tensors_{year}.parquet"
        LOGGER.debug("year=%d staged=%s out=%s", int(year), staged, out_path)
        outputs.append(
            build_county_tensor_parquet(
                raster_path=staged,
                county_shapefile=config.paths.county_shapefile,
                out_path=out_path,
                year=year,
                modality="s5p_no2",
                temp_root=config.paths.temp_root,
            )
        )
    return outputs
