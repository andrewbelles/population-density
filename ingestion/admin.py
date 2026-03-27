#!/usr/bin/env python3
#
# admin.py  Andrew Belles  Mar 27th, 2026
#
# Administrative feature orchestration for LAUS and housing.
#

import logging
from pathlib import Path

from ingestion import housing, laus
from ingestion.config import IngestConfig


LOGGER = logging.getLogger("ingestion.admin")


def run(config: IngestConfig, *, skip_existing: bool = False) -> dict[str, Path]:
    if not config.admin.enabled:
        return {}
    outputs: dict[str, Path] = {}
    laus_path = laus.run(config, skip_existing=bool(skip_existing))
    if laus_path is not None:
        outputs["laus"] = laus_path
    housing_path = housing.run(config, skip_existing=bool(skip_existing))
    if housing_path is not None:
        outputs["housing"] = housing_path
    LOGGER.debug("admin outputs=%s", {k: str(v) for k, v in outputs.items()})
    return outputs
