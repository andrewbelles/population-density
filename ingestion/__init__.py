#!/usr/bin/env python3
#
# __init__.py  Andrew Belles  Mar 27th, 2026
#
# Ingestion stage exports.
#

from ingestion.config import IngestConfig, load_config

__all__ = ["IngestConfig", "load_config"]
