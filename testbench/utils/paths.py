#!/usr/bin/env python3 
# 
# paths.py  Andrew Belles  Jan 7th, 2025 
# 
# Global Path Constants for Testbenches 
# 
# 

from pathlib import Path

from utils.helpers import project_path

MOBILITY_PATH          = project_path("data", "datasets", "travel_proxy.mat")
SHAPEFILE              = project_path("data", "geography", "county_shapefile", 
                                      "tl_2020_us_county.shp")
CONFIG_PATH            = project_path("testbench", "model_config.yaml")
LABELS_PATH            = project_path("data", "datasets", "viirs_nchs_2023.mat")
PROBA_PATH             = project_path("data", "results", "final_stacked_predictions.mat")
PROBA_PASSTHROUGH_PATH = project_path("data", "results", "final_stacked_passthrough.mat")
PROBA_DIR              = project_path("data", "stacking")
RESULTS_DIR            = project_path("data", "results")

def check_paths_exist(paths, label): 
    missing = [p for p in paths if not Path(p).exists()]
    if missing: 
        raise FileNotFoundError(f"{label} missing: {missing}")
