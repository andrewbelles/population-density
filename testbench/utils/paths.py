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
ROUND_ROBIN_CONFIG     = project_path("testbench", "round_robin_config.yaml")
ROUND_ROBIN_PROBA      = project_path("data", "results", "round_robin_stacked_probs.mat")
ROUND_EXPERT_PROBA     = {
    "VIIRS": project_path("data", "stacking", "rr_viirs_probs.mat"),
    # "TIGER": project_path("data", "stacking", "rr_tiger_probs.mat"),
    "NLCD":  project_path("data", "stacking", "rr_nlcd_probs.mat"),
    "SAIPE": project_path("data", "stacking", "rr_saipe_probs.mat")
}
ROUND_ROBIN_OVR_PROBA  = {
    label: project_path("data", "stacking", f"rr_label_{label}_probs.mat")
    for label in range(1, 7)
}

EXPERT_DATA            = {
    "VIIRS": project_path("data", "datasets", "viirs_nchs_2023.mat"),
    "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"),
    "NLCD":  project_path("data", "datasets", "nlcd_nchs_2023.mat")
}
PAIRWISE_CSV           = project_path("data", "results", "pairwise_vif.csv")
FULL_CSV               = project_path("data", "results", "full_vif.csv")

def check_paths_exist(paths, label): 
    missing = [p for p in paths if not Path(p).exists()]
    if missing: 
        raise FileNotFoundError(f"{label} missing: {missing}")
