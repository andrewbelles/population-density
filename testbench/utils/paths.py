#!/usr/bin/env python3 
# 
# paths.py  Andrew Belles  Jan 7th, 2025 
# 
# Global Path Constants for Testbenches 
# 
# 

from pathlib import Path

from utils.helpers import project_path

DATASETS               = ("SAIPE", "VIIRS", "NLCD")
MOBILITY_PATH          = project_path("data", "datasets", "travel_proxy.mat")
SHAPEFILE              = project_path("data", "geography", "county_shapefile", 
                                      "tl_2020_us_county.shp")
CONFIG_PATH            = project_path("testbench", "model_config.yaml")
LABELS_PATH            = project_path("data", "datasets", "saipe_scalar_2019.mat")
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
    # "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"),
    "NLCD":  project_path("data", "datasets", "nlcd_nchs_2023.mat"),
    "SAIPE": project_path("data", "datasets", "saipe_nchs_2023.mat")
}
PAIRWISE_CSV           = project_path("data", "results", "pairwise_vif.csv")
FULL_CSV               = project_path("data", "results", "full_vif.csv")

DEFAULT_PROB_PATHS     = {
    "VIIRS": project_path("data", "stacking", "viirs_optimized_probs.mat"),
    "NLCD":  project_path("data", "stacking", "nlcd_optimized_probs.mat"),
    "SAIPE": project_path("data", "stacking", "saipe_optimized_probs.mat")
}

BORUTA_CSV            = project_path("data", "results", "boruta_summary.csv")

def check_paths_exist(paths, label): 
    missing = [p for p in paths if not Path(p).exists()]
    if missing: 
        raise FileNotFoundError(f"{label} missing: {missing}")

def read_feature_list(path: str | None): 
    if not path: 
        return None 
    lines = Path(path).read_text().splitlines() 
    return [l.strip() for l in lines if l.strip()]

def keep_list(filter_dir: str | None, name: str) -> list[str] | None: 
    if not filter_dir: 
        return None 
    keep_path = Path(filter_dir) / f"boruta_keep_{name.lower()}.txt"
    return read_feature_list(str(keep_path))

def stacking_context(passthrough: bool): 
    key   = "StackingPassthrough"  if passthrough else "Stacking"
    proba = PROBA_PASSTHROUGH_PATH if passthrough else PROBA_PATH 
    name  = "StackingPassthrough"  if passthrough else "Stacking"
    return key, proba, name 

def expert_prob_files(datasets=DATASETS, prob_map=DEFAULT_PROB_PATHS): 
    return [prob_map[d] for d in datasets]
