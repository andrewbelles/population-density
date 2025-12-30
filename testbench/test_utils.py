#!/usr/bin/env python3 
# 
# test_utils.py  Andrew Belles  Dec 29th, 2025 
# 
# Test utilities to clean up main testbench files 
# 
# 


from typing import Sequence
import numpy as np

from pathlib import Path

from models.metric import IDMLGraphLearner

from support.helpers import project_path

from analysis.hyperparameter import (
    _load_yaml_config,

)

from preprocessing.loaders import (
    load_viirs_nchs,
    load_coords_from_mobility,
    load_compact_dataset,
    make_oof_dataset_loader,
    ConcatSpec
)

from itertools import combinations

BASE: dict[str, ConcatSpec] = {
    "VIIRS": {
        "name": "VIIRS",
        "path": project_path("data", "datasets", "viirs_nchs_2023.mat"),
        "loader": load_viirs_nchs
    },
    "TIGER": {
        "name": "TIGER",
        "path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
        "loader": load_compact_dataset
    },
    "NLCD": {
        "name": "NLCD",
        "path": project_path("data", "datasets", "nlcd_nchs_2023.mat"),
        "loader": load_compact_dataset
    },
    "OOF": {
        "name": "OOF",
        "path": project_path("data", "results", "final_stacked_predictions.mat"),
        "loader": make_oof_dataset_loader()
    },
    "COORDS": {
        "name": "COORDS",
        "path": project_path("data", "datasets", "travel_proxy.mat"),
        "loader": load_coords_from_mobility
    } 
}


def _load_model_params(config_path: str, key: str) -> dict: 

    config = _load_yaml_config(Path(config_path))
    params = config.get("models", {}).get(key)
    if params is None: 
        raise ValueError(f"missing model config for key: {key}")
    return dict(params)


def _select_specs_psv(dataset_key: str) -> Sequence[ConcatSpec]: 
    names = dataset_key.split("+")
    return [BASE[n] for n in names]


def _get_cached_params(cache: dict, key: str): 
    models = cache.get("models", {})
    if isinstance(models, dict): 
        return models.get(key)
    return None


def _normalize_params(model_type: str, params: dict) -> dict: 
    if model_type != "SVM": 
        return params 
    cleaned = dict(params)
    if "gamma" not in cleaned: 
        for key in ("gamma_poly", "gamma_sigmoid", "gamma_rbf", "gamma_custom"):
            if key in cleaned: 
                cleaned["gamma"] = cleaned.pop(key)
                
                break 
    cleaned.pop("gamma_mode", None)
    return cleaned 


def _score_from_summary(summary): 
    row = summary.iloc[0]
    for key in ("f1_macro_mean", "accuracy_mean", "roc_auc_mean", "r2_mean"): 
        if key in row and not np.isnan(row[key]): 
            return float(row[key])
    return float("nan")


def _metrics_from_summary(summary): 
    row = summary.iloc[0]
    return {
        "accuracy": float(row["accuracy_mean"]) if "accuracy_mean" in row else np.nan, 
        "f1_macro": float(row["f1_macro_mean"]) if "f1_macro_mean" in row else np.nan,
        "roc_auc": float(row["roc_auc_mean"]) if "roc_auc_mean" in row else np.nan
    }


def _best_score(metrics): 
    for key in ("f1_macro", "accuracy", "roc_auc"): 
        val = metrics.get(key, np.nan)
        if not np.isnan(val): 
            return float(val)
    return float("-inf")


def _select_specs_csv(sources_csv: str):
    wanted = {s.strip().lower() for s in sources_csv.split(",") if s.strip()}
    return [s for s in BASE.values() if s["name"].lower() in wanted]


def _power_set(specs): 
    for r in range(1, len(specs) + 1): 
        for combo in combinations(specs, r): 
            yield combo 


def _map_fracs(params: dict, X): 
    max_components = max(1, min(128, X.shape[1]))
    max_neighbors  = max(1, min(100, X.shape[0] - 1))

    if "n_components_frac" in params: 
        frac = params.pop("n_components_frac")
        params["n_components"] = 1 + int(round(frac * (max_components - 1)))
    if "n_neighbors_frac" in params: 
        frac = params.pop("n_neighbors_frac")
        params["n_neighbors"] = 1 + int(round(frac * (max_neighbors - 1)))
    return params 


def _make_idml(**kwargs): 
    return IDMLGraphLearner(**kwargs)
