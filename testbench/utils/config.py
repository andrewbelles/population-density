#!/usr/bin/env python3 
# 
# config.py  Andrew Belles  Jan 7th, 2025 
# 
# Configuration Helper functions for Testbench 
# 
# 

from pathlib import Path

from utils.helpers import load_yaml_config 

from analysis.cross_validation import CVConfig

def load_model_params(config_path: str, key: str) -> dict: 

    config = load_yaml_config(Path(config_path))
    params = config.get("models", {}).get(key)
    if params is None: 
        raise ValueError(f"missing model config for key: {key}")
    return dict(params)

def get_cached_params(cache: dict, key: str): 
    models = cache.get("models", {})
    if isinstance(models, dict): 
        return models.get(key)
    return None

def normalize_params(model_type: str, params: dict) -> dict: 
    if model_type != "SVM": 
        return params 
    cleaned = dict(params)

    if model_type == "CNN" and "conv_channels" in params: 
        v = params["conv_channels"]
        if isinstance(v, str): 
            params["conv_channels"] = tuple(int(x) for x in v.split("-") if x)
        elif isinstance(v, list): 
            params["conv_channels"] = tuple(v)

    if "gamma" not in cleaned: 
        for key in ("gamma_poly", "gamma_sigmoid", "gamma_rbf", "gamma_custom"):
            if key in cleaned: 
                cleaned["gamma"] = cleaned.pop(key)
                
                break 
    cleaned.pop("gamma_mode", None)
    return cleaned 

def eval_config(random_state: int = 0): 
    cfg = CVConfig(
        n_splits=5,
        n_repeats=1,
        stratify=True,
        random_state=random_state
    ) 
    cfg.verbose = False 
    return cfg 

