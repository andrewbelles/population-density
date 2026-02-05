#!/usr/bin/env python3 
# 
# config.py  Andrew Belles  Jan 7th, 2025 
# 
# Configuration Helper functions for Testbench 
# 
# 

from functools import partial 

from pathlib import Path

import numpy as np 

from models.estimators import TFTabular, SpatialGATClassifier

from utils.helpers import load_yaml_config 

from analysis.cross_validation import CVConfig

from models.graph.construction import (
    LOGRADIANCE_GATE_LOW,
    LOGRADIANCE_GATE_HIGH,
    LOGCAPACITY_GATE_HIGH,
    LOGCAPACITY_GATE_LOW
)

from utils.helpers   import bind 

from utils.resources import ComputeStrategy

def load_model_params(config_path: str, key: str) -> dict: 

    config = load_yaml_config(Path(config_path))
    params = config.get("models", {}).get(key)
    if params is None: 
        raise ValueError(f"missing model config for key: {key}")
    return dict(params)

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

def cv_config(folds: int, random_state: int) -> CVConfig:
    config = CVConfig(n_splits=folds, n_repeats=1, stratify=True, random_state=random_state)
    config.verbose = False 
    return config 

def normalize_spatial_params(params, *, random_state: int, collate_fn): 
    params.setdefault("random_state", random_state)
    params.setdefault("collate_fn", collate_fn)
    params.setdefault("epochs", 350)
    params.setdefault("early_stopping_rounds", 15)
    params.setdefault("eval_fraction", 0.2)
    params.setdefault("min_delta", 1e-4)
    params.setdefault("target_global_batch", 2048)
    return params 

def spatial_gat_factory(
    *,
    collate_fn,
    compute_strategy,
    fixed,
    in_channels=None,
    **params
): 
    merged = dict(fixed)
    merged.update(params)

    if in_channels is not None: 
        merged.setdefault("in_channels", in_channels)

    scale = merged.pop("threshold_scale", 1.0)
    merged.setdefault("thresh_low", LOGRADIANCE_GATE_LOW * scale)
    merged.setdefault("thresh_high", LOGRADIANCE_GATE_HIGH * scale)

    collate = merged.pop("collate_fn", collate_fn)
    return SpatialGATClassifier(
        collate_fn=collate,
        compute_strategy=compute_strategy,
        **merged
    )

def make_spatial_gat(
    *,
    collate_fn=None,
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
    **fixed
): 
    return bind(
        spatial_gat_factory,
        collate_fn=collate_fn,
        compute_strategy=compute_strategy,
        fixed=fixed 
    )

def _spatial_factory(
    factory, 
    in_ch,
    **kwargs
): 
    return factory(
        in_channels=in_ch,
        **kwargs 
    )

def with_spatial_channels(factory, spatial_data, **kwargs): 
    in_ch = spatial_data.get("in_channels")
    return partial(
        _spatial_factory,
        factory=factory,
        in_ch=in_ch,
        **kwargs 
    )

def make_residual_tabular(**fixed):
    def _factory(**params): 
        merged = dict(fixed)
        merged.update(params)

        return TFTabular(**merged)

    return _factory

def load_node_anchors(anchor_path: str | None): 
    if not anchor_path: 
        return None 

    path = Path(anchor_path)
    if not path.exists(): 
        raise FileNotFoundError(f"anchors file not found: {path}")

    anchors = np.load(path)
    arr     = np.asarray(anchors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 3: 
        raise ValueError(f"anchors must have shape (3, C), got {arr.shape}")

    stats_path = path.with_suffix("")
    stats_path = stats_path.with_name(stats_path.name + "_stats.npy")
    if not Path(stats_path).exists(): 
        raise FileNotFoundError(f"anchor stats file not found: {stats_path}")

    stats = np.asarray(np.load(stats_path), dtype=np.float32) 

    return arr.tolist(), stats.tolist() 
