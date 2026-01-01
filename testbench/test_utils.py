#!/usr/bin/env python3 
# 
# test_utils.py  Andrew Belles  Dec 29th, 2025 
# 
# Test utilities to clean up main testbench files 
# 
# 


from enum import unique
from typing import Sequence
import numpy as np

from pathlib import Path

from models.metric import GradientBoostingMetricLearner, IDMLGraphLearner

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

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import kneighbors_graph 
from models.graph_utils import normalize_adjacency

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
    "COORDS": {
        "name": "COORDS",
        "path": project_path("data", "datasets", "travel_proxy.mat"),
        "loader": load_coords_from_mobility
    },
    "PASSTHROUGH": {
        "name": "PASSTHROUGH",
        "path": project_path("data", "datasets", "cross_modal_2023.mat"),
        "loader": load_viirs_nchs
    },
    "OOF": {
        "name": "OOF",
        "path": project_path("data", "results", "final_stacked_predictions.mat"),
        "loader": make_oof_dataset_loader()
    },
}


def _align_by_fips(fips_order, fips_vec):
    idx_map = {f: i for i, f in enumerate(fips_vec)}
    return np.array([idx_map[f] for f in fips_order], dtype=int)


def _coords_for_fips(coords_path: str, fips_order):
    data = load_coords_from_mobility(coords_path)
    idx = _align_by_fips(fips_order, data["sample_ids"])
    return np.asarray(data["coords"])[idx]


def _apply_transforms(X, transforms):
    for t in transforms: 
        if hasattr(t, "fit_transform"):
            X = t.fit_transform(X)
        else: 
            t.fit(X)
            X = t.transform(X)
    return X 


def _iter_metric_models(cfg): 
    models = cfg.get("models", {})
    for key, params in models.items(): 
        if not isinstance(params, dict): 
            continue 
        if not key.endswith(("/IDML", "/GBM")):
            continue 
        if "dataset" not in params: 
            continue 
        yield key, dict(params)


def _override_oof_path(specs, oof_path: str):
    out = []
    for s in specs: 
        if s["name"].upper() == "OOF": 
            s = dict(s)
            s["path"] = oof_path 
        out.append(s)
    return out 

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


def _extract_oof_preds(oof, model_name=None): 
    preds = np.asarray(oof["preds"])
    model_names = oof["model_names"]
    if preds.ndim == 2: 
        if model_name and model_name in model_names: 
            m_idx = int(np.where(model_names == model_name)[0][0])
        else: 
            m_idx = 0
        return preds[:, m_idx]
    return preds.reshape(-1)


def _majority_vote(pred_matrix): 
    out = np.empty(pred_matrix.shape[0], dtype=pred_matrix.dtype)
    for i, row in enumerate(pred_matrix):
        vals, counts = np.unique(row, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out 


def _metrics_from_preds(y_true, y_pred, class_labels=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, average="macro"))

    labels = (np.asarray(class_labels).reshape(-1) if class_labels is not None and 
        len(class_labels) else np.unique(y_true))
    labels = np.array(sorted(labels.tolist()))
    scores = np.zeros((y_pred.shape[0], labels.size), dtype=np.float64)
    for i, lbl in enumerate(labels): 
        scores[y_pred == lbl, i] = 1.0 
    roc = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=labels))

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": roc}


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


def _make_gb_metric(**kwargs):
    return GradientBoostingMetricLearner(**kwargs)
