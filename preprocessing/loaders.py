#!/usr/bin/env python3 
# 
# loaders.py  Andrew Belles  Dec 19th, 2025 
# 
# Loader functions that return dataset dicts to 
# supervised/unsupervised datasets for ues by models 
# 

import json, os, torch 

import numpy as np 

import pandas as pd 

from numpy.typing import NDArray

from typing import Callable, Optional, Sequence, TypedDict, List, Any 

from dataclasses import dataclass

from pathlib import Path

from scipy.io import loadmat
from torch.utils.data import Dataset

from utils.helpers import (
    bind,
    _mat_str_vector,
    align_on_fips
)

from preprocessing.tensors import TileLoader 

_CLIMATE_GROUPS: tuple[str, ...] = ("degree_days", "palmer_indices")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# ---------------------------------------------------------
# Supervised Loader Interface 
# ---------------------------------------------------------

class DatasetDict(TypedDict): 
    features: NDArray
    labels: NDArray
    coords: NDArray
    feature_names: NDArray[np.str_]
    sample_ids: NDArray[np.str_]

DatasetLoader = Callable[[str], DatasetDict]


def load_climate_population(filepath: str, *, decade: int, groups: List[str]) -> DatasetDict: 
    
    '''
    Loader Helper for Climate against Population Density. 

    Caller Provides: 
        Path to dataset, 
        decade to load, 
        groups to include in feature set 

    We return: 
        A DatasetDict containing the requested features and labels 
    '''

    group_set = set(groups)

    data    = loadmat(filepath)
    decades = data["decades"]
    decade_data = decades[f"decade_{decade}"][0, 0]

    X = np.asarray(decade_data["features"][0, 0], dtype=np.float64)
    y = np.asarray(decade_data["labels"][0, 0], dtype=np.float64).reshape(-1)
    coords = np.asarray(data["coords"], dtype=np.float64)

    if "coords" in group_set and "climate" in group_set: 
        features = np.hstack([X, coords], dtype=np.float64)
    elif "coords" in group_set: 
        features = coords
    elif "climate" in group_set: 
        features = X 
    else: 
        raise ValueError(f"{groups} does not contain any valid group labels for data")

    if "fips_codes" in data: 
        fips = _mat_str_vector(data["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"dataset failed to extract fips codes")
        
    return {
        "features": features, 
        "labels": y, 
        "coords": coords,
        "feature_names": np.array([]), 
        "sample_ids": fips
    }


def load_climate_geospatial(filepath: str, *, target: str, groups: List[str]) -> DatasetDict: 
    '''
    Loader Helper for Climate against (lat, lon). 

    Caller Provides: 
        filepath to dataset, 
        target (lat, lon, or all)
        groups to include in dataset 

    We return: 
        A DatasetDict containing the requested feature groups and target label 
    '''

    group_set = set(groups)
    label_set = {"lat", "lon", "all"}

    if target not in label_set: 
        raise ValueError(f"target: {target} must be in {sorted(label_set)} to be requested")

    data = loadmat(filepath)
    c = np.asarray(data["labels"], dtype=np.float64)   # expects shape (n, 2)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"expected labels to have shape (n, 2); got {c.shape}")

    if "fips_codes" in data: 
        fips = _mat_str_vector(data["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"expected fips_codes in climate_geospatial dataset")

    if target == "all":
        y = c.astype(np.float64, copy=False)
    else:
        idx = 0 if target == "lat" else 1
        y = c[:, idx].reshape(-1)

    features = []
    for name in _CLIMATE_GROUPS: 
        if name not in group_set: 
            continue 
        features.append(np.asarray(data[f"features_{name}"], dtype=np.float64))

    if not features: 
        raise ValueError(f"{groups} does not contain any valid group labels for data")

    features = np.hstack(features) if len(features) > 1 else features[0] 
    return {
        "features": features, 
        "labels": y, 
        "coords": c,
        "feature_names": np.array([]),
        "sample_ids": fips
    }


def load_geospatial_climate(filepath, *, target: str, groups: List[str] = ["lat", "lon"]) -> DatasetDict: 
    
    data = loadmat(filepath)
    coords = np.asarray(data["labels"], dtype=np.float64) 
    if coords.ndim != 2 or coords.shape[1] != 2: 
        raise ValueError(f"expected shape (n,2): got {coords.shape}")
    
    if "fips_codes" in data: 
        fips = _mat_str_vector(data["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"expected fips_codes in climate_geospatial dataset")

    names = _mat_str_vector(data["feature_names"])
    F = np.asarray(data["features"], dtype=np.float64)

    idx = {"lat": 0, "lon": 1}
    X = coords[:, [idx[c] for c in groups]].astype(np.float64, copy=False)

    if target == "all": 
        y = F.astype(np.float64, copy=False) 
        return {
            "features": X, 
            "labels": y, 
            "coords": coords, 
            "feature_names": names, 
            "sample_ids": fips
        }

    cols = [f"{target}_{m}" for m in MONTHS]
    col_idx = [int(np.where(names == c)[0][0]) for c in cols]
    Ym = F[:, col_idx] 
    y = np.column_stack([Ym, Ym.mean(axis=1)])

    return {
        "features": X, 
        "labels": y, 
        "coords": coords,
        "feature_names": names, 
        "sample_ids": fips
    }


def load_saipe_population(
    filepath: str, 
    *, decade: int, 
    groups: List[str],
    label_transform: str | None = None 
)-> DatasetDict: 

    group_set = set(groups)

    data  = loadmat(filepath)
    years = data["years"]
    year_data = years[f"year_{decade}"][0, 0]

    X = np.asarray(year_data["features"][0, 0], dtype=np.float64)
    y = np.asarray(year_data["labels"][0, 0], dtype=np.float64).reshape(-1)

    if label_transform == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p label_transform requires non-negative labels")
        y = np.log1p(y)
    elif label_transform is not None: 
        raise ValueError(f"unknown label_transform: {label_transform}")
    
    coords = np.asarray(data["coords"], dtype=np.float64)
    # Might possible be inserted in wrong shape, transpose if so 
    if coords.ndim == 2 and coords.shape == (2, y.shape[0]): 
        coords = coords.T 

    # TODO: SPLIT UP GROUPS ON FEATURE COLUMN INSTEAD OF JUST SAIPE 
    if "all" in group_set: 
        features = X 
    else: 
        _ = group_set 
        raise ValueError(f"{groups} does not contain any valid group labels")

    if "fips_codes" in data: 
        fips = _mat_str_vector(data["fips_codes"]).astype("U5")
    else: 
        raise ValueError("dataset failed to extract fips codes")

    return {
        "features": features, 
        "labels": y, 
        "coords": coords, 
        "feature_names": np.array([]), 
        "sample_ids": fips
    }


def load_viirs_nchs(filepath: str) -> DatasetDict: 

    mat = loadmat(filepath)

    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"{filepath} missing required keys 'features'/'labels'")

    X = np.asarray(mat["features"], dtype=np.float64) 
    y = np.asarray(mat["labels"], dtype=np.int64).reshape(-1)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({y.shape[0]})")

    if "fips_codes" in mat: 
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"{filepath} missing fips_codes")

    if "feature_names" in mat: 
        feature_names = _mat_str_vector(mat["feature_names"]).astype("U")
        feature_names = np.array([n.strip() for n in feature_names], dtype="U")
    else: 
        feature_names = np.array([], dtype=np.str_) 

    coords = np.zeros((X.shape[0],2), dtype=np.float64)
    return {
        "features": X, 
        "labels": y, 
        "coords": coords,
        "feature_names": feature_names,
        "sample_ids": fips
    }


def load_tiger_noncore_binary(filepath: str, *, noncore_label: int = 5) -> DatasetDict: 
    mat = loadmat(filepath)

    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"{filepath} missing required keys 'features'/'labels'")

    X = np.asarray(mat["features"], dtype=np.float64) 
    y = np.asarray(mat["labels"], dtype=np.int64).reshape(-1)

    # Collapse {0,4} into label 0, label 5 maps to 1 
    if noncore_label not in np.unique(y): 
        raise ValueError(f"noncore_label {noncore_label} not found in labels {np.unique(y)}")
    y = (y == noncore_label).astype(np.int64)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({y.shape[0]})")

    if "fips_codes" in mat: 
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"{filepath} missing fips_codes")

    if "feature_names" in mat: 
        feature_names = _mat_str_vector(mat["feature_names"]).astype("U")
        feature_names = np.array([n.strip() for n in feature_names], dtype="U")
    else: 
        feature_names = np.array([], dtype=np.str_) 

    coords = np.zeros((X.shape[0],2), dtype=np.float64)
    return {
        "features": X, 
        "labels": y, 
        "coords": coords,
        "feature_names": feature_names,
        "sample_ids": fips
    }


def load_residual_dataset(residual_filepath: str, original_filepath: str) -> DatasetDict: 

    residual_data = loadmat(residual_filepath)
    base = load_geospatial_climate(original_filepath, target="all", groups=["lat", "lon"])

    if "features" not in residual_data: 
        raise ValueError(f"{residual_filepath} missing features")

    X_resi = np.asarray(residual_data["features"], dtype=np.float64)
    X_orig = np.asarray(base["features"], dtype=np.float64)
    y = np.asarray(base["labels"], dtype=np.float64)

    if X_resi.ndim == 1:
        X_resi = X_resi.reshape(-1, 1)
    if X_orig.ndim == 1:
        X_orig = X_orig.reshape(-1, 1)

    if X_resi.shape[0] != X_orig.shape[0]:
        raise ValueError(
            f"residual rows ({X_resi.shape[0]}) != original rows ({X_orig.shape[0]})"
        )

    X = np.hstack([X_orig, X_resi]).astype(np.float64, copy=False)

    # Preserve multi-output labels (n, k). Only squeeze the trivial (n, 1) case.
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim not in (1, 2):
        raise ValueError(f"labels must be 1D or 2D; got shape {y.shape}")

    n = X.shape[0]
    if y.shape[0] != n: 
        raise ValueError(f"features rows ({n}) != labels row ({y.shape[0]})")

    coords = np.asarray(base["coords"], dtype=np.float64)

    return {
        "features": X, 
        "labels": y, 
        "coords": coords, 
        "feature_names": np.array([]), 
        "sample_ids": base["sample_ids"]
    }


def load_compact_dataset(filepath: str) -> DatasetDict: 

    '''
    Note, also loads 
    '''
    
    mat = loadmat(filepath)

    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"{filepath} missing required keys 'features'/'labels'")

    X = np.asarray(mat["features"], dtype=np.float64) 
    y = np.asarray(mat["labels"], dtype=np.float64)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({y.shape[0]})")

    if "coords" in mat:
        coords = np.asarray(mat["coords"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape == (2, y.shape[0]):
            coords = coords.T
        if coords.ndim != 2 or coords.shape != (y.shape[0], 2):
            raise ValueError(f"expected coords shape (n,2), got {coords.shape}")
    else:
        coords = np.zeros((y.shape[0], 2), dtype=np.float64)

    if "feature_names" in mat: 
        feature_names = _mat_str_vector(mat["feature_names"]).astype("U")
        feature_names = np.array([n.strip() for n in feature_names], dtype="U")
    else: 
        feature_names = np.array([], dtype=np.str_)

    if "fips_codes" in mat: 
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
    else: 
        raise ValueError(f"expected fips_codes in climate_geospatial dataset")

    return {
        "features": X, 
        "labels": y, 
        "coords": coords,
        "feature_names": feature_names, 
        "sample_ids": fips 
    }


def load_stacking(filepaths: Sequence[str]) -> DatasetDict:  
    mats = [loadmat(p) for p in filepaths]

    feats = []
    for m in mats: 
        X = np.asarray(m["features"], dtype=np.float64)
        if X.ndim != 2: 
            raise ValueError(f"expected 2d features, got {X.shape}")
        # only yield the true probs for binary classes 
        if X.shape[1] == 2: 
            X = X[:, [1]] 
        feats.append(X)

    labels = np.asarray(mats[0]["labels"]).reshape(-1) 
    fips_list = [_mat_str_vector(m["fips_codes"]).astype("U5") for m in mats]

    idx_maps = [{f: i for i, f in enumerate(fips)} for fips in fips_list]

    common = [f for f in fips_list[0] if all(f in m for m in idx_maps[1:])]

    idx_lists = [[m[f] for f in common] for m in idx_maps]

    feats  = [X[idx] for X, idx in zip(feats, idx_lists)]
    labels = labels[idx_lists[0]]
    fips   = np.array(common, dtype="U5")

    X = np.hstack(feats)
    coords = np.zeros((X.shape[0], 2), dtype=np.float64)

    return {
        "features": X, 
        "labels": labels, 
        "coords": coords, 
        "feature_names": np.array([]),   
        "sample_ids": fips}


def load_coords_from_mobility(filepath: str) -> DatasetDict: 
    mat = loadmat(filepath)
    if "fips_codes" not in mat or "coords" not in mat: 
        raise ValueError(f"{filepath} missing 'fips_codes'/'coords'")

    fips   = _mat_str_vector(mat["fips_codes"]).astype("U5")
    coords = np.asarray(mat["coords"], dtype=np.float64)
    if coords.ndim == 2 and coords.shape == (2, coords.shape[0]): 
        coords = coords.T 
    return {
        "features": coords, 
        "labels": np.zeros(coords.shape[0], dtype=np.int64), 
        "coords": coords, 
        "feature_names": np.array(["lat", "lon"], dtype="U"), 
        "sample_ids": fips
    }


def load_tensor_data(path): 
    mat = loadmat(path)
    required = {"spatial", "mask", "gaf", "labels"}
    if not required.issubset(mat): 
        raise ValueError(f"{path} missing required keys: {sorted(required)}")

    spatial = np.asarray(mat["spatial"], dtype=np.float32)
    mask    = np.asarray(mat["mask"], dtype=np.float32)
    gaf     = np.asarray(mat["gaf"], dtype=np.float32)
    labels  = np.asarray(mat["labels"], dtype=np.int64).reshape(-1)

    if "fips" in mat: 
        fips = _mat_str_vector(mat["fips"]).astype("U5")
    elif "fips_codes" in mat: 
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
    else: 
        fips = np.asarray([f"{i:05d}" for i in range(labels.shape[0])], dtype="U5")

    return spatial, mask, gaf, labels, fips

# ---------------------------------------------------------
# Unsupervised Loader Interface 
# ---------------------------------------------------------

@dataclass
class FeatureMatrix: 
    X: NDArray[np.float64] 
    coords: NDArray[np.float64] 
    feature_names: NDArray[np.str_]
    sample_ids: NDArray[np.str_]

    def subset(self, cols: list[int]) -> "FeatureMatrix": 
        return FeatureMatrix(
            X=self.X[:, cols], 
            coords=self.coords,
            feature_names=self.feature_names[cols], 
            sample_ids=self.sample_ids, 
        )

UnsupervisedLoader = Callable[[str], FeatureMatrix]

def make_pair_loader(
    data: FeatureMatrix,
    target_idx: int, 
    feature_idx: int 
) -> Callable[[str], dict]: 
    return bind(
        _pair_loader,
        data=data,
        target_idx=target_idx,
        feature_idx=feature_idx
    )

def _pair_loader(_filepath: str, *, data: FeatureMatrix, target_idx: int, feature_idx: int): 
    return {
        "features": data.X[:, [feature_idx]], 
        "labels": data.X[:, target_idx], 
        "coords": data.coords, 
        "feature_names": np.array([data.feature_names[feature_idx]]),
        "sample_idx": data.sample_ids
    }

def load_tiger_nlcd_viirs_feature_matrix(filepaths: Sequence[str]): 
    if len(filepaths) != 3: 
        raise ValueError("filepaths must be for viirs, nlcd, and tiger datasets")

    PREFIXES = ("viirs", "nlcd", "tiger")

    feats: list[NDArray] = []
    names_list: list[NDArray[np.str_]] = []
    fips_list: list[NDArray[np.str_]]  = []

    for path, prefix in zip(filepaths, PREFIXES): 
        mat = loadmat(path)

        if "features" not in mat: 
            raise ValueError(f"{path} missing 'features'")
        X = np.asarray(mat["features"], dtype=np.float64)
        if X.ndim == 1: 
            X = X.reshape(-1, 1) 

        if "fips_codes" not in mat: 
            raise ValueError(f"{path} missing 'fips_codes'")
        fips = _mat_str_vector(mat["fips_codes"]).astype("U5")
        
        if "feature_names" in mat: 
            names = _mat_str_vector(mat["feature_names"]).astype("U64")
            names = np.array([n.strip() for n in names], dtype="U64")
            # Coerce instead of failing 
            if names.shape[0] != X.shape[1]: 
                names = np.array([f"{prefix}_{i}" for i in range(X.shape[1])], dtype="U64")
        else: 
            names = np.array([f"{prefix}_{i}" for i in range(X.shape[1])], dtype="U64")

        feats.append(X)
        names_list.append(names)
        fips_list.append(fips)

    idx_maps = [{f: i for i, f in enumerate(fips)} for fips in fips_list]
    common   = [f for f in fips_list[0] if all(f in m for m in idx_maps[1:])]
    if not common: 
        raise ValueError("no common FIPS codes across datasets")

    idx_lists = [[m[f] for f in common] for m in idx_maps]
    feats     = [X[idx] for X, idx in zip(feats, idx_lists)]
    
    X_all = np.hstack(feats)
    feature_names = np.concatenate(names_list)
    sample_ids = np.array(common, dtype="U5")
    coords = np.zeros((X_all.shape[0], 2), dtype=np.float64)

    return FeatureMatrix(
        X=X_all, 
        coords=coords,
        feature_names=feature_names, 
        sample_ids=sample_ids
    )

# --------------------------------------------------------- 
# OOF, Probability Datasets 
# --------------------------------------------------------- 

class OOFDatasetDict(TypedDict):
    probs: NDArray
    preds: NDArray
    labels: NDArray
    fips_codes: NDArray[np.str_]
    model_names: NDArray[np.str_]
    class_labels: NDArray

def load_oof_predictions(filepath: str) -> OOFDatasetDict: 

    mat = loadmat(filepath) 

    if "fips_codes" not in mat: 
        if "sample_ids" not in mat: 
            raise ValueError(f"{filepath} missing fips_codes and sample_ids")
        else: 
            key = "sample_ids"
    else: 
        key = "fips_codes"
    fips = _mat_str_vector(mat[key]).astype("U5")

    if "model_names" in mat: 
        model_names = _mat_str_vector(mat["model_names"]).astype("U64")
    else: 
        model_names = np.zeros((0,), dtype="U64")

    if "class_labels" in mat:
        class_labels = np.asarray(mat["class_labels"]).reshape(-1).astype(np.int64)
    else: 
        class_labels = np.array([], dtype=np.int64)

    if "probs" in mat: 
        probs = np.asarray(mat["probs"], dtype=np.float64)
        if probs.ndim != 3: 
            raise ValueError(f"{filepath} expected probs with shape (n, m, c), got {probs.shape}")
    else: 
        if "features" not in mat: 
            raise ValueError(f"{filepath} missing probs/features for OOF")
        feats = np.asarray(mat["features"], dtype=np.float64)
        n_models  = model_names.shape[0]
        n_classes = class_labels.shape[0] if class_labels.size else 1 
        expected = n_models * n_classes 
        if feats.shape[1] != expected: 
            raise ValueError(f"{filepath} features cols ({feats.shape[1]} != {expected})") 
        probs = feats.reshape(feats.shape[0], n_models, n_classes)

    if "preds" in mat: 
        preds = np.asarray(mat["preds"], dtype=np.int64)
    else: 
        pred_idx = np.argmax(probs, axis=2)
        preds = class_labels[pred_idx] if class_labels.size else pred_idx 

    if "labels" not in mat: 
        raise ValueError(f"{filepath} missing labels")
    labels = np.asarray(mat["labels"]).reshape(-1)

    if probs.shape[0] != labels.shape[0]: 
        raise ValueError(f"{filepath} probs rows ({probs.shape[0]}) !- label rows ({labels.shape[0]})")

    return {
        "probs": probs, 
        "preds": preds, 
        "labels": labels, 
        "fips_codes": fips, 
        "model_names": model_names, 
        "class_labels": class_labels 
    }

def load_oof_errors(
    oof_path: str, 
    label_path: str | None = None, 
    coords_path: str | None = None, 
    model_name: str | None = None 
) -> pd.DataFrame: 

    oof   = loadmat(oof_path)
    label = loadmat(label_path)

    fips_oof   = _mat_str_vector(oof["fips_codes"]).astype("U5")
    fips_label = _mat_str_vector(label["fips_codes"]).astype("U5")

    idx_oof   = {f: i for i, f in enumerate(fips_oof)}
    idx_label = {f: i for i, f in enumerate(fips_label)}

    common    = [f for f in fips_label if f in idx_oof]
    oof_idx   = np.array([idx_oof[f] for f in common], dtype=int)
    label_idx = np.array([idx_label[f] for f in common], dtype=int)

    y_true = np.asarray(label["labels"]).reshape(-1)[label_idx]

    preds = np.asarray(oof["preds"])
    model_names = _mat_str_vector(oof["model_names"]).astype("U64")

    if preds.ndim == 2 and preds.shape[1] > 1: 
        if model_name is None: 
            raise ValueError(f"model_name required when OOF has multiple models")
        m_idx  = int(np.where(model_names == model_name)[0][0])
        y_pred = preds[oof_idx, m_idx]
    else: 
        y_pred = preds[oof_idx].reshape(-1)

    if "coords" in label: 
        coords = np.asarray(label["coords"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape == (2, coords.shape[1]):
            coords = coords.T 
        coords = coords[label_idx]
    elif coords_path is not None:
        cm     = loadmat(coords_path)
        fips_c = _mat_str_vector(cm["fips_codes"]).astype("U5")
        coords = np.asarray(cm["coords"], dtype=np.float64)
        if coords.ndim == 2 and coords.shape == (2, coords.shape[1]): 
            coords = coords.T 
        idx_c = {f: i for i, f in enumerate(fips_c)}
        coords = np.array([coords[idx_c[f]] for f in common], dtype=np.float64)
    else: 
        coords = np.full((len(common), 2), np.nan, dtype=np.float64)

    return pd.DataFrame({
        "FIPS": common, 
        "Lat": coords[:, 0], 
        "Lon": coords[:, 1], 
        "True_Class": y_true, 
        "Predicted_Class": y_pred,
        "Class_Distance": y_true - y_pred
    })

# ---------------------------------------------------------
# Meta-Learner Dataset Loader  
# ---------------------------------------------------------

class MetaSpec(TypedDict): 
    name:         str 
    proba_path:   str 
    proba_loader: Callable[[str], OOFDatasetDict] 
    model_name:   str | None 

class PassthroughSpec(TypedDict): 
    name:       str 
    raw_path:   str 
    raw_loader: DatasetLoader

def load_passthrough(
    meta_specs: Sequence[MetaSpec],
    *,
    label_path: str,
    label_loader: DatasetLoader, 
    passthrough_specs: Sequence[PassthroughSpec] | None = None, 
    passthrough_features: Sequence[str] | None 
) -> DatasetDict: 

    passthrough_specs = list(passthrough_specs or [])

    label_data = label_loader(label_path)
    label_fips = list(label_data["sample_ids"])
    y          = np.asarray(label_data["labels"]).reshape(-1)

    proba_data = {s["name"]: s["proba_loader"](s["proba_path"]) for s in meta_specs}
    raw_data   = {s["name"]: s["raw_loader"](s["raw_path"]) for s in passthrough_specs}

    fips_sets  = [set(label_fips)]
    for d in proba_data.values(): 
        fips_sets.append(set(d["fips_codes"]))
    for d in raw_data.values(): 
        fips_sets.append(set(d["sample_ids"]))

    common = [f for f in label_fips if all(f in s for s in fips_sets)]
    if not common: 
        raise ValueError("no common sample_ids across labels/probs/passthrough")

    idx_label   = align_on_fips(common, label_data["sample_ids"])
    y           = y[idx_label]
    prob_blocks = []
    prob_names  = []
    for s in meta_specs: 
        prob_ds     = proba_data[s["name"]]
        idx_probs   = align_on_fips(common, prob_ds["fips_codes"])
        model_names = prob_ds["model_names"].tolist() 
        model_name  = s.get("model_name")
        if model_name is None: 
            if len(model_names) != 1: 
                raise ValueError(f"{s['name']} probs has multiple models")
            m_idx  = 0 
            m_name = model_names[0]
        else: 
            if model_name not in model_names: 
                raise ValueError(f"{s['name']} model_name not in {model_names}")
            m_idx  = model_names.index(model_name)
            m_name = model_name 

        probs = prob_ds["probs"][idx_probs, m_idx, :]
        if probs.ndim != 2: 
            raise ValueError(f"{s['name']} expected 2d probs, got {probs.shape}")

        if probs.shape[1] == 2: 
            probs = probs[:, [1]]
            prob_blocks.append(probs)
            prob_names.append(f"{s['name']}__{m_name}_p1")
        else: 
            prob_blocks.append(probs)
            class_labels = prob_ds["class_labels"].reshape(-1)
            for c in class_labels: 
                prob_names.append(f"{s['name']}__{m_name}__p{int(c)}")

    if passthrough_features is None: 
        passthrough_features = []
        for name, raw in raw_data.items(): 
            names = raw.get("feature_names")
            if names is None or len(names) == 0: 
                raise ValueError(f"passthrough dataset {name} missing feature_names")
            for feat in names: 
                passthrough_features.append(f"{name}__{feat}")

    pass_blocks = []
    pass_names  = []
    for feat in passthrough_features: 
        if "__" in feat: 
            ds_name, raw_name = feat.split("__", 1)
        else: 
            ds_name, raw_name = None, feat 

        found = False 
        for name, raw in raw_data.items(): 
            if ds_name is not None and name != ds_name: 
                continue 
            names = list(raw["feature_names"])
            if raw_name in names: 
                col     = names.index(raw_name)
                idx_raw = align_on_fips(common, raw["sample_ids"])
                pass_blocks.append(raw["features"][idx_raw, col].reshape(-1, 1))
                pass_names.append(f"{name}__{raw_name}")
                found = True 
                break 

        if not found: 
            raise ValueError(f"passthrough feature not found: {feat}")

    X             = np.hstack(prob_blocks + pass_blocks)
    feature_names = np.array(prob_names + pass_names, dtype="U64")

    return {
        "features": X, 
        "labels": y,
        "coords": np.zeros((X.shape[0], 2), dtype=np.float64),
        "feature_names": feature_names, 
        "sample_ids": np.array(common, dtype="U5")
    }

# --------------------------------------------------------- 
# Datasets that work specifically with Metric Learners 
# --------------------------------------------------------- 

class ConcatSpec(TypedDict): 
    name: str 
    path: str 
    loader: Callable[[str], DatasetDict]

def _oof_feature_names(model_names, class_labels, n_classes): 
    if class_labels.size: 
        labels = [int(c) for c in class_labels]
    else: 
        labels = list(range(n_classes))
    names = []
    for m in model_names: 
        for c in labels: 
            names.append(f"{m}__p{c}")
    return np.array(names, dtype="U64")

def make_oof_dataset_loader(
    *,
    model_name: str | None = None 
) -> DatasetLoader: 
    return bind(_oof_dataset_loader, model_name=model_name)

def _oof_dataset_loader(path: str, *, model_name: str | None) -> DatasetDict: 
    oof   = load_oof_predictions(path)
    probs = np.asarray(oof["probs"], dtype=np.float64)
    model_names  = oof["model_names"].tolist()
    class_labels = np.asarray(oof["class_labels"]).reshape(-1)
    labels       = np.asarray(oof["labels"]).reshape(-1)

    if probs.ndim == 3: 
        if model_name is not None: 
            if model_name not in model_names: 
                raise ValueError(f"oof model '{model_name}' not in {model_names}")
            m_idx = model_names.index(model_name)
            probs = probs[:, m_idx, :]
            feature_names = _oof_feature_names([model_name], class_labels, probs.shape[1])
        else: 
            feature_names = _oof_feature_names(model_names, class_labels, probs.shape[2])
            probs = probs.reshape(probs.shape[0], -1)
    else: 
        feature_names = _oof_feature_names(model_names, class_labels, probs.shape[1])
    
    return {
        "features": probs, 
        "labels": labels, 
        "coords": np.zeros((probs.shape[0], 2), dtype=np.float64),
        "feature_names": feature_names, 
        "sample_ids": np.asarray(oof["fips_codes"]).astype("U5")
    }

def load_concat_datasets(
    specs: Sequence[ConcatSpec], 
    *,
    labels_path: str, 
    labels_loader: Callable[[str], DatasetDict], 
) -> DatasetDict: 
    if not specs: 
        raise ValueError("specs must not be empty")


    label_data = labels_loader(labels_path) 
    label_ids  = list(label_data["sample_ids"])
    y = np.asarray(label_data["labels"])
    if y.ndim == 2 and y.shape[1] == 1: 
        y = y.reshape(-1)

    data    = {s["name"]: s["loader"](s["path"]) for s in specs}
    id_sets = [set(d["sample_ids"]) for d in data.values()]
    common  = [f for f in label_ids if all(f in s for s in id_sets)]
    if not common: 
        raise ValueError("no common sample_ids across specs")

    idx_label  = align_on_fips(common, label_data["sample_ids"])
    y = y[idx_label]
    
    coords = np.zeros((len(common), 2), dtype=np.float64)

    X_blocks    = []
    name_blocks = []
    for s in specs: 
        d   = data[s["name"]]
        idx = align_on_fips(common, d["sample_ids"])
        X   = np.asarray(d["features"])[idx]

        names = d.get("feature_names")
        if names is None or len(names) != X.shape[1]: 
            names = np.array([f"f{i}" for i in range(X.shape[1])], dtype="U64")
        else: 
            names = np.asarray(names)

        if s.get("prefix", True): 
            names = np.array([f"{s['name']}__{n}" for n in names], dtype="U64")

        X_blocks.append(X)
        name_blocks.append(names)

    X_all = np.hstack(X_blocks) if len(X_blocks) > 1 else X_blocks[0]
    feature_names = np.concatenate(name_blocks) if len(name_blocks) > 1 else name_blocks[0]

    return {
        "features": X_all, 
        "labels": y, 
        "coords": coords, 
        "feature_names": feature_names, 
        "sample_ids": np.array(common, dtype="U5")
    }

# ---------------------------------------------------------
# CNN Specific Datasets (include a collate function)
# ---------------------------------------------------------

class SpatialDatasetDict(TypedDict): 
    dataset: Any 
    labels: NDArray 
    sample_ids: NDArray[np.str_]
    collate_fn: Callable | None 
    in_channels: int 

SpatialDatasetLoader = Callable[[str], SpatialDatasetDict]

def dynamic_tile_collate(batch): 

    tiles_list, stats_list, labels, num_tiles_list = [], [], [], []

    for item in batch: 
        tiles, stats, label, = item[:3]

        t   = torch.from_numpy(np.array(tiles, dtype=np.float32).copy())
        s   = torch.from_numpy(np.array(stats, dtype=np.float32).copy())

        if s.ndim != 3: 
            raise ValueError(f"expected stats shape (T, L, D), got {tuple(s.shape)}")
        if s.shape[0] != t.shape[0]: 
            raise ValueError("tile/stats count mismatch in collate")

        tiles_list.append(t)
        stats_list.append(s)
        labels.append(label)
        num_tiles_list.append(t.shape[0])

    flat_inputs = torch.cat(tiles_list, dim=0).contiguous(memory_format=torch.channels_last)
    flat_stats  = torch.cat(stats_list, dim=0).contiguous()

    labels      = torch.tensor(labels, dtype=torch.float32)
    sections    = torch.tensor(num_tiles_list, dtype=torch.long)
    batch_idx   = torch.repeat_interleave(torch.arange(len(batch)), sections)

    return flat_inputs, labels, batch_idx, flat_stats 

def load_spatial_mmap_manifest(
    root_dir: str, 
    *,
    tile_shape: tuple[int, int, int], 
    patch_size: int = 32, 
    random_state: int = 0, 
    should_validate_index: bool = False,
    **_ 
) -> SpatialDatasetDict:
    root       = Path(root_dir)
    index_csv  = root / "index.csv"
    bin_path   = root / "dataset.bin"
    stats_path = root / "stats.bin" 

    if not index_csv.exists() or not bin_path.exists(): 
        raise FileNotFoundError(f"missing dataset.bin/index.csv in {root}")

    ds = TileLoader(
        index_csv=str(index_csv),
        bin_path=str(bin_path),
        stats_bin_path=str(stats_path),
        tile_shape=tile_shape,
        patch_size=patch_size,
        random_state=random_state,
        return_fips=True,
        return_num_tiles=False,
        should_validate_index=should_validate_index
    )

    labels = np.asarray(ds.labels, dtype=np.float32)
    fips   = np.asarray(ds.fips, dtype="U5")

    return {
        "dataset": ds,
        "labels": labels, 
        "sample_ids": fips, 
        "collate_fn": dynamic_tile_collate, 
        "in_channels": int(tile_shape[0])
    }

# Convenience wrapper over loader 
def mmap_loader(
    path,
    *,
    tile_shape,
    patch_size=32, 
    random_state=0,
    shuffle_tiles=True,
    should_validate_index=False
):
    return load_spatial_mmap_manifest(
        path,
        tile_shape=tile_shape,
        patch_size=patch_size, 
        random_state=random_state,
        shuffle_tiles=shuffle_tiles,
        should_validate_index=should_validate_index
    )

# ---------------------------------------------------------
# Fusion Dataset Definition  
# ---------------------------------------------------------

class FusionDataset(Dataset): 

    def __init__(
        self,
        experts: dict[str, torch.Tensor], 
        wide_x: torch.Tensor, 
        y_rank: torch.Tensor, 
        sample_weight: Optional[torch.Tensor] = None 
    ): 
        self.experts       = experts 
        self.wide_x        = wide_x 
        self.y_rank        = y_rank 
        self.sample_weight = sample_weight

    def __len__(self) -> int: 
        return int(self.y_rank.shape[0])

    def __getitem__(self, idx: int): 
        xb = {k: v[idx] for k, v in self.experts.items()}
        wb = self.wide_x[idx]
        yb = self.y_rank[idx]
        if self.sample_weight is None: 
            sw = torch.tensor(1.0, dtype=self.y_rank.dtype)
        else: 
            sw = self.sample_weight[idx]
        return xb, wb, yb, sw

class WideDeepInputs(TypedDict): 
    experts: dict[str, NDArray[np.float32]]
    wide: NDArray[np.float32]
    y_rank: NDArray[np.float32]
    coords: NDArray[np.float32]
    sample_ids: NDArray[np.str_]
    expert_dims: dict[str, int]
    wide_in_dim: int 

def canon_fips_vec(x) -> NDArray[np.str_]: 
    arr = np.asarray(x).reshape(-1)
    out = []
    for v in arr:
        s = str(v).strip() 
        if s.isdigit():
            s = s.zfill(5)
        out.append(s)
    return np.asarray(out, dtype="U5")

def load_rank_source(
    path: str, 
    *, 
    rank_key: str
) -> tuple[NDArray[np.str_], NDArray[np.float32]]:

    mat = loadmat(path)
    if "fips_codes" not in mat: 
        raise ValueError(f"{path} missing fips_codes.")
    if rank_key not in mat and "labels" not in mat: 
        raise ValueError(f"{path} missing {rank_key} and labels.")

    fips = canon_fips_vec(_mat_str_vector(mat["fips_codes"]))
    key  = rank_key if rank_key in mat else "labels" 
    y    = np.asarray(mat[key], dtype=np.float32).reshape(-1)

    if fips.shape[0] != y.shape[0]: 
        raise ValueError(f"{path} rank/fips mismatch.")
    return fips, y

def load_wide_deep_inputs(
    *,
    expert_paths: dict[str, str],
    wide_path: str,
    rank_path: str,
    coords_path: Optional[str] = None, 
    rank_key: str = "soft_rank"
) -> WideDeepInputs: ...
