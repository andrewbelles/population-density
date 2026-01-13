#!/usr/bin/env python3 
# 
# data.py  Andrew Belles  Jan 7th, 2025 
# 
# Dataset Specification and loader creation. 
# 
# 

import numpy as np

from typing import Sequence

from utils.helpers import (
    make_cfg_gap_factory, 
    project_path,
    _mat_str_vector
)

from testbench.utils.paths import (
    LABELS_PATH
)

from testbench.utils.transforms import apply_transforms

from scipy.io import loadmat 

from preprocessing.loaders import (
    DatasetDict,
    MetaSpec,
    PassthroughSpec,
    load_stacking,
    load_viirs_nchs,
    load_coords_from_mobility,
    load_compact_dataset,
    load_concat_datasets,
    make_oof_dataset_loader,
    load_oof_predictions, 
    ConcatSpec
)

from preprocessing.loaders import load_passthrough

DATASETS = ("VIIRS", "NLCD", "SAIPE")
LEGACY   = ("TIGER",)

BASE: dict[str, ConcatSpec] = {
    "VIIRS": {
        "name": "VIIRS",
        "path": project_path("data", "datasets", "viirs_nchs_2023.mat"),
        "loader": load_viirs_nchs
    },
    "TIGER": {
        "name": "tiger",
        "path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
        "loader": load_compact_dataset
    },
    "NLCD": {
        "name": "NLCD",
        "path": project_path("data", "datasets", "nlcd_nchs_2023.mat"),
        "loader": load_compact_dataset
    },
    "SAIPE": {
        "name": "SAIPE", 
        "path": project_path("data", "datasets", "saipe_nchs_2023.mat"),
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

def select_specs_psv(dataset_key: str) -> Sequence[ConcatSpec]: 
    names = dataset_key.split("+")
    return [BASE[n] for n in names]

def select_specs_csv(sources_csv: str) -> Sequence[ConcatSpec]:
    wanted = {s.strip().lower() for s in sources_csv.split(",") if s.strip()}
    return [s for s in BASE.values() if s["name"].lower() in wanted]

# Specifically when we need to load passthrough instead of raw probability matrices 
def override_proba_path(specs, proba_path: str) -> Sequence[ConcatSpec]:
    out = []
    for s in specs: 
        if s["name"].upper() == "OOF": 
            s = dict(s)
            s["path"] = proba_path 
        out.append(s)
    return out 

def dataset_specs(dataset_key: str, proba_path: str): 
    specs = select_specs_psv(dataset_key)
    return override_proba_path(specs, proba_path)

def make_dataset_loader(dataset_key: str, proba_path: str): 
    specs = dataset_specs(dataset_key, proba_path)

    def _loader(_): 
        return load_concat_datasets(
            specs=specs,
            labels_path=LABELS_PATH,
            labels_loader=load_viirs_nchs
        )
    return {dataset_key: _loader}

def load_dataset_raw(dataset_key: str, proba_path: str): 
    specs = dataset_specs(dataset_key, proba_path)
    data  = load_concat_datasets(
        specs=specs,
        labels_path=LABELS_PATH,
        labels_loader=load_viirs_nchs
    )
    X    = data["features"]
    y    = np.asarray(data["labels"]).reshape(-1)
    fips = np.asarray(data["sample_ids"]).astype("U5")
    feature_names = data.get("feature_names")
    return X, y, fips, feature_names

def build_specs(prob_files) -> list[MetaSpec]: 
    specs = []
    for name, prob_path in zip(DATASETS, prob_files): 
        specs.append({
            "name": name.lower(), 
            "proba_path": prob_path,
            "proba_loader": load_oof_predictions
        })
    return specs 

def stacking_loader(prob_files): 
    def _loader(_): 
        return load_stacking(prob_files)
    return _loader 

def passthrough_loader(prob_files):
    passthrough_base  = BASE["PASSTHROUGH"]
    passthrough_specs: list[PassthroughSpec] = [
        {
            "name": "cross", 
            "raw_path": passthrough_base["path"],
            "raw_loader": passthrough_base["loader"]
        }
    ] 

    def _loader(_): 
        data = load_passthrough(
            build_specs(prob_files),
            label_path=LABELS_PATH,
            label_loader=BASE["VIIRS"]["loader"],
            passthrough_specs=passthrough_specs,
            passthrough_features=None 
        )
        transforms = make_cfg_gap_factory(data.get("feature_names"))() 
        if transforms: 
            data["features"] = apply_transforms(data["features"], transforms)
            
            names = np.asarray(data.get("feature_names")) 
            if names is None: 
                names = np.array([], dtype="U64")
            else: 
                names = np.asarray(names, dtype="U64")

            if names.size != data["features"].shape[1]: 
                extra = data["features"].shape[1] - names.size 
                if extra > 0: 
                    suffix = (["cfg_gap"] if extra == 1 else 
                    [f"cfg_gap_{i+1}" for i in range(extra)])

                    data["feature_names"] = np.concatenate(
                        [names, np.asarray(suffix, dtype="U64")]
                    )

        return data 
    return _loader 

def make_binary_loader(data, label: int): 
    X     = data["features"]
    y     = np.asarray(data["labels"]).reshape(-1)
    y_bin = (y == label).astype(np.int64)

    def _loader(_) -> DatasetDict: 
        return {
            "features": X, 
            "labels": y_bin, 
            "coords": data.get("coords"),
            "feature_names": data.get("feature_names"),
            "sample_ids": data.get("sample_ids")
        }
    return _loader 

def load_dataset(dataset_key: str): 
    spec = BASE[dataset_key] 
    data = spec["loader"](spec["path"])
    y    = np.asarray(data["labels"]).reshape(-1).astype(np.int64)
    out  = dict(data)
    out["labels"] = y 
    return out 


def feature_names_from_mat(mat, prefix: str, keep_idx: list[int] | None, n_cols: int): 
    names = None 
    if "feature_names" in mat: 
        names = _mat_str_vector(mat["feature_names"]).astype("U64")
        if names.shape[0] != n_cols: 
            names = None 
    if names is None: 
        names = np.array([f"p{i}" for i in range(n_cols)], dtype="U64")
    if keep_idx is not None: 
        names = names[keep_idx]
    return np.array([f"{prefix}::{n}" for n in names], dtype="U128")

def load_oof_features(path: str, prefix: str): 
    mat = loadmat(path)
    if "features" not in mat: 
        raise ValueError(f"{path} missing 'features'")
    if "fips_codes" not in mat: 
        raise ValueError(f"{path} missing 'fips_codes'")

    X = np.asarray(mat["features"], dtype=np.float64)
    if X.ndim == 1: 
        X = X.reshape(-1, 1)
    if X.ndim != 2: 
        raise ValueError(f"{path} expected 2d features, got {X.shape}")

    keep_idx = None 
    if X.shape[1] == 2: 
        keep_idx = [1]
        X = X[:, keep_idx]

    fips  = _mat_str_vector(mat["fips_codes"]).astype("U5")
    names = feature_names_from_mat(mat, prefix, keep_idx, X.shape[1])

    return {
        "features": X, 
        "coords": np.zeros((X.shape[0], 2), dtype=np.float64),
        "feature_names": names, 
        "sample_ids": fips 
    }

def load_raw(key: str) -> dict: 
    spec  = BASE[key]
    mat   = loadmat(spec["path"])

    X     = np.asarray(mat["features"], dtype=np.float64)
    fips  = _mat_str_vector(mat["fips_codes"]).astype("U5")
    names = _mat_str_vector(mat["feature_names"]).astype("U64")
    names = np.array([n.strip() for n in names], dtype="U64")

    return {
        "features": X,
        "coords": np.zeros((X.shape[0], 2), dtype=np.float64),
        "feature_names": names, 
        "sample_ids": fips  
    }
