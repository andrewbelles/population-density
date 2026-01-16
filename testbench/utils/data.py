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
    LABELS_PATH,
    DATASETS
)

from pathlib import Path

from testbench.utils.transforms import apply_transforms

from testbench.utils.etc        import flatten_imaging

from scipy.io import loadmat 

from preprocessing.loaders import (
    DatasetDict,
    MetaSpec,
    PassthroughSpec,
    load_stacking,
    load_tensor_data,
    load_viirs_nchs,
    load_coords_from_mobility,
    load_compact_dataset,
    load_concat_datasets,
    make_oof_dataset_loader,
    load_oof_predictions, 
    ConcatSpec
)

from preprocessing.loaders import load_passthrough

LEGACY   = ("TIGER",)

PREFIXES = (
    "viirs__",
    "nlcd__",
    "saipe__",
    "tiger__",
    "cross__",
    "coords__",
    "oof__",
    "passthrough__"
)

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

def passthrough_loader(prob_files, passthrough_features = None):
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

def read_feature_list(path: str | None): 
    if not path: 
        return None 
    lines = Path(path).read_text().splitlines() 
    return [l.strip() for l in lines if l.strip()]

def norm_name(name: str) -> str: 
    s = str(name).strip().lower() 
    for prefix in PREFIXES: 
        if s.startswith(prefix): 
            return s[len(prefix):]
    return s

def filter_features(data: dict, keep_list: list[str] | None) -> dict: 
    if not keep_list: 
        return data 
    names = np.asarray(data.get("feature_names"))
    if names is None or names.size == 0: 
        raise ValueError("feature_names missing for filtering")

    keep_norm = {norm_name(n) for n in keep_list}
    cols      = [i for i, n in enumerate(names) if norm_name(n) in keep_norm]
    if not cols: 
        raise ValueError("no features kept after boruta filtering")

    out = dict(data)
    out["features"] = np.asarray(out["features"])[:, cols]
    out["feature_names"] = names[cols]
    return out 

def make_filtered_loader(base_loader, keep_list: list[str] | None): 
    if not keep_list: 
        return base_loader 
    def _loader(path): 
        data = base_loader(path)
        return filter_features(data, keep_list)
    return _loader 

def make_tensor_loader(mode, canvas_h, canvas_w, gaf_size): 
    def _loader(filepath): 
        spatial, mask, gaf, labels, fips = load_tensor_data(filepath)
        X = flatten_imaging(spatial, mask, gaf, mode)
        
        coords = np.zeros((X.shape[0], 2), dtype=np.float64)
        return {
            "features": X, 
            "labels": labels, 
            "coords": coords, 
            "feature_names": np.array([], dtype="U"),
            "sample_ids": fips 
        }
    return _loader 

def make_tensor_adapter(mode, canvas_h, canvas_w, gaf_size): 
    n_pix = canvas_h * canvas_w 
    g_pix = gaf_size * gaf_size 

    def _adapter(X): 
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]

        if mode == "spatial": 
            spatial = X[:, :n_pix].reshape(n, 1, canvas_h, canvas_w)
            mask    = X[:, n_pix:2*n_pix].reshape(n, 1, canvas_h, canvas_w)
            return np.concatenate([spatial, mask], axis=1)

        if mode == "gaf": 
            return X[:, :g_pix].reshape(n, 1, gaf_size, gaf_size)

        if mode == "dual": 
            spatial = X[:, :n_pix].reshape(n, 1, canvas_h, canvas_w)
            mask    = X[:, n_pix:2*n_pix].reshape(n, 1, canvas_h, canvas_w)
            x_main  = np.concatenate([spatial, mask], axis=1)
            x_aux   = X[:, 2*n_pix:2*n_pix + g_pix].reshape(n, 1, gaf_size, gaf_size)
            return x_main, x_aux 

        raise ValueError("mode must be spatial/gaf/dual")

    return _adapter 
