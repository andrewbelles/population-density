#!/usr/bin/env python3 
# 
# data.py  Andrew Belles  Jan 7th, 2025 
# 
# Dataset Specification and loader creation. 
# 
# 

import numpy as np

from typing import Sequence

from utils.helpers import make_cfg_gap_factory, project_path

from testbench.utils.paths import (
    LABELS_PATH
)

from testbench.utils.transforms import apply_transforms

from preprocessing.loaders import (
    load_stacking,
    load_viirs_nchs,
    load_coords_from_mobility,
    load_compact_dataset,
    load_concat_datasets,
    make_oof_dataset_loader,
    load_oof_predictions, 
    ConcatSpec
)

from preprocessing.disagreement import DisagreementSpec, load_pass_through_stacking


DATASETS = ("VIIRS", "TIGER", "NLCD")

PASSTHROUGH_FEATURES = [
    "cross__cross_viirs_log_mean", 
    "cross__cross_tiger_integ", 
    "cross__cross_radiance_entropy",
    "cross__cross_dev_intensity_gradient",
    "cross__cross_vanui_proxy",
    "cross__cross_effective_mesh_proxy",
    "cross__cross_road_effect_intensity"
]

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

def build_specs(prob_files) -> list[DisagreementSpec]: 
    specs = []
    for name, prob_path in zip(DATASETS, prob_files): 
        base = BASE[name]
        specs.append({
            "name": name.lower(), 
            "raw_path": base["path"],
            "raw_loader": base["loader"],
            "oof_path": prob_path,
            "oof_loader": load_oof_predictions
        })
    return specs 

def stacking_loader(prob_files): 
    def _loader(_): 
        return load_stacking(prob_files)
    return _loader 

def passthrough_loader(prob_files):
    passthrough_base  = BASE["PASSTHROUGH"]
    passthrough_specs = [
        {
            "name": "cross", 
            "raw_path": passthrough_base["path"],
            "raw_loader": passthrough_base["loader"]
        }
    ] 

    def _loader(_): 
        data = load_pass_through_stacking(
            build_specs(prob_files),
            label_path=LABELS_PATH,
            label_loader=BASE["VIIRS"]["loader"],
            passthrough_features=PASSTHROUGH_FEATURES,
            passthrough_specs=passthrough_specs 
        )
        transforms = make_cfg_gap_factory(data.get("feature_names"))() 
        if transforms: 
            data["features"] = apply_transforms(data["features"], transforms)
        return data 
    return _loader 
