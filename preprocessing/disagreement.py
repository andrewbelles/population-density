#!/usr/bin/env python3 
# 
# disagreement.py  Andrew Belles  Dec 24th, 2025 
# 
# Instantiates a disagreement based supervised dataset for 
# classifiers that are aiming to be stacked by a meta-learner 
# to understand feature importance 
# 

import numpy as np 

from typing import Callable, TypedDict

from preprocessing.loaders import (
        OOFDatasetDict, 
        DatasetDict 
)

class DisagreementSpec(TypedDict): 
    name: str 
    raw_path: str 
    raw_loader: Callable[[str], DatasetDict]
    oof_path: str 
    oof_loader: Callable[[str], OOFDatasetDict]


def _align_on_fips(fips_order, fips_vec): 
    idx_map = {f: i for i, f in enumerate(fips_vec)}
    return np.array([idx_map[f] for f in fips_order], dtype=int)

def build_disagreement_dataset(specs: list[DisagreementSpec]) -> DatasetDict: 

    raw_data = {}
    oof_data = {}

    for s in specs: 
        raw_data[s["name"]] = s["raw_loader"](s["raw_path"])
        oof_data[s["name"]] = s["oof_loader"](s["oof_path"])

    fips_sets = []
    for name in raw_data: 
        fips_sets.append(set(raw_data[name]["sample_ids"])) 
        fips_sets.append(set(oof_data[name]["fips_codes"]))

    common = sorted(set.intersection(*fips_sets))

    X_blocks      = []
    pred_blocks   = []
    feature_names = []

    for s in specs: 
        name = s["name"]
        raw  = raw_data[name]
        oof  = oof_data[name]

        idx_raw = _align_on_fips(common, raw["sample_ids"])
        idx_oof = _align_on_fips(common, oof["fips_codes"])

        X = raw["features"][idx_raw]
        X_blocks.append(X)

        if "feature_names" in raw: 
            names = [f"{name}__{n}" for n in raw["feature_names"]]
        else:
            names = [f"{name}__f{i}" for i in range(X.shape[1])]
        feature_names.extend(names)

        preds = oof["preds"][idx_oof]
        pred_blocks.append(preds)

    preds_all = np.stack(pred_blocks, axis=1)
    if preds_all.ndim == 3 and preds_all.shape[2] == 1: 
        preds_all = preds_all.squeeze(-1)

    agreement  = np.all(preds_all == preds_all[:, [0]], axis=1)
    y_conflict = (~agreement).astype(np.int64)

    X_raw = np.hstack(X_blocks)

    return {
        "features": X_raw, 
        "labels": y_conflict,
        "coords": np.zeros((X_raw.shape[0], 2), dtype=np.float64), 
        "feature_names": np.array(feature_names, dtype="U"), 
        "sample_ids": np.array(common, dtype="U5")
    }



def load_pass_through_stacking(
    specs: list[DisagreementSpec],
    label_path: str, 
    label_loader: Callable[[str], DatasetDict], 
    passthrough_features: list[str]
) -> DatasetDict: 

    label_data = label_loader(label_path)
    label_fips = list(label_data["sample_ids"])
    y = np.asarray(label_data["labels"]).reshape(-1)

    raw_data = {}
    oof_data = {}

    for s in specs: 
        raw_data[s["name"]] = s["raw_loader"](s["raw_path"])
        oof_data[s["name"]] = s["oof_loader"](s["oof_path"])

    fips_sets = [set(label_fips)]
    for name in raw_data: 
        fips_sets.append(set(raw_data[name]["sample_ids"])) 
        fips_sets.append(set(oof_data[name]["fips_codes"]))
    common = [f for f in label_fips if all(f in s for s in fips_sets)]
    
    idx_label = _align_on_fips(common, label_data["sample_ids"])
    y = y[idx_label]

    prob_blocks = []
    prob_names  = []
    for s in specs: 

        name = s["name"]
        oof  = oof_data[name]
        idx_oof = _align_on_fips(common, oof["fips_codes"])

        model_names = oof["model_names"].tolist() 
        if "model_name" in s: 
            m_idx  = model_names.index(s["model_name"])
            m_name = s["model_name"] 
        else: 
            if len(model_names) != 1: 
                raise ValueError(f"{name} OOF has multiple models, set model_name")
            m_idx  = 0 
            m_name = model_names[0] 

        probs = oof["probs"][idx_oof, m_idx, :]
        prob_blocks.append(probs)

        class_labels = oof["class_labels"].reshape(-1)
        for c in class_labels: 
            prob_names.append(f"{name}__{m_name}__p{int(c)}")

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
            if "feature_names" not in raw: 
                continue 
            names = list(raw["feature_names"])
            if raw_name in names: 
                col = names.index(raw_name)
                idx_raw = _align_on_fips(common, raw["sample_ids"])
                pass_blocks.append(raw["features"][idx_raw, col].reshape(-1, 1))
                pass_names.append(f"{name}__{raw_name}")
                found = True 
                break 
        if not found: 
            raise ValueError(f"passthrough feature not found: {feat}")

    X = np.hstack(prob_blocks + pass_blocks)
    feature_names = np.array(prob_names + pass_names, dtype="U")

    return {
        "features": X, 
        "labels": y, 
        "coords": np.zeros((X.shape[0], 2), dtype=np.float64), 
        "feature_names": feature_names, 
        "sample_ids": np.array(common, dtype="U5")
    }
