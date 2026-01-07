#!/usr/bin/env python3 
# 
# oof.py  Andrew Belles  Jan 7th, 2025 
# 
# Helper functions for handling produced out of fold predictions used downstream 
# in C+S and second model stage 
# 

import numpy as np 

from preprocessing.loaders import (
    load_oof_predictions
)

from numpy.typing import NDArray

from testbench.utils.paths import (
    PROBA_PATH,
    check_paths_exist
)

from utils.helpers import (
    align_on_fips
)

def load_probs_labels_fips(proba_path=PROBA_PATH): 
    check_paths_exist([proba_path], "stacking OOF file")
    oof   = load_oof_predictions(proba_path) 
    probs = np.asarray(oof["probs"], dtype=np.float64)
    if probs.ndim != 3: 
        raise ValueError(f"expected probs (n, m, c), got {probs.shape}")
    if probs.shape[1] == 1: 
        P = probs[:, 0, :]
    else: 
        P = probs.mean(axis=1)
    y    = np.asarray(oof["labels"]).reshape(-1)
    fips = np.asarray(oof["fips_codes"]).astype("U5")
    class_labels = np.asarray(oof["class_labels"]).reshape(-1) 
    return P, y, fips, class_labels  

def load_probs_for_fips(fips: NDArray[np.str_], proba_path=PROBA_PATH) -> NDArray: 
    P, _, oof_fips, _ = load_probs_labels_fips(proba_path)
    idx = align_on_fips(fips, oof_fips)
    return P[idx]

def extract_oof_preds(oof, model_name=None): 
    preds = np.asarray(oof["preds"])
    model_names = oof["model_names"]
    if preds.ndim == 2: 
        if model_name and model_name in model_names: 
            m_idx = int(np.where(model_names == model_name)[0][0])
        else: 
            m_idx = 0
        return preds[:, m_idx]
    return preds.reshape(-1)
