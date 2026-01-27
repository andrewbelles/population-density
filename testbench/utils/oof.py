#!/usr/bin/env python3 
# 
# oof.py  Andrew Belles  Jan 7th, 2025 
# 
# Helper functions for handling produced out of fold predictions used downstream 
# in C+S and second model stage 
# 

import numpy as np 

import gc, torch

from torch.utils.data        import Subset 

from sklearn.model_selection import StratifiedGroupKFold


from preprocessing.loaders   import (
    load_oof_predictions
)

from numpy.typing            import NDArray

from testbench.utils.paths   import (
    PROBA_PATH,
    check_paths_exist
)

from utils.helpers           import (
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

def load_probs_for_fips(fips: NDArray[np.str_], proba_path=PROBA_PATH): 
    P, _, oof_fips, class_labels = load_probs_labels_fips(proba_path)
    idx = align_on_fips(fips, oof_fips)
    return P[idx], class_labels

def stacking_metadata(proba_path: str):
    P, y, fips, class_labels = load_probs_labels_fips(proba_path)
    return {
        "probs": P, 
        "labels": y,
        "fips": fips,
        "class_labels": class_labels 
    }

# ---------------------------------------------------------
# Holdout Dataset Generation
# ---------------------------------------------------------

class PackedGroupSplitter: 

    def __init__(
        self,
        y_full,
        groups,
        n_splits,
        random_state=0
    ):
        self.y_full       = np.asarray(y_full)
        self.groups       = np.asarray(groups)
        self.n_splits     = int(n_splits)
        self.random_state = int(random_state)

    def split(self, X=None, y=None): 
        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        dummy = np.zeros_like(self.y_full, dtype=np.int64)
        for tr, va in splitter.split(dummy, self.y_full, groups=self.groups): 
            tr_g = np.unique(self.groups[tr])
            va_g = np.unique(self.groups[va])
            yield np.sort(tr_g), np.sort(va_g)


def holdout_embeddings(
    ds,
    labels,
    splits,
    model_factory, 
    extract_fn,
    postprocess=None,
    random_state: int = 0, 
    subset_fn=None,
    fit_fn=None 
):
    n   = len(labels)
    out = None 

    if subset_fn is None: 
        subset_fn = lambda data, idx: Subset(data, idx)
    if fit_fn is None: 
        fit_fn = lambda model, data, idx: model.fit(data, labels[idx])

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_data = subset_fn(ds, train_idx)
        val_data   = subset_fn(ds, val_idx)

        model = model_factory()
        if callable(model) and not hasattr(model, "fit"): 
            model = model() 
        fit_fn(model, train_data, train_idx)

        train_emb = extract_fn(model, train_data) if postprocess else None 
        val_emb   = extract_fn(model, val_data)

        if postprocess:
            val_emb = postprocess(
                train_emb, labels[train_idx],
                val_emb, labels[val_idx],
                random_state=random_state, 
                fold=fold_idx 
            )

        if out is None: 
            out = np.zeros((n, val_emb.shape[1]), dtype=val_emb.dtype)
        out[val_idx] = val_emb 

        del model, train_emb, val_emb 
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        gc.collect() 

    if out is None: 
        raise ValueError("no embeddings returned")
    return out 
