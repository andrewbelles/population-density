#!/usr/bin/env python3 
# 
# oof.py  Andrew Belles  Jan 7th, 2025 
# 
# Helper functions for handling produced out of fold predictions used downstream 
# in C+S and second model stage 
# 

import numpy as np 

import multiprocessing as mp 

import gc, torch

from functools               import partial

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

    if probs.ndim == 2: 
        P = probs 
    elif probs.ndim == 3: 
        if probs.shape[1] == 1: 
            P = probs[:, 0, :]
        else: 
            P = probs.mean(axis=1)
    else: 
        raise ValueError(f"got {probs.shape}, invalid.")

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

def extract_pooled(model, subset): 
    return model.extract_pooled(subset)

def extract_with_logits(model, subset): 
    return model.extract_with_logits(subset)

def subset_default(data, idx): 
    return Subset(data, idx)

def subset_by_groups(data, idx, groups): 
    return Subset(data, np.unique(groups[idx]))

def fit_with_labels(model, data, idx, labels): 
    return model.fit(data, labels[idx])

def fit_without_labels(model, data, idx): 
    return model.fit(data, y=None)

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

def _holdout_worker(
    ds, 
    labels,
    jobs,
    results,
    model_factory,
    extract_fn,
    postprocess,
    random_state,
    subset_fn,
    fit_fn,
    device_id
):
    if device_id is not None and torch.cuda.is_available(): 
        torch.cuda.set_device(device_id)

    while True: 
        item = jobs.get() 
        if item is None: 
            break 

        fold_idx, train_idx, val_idx = item 
        train_data = subset_fn(ds, train_idx)
        val_data   = subset_fn(ds, val_idx)

        model = model_factory()
        if callable(model) and not hasattr(model, "fit"): 
            model = model() 

        if device_id is not None: 
            if hasattr(model, "device"): 
                model.device = torch.device(f"cuda:{device_id}")
            if hasattr(model, "compute_strategy"): 
                    model.compute_strategy.device = "cuda"
                    model.compute_strategy.gpu_id = device_id

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

        results.put((val_idx, val_emb))

        del model, train_emb, val_emb 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 

def holdout_embeddings(
    ds,
    labels,
    splits,
    model_factory, 
    extract_fn,
    postprocess=None,
    random_state: int = 0, 
    subset_fn=None,
    fit_fn=None,
    devices: list[int] | None = None 
):
    n   = len(labels)
    out = None 

    if subset_fn is None: 
        subset_fn = subset_default  
    if fit_fn is None: 
        fit_fn = partial(fit_with_labels, labels=labels)

    splits_list = list(splits)

    if devices is not None and len(devices) > 1 and torch.cuda.is_available():
        context = mp.get_context("spawn")
        jobs    = context.Queue()
        results = context.Queue()

        for fold_idx, (train_idx, val_idx) in enumerate(splits_list): 
            jobs.put((fold_idx, train_idx, val_idx))

        for _ in devices: 
            jobs.put(None)

        procs = []
        for dev in devices: 
            p = context.Process(
                target=_holdout_worker,
                args=(
                    ds, labels, jobs,
                    results,
                    model_factory,
                    extract_fn,
                    postprocess,
                    random_state,
                    subset_fn,
                    fit_fn, 
                    dev
                )
            )
            p.start() 
            procs.append(p)

        out = None 
        received = 0 
        while received < len(splits_list): 
            val_idx, val_emb = results.get() 
            if out is None: 
                out = np.zeros((n, val_emb.shape[1]), dtype=val_emb.dtype)
            out[val_idx] = val_emb 
            received += 1 

        for p in procs: 
            p.join()

        if out is None: 
            raise ValueError("no embeddings returned")
        return out 

    out = None 
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits_list): 
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
