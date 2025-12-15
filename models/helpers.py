#!/usr/bin/env python3 
# 
# helpers.py  Andrew Belles  Dec 11th, 2025 
# 
# Module of helper functions for models to utilize, mostly for path concatenation
# 
# 

import os, re

import numpy as np 
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

NCLIMDIV_RE = re.compile(r"^climdiv-([a-z0-9]+)cy-v[0-9.]+-[0-9]{8}.*$")

def project_path(*args):
    root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *args)

def split_indices(n_samples: int, test_size: float, seed: int | None = None): 
    if n_samples <= 0: 
        raise ValueError("n_samples must be > 0")
    if not (0.0 < test_size < 1.0): 
        raise ValueError("test_size must be in (0.0, 1.0)")
    if seed is None: 
        seed = int(np.random.randint(0, 2**32 - 1))

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_samples) 
    
    n_test    = int(round(n_samples * test_size))
    test_idx  = perm[:n_test] 
    train_idx = perm[n_test:]

    return train_idx, test_idx 

def fit_scaler(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64) 

    if X_train.shape[0] != y_train.shape[0]: 
        raise ValueError(f"X_train rows ({X_train.shape[0]}) != y_train ({y_train.shape[0]})")

    X_scaler = StandardScaler() 
    y_scaler = StandardScaler() 

    X_scaler.fit(X_train) 
    y_scaler.fit(y_train.reshape(-1, 1))

    return X_scaler, y_scaler 

def transform_with_scalers(X: ArrayLike, y: ArrayLike, X_scaler: StandardScaler, 
                           y_scaler: StandardScaler) -> tuple[NDArray[np.float64], NDArray[np.float64]]: 
    X = np.asarray(X, dtype=np.float64) 
    y = np.asarray(y, dtype=np.float64) 

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"X rows ({X.shape[0]}) != y ({y.shape[0]})") 

    X_scaled = np.asarray(X_scaler.transform(X), dtype=np.float64) 

    y_scaled = y_scaler.transform(y.reshape(-1, 1))
    y_scaled = np.asarray(y_scaled, dtype=np.float64).ravel() 

    return X_scaled, y_scaled 


def kfold_indices(n_samples: int, n_folds: int = 5, seed: int | None = None): 

    rng       = np.random.default_rng(seed)
    indices   = rng.permutation(n_samples)
    fold_size = n_samples // n_folds 

    folds = []
    for i in range(n_folds):
        start = i * fold_size 
        end   = start + fold_size if i < n_folds - 1 else n_samples 
        test_idx  = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))

    return folds 
