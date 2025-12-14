#!/usr/bin/env python3 
# 
# helpers.py  Andrew Belles  Dec 11th, 2025 
# 
# Module of helper functions for models to utilize, mostly for path concatenation
# 
# 

import os, re

import numpy as np 
from sklearn.preprocessing import StandardScaler

NCLIMDIV_RE = re.compile(r"^climdiv-([a-z0-9]+)cy-v[0-9.]+-[0-9]{8}.*$")

def project_path(*args):
    root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *args)


def split_and_scale(features, labels, test_size: float = 0.25):
    
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels ({y.shape[0]})")

    if not (0.0 < test_size < 1.0): 
        raise ValueError("test size must be in (0.0, 1.0)")

    seed = int(np.random.randint(0, 2**32 - 1))
    rng  = np.random.default_rng(seed)
    n    = X.shape[0]
    perm = rng.permutation(n) 

    n_test    = int(round(n * test_size)) 
    test_idx  = perm[:n_test] 
    train_idx = perm[n_test:]
    
    X_train, X_test = X[train_idx], X[test_idx] 
    y_train, y_test = y[train_idx], y[test_idx]

    X_scaler = StandardScaler() 
    X_train  = X_scaler.fit_transform(X_train)
    X_test   = X_scaler.transform(X_test)  

    y_scaler = StandardScaler() 
    y_train  = y_scaler.fit_transform(y_train) 
    y_test   = y_scaler.transform(y_test)

    return (X_train, X_test), (y_train, y_test), (train_idx, test_idx), (X_scaler, y_scaler)
