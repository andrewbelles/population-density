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
from typing import Any, Callable, TypedDict, List 
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod
from scipy.io import loadmat 

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


def unwrap_scalar(v): 
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v 


def make_xgb_model(
    cls: type, 
    *, 
    gpu: bool, 
    base_params: dict[str, Any], 
    default_tree_method: str = "hist"
):
    params = dict(base_params) 
    params.setdefault("tree_method", default_tree_method)

    params.setdefault("device", "cuda" if gpu else "cpu")

    try: 
        return cls(**params) 
    except TypeError as e: 
        raise e


class ModelInterface(ABC):
    '''
    Abstract Method for model to be trained and predict some y_hat for the provided 
    features and labels.

    Returns y_hat scaled via scikit-learn StandardScaler()  
    '''

    @abstractmethod 
    def fit_and_predict(self, 
                        features: tuple[NDArray[np.float64], NDArray[np.float64]], 
                        labels: tuple[NDArray[np.float64], NDArray[np.float64]], 
                        coords: tuple[NDArray[np.float64], NDArray[np.float64]],
                        **kwargs) -> NDArray[np.float64]:
        raise NotImplementedError


class DatasetDict(TypedDict): 
    features: NDArray[np.float64]
    labels: NDArray[np.float64]
    coords: NDArray[np.float64]

DataseLoader = Callable[[str], DatasetDict]

def load_climate_population(filepath: str, decade: int, groups: List[str]) -> DatasetDict: 
    
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

    return {"features": features, "labels": y, "coords": coords}

CLIMATE_GROUPS = {"degree_days", "palmer_indices"}

def load_climate_geospatial(filepath: str, target: str, groups: List[str]) -> DatasetDict: 
    '''
    Loader Helper for Climate against (lat, lon). 

    Caller Provides: 
        filepath to dataset, 
        target (lat or lon)
        groups to include in dataset 

    We return: 
        A DatasetDict containing the requested feature groups and target label 
    '''

    group_set = set(groups)
    label_set = {"lat", "lon"}

    if target not in label_set: 
        raise ValueError(f"target: {target} must be in {sorted(label_set)} to be requested")

    data = loadmat(filepath)
    c = np.asarray(data["labels"], dtype=np.float64)   # expects shape (n, 2)

    idx = 0 if target == "lat" else 1 
    y = c[:, idx].reshape(-1)

    features = []
    for name in sorted(CLIMATE_GROUPS): 
        if name not in group_set: 
            continue 
        features.append(np.asarray(data[f"features_{name}"], dtype=np.float64))

    if not features: 
        raise ValueError(f"{groups} does not contain any valid group labels for data")

    features = np.hstack(features) if len(features) > 1 else features[0] 
    return {"features": features, "labels": y, "coords": np.zeros_like(y)}
