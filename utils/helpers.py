#!/usr/bin/env python3 
# 
# helpers.py  Andrew Belles  Dec 11th, 2025 
# 
# Module of helper functions for models to utilize
# 
# 

import os, re, yaml

import pandas as pd 

from pathlib import Path

import numpy as np 
from numpy.typing import ArrayLike, NDArray
from typing import Any, Callable, Sequence, Dict  
from sklearn.preprocessing import StandardScaler

from functools import partial 

from abc import ABC, abstractmethod

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


NCLIMDIV_RE = re.compile(r"^climdiv-([a-z0-9]+)cy-v[0-9.]+-[0-9]{8}.*$")
MONTHS      = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# ---------------------------------------------------------
# Generic Helper Functions 
# ---------------------------------------------------------

def to_num(s: pd.Series) -> pd.Series: 
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

def bind(fn, **kwargs):
    return partial(fn, **kwargs)

def project_path(*args):
    root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *args)

def align_on_fips(fips_order, fips_vec): 
    idx_map = {f: i for i, f in enumerate(fips_vec)}
    return np.array([idx_map[f] for f in fips_order], dtype=int) 

def unwrap_scalar(v): 
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v 

def _mat_str_vector(x) -> np.ndarray: 

    arr = np.asarray(x) 
    arr = arr.reshape(-1) 
    out = []
    for v in arr: 
        vv = np.asarray(v).reshape(-1) 
        out.append(str(vv[0]) if vv.size > 0 else str(v))
    return np.asarray(out, dtype=str)

def _mat_scalar(value):
    if value is None: 
        return None 
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0: 
        return None 
    return arr[0]

def _as_tuple_str(x: str | Sequence[str] | None) -> tuple[str, ...]: 
    if x is None: 
        return tuple() 
    if isinstance(x, str): 
        return (x,)
    return tuple(str(v) for v in x)

def normalize_hidden_dims(value): 
    if value is None: 
        return None 
    if isinstance(value, str): 
        return tuple(int(v) for v in value.split("-") if v)
    if isinstance(value, (list, tuple)): 
        return tuple(int(v) for v in value)
    if isinstance(value, int): 
        return (int(value),)
    return value 

def load_probs_labels_fips(proba_path): 
    # local import, avoids circular + I'm lazy 
    from preprocessing.loaders import load_oof_predictions

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

def load_probs_for_fips(fips: NDArray[np.str_], proba_path): 
    P, _, oof_fips, class_labels = load_probs_labels_fips(proba_path)
    idx = align_on_fips(fips, oof_fips)
    return P[idx], class_labels

# ---------------------------------------------------------
# Dataset Manipulation  
# ---------------------------------------------------------

def resolve_feature_subset(feature_names, subset): 
    if subset is None: 
        return list(range(len(feature_names))) 
    idx = [int(np.where(feature_names == s)[0][0]) for s in subset]
    return idx 

class ConfigGapTransformer: 
    def __init__(self, integ_idx, viirs_idx): 
        self.integ_idx = int(integ_idx)
        self.viirs_idx = int(viirs_idx)

    def fit(self, X, y=None): 
        integ = X[:, self.integ_idx]
        viirs = X[:, self.viirs_idx]
        self.mu_i = integ.mean() 
        self.sd_i = integ.std() + 1e-9 
        self.mu_v = viirs.mean() 
        self.sd_v = viirs.std() + 1e-9 
        return self 

    def transform(self, X):
        integ = X[:, self.integ_idx]
        viirs = X[:, self.viirs_idx]
        cfg_gap = ((integ - self.mu_i) / self.sd_i) - ((viirs - self.mu_v) / self.sd_v)
        return np.hstack([X, cfg_gap.reshape(-1, 1)])


def make_cfg_gap_factory(feature_names): 
    if feature_names is None: 
        return lambda: []
    names = np.asarray(feature_names)
    if "cross__cross_tiger_integ" not in names or "cross__cross_viirs_log_mean" not in names: 
        return lambda: []
    integ_idx = int(np.where(names == "cross__cross_tiger_integ")[0][0])
    viirs_idx = int(np.where(names == "cross__cross_viirs_log_mean")[0][0])
    return lambda: [ConfigGapTransformer(integ_idx, viirs_idx)]

def load_yaml_config(path: Path) -> Dict[str, Any]: 
    if not path.exists(): 
        return {}
    with path.open("r", encoding="utf-8") as handle: 
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict): 
        raise ValueError(f"config file {path} must contain a mapping at root")
    return data 

def save_yaml_config(path: Path, data: Dict[str, Any]): 
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle: 
        yaml.safe_dump(data, handle, sort_keys=False)

def save_model_config(path: str, model_key: str, params: Dict[str, Any]): 
    config_path = Path(path) 
    data = load_yaml_config(config_path)

    models = data.get("models")
    if models is None: 
        models = {}
    elif not isinstance(models, dict): 
        raise ValueError(f"config file {config_path} has a non-mapping 'models' entry")

    models[model_key] = params 
    data["models"] = models 
    save_yaml_config(config_path, data)

# ---------------------------------------------------------
# Fold/Fit Functions 
# ---------------------------------------------------------

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
    if y_train.ndim == 1:
        y_scaler.fit(y_train.reshape(-1, 1))
    elif y_train.ndim == 2:
        y_scaler.fit(y_train)
    else:
        raise ValueError(f"y_train must be 1D or 2D; got shape {y_train.shape}")

    return X_scaler, y_scaler 


def transform_with_scalers(
    X: ArrayLike, 
    y: ArrayLike, 
    X_scaler: StandardScaler, 
    y_scaler: StandardScaler
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: 
    X = np.asarray(X, dtype=np.float64) 
    y = np.asarray(y, dtype=np.float64) 

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"X rows ({X.shape[0]}) != y ({y.shape[0]})") 

    X_scaled = np.asarray(X_scaler.transform(X), dtype=np.float64) 

    if y.ndim == 1:
        y_scaled = y_scaler.transform(y.reshape(-1, 1))
        y_scaled = np.asarray(y_scaled, dtype=np.float64).ravel()
    elif y.ndim == 2:
        y_scaled = y_scaler.transform(y)
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
    else:
        raise ValueError(f"y must be 1D or 2D; got shape {y.shape}")

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

def make_train_mask(
    y,
    *,
    train_size: float | int = 0.3, 
    random_state: int = 0, 
    stratify: bool = True 
) -> NDArray: 

    y = np.asarray(y).reshape(-1)
    n = y.shape[0] 
    if n == 0: 
        return np.zeros(0, dtype=bool)

    if isinstance(train_size, int):
        if train_size <= 0 or train_size >= n:
            raise ValueError("train_size must be in [1, n-1] when int")
        train_size = train_size / n
    elif isinstance(train_size, float):
        if not (0.0 < train_size < 1.0):
            raise ValueError("train_size must be in (0, 1) when float")

    if stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=random_state
        )
        train_idx, _ = next(splitter.split(np.zeros((n, 1)), y))
    else:
        splitter = ShuffleSplit(
            n_splits=1, train_size=train_size, random_state=random_state
        )
        train_idx, _ = next(splitter.split(np.zeros((n, 1))))

    mask = np.zeros(n, dtype=bool)
    mask[train_idx] = True
    return mask

# ---------------------------------------------------------
# Model Interface 
# ---------------------------------------------------------


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

ModelFactory = Callable[[], ModelInterface]


# ---------------------------------------------------------
# Distance Functions (vectorized) 
# ---------------------------------------------------------


def _haversine_dist(coords_a: NDArray, coords_b: NDArray) -> NDArray: 
    '''
    Haversine distance between two sets of coordinates (lat, lon) in degrees 
    
    Caller Provides: 
        Set of (lat, lon) for two counties 

    We return: 
        Distance in KM 
    '''

    R = 6371.0 

    rad_a = np.radians(coords_a)
    rad_b = np.radians(coords_b)

    dlat = rad_a[:, 0:1] - rad_b[:, 0:1].T 
    dlon = rad_a[:, 1:2] - rad_b[:, 1:2].T 

    a = np.sin(dlat / 2)**2 + (np.cos(rad_a[:, 0:1]) * np.cos(rad_b[:, 0:1].T) * np.sin(dlon / 2)**2) 
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

