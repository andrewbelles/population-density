#!/usr/bin/env python3 
# 
# helpers.py  Andrew Belles  Dec 11th, 2025 
# 
# Module of helper functions for models to utilize, mostly for path concatenation
# 
# 

import os, re, yaml

import numpy as np 
from numpy.typing import ArrayLike, NDArray
from typing import Any, Callable, Sequence, TypedDict, List 
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod
from scipy.io import loadmat


_CLIMATE_GROUPS: tuple[str, ...] = ("degree_days", "palmer_indices")
NCLIMDIV_RE = re.compile(r"^climdiv-([a-z0-9]+)cy-v[0-9.]+-[0-9]{8}.*$")
MONTHS      = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# ---------------------------------------------------------
# Generic Helper Functions 
# ---------------------------------------------------------

def project_path(*args):
    root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *args)


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


def _mat_str_vector(x) -> np.ndarray: 

    arr = np.asarray(x) 
    arr = arr.reshape(-1) 
    out = []
    for v in arr: 
        vv = np.asarray(v).reshape(-1) 
        out.append(str(vv[0]) if vv.size > 0 else str(v))
    return np.asarray(out, dtype=str)


def _as_tuple_str(x: str | Sequence[str] | None) -> tuple[str, ...]: 
    if x is None: 
        return tuple() 
    if isinstance(x, str): 
        return (x,)
    return tuple(str(v) for v in x)


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

def load_models_from_yaml(yaml_path: str, **kwargs) -> dict[str, ModelFactory]: 

    try: 
        with open(yaml_path, 'r') as f: 
            config = yaml.safe_load(f) 
    except FileNotFoundError: 
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    except yaml.YAMLError as e: 
        raise ValueError(f"Invalid YAML format: {e}")

    if 'models' not in config: 
        raise ValueError("YAML must contain 'models' section")

    factories = {}
    for model_name, model_config in config['models'].items(): 
        if model_config is None: 
            model_config = {}

        merged_params = {**model_config, **kwargs} 
        factories[model_name] = create_model_factory(model_name, merged_params)

    return factories 

def create_model_factory(model_type: str, params: dict) -> ModelFactory: 

    if model_type == "LinearModel": 
        from models.linear_model import LinearModel
        return lambda: LinearModel(**params) 
    elif model_type == "RandomForest": 
        from models.random_forest_model import RandomForest
        return lambda: RandomForest(**params)
    elif model_type == "XGBoost": 
        from models.xgboost_model import XGBoost 
        return lambda: XGBoost(**params) 
    else: 
        raise ValueError(f"unsupported model type: {model_type}")


# ---------------------------------------------------------
# Supervised Loader Interface 
# ---------------------------------------------------------


class DatasetDict(TypedDict): 
    features: NDArray[np.float64]
    labels: NDArray[np.float64]
    coords: NDArray[np.float64]

DatasetLoader = Callable[[str], DatasetDict]

def load_climate_population(filepath: str, *, decade: int, groups: List[str]) -> DatasetDict: 
    
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

def load_climate_geospatial(filepath: str, *, target: str, groups: List[str]) -> DatasetDict: 
    '''
    Loader Helper for Climate against (lat, lon). 

    Caller Provides: 
        filepath to dataset, 
        target (lat, lon, or all)
        groups to include in dataset 

    We return: 
        A DatasetDict containing the requested feature groups and target label 
    '''

    group_set = set(groups)
    label_set = {"lat", "lon", "all"}

    if target not in label_set: 
        raise ValueError(f"target: {target} must be in {sorted(label_set)} to be requested")

    data = loadmat(filepath)
    c = np.asarray(data["labels"], dtype=np.float64)   # expects shape (n, 2)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"expected labels to have shape (n, 2); got {c.shape}")

    if target == "all":
        y = c.astype(np.float64, copy=False)
    else:
        idx = 0 if target == "lat" else 1
        y = c[:, idx].reshape(-1)

    features = []
    for name in _CLIMATE_GROUPS: 
        if name not in group_set: 
            continue 
        features.append(np.asarray(data[f"features_{name}"], dtype=np.float64))

    if not features: 
        raise ValueError(f"{groups} does not contain any valid group labels for data")

    features = np.hstack(features) if len(features) > 1 else features[0] 
    return {"features": features, "labels": y, "coords": c}


def load_geospatial_climate(filepath, *, target: str, groups: List[str] = ["lat", "lon"]) -> DatasetDict: 
    
    data = loadmat(filepath)
    coords = np.asarray(data["labels"], dtype=np.float64) 
    if coords.ndim != 2 or coords.shape[1] != 2: 
        raise ValueError(f"expected shape (n,2): got {coords.shape}")
    
    names = _mat_str_vector(data["feature_names"])
    F = np.asarray(data["features"], dtype=np.float64)

    idx = {"lat": 0, "lon": 1}
    X = coords[:, [idx[c] for c in groups]].astype(np.float64, copy=False)

    if target == "all": 
        y = F.astype(np.float64, copy=False) 
        return {"features": X, "labels": y, "coords": coords}

    cols = [f"{target}_{m}" for m in MONTHS]
    col_idx = [int(np.where(names == c)[0][0]) for c in cols]
    Ym = F[:, col_idx] 
    y = np.column_stack([Ym, Ym.mean(axis=1)])

    return {"features": X, "labels": y, "coords": coords}

def load_residual_dataset(residual_filepath: str, original_filepath: str) -> DatasetDict: 

    residual_data = loadmat(residual_filepath)
    base = load_geospatial_climate(original_filepath, target="all", groups=["lat", "lon"])

    if "features" not in residual_data: 
        raise ValueError(f"{residual_filepath} missing features")

    X_resi = np.asarray(residual_data["features"], dtype=np.float64)
    X_orig = np.asarray(base["features"], dtype=np.float64)
    y = np.asarray(base["labels"], dtype=np.float64)

    if X_resi.ndim == 1:
        X_resi = X_resi.reshape(-1, 1)
    if X_orig.ndim == 1:
        X_orig = X_orig.reshape(-1, 1)

    if X_resi.shape[0] != X_orig.shape[0]:
        raise ValueError(
            f"residual rows ({X_resi.shape[0]}) != original rows ({X_orig.shape[0]})"
        )

    X = np.hstack([X_orig, X_resi]).astype(np.float64, copy=False)

    # Preserve multi-output labels (n, k). Only squeeze the trivial (n, 1) case.
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim not in (1, 2):
        raise ValueError(f"labels must be 1D or 2D; got shape {y.shape}")

    n = X.shape[0]
    if y.shape[0] != n: 
        raise ValueError(f"features rows ({n}) != labels row ({y.shape[0]})")

    coords = np.asarray(base["coords"], dtype=np.float64)

    return {"features": X, "labels": y, "coords": coords}

def load_contrastive_dataset(filepath: str) -> DatasetDict: 
    mat = loadmat(filepath)

    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"{filepath} missing required keys 'features'/'labels'")

    X = np.asarray(mat["features"], dtype=np.float64) 
    y = np.asarray(mat["labels"], dtype=np.float64)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({y.shape[0]})")

    coords = np.zeros((y.shape[0], 2), dtype=np.float64) # Satisfy DatasetDict 
    return {"features": X, "labels": y, "coords": coords}


# ---------------------------------------------------------
# Unsupervised Loader Interface 
# ---------------------------------------------------------


class UnsupervisedDatasetDict(TypedDict): 
    X: NDArray[np.float64] 
    coords: NDArray[np.float64] 
    feature_names: NDArray[np.str_]
    coord_names: NDArray[np.str_]
    sample_ids: NDArray[np.str_]
    groups: dict[str, slice]

UnsupervisedLoader = Callable[[str], UnsupervisedDatasetDict]

def load_climate_and_geospatial_unsupervised(
    filepath: str, 
    *, 
    groups: Sequence[str] = ("degree_days", "palmer_indices"), 
    include_coords: bool = True 
) -> UnsupervisedDatasetDict: 

    mat = loadmat(filepath)

    if "labels" not in mat: 
        raise ValueError(f"{filepath} missing 'labels' (expected coordinates)")

    coords = np.asarray(mat["labels"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2: 
        raise ValueError(f"expected shape (n,2), got {coords.shape}")

    if "fips_codes" not in mat: 
        raise ValueError(f"{filepath} missing 'fips_codes'")

    sample_ids = _mat_str_vector(mat["fips_codes"]).astype("U5", copy=False) 
    
    if not groups: 
        raise ValueError("groups cannot be empty")
    if len(set(groups)) != len(groups): 
        raise ValueError(f"groups contains duplicates: {groups}")

    resolved: list[str] = []
    for g in groups: 
        if g == "all": 
            resolved.extend(list(_CLIMATE_GROUPS))
        else: 
            resolved.append(g)
    groups = tuple(resolved)

    X_parts: list[NDArray[np.float64]] = []
    name_parts: list[NDArray[np.str_]] = []
    group_slices: dict[str, slice] = {}
    column_offset: int = 0 # index denoting last appended column idx 

    for g in groups: 

        key  = f"features_{g}" 
        name = f"feature_names_{g}"

        if key not in mat: 
            raise ValueError(f"{filepath} missing '{key}'. available: {_CLIMATE_GROUPS}")

        Xg = np.asarray(mat[key], dtype=np.float64) 
        if Xg.ndim != 2: 
            raise ValueError(f"{key} must be 2d, got {Xg.ndim}d with shape {Xg.shape}")
        if Xg.shape[0] != coords.shape[0]:
            raise ValueError(f"{key} rows (Xg.shape[0]) != coords rows ({coords.shape[0]})")

        if name in mat: 
            names_g = _mat_str_vector(mat[name]).astype("U64", copy=False)
        elif "feature_names" in mat: 
            all_names = _mat_str_vector(mat["feature_names"]).astype("U64", copy=False)
            if all_names.shape[0] == Xg.shape[1]:
                names_g = all_names
            elif all_names.shape[0] >= column_offset + Xg.shape[1]:
                names_g = all_names[column_offset : column_offset + Xg.shape[1]]
            else:
                names_g = np.asarray([f"{g}_{i}" for i in range(Xg.shape[1])], dtype="U64")
        else: 
            names_g = np.asarray([f"{g}_{i}" for i in range(Xg.shape[1])], dtype="U64")

        if names_g.shape[0] != Xg.shape[1]: 
            raise ValueError(f"{name} length ({names_g.shape[0]}) != {key} cols ({Xg.shape[1]})")

        X_parts.append(Xg) 
        name_parts.append(names_g)

        group_slices[str(g)] = slice(column_offset, column_offset + Xg.shape[1])
        column_offset += Xg.shape[1]

    X = np.hstack(X_parts).astype(np.float64, copy=False)
    feature_names = np.concatenate(name_parts).astype("U64", copy=False)

    if include_coords:
        X = np.hstack([X, coords]).astype(np.float64, copy=False)
        group_slices["coords"] = slice(column_offset, column_offset + 2)
        feature_names = np.concatenate([feature_names, np.asarray(["lat", "lon"], dtype="U64")])

    return {
        "X": X, 
        "feature_names": feature_names, 
        "sample_ids": sample_ids, 
        "groups": group_slices, 
        "coords": np.empty((0,2), dtype=np.float64), 
        "coord_names": np.empty((0,), dtype="U1")
    }
