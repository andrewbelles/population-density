#!/usr/bin/env python3 
# 
# loaders.py  Andrew Belles  Dec 19th, 2025 
# 
# Loader functions that return dataset dicts to 
# supervised/unsupervised datasets for ues by models 
# 


import numpy as np 

from numpy.typing import NDArray
from typing import Callable, Sequence, TypedDict, List 

from scipy.io import loadmat

from support.helpers import (
    _mat_str_vector
)

_CLIMATE_GROUPS: tuple[str, ...] = ("degree_days", "palmer_indices")
MONTHS      = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

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


def load_compact_dataset(filepath: str) -> DatasetDict: 

    '''
    Note, also loads 
    '''
    
    mat = loadmat(filepath)

    if "features" not in mat or "labels" not in mat: 
        raise ValueError(f"{filepath} missing required keys 'features'/'labels'")

    X = np.asarray(mat["features"], dtype=np.float64) 
    y = np.asarray(mat["labels"], dtype=np.float64)

    if X.shape[0] != y.shape[0]: 
        raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({y.shape[0]})")

    coords = np.zeros((y.shape[0], 2), dtype=np.float64) # Satisfy DatasetDict 
    return {"features": X, "labels": y, "coords": coords}

def load_neighbors_by_density(filepath: str, threshold: float = 0.1): 
    pass 

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
