#!/usr/bin/env python3 
# 
# feature_matrix.py  Andrew Belles  Jan 13th, 2026 
# 
# Shared helpers for aligning feature matrices across datasets 
# 
# 

import numpy as np 

from dataclasses import dataclass 

from typing import Sequence

from numpy.typing import NDArray 

from preprocessing.loaders import FeatureMatrix, DatasetDict 


@dataclass 
class SupervisedMatrix: 
    X:             NDArray[np.float64] 
    y:             NDArray[np.int64]
    feature_names: NDArray[np.str_]
    sample_ids:    NDArray[np.str_]

def coerce_feature_mat(data: FeatureMatrix | DatasetDict) -> FeatureMatrix: 
    if isinstance(data, FeatureMatrix): 
        return data 

    X      = np.asarray(data["features"], dtype=np.float64)
    coords = np.asarray(data.get("coords", np.zeros((X.shape[0], 2))), dtype=np.float64)
    if coords.ndim == 2 and coords.shape == (2, X.shape[0]): 
        coords = coords.T 

    feature_names = data.get("feature_names")
    if feature_names is None or len(feature_names) == 0: 
        feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])], dtype="U64")
    else: 
        feature_names = np.asarray(feature_names, dtype="U64")

    if feature_names.shape[0] != X.shape[1]: 
        raise ValueError(f"feature_names length ({feature_names.shape[0]}) != " 
                         f"X cols ({X.shape[1]})")

    sample_ids = data.get("sample_ids")
    if sample_ids is None or len(sample_ids) == 0: 
        sample_ids = np.array([str(i) for i in range(X.shape[0])], dtype="U16")
    else: 
        sample_ids = np.asarray(sample_ids, dtype="U16")

    return FeatureMatrix(
        X=X,
        coords=coords,
        feature_names=feature_names,
        sample_ids=sample_ids
    )

def coerce_supervised(data: DatasetDict) -> SupervisedMatrix: 
    X = np.asarray(data["features"], dtype=np.float64)
    y = np.asarray(data["labels"]).reshape(-1).astype(np.int64)

    feature_names = data.get("feature_names")
    if feature_names is None or len(feature_names) == 0: 
        feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])], dtype="U64")
    else: 
        feature_names = np.asarray(feature_names, dtype="U64")

    if feature_names.shape[0] != X.shape[1]: 
        raise ValueError(f"feature_names length ({feature_names.shape[0]}) != "
                         f"X cols ({X.shape[1]})")

    sample_ids = data.get("sample_ids")
    if sample_ids is None or len(sample_ids) == 0: 
        sample_ids = np.array([str(i) for i in range(X.shape[0])], dtype="U16")
    else: 
        sample_ids = np.asarray(sample_ids, dtype="U16")

    return SupervisedMatrix(
        X=X,
        y=y,
        feature_names=feature_names,
        sample_ids=sample_ids
    )

def drop_nan_rows(matrix: SupervisedMatrix) -> SupervisedMatrix:
    mask = np.isfinite(matrix.X).all(axis=1)
    return SupervisedMatrix(
        X=matrix.X[mask],
        y=matrix.y[mask],
        feature_names=matrix.feature_names,
        sample_ids=matrix.sample_ids[mask]
    )

def align_and_merge_features(
    datasets: Sequence[FeatureMatrix],
    *,
    prefixes: Sequence[str] | None = None, 
    feature_groups: Sequence[str] | None = None, 
    sample_order: Sequence[str] | None = None 
) -> tuple[FeatureMatrix, NDArray[np.str_]]: 
    if not datasets: 
        raise ValueError("datasets cannot be empty")

    matrices = [coerce_feature_mat(d) for d in datasets]

    if prefixes is not None and len(prefixes) != len(matrices): 
        raise ValueError("prefixes must match datasets length")

    id_maps = [{sid: i for i, sid in enumerate(m.sample_ids)} for m in matrices]

    if sample_order is None: 
        common = [sid for sid in matrices[0].sample_ids if all(sid in m for m in id_maps[1:])]
    else: 
        common = [sid for sid in sample_order if all(sid in m for m in id_maps)]

    if not common: 
        raise ValueError("no overlapping sample_ids across datasets")

    idx_lists = [[m[sid] for sid in common] for m in id_maps]
    X_list    = [mat.X[idx] for mat, idx in zip(matrices, idx_lists)]

    names_list: list[NDArray[np.str_]] = []
    groups: list[str]                  = []
    for i, mat in enumerate(matrices): 
        prefix = prefixes[i] if prefixes is not None else None 
        if prefix: 
            names = np.array([f"{prefix}::{n}" for n in mat.feature_names], dtype="U128")
            group = prefix 
        else: 
            names = mat.feature_names.astype("U128")
            group = feature_groups[i] if feature_groups is not None else f"set_{i}"
        names_list.append(names)
        groups.extend([group] * names.shape[0])

    X_all         = np.hstack(X_list).astype(np.float64, copy=False)
    feature_names = np.concatenate(names_list)
    sample_ids    = np.asarray(common, dtype="U16")
    coords        = matrices[0].coords[idx_lists[0]]
    merged        = FeatureMatrix(
        X=X_all,
        coords=coords,
        feature_names=feature_names,
        sample_ids=sample_ids
    )

    return merged, np.asarray(groups, dtype="U64")

def apply_feature_subset(
    matrix: SupervisedMatrix,
    subset: Sequence[int | str] | None 
) -> SupervisedMatrix:
    if subset is None: 
        return matrix 
    if len(subset) == 0: 
        raise ValueError("subset cannot be empty")

    first = subset[0]
    if isinstance(first, (int, np.integer)): 
        cols = [int(i) for i in subset] 
    else:
        name_map = {str(n): i for i, n in enumerate(matrix.feature_names)}
        cols     = []
        for name in subset: 
            key = str(name)
            if key not in name_map:
                raise KeyError(f"feature name not found: {key}")
            cols.append(name_map[key])

    return SupervisedMatrix(
        X=matrix.X[:, cols],
        y=matrix.y,
        feature_names=matrix.feature_names[cols],
        sample_ids=matrix.sample_ids
    )
