#!/usr/bin/env python3 
# 
# vif.py  Andrew Belles  Dec 25th, 2025 
# 
# Given an unsupervised matrix of features, performs a variation 
# inflation factor test pairwise to analyze pairwise collinearity 
# among features 
# 

import numpy as np 
from numpy.typing import NDArray
import pandas as pd 

from dataclasses import dataclass 
from typing import Callable, Sequence 

from scipy.io import savemat
from sklearn.base import BaseEstimator 

from analysis.cross_validation import (
    CVConfig,
    r2_cv_from_array,
)

from preprocessing.loaders import (
    DatasetDict,
    FeatureMatrix, 
    load_compact_dataset
)

from preprocessing.encodings import Encoder

def _vif_from_r2(r2: float) -> float: 
    r2 = max(0.0, min(0.999, r2))
    return 1.0 / (1.0 - r2)


@dataclass 
class VIFResult: 
    vif:            pd.DataFrame 
    r2:             pd.DataFrame | None 
    feature_names:  NDArray[np.str_] 
    sample_ids:     NDArray[np.str_]
    feature_groups: NDArray[np.str_] | None 
    meta:           dict[str, object]


@dataclass 
class PairwiseVIF:
    '''
    Computes Variance Inflation Factor pairwise for all features in a given feature matrix 
    safely using CrossValidator. 

    VIF can be used to analyze how collinear two features are with each other so decisions about 
    feature inclusion for stacked classifiers can be made. 
    '''

    model_factory: Callable[[], BaseEstimator] 
    config: CVConfig 
    verbose: bool = False 

    def compute(
        self, 
        filepath: str, 
        base_loader_func: Callable[[str], FeatureMatrix | dict], 
        feature_subset: Sequence[int] | None = None,
        *,
        return_r2: bool = True, 
        feature_groups: Sequence[str] | None = None 
    ) -> VIFResult:

        data   = base_loader_func(filepath)
        matrix = self._coerce_feature_mat(data) 
        matrix = self._apply_subset(matrix, feature_subset)

        n_features = matrix.X.shape[1]
        vif_scores = np.full((n_features, n_features), np.nan, dtype=np.float64)
        np.fill_diagonal(vif_scores, np.inf) 

        r2_scores = None 
        if return_r2: 
            r2_scores = np.full((n_features, n_features), np.nan, dtype=np.float64)

        for i in range(n_features): 
            if self.verbose: 
                name = str(matrix.feature_names[i])
                print(f"[{i+1}/{n_features}] target={name}")
            y = matrix.X[:, i]
            for j in range(i + 1, n_features): 
                X = matrix.X[:, [j]]
                r2 = r2_cv_from_array(
                    X,
                    y,
                    matrix.coords,
                    self.model_factory,
                    self.config 
                )
                vif = _vif_from_r2(r2)
                vif_scores[i, j] = vif 
                vif_scores[j, i] = vif
                if r2_scores is not None: 
                    r2_scores[i, j] = r2 
                    r2_scores[j, i] = r2

        vif_df = pd.DataFrame(
            vif_scores, 
            index=matrix.feature_names, 
            columns=matrix.feature_names
        )
        r2_df  = None 
        if r2_scores is not None: 
            r2_df = pd.DataFrame(
                r2_scores, 
                index=matrix.feature_names,
                columns=matrix.feature_names 
            )

        groups = None 
        if feature_groups is not None: 
            groups = np.asarray(feature_groups, dtype="U64")

        meta = {
            "name": "pairwise", 
            "method": "cv", 
            "n_features": int(n_features),
            "n_samples": int(matrix.X.shape[0])
        }

        return VIFResult(
            vif=vif_df,
            r2=r2_df,
            feature_names=matrix.feature_names,
            sample_ids=matrix.sample_ids,
            feature_groups=groups,
            meta=meta
        )

    def _coerce_feature_mat(self, data: FeatureMatrix | dict) -> FeatureMatrix: 

        if isinstance(data, FeatureMatrix): 
            return data 

        if "features" in data: 
            X = np.asarray(data["features"], dtype=np.float64)
        elif "X" in data: 
            X = np.asarray(data["X"], dtype=np.float64)
        else: 
            raise KeyError("loader output missing 'features' or 'X'")

        # Don't care if coords dne, stacking models agnostic to geography (FOR NOW?) 
        coords = np.asarray(data.get("coords", np.zeros((X.shape[0], 2))), dtype=np.float64)
        if coords.ndim == 2 and coords.shape == (2, X.shape[0]): 
            coords = coords.T # coerce shape .mat may fuck this up occasionally 

        feature_names = data.get("feature_names")
        if feature_names is None or len(feature_names) == 0: 
            feature_names = np.array(
                [f"feature_{i}" for i in range(X.shape[1])], dtype="U64"
            )
        else: 
            feature_names = np.asarray(feature_names, dtype="U64")

        if feature_names.shape[0] != X.shape[1]: 
            n_features = feature_names.shape[0]
            raise ValueError(f"feature_names length ({n_features})!= X cols ({X.shape[1]})")

        sample_ids = data.get("sample_ids")
        if sample_ids is None or len(sample_ids) == 0: 
            sample_ids = np.array(
                [str(i) for i in range(X.shape[0])], dtype="U16"
            )
        else: 
            sample_ids = np.asarray(sample_ids, dtype="U16")

        return FeatureMatrix(
            X=X, 
            coords=coords,
            feature_names=feature_names,
            sample_ids=sample_ids
        )

    def _apply_subset(
        self, 
        matrix: FeatureMatrix, 
        feature_subset: Sequence[int] | None
    ) -> FeatureMatrix: 
        if feature_subset is None: 
            return matrix 
        if len(feature_subset) == 0: 
            raise ValueError("feature_subset cannot be empty")

        first = feature_subset[0]
        if isinstance(first, (int, np.integer)):
            cols = [int(i) for i in feature_subset]
        else: 
            name_map = {str(n): i for i, n in enumerate(matrix.feature_names)}
            cols = []
            for name in feature_subset: 
                key = str(name)
                if key not in name_map: 
                    raise KeyError(f"feature name not found: {key}")
                cols.append(name_map[key])

        return matrix.subset(cols)

# ---------------------------------------------------------
# Combine Collinear Features Pairwise 
# ---------------------------------------------------------

# Binding for a pair that must be reduced via PCA 
PairSpec = Sequence[tuple[str, str]]

@dataclass 
class PairwiseReducer: 
    standarize: bool = True 
    with_mean: bool = True 
    with_std: bool = True 
    verbose: bool = False 

    def combine(
        self, 
        matrix: FeatureMatrix, 
        pairs: PairSpec 
    ) -> FeatureMatrix: 
        
        if not pairs: 
            raise ValueError("pairs cannot be empty")

        pair_idx = self._resolve_pairs(matrix.feature_names, pairs)
        
        out_cols: list[NDArray[np.float64]] = []
        out_names: list[str] = []

        for k, (i, j) in enumerate(pair_idx, start=1): 
            if i == j: 
                raise ValueError(f"pair {i} == {j}. pairs must be distinct")

            Xi = matrix.X[:, [i, j]]
            pair_names = np.array([str(matrix.feature_names[i]),
                                   str(matrix.feature_names[j])], dtype="U64")

            dataset = FeatureMatrix(
                X=Xi, 
                coords=matrix.coords, 
                feature_names=pair_names, 
                sample_ids=matrix.sample_ids, 
            )

            enc = Encoder(
                dataset=dataset, 
                standardize=self.standarize, 
                with_mean=self.with_mean, 
                with_std=self.with_std
            )

            scores = enc.fit_transform_pca(n_components=1)

            if scores.ndim == 1: 
                scores = scores.reshape(-1, 1)

            out_cols.append(scores.astype(np.float64, copy=False))

            if scores.shape[1] == 1: 
                out_names.append(f"pca_{pair_names[0]}__{pair_names[1]}")

            if self.verbose:
                print(f"[{k}/{len(pair_idx)}] {pair_names[0]} + {pair_names[1]} "
                      f"-> {out_names[-1]}")

        X_new     = np.hstack(out_cols).astype(np.float64, copy=False)
        names_new = np.asarray(out_names, dtype="U128")

        return FeatureMatrix(
            X=X_new, 
            coords=matrix.coords, 
            feature_names=names_new, 
            sample_ids=matrix.sample_ids
        )

    def save(
        self, 
        out_path: str, 
        combined: FeatureMatrix, 
        labels_filepath: str, 
        labels_loader: Callable[[str], DatasetDict] = load_compact_dataset, 
        *, 
        original: FeatureMatrix, 
        pairs: PairSpec,
        labels_key: str = "labels" 
    ) -> FeatureMatrix | None: 
        labels = labels_loader(labels_filepath)

        if labels_key not in labels: 
            raise KeyError(f"labels_key '{labels_key}' not found in labels")

        y = np.asarray(labels[labels_key])
        if y.ndim == 2 and y.shape[1] == 1: 
            y = y.reshape(-1)
        elif y.ndim not in (1, 2): 
            raise ValueError(f"labels must be 1d or 2d, got {y.shape}")

        label_ids = labels.get("sample_ids")
        if label_ids is None: 
            raise ValueError("labels missing sample_ids")
        
        label_ids = np.asarray(label_ids, dtype="U5")
        feat_ids  = np.asarray(combined.sample_ids, dtype="U5")

        label_map = {f: i for i, f in enumerate(label_ids)}
        feat_map  = {f: i for i, f in enumerate(feat_ids)}
        common    = [f for f in feat_ids if f in label_map]

        if not common: 
            raise ValueError("no overlapping sample_ids between combined features and labels")

        feat_idx  = [feat_map[f] for f in common]
        label_idx = [label_map[f] for f in common]
        
        X = combined.X[feat_idx]
        coords = combined.coords[feat_idx]
        y_aligned = y[label_idx]

        mat = {
            "features": X, 
            "labels": y_aligned, 
            "coords": coords, 
            "feature_names": combined.feature_names, 
            "fips_codes": np.asarray(common, dtype="U5")
        }
        savemat(out_path, mat)

        return self._remaining_matrix(original, pairs, common)
        
    def _remaining_matrix(
        self,
        original: FeatureMatrix, 
        pairs: PairSpec,
        keep_ids: Sequence[str]
    ) -> FeatureMatrix: 
        pair_idx = self._resolve_pairs(original.feature_names, pairs)
        drop = sorted({i for pair in pair_idx for i in pair})
        keep_cols = [i for i in range(original.X.shape[1]) if i not in drop]

        id_map  = {f: i for i, f in enumerate(original.sample_ids)}
        row_idx = [id_map[f] for f in keep_ids if f in id_map] 

        X      = original.X[row_idx][:, keep_cols]
        coords = original.coords[row_idx]
        names  = original.feature_names[keep_cols]
        sample_ids = np.asarray(keep_ids, dtype="U5")

        return FeatureMatrix(
            X=X,
            coords=coords,
            feature_names=names,
            sample_ids=sample_ids
        )

    def _resolve_pairs(
        self,
        feature_names: NDArray[np.str_], 
        pairs: PairSpec 
    ) -> list[tuple[int, int]]: 
        name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
        out: list[tuple[int, int]] = []
        for a, b in pairs: 
            key = str(a)
            if key not in name_to_idx: 
                raise KeyError(f"feature name not found: {key}")
            ia = name_to_idx[key]

            key = str(b)
            if key not in name_to_idx: 
                raise KeyError(f"feature name not found: {key}")
            ib = name_to_idx[key]

            out.append((ia, ib))
        return out 


@dataclass 
class FullVIF: 
    model_factory: Callable[[], BaseEstimator]
    config:        CVConfig 
    verbose:       bool = False 

    def compute(
        self,
        filepath: str,
        base_loader_func: Callable[[str], FeatureMatrix | dict],
        feature_subset: Sequence[int] | None = None, 
        *,
        return_r2: bool = True, 
        feature_groups: Sequence[str] | None = None 
    ) -> VIFResult:
        data   = base_loader_func(filepath)
        matrix = PairwiseVIF(self.model_factory, self.config)._coerce_feature_mat(data)
        matrix = PairwiseVIF(self.model_factory, self.config)._apply_subset(
            matrix, 
            feature_subset
        )

        n_features = matrix.X.shape[1]
        vif_vals   = np.full(n_features, np.nan, dtype=np.float64)
        r2_vals    = np.full(n_features, np.nan, dtype=np.float64) if return_r2 else None 

        for i in range(n_features): 
            if self.verbose: 
                name = str(matrix.feature_names[i])
                print(f"[{i+1}/{n_features}] target={name}")
            y    = matrix.X[:, i]
            keep = [j for j in range(n_features) if j != i]
            X    = matrix.X[:, keep]
            r2   = r2_cv_from_array(X, y, matrix.coords, self.model_factory, self.config)
            vif_vals[i] = _vif_from_r2(r2)
            if r2_vals is not None: 
                r2_vals[i] = r2 

        vif_df = pd.DataFrame({"vif": vif_vals}, index=matrix.feature_names)
        r2_df  = None 
        if r2_vals is not None: 
            r2_df = pd.DataFrame({"r2": r2_vals}, index=matrix.feature_names)

        groups = None 
        if feature_groups is not None: 
            groups = np.asarray(feature_groups, dtype="U64")

        meta = {
            "mode": "full",
            "method": "cv", 
            "n_features": int(n_features),
            "n_samples": int(matrix.X.shape[0])
        }

        return VIFResult(
            vif=vif_df,
            r2=r2_df,
            feature_names=matrix.feature_names,
            sample_ids=matrix.sample_ids,
            feature_groups=groups,
            meta=meta
        )
