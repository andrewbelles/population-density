#!/usr/bin/env python 
# 
# encodings.py  Andrew Belles  Dec 16th, 2025 
# 
# Interfaces through mature libraries (such as scikit)
# for preprocessing data and learned encodings via PCA, AE, etc. 
# 

import numpy as np
import matplotlib.pyplot as plt 

from support.helpers import (
    UnsupervisedDatasetDict, 
    load_climate_and_geospatial_unsupervised, 
    project_path,
    _as_tuple_str
)

from dataclasses import dataclass 
from typing import Any, Sequence 

from numpy.typing import NDArray

from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler 


@dataclass(frozen=True)
class View: 
    X: NDArray[np.float64]
    feature_names: NDArray[np.str_]
    groups: dict[str, slice]


class Encoder: 

    def __init__(
        self, 
        *, 
        dataset: UnsupervisedDatasetDict,
        standardize: bool = True, 
        with_mean: bool = True, 
        with_std: bool = True, 
    ): 
        self.dataset = dataset 

        X = np.asarray(dataset["X"], dtype=np.float64)
        if X.ndim != 2: 
            raise ValueError(f"dataset['X'] must be 2D, got shape {X.shape}")

        if not np.isfinite(X).all():
            raise ValueError("dataset['X'] must be complete (without NaN/Inf)")

        feature_names = np.asarray(dataset["feature_names"])
        if feature_names.ndim != 1 or feature_names.shape[0] != X.shape[1]:
            raise ValueError(f"feature_names must be (n_features,)")

        self.X = X 
        self.feature_names = feature_names.astype("U64", copy=False)

        self.coords = np.asarray(dataset.get("coords", np.empty((0,2))), dtype=np.float64)
        self.coord_names = np.asarray(dataset.get("coord_names", np.empty((0,))), dtype="U64")
        self.sample_ids = np.asarray(dataset.get("sample_ids", np.empty((0,))),
                                     dtype="U64").reshape(-1)


        if self.sample_ids.shape[0] not in (0, self.n_samples): 
            raise ValueError(f"sample_ids length ne n_samples")

        self.groups = dict(dataset.get("groups", {}))
        self.standardize = bool(standardize)
        if self.standardize: 
            self._scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
            self._scaler.fit(self.X)

        self.pca: PCA | None = None 
        self.scores_: NDArray[np.float64] | None = None 


    @property
    def n_samples(self) -> int: 
        return int(self.X.shape[0])

    @property 
    def n_features(self) -> int: 
        return int(self.X.shape[1])

    @property 
    def group_names(self) -> list[str]: 
        return sorted(self.groups.keys())

    def view(self, groups: str | Sequence[str] | None = None) -> View: 

        '''
        Returns a representation of dataset on the specified group labels. 
        
        If groups is None then the View returned is the full data matrix 

        Caller Provides: 
            groups (as union) defining which groups to take for view. 

        We return: 
            View dataclass as a subset of full data matrix 
        '''
        if groups is None: 
            return View(X=self.X, feature_names=self.feature_names, groups=dict(self.groups))

        names   = _as_tuple_str(groups)
        missing = [g for g in names if g not in self.groups]
        if missing: 
            raise KeyError(f"unknown groups: {missing} not in {self.group_names}")

        slices = [self.groups[g] for g in names]
        if any(s.step not in (None,1) for s in slices): 
            raise ValueError("group slices must be contiguous with step=1 or None")

        cols = np.concatenate([np.arange(s.start or 0, s.stop, dtype=int) for s in slices])
        Xv = self.X[:, cols]
        Nv = self.feature_names[cols]

        out_groups: dict[str, slice] = {}
        offset = 0 
        for g, s in zip(names, slices): 
            width = int(s.stop - (s.start or 0)) 
            out_groups[g] = slice(offset, offset + width)
            offset += width 

        return View(X=Xv ,feature_names=Nv, groups=out_groups)

    def fit_pca(
            self, 
            *,
            n_components: int | float | None = None,
            **pca_kwargs: Any
    ) -> PCA:
        
        '''
        Fits sklearn PCA on X. 

        caller provides: 
            n_components: 
                - none, all components 
                - int: exact number of components 
                - float (in 0,1): variance fraction to retain 
            kwargs specific to PCA __init__
        '''

        Xp = self._X_for_pca()
        self.pca = PCA(n_components=n_components, **pca_kwargs)
        self.scores_ = np.asarray(self.pca.fit_transform(Xp), dtype=np.float64) 
        return self.pca 

    def transform(self, X: NDArray[np.float64] | None = None) -> NDArray[np.float64]: 
        pca = self._require_pca()
        Xp = self._X_for_pca(X)
        return np.asarray(pca.transform(Xp), dtype=np.float64)

    def fit_transform_pca(
            self, 
            *, 
            n_components: int | float | None = None,
            **pca_kwargs: Any
    ) -> NDArray[np.float64]:  
        
        '''
        Fits pca to dataset, and transforms dataset. 

        caller provides: 
            n_components: 
                - none, all components 
                - int: exact number of components 
                - float (in 0,1): variance fraction to retain 
            kwargs specific to PCA __init__
        '''

        self.fit_pca(n_components=n_components, **pca_kwargs) 
        if self.scores_ is None: 
            raise RuntimeError("scores_ missing after fit_pca()")
        return self.scores_ 

    def eigenvalues(self) -> NDArray[np.float64]: 
        return np.asarray(self._require_pca().explained_variance_, dtype=np.float64)

    def explained_variance_ratio(self) -> NDArray[np.float64]: 
        return np.asarray(self._require_pca().explained_variance_ratio_, dtype=np.float64)

    def broken_stick(self) -> NDArray[np.float64]: 
        return Encoder.broken_stick_expectation(
            int(self.explained_variance_ratio().shape[0])
        )

    # ------- Plots 

    def plot_eigenvalue_decay(
            self, 
            *,
            ax=None,
            logy: bool = False, 
            title: str | None = None
    ): 
        import matplotlib.pyplot as plt 

        if ax is None: 
            _, ax = plt.subplots(figsize=(10,7))

        eig = self.eigenvalues() 
        x   = np.arange(1, eig.shape[0] + 1)
        ax.plot(x, eig, marker="o", linewidth=1.25)
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue (explained variance)")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(title or "Eigenvalue decay")
        return ax 

    def plot_variance_analysis(
        self,
        *,
        ax=None,
        threshold: float | None = 0.9, 
        title: str | None = None
    ): 
        import matplotlib.pyplot as plt 
        
        if ax is None: 
            _, ax = plt.subplots(figsize=(10,7)) 

        evr = self.explained_variance_ratio() 
        x   = np.arange(1, evr.shape[0] + 1)
        cum = np.cumsum(evr)

        ax.bar(x, evr, alpha=0.5, label="Explained variance ratio")
        ax.plot(x, cum, marker="o", linewidth=1.25, label="Cumulative")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance ratio")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)

        if threshold is not None: 
            t = float(threshold)
            if not (0.0 <= t <= 1.0):
                raise ValueError("threshold must be in (0, 1]")

            k = int(np.searchsorted(cum, t) + 1)
            ax.axhline(t, linestyle="--", linewidth=0.8, color="green", alpha=0.8, 
                       label=f"threshold {t:.0%}")
            ax.axvline(k, linestyle="--", linewidth=0.8, color="red", alpha=0.7,
                       label=f"k={k}")

        ax.legend(frameon=False)
        ax.set_title(title or "Explained Variance")
        return ax 

    def plot_broken_stick(
        self, 
        *,
        ax=None, 
        title: str | None = None
    ): 
        import matplotlib.pyplot as plt 

        if ax is None: 
            _, ax = plt.subplots(figsize=(10,7))

        evr = self.explained_variance_ratio()
        bs  = self.broken_stick()
        x   = np.arange(1, evr.shape[0] + 1)

        ax.plot(x, evr, marker="o", linewidth=1.25, label="Observed (PCA)")
        ax.plot(x, bs, marker="o", linewidth=1.25, label="Broken-stick Expectation")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Ratio")
        ax.set_ylim(0.0, max(0.25, float(max(evr.max(), bs.max())) * 1.1))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.set_title(title or "Broken-stick Test")
        return ax 

    # ------- Private Helpers 

    def _X_for_pca(self, X: NDArray[np.float64] | None = None) -> NDArray[np.float64]: 
        X_use = self.X if X is None else np.asarray(X, dtype=np.float64)
        if X_use.ndim != 2: 
            raise ValueError(f"X must be 2d, got shape {X_use.shape}")
        if self.standardize:
            if self._scaler is None: 
                raise RuntimeError("standardize=True but scaler is not initialized")
            return np.asarray(self._scaler.transform(X_use), dtype=np.float64)
        return X_use 

    def _require_pca(self) -> PCA: 
        if self.pca is None: 
            raise RuntimeError("PCA not fit, call fit_pca() first")
        return self.pca 

    # ------- Static Methods 
    
    @staticmethod 
    def broken_stick_expectation(n: int) -> NDArray[np.float64]: 
        '''
        Returns expected variance ratios for components 1..n by the formula:
            E[p_j] = (1/n) * sum_{k=j..n} (1/k)
        '''
        if n <= 0: 
            raise ValueError("n must be > 0")

        j = np.arange(1, n + 1, dtype=np.float64)
        summations = np.cumsum(1.0 / j[::-1])[::-1]
        return (summations / float(n)).astype(np.float64, copy=False)

def main():
    
    '''
    PCA Analysis on Climate + Coordinate View 

    '''

    dataset = load_climate_and_geospatial_unsupervised(
        filepath=project_path("data", "climate_geospatial.mat"),
        include_coords=True,
        groups=("all",)
    )

    encoder = Encoder(dataset=dataset, standardize=True)
    encoder.fit_pca()

    image_dir = project_path("analysis", "images")

    # ------- Plots for Analysis from fit PCA 

    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_eigenvalue_decay(ax=ax, logy=True)
    fig.savefig(
        project_path(image_dir, "eigen_decay_logy.png"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_eigenvalue_decay(ax=ax, logy=False)
    fig.savefig(
        project_path(image_dir, "eigen_decay.png"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_variance_analysis(ax=ax, threshold=0.95) 
    fig.savefig(
        project_path(image_dir, "variance.png"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_broken_stick(ax=ax)
    plt.savefig(
        project_path(image_dir, "broken_stick.png"), 
        dpi=200, 
        bbox_inches="tight"
    ) 
    plt.close(fig)

if __name__ == "__main__":
    main() 
