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
    project_path,
    _as_tuple_str
)

from preprocessing.loaders import (
    UnsupervisedDatasetDict, 
    load_climate_and_geospatial_unsupervised, 
)

from dataclasses import dataclass 
from typing import Any, Literal, Sequence 

from numpy.typing import NDArray

from sklearn.decomposition import PCA, KernelPCA  
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler 

from scipy.io import savemat 


@dataclass(frozen=True)
class View: 
    X: NDArray[np.float64]
    feature_names: NDArray[np.str_]
    groups: dict[str, slice]

PairMode     = Literal["concat", "diff", "absdiff", "concat_absdiff"]
NegativeMode = Literal["mismatch", "shuffle"]
PositiveMode = Literal["augment"]

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
            pca_class: Any | None = None,
            n_components: int | float | None = None,
            **pca_kwargs: Any
    ) -> Any:
        
        '''
        Fits sklearn PCA on X. 

        caller provides: 
            n_components: 
                - none, all components 
                - int: exact number of components 
                - float (in 0,1): variance fraction to retain 
            kwargs specific to PCA __init__
        '''
        if pca_class is None: 
            pca_class = PCA

        Xp = self._X_for_pca()
        self.pca = pca_class(n_components=n_components, **pca_kwargs)
        if self.pca is None: 
            raise TypeError("failed to pass PCA type, found none on encoder.pca")

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

        Caller provides: 
            n_components: 
                - none, all components 
                - int: exact number of components 
                - float (in 0,1): variance fraction to retain 
            kwargs specific to PCA __init__
        
        We return: 
            The transformed dataset back to the user 

        '''

        self.fit_pca(n_components=n_components, **pca_kwargs) 
        if self.scores_ is None: 
            raise RuntimeError("scores_ missing after fit_pca()")
        return self.scores_ 

    def get_reduced_pca_scores(
        self, 
        threshold: float = 0.95 
    ) -> tuple[NDArray[np.float64], int]: 

        '''
        Determines whether Encoder was fit off of KernelPCA or PCA and returns 
        compact representation on provided threshold 
        
        Caller Provides: 
            threshold to reject on for compact representation  

        We return: 
            tuple of (reduced_scores, k)
        '''

        if self.pca is None or self.scores_ is None: 
            raise RuntimeError("Must call fit_pca() first")

        is_kpca = hasattr(self.pca, "eigenvalues_") or hasattr(self.pca, "lambdas_")

        if is_kpca: 
            return self._kpca_components_by_lambda(self.pca, threshold)
        else: 
            return self._pca_components_by_variance(self.pca, threshold)

    def eigenvalues(self) -> NDArray[np.float64]: 
        return np.asarray(self._require_pca().explained_variance_, dtype=np.float64)

    def explained_variance_ratio(self) -> NDArray[np.float64]: 
        return np.asarray(self._require_pca().explained_variance_ratio_, dtype=np.float64)

    def broken_stick(self) -> NDArray[np.float64]: 
        return Encoder.broken_stick_expectation(
            int(self.explained_variance_ratio().shape[0])
        )

    def kpca_lambdas(self) -> NDArray[np.float64]: 
        m = self._require_pca() 
        
        if hasattr(m, "eigenvalues_"):
            vals = getattr(m, "eigenvalues_")
        elif hasattr(m, "lambdas_"):
            vals = getattr(m, "lambdas_")
        else: 
            raise TypeError("current model is not KernelPCA-like")
        
        return np.asarray(vals, dtype=np.float64)

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

    def plot_kpca_lambda(
        self, 
        *, 
        ax=None, 
        logy: bool = False, 
        normalize: bool = False, 
        title: str | None = None 
    ): 
        import matplotlib.pyplot as plt 

        if ax is None: 
            _, ax = plt.subplots(figsize=(10,7))

        lam = self.kpca_lambdas()
        y   = lam / lam.sum() if normalize and lam.sum() > 0 else lam 
        x   = np.arange(1, y.shape[0] + 1)

        ax.plot(x, y, marker="o", linewidth=1.25)
        ax.set_xlabel("Component")
        ax.set_ylabel("Normalized eigenvalues" if normalize else "Kernel eigenvalues")
        if logy:
            ax.set_yscale("log")

        ax.grid(True, alpha=0.25)
        ax.set_title(title or ("Kernel eigenvalue decay" + (" (normalied)" if normalize else "")))
        return ax 

    def plot_kpca_cumulative(
        self, 
        *, 
        ax=None, 
        threshold: float | None = 0.9, 
        title: str | None = None 
    ): 
        import matplotlib.pyplot as plt 

        if ax is None: 
            _, ax = plt.subplots(figsize=(10,7))

        lam   = self.kpca_lambdas()
        total = float(lam.sum())
        if total <= 0: 
            raise ValueError("KernelPCA eigenvalues sum to <= 0. cannot form variance proxy")

        r = lam / total 
        x = np.arange(1, r.shape[0] + 1)
        cum = np.cumsum(r)

        ax.bar(x, r, alpha=0.5, label="KPCA Lambda Ratio")
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
        ax.set_title(title or "KernelPCA Lambda Ratio against Cumulative")
        return ax 

    # ------- Save Constrastive Dataset 

    @staticmethod 
    def save_as_constrastive(
        X_repr: NDArray[np.float64],
        out_path: str | None = None,
        *,
        pair_mode: PairMode = "concat_absdiff", 
        # positive_mode: PositiveMode = "augment", # does not do anything as of right now  
        negative_mode: NegativeMode = "mismatch", 
        n_pos_per_sample: int = 1, 
        neg_ratio: int = 1, 
        noise_std: float = 0.05, 
        mask_prob: float = 0.05,
        seed: int = 0, 
    ): 

        '''
        Writes a contrastive pair dataset to "output_path" with keys: 

        Caller Provides: 
           Some representation (or raw) of the features matrix 
           Output Path 
           And optional kwargs that modify the behavior of the generated dataset.
             A more detailed explanation can be found in the preprocessing README.md 

        Function Does: 
            Generates the contrastive dataset and outputs it at .mat to the specified location
        '''

        X = np.asarray(X_repr, dtype=np.float64)
        if X.ndim != 2: 
            raise ValueError(f"X must be 2d, got {X.shape}")
        if not np.isfinite(X).all(): 
            raise ValueError("X contains NaN/Inf")

        n, d = X.shape 
        if n < 2: 
            raise ValueError("need at least 2 samples")

        if n_pos_per_sample <= 0: 
            raise ValueError("n_pos_per_sample must be > 0")
        if neg_ratio <= 0: 
            raise ValueError("neg_ratio must be > 0")
        if not (0.0 <= mask_prob < 1.0):
            raise ValueError("mask_prob must be in [0, 1)")
        if noise_std < 0.0: 
            raise ValueError("noise_std must be >= 0")

        rng = np.random.default_rng(seed)

        feats: list[NDArray[np.float64]] = []
        labels: list[float] = [] 
        pair_i: list[int]   = []
        pair_j: list[int]   = []

        for i in range(n): 
            xi = X[i]
            for _ in range(n_pos_per_sample): 
                a = Encoder._augment(xi, mask_prob, rng, noise_std, d)
                b = Encoder._augment(xi, mask_prob, rng, noise_std, d)
                feats.append(Encoder._pair_features(a, b, pair_mode))
                labels.append(1.0)
                pair_i.append(i)
                pair_j.append(i)

        n_pos = len(labels) 
        n_neg = n_pos * neg_ratio 

        # Handle negatives from kwarg 

        if negative_mode == "mismatch": 
            for _ in range(n_neg): 
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n - 1))
                if j >= i: 
                    j += 1 
                a = Encoder._augment(X[i], mask_prob, rng, noise_std, d)
                b = Encoder._augment(X[j], mask_prob, rng, noise_std, d)
                feats.append(Encoder._pair_features(a, b, pair_mode))
                labels.append(0.0)
                pair_i.append(i)
                pair_j.append(j)
        elif negative_mode == "shuffle": 
            perm = np.arange(d)
            for _ in range(n_neg): 
                i = int(rng.integers(0, n))
                rng.shuffle(perm)
                a = Encoder._augment(X[i], mask_prob, rng, noise_std, d)
                b = Encoder._augment(X[i], mask_prob, rng, noise_std, d)[perm]
                feats.append(Encoder._pair_features(a, b, pair_mode))
                labels.append(0.0)
                pair_i.append(i)
                pair_j.append(i)

        features = np.vstack(feats).astype(np.float64, copy=False) 
        y = np.asarray(labels, dtype=np.int64).reshape(-1, 1)

        savemat(
            out_path,
            {
                "features": features, 
                "labels": y, 
                "pair_i": np.asarray(pair_i, dtype=np.int32), 
                "pair_j": np.asarray(pair_j, dtype=np.int32), 
            }
        )

    @staticmethod 
    def save_as_compact_supervised(
        out_path: str,  
        X_repr: NDArray[np.float64], 
        y: NDArray[np.float64] 
    ): 

        X = np.asarray(X_repr, dtype=np.float64)
        if X.ndim != 2: 
            raise ValueError(f"X must be 2d, got {X.shape}")
        if not np.isfinite(X).all(): 
            raise ValueError("X contains NaN/Inf")
        
        if y.ndim == 2 and y.shape[1] == 1: 
            y = y.ravel() 
        elif y.ndim == 1: 
            y = y.reshape(-1, 1)
        elif y.ndim != 2: 
            raise ValueError(f"labels must be 1d/2d got shape {y.shape}")

        n = y.shape[0]

        if X.shape[0] != n and X.shape[1] == n: 
            y = y.T 

        if X.shape[0] != n: 
            raise ValueError(f"features rows ({X.shape[0]}) != labels rows ({n})")

        savemat(
            out_path, 
            {"features": X, "labels": y}
        )
        

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

    def _require_pca(self) -> Any: 
        if self.pca is None: 
            raise RuntimeError("PCA not fit, call fit_pca() first")
        return self.pca 

    @staticmethod 
    def _augment(
        x: NDArray[np.float64],
        mask_prob: float, 
        rng, 
        noise_std: float,
        d: int
    ) -> NDArray[np.float64]: 
        y = x.copy() 
        if mask_prob > 0.0: 
            mask = rng.random(d) < mask_prob 
            y[mask] = 0.0 
        if noise_std > 0.0: 
            y = y + rng.normal(0.0, noise_std, size=d)
        return y 

    @staticmethod 
    def _pair_features(
        a: NDArray[np.float64], 
        b: NDArray[np.float64], 
        pair_mode: PairMode, 
    ) -> NDArray[np.float64]: 
        if pair_mode == "concat": 
            return np.concatenate([a, b])
        elif pair_mode == "diff": 
            return a - b 
        elif pair_mode == "absdiff": 
            return np.abs(a - b) 
        elif pair_mode == "concat_absdiff": 
            return np.concatenate([a, b, np.abs(a - b)])

    def _pca_components_by_variance(
        self, 
        pca: Any, 
        threshold: float = 0.95 
    ) -> tuple[NDArray[np.float64], int]: 

        '''
        Returns PCA scores truncated to the minimum components needed to explain some threshold 
        of variance 

        Caller Provides: 
            prefit PCA (w/ attr explained_variance_ratio_)
            threshold of cumulative variance ratio to retain

        We return: 
            truncated scores in shape (n_samples, k)
            k retained scores 
        '''

        if not hasattr(pca, "explained_variance_ratio_"): 
            raise AttributeError("PCA object must have explained_variance_ratio_")
        if self.scores_ is None:
            raise AttributeError("self.scores_ cannot be none")

        if self.scores_.ndim != 2: 
            raise ValueError(f"scores must be 2d, got shape {self.scores_.shape}")

        evr = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cumsum = np.cumsum(evr)

        k = int(np.searchsorted(cumsum, threshold) + 1) 
        k = min(k, self.scores_.shape[1])

        return self.scores_[:, :k].astype(np.float64, copy=False), k

    def _kpca_components_by_lambda(
        self, 
        kpca: Any, 
        threshold: float = 0.95 
    ) -> tuple[NDArray[np.float64], int]: 

        '''
        Returns PCA scores truncated to the minimum components needed to explain some threshold 
        of variance 

        Caller Provides: 
            prefit KernelPCA (w/ attr eigenvalues_ or lambdas_)
            threshold of cumulative normalized lambdas to retain

        We return: 
            truncated scores in shape (n_samples, k)
            k retained scores 
        '''

        if hasattr(kpca, "eigenvalues_"):
            lambdas = np.asarray(kpca.eigenvalues_, dtype=np.float64)
        elif hasattr(kpca, "lambdas_"):
            lambdas = np.asarray(kpca.lambdas_, dtype=np.float64)
        else: 
            raise AttributeError("KernelPCA must have eigenvalues_ or lambdas_")

        if self.scores_ is None:
            raise AttributeError("self.scores_ cannot be none")

        if self.scores_.ndim != 2: 
            raise ValueError(f"scores must be 2d, got shape {self.scores_.shape}")

        total = lambdas.sum() 
        if total <= 0: 
            raise ValueError("Eigenvalues sum to <= 0, cannot compute ratio")

        ratios = lambdas / total 
        cumsum = np.cumsum(ratios)

        k = int(np.searchsorted(cumsum, threshold) + 1) 
        k = min(k, self.scores_.shape[1])

        return self.scores_[:, :k].astype(np.float64, copy=False), k 


def main():
    
    '''
    PCA Analysis on Climate + Coordinate View 

    '''

    dataset = load_climate_and_geospatial_unsupervised(
        filepath=project_path("data", "climate_geospatial.mat"),
        include_coords=False,
        groups=("all",)
    )

    encoder  = Encoder(dataset=dataset, standardize=True)
    
    Xp = encoder._X_for_pca()
    d = pairwise_distances(Xp, metric="euclidean")
    med = np.median(d[d > 0])
    gamma = 1.0 / (2.0 * med * med)

    encoder.fit_pca(
        pca_class=PCA, 
    )

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

    encoder.fit_pca(
        pca_class=KernelPCA,
        kernel="rbf",
        gamma=gamma, 
        eigen_solver="auto", 
        remove_zero_eig=True
    )

    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_kpca_lambda(ax=ax, logy=True)
    fig.savefig(
        project_path(image_dir, "kpca_lambda_decay.png"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(10,7))
    encoder.plot_kpca_cumulative(ax=ax, threshold=0.95) 
    fig.savefig(
        project_path(image_dir, "kpca_variance_proxy.png"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main() 
