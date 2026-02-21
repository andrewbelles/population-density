#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import numpy as np 

import torch, shap  

import torch.nn.functional as F 

from torch import nn 

import torchvision.models as tvm

from typing import (
    Optional,
    Any 
)

from numpy.typing import NDArray

from torch_scatter import (
    scatter_add, 
    scatter_softmax, 
)

from sklearn.base import (
    BaseEstimator,
    RegressorMixin
)

from sklearn.ensemble import (
    GradientBoostingRegressor 
)

# ---------------------------------------------------------
# Quantile-GBR triplet module 
# ---------------------------------------------------------

class QuantileGBRTriplet(BaseEstimator, RegressorMixin): 

    def __init__(
        self,
        *,
        n_estimators: int, 
        lr: float, 
        max_depth: int, 
        min_samples_split: int, 
        min_samples_leaf: int, 
        subsample: float, 
        max_features: Optional[str | int] = None, 
        alpha_lo: float = 0.16, 
        alpha_mid: float = 0.50, 
        alpha_hi: float = 0.84,
        random_state: int = 0 
    ):
        self.n_estimators      = n_estimators
        self.lr                = lr 
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf 
        self.subsample         = subsample 
        self.max_features      = max_features
        self.alpha_lo          = alpha_lo 
        self.alpha_mid         = alpha_mid 
        self.alpha_hi          = alpha_hi 
        self.random_state      = random_state

    def fit(self, X, y): 
        X, y = self.validate_input(X, y)
        assert y is not None 
        
        self.est_lo_  = self.new(self.alpha_lo)
        self.est_mid_ = self.new(self.alpha_mid)
        self.est_hi_  = self.new(self.alpha_hi)

        self.est_lo_.fit(X, y)
        self.est_mid_.fit(X, y)
        self.est_hi_.fit(X, y)
        
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True 
        return self 

    def predict(self, X) -> NDArray[np.float64]: 
        self.check_fitted()
        X, _ = self.validate_input(X)
        if X.shape[1] != self.n_features_in_: 
            raise ValueError("X feature count mismatch")
        return np.asarray(self.est_mid_.predict(X), dtype=np.float64)

    def predict_distribution(
        self,
        X,
        *,
        sigma_floor: float = 1e-6, 
        var_floor: float = 1e-6
    ) -> dict[str, NDArray[np.float64]]: 
        q_lo, mu, q_hi = self.predict_quantiles(X)
        _, sigma       = self.predict_mu_sigma(
            X, sigma_floor=sigma_floor, 
            q_lo=q_lo, mu=mu, q_hi=q_hi
        )
        var = np.maximum(np.power(sigma, 2), var_floor)

        return {
            "mu": mu, 
            "sigma": sigma, 
            "var": var,
            "log_var": np.log(var),
            "q_lo": q_lo,
            "q_mid": mu,
            "q_hi": q_hi 
        }

    def predict_mu_sigma(
        self,
        X,
        *,
        sigma_floor: float = 1e-6, 
        q_lo: Optional[NDArray[np.float64]] = None,
        mu: Optional[NDArray[np.float64]] = None,
        q_hi: Optional[NDArray[np.float64]] = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: 
        if q_lo is None or mu is None or q_hi is None: 
            q_lo, mu, q_hi = self.predict_quantiles(X)
        q_lo = np.minimum(q_lo, q_hi)
        q_hi = np.maximum(q_hi, q_lo)

        sigma = (q_hi - q_lo) / 1.988 # normal 
        sigma = np.maximum(sigma, sigma_floor)
        return mu, sigma 

    def predict_quantiles(
        self, 
        X
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: 
        self.check_fitted()
        X, _ = self.validate_input(X)
        if X.shape[1] != self.n_features_in_: 
            raise ValueError("X feature count mismatch")
        q_lo  = np.asarray(self.est_lo_.predict(X), dtype=np.float64)
        q_mid = np.asarray(self.est_mid_.predict(X), dtype=np.float64)
        q_hi  = np.asarray(self.est_hi_.predict(X), dtype=np.float64)
        return q_lo, q_mid, q_hi 

    def shap(
        self,
        X,
        *,
        which: str = "mid",
        feature_names: Optional[list[str]] = None, 
    ) -> dict[str, Any]:
        '''
        SHAP explanation for single branch lo, mid, or hi 
        '''
        self.check_fitted()
        if which not in {"lo", "mid", "hi"}: 
            raise ValueError("which must be one of {lo, mid, hi}")

        X, _ = self.validate_input(X)
        if X.shape[1] != self.n_features_in_: 
            raise ValueError("X feature count mismatch")

        idx    = np.arange(X.shape[0], dtype=np.int64)
        X_eval = X 

        est = {"lo": self.est_lo_, "mid": self.est_mid_, "hi": self.est_hi_}[which]

        explainer = shap.TreeExplainer(est)
        values    = np.asarray(explainer.shap_values(X_eval), dtype=np.float64)
        base      = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        if feature_names is None: 
            feature_names = [f"f{i}" for i in range(X_eval.shape[1])]

        return {
            "which": which, 
            "sample_index": idx, 
            "values": values, 
            "base_values": base, 
            "feature_names": feature_names 
        }

    def uncertainty(
        self,
        X,
        *,
        feature_names: Optional[list[str]] = None
    ) -> dict[str, Any]: 
        lo = self.shap(X, which="lo", feature_names=feature_names)
        hi = self.shap(X, which="hi", feature_names=feature_names)
        
        values = (np.asarray(hi["values"]) - np.asarray(lo["values"])) / 1.988 
        base   = (hi["base_values"] - lo["base_values"]) / 1.988 

        return {
            "which": "sigma_proxy", 
            "sample_index": hi["sample_index"],
            "values": values, 
            "base_values": base, 
            "feature_names": hi["feature_names"]
        }

    def new(self, alpha: float) -> GradientBoostingRegressor: 
        return GradientBoostingRegressor(
            loss="quantile", 
            alpha=alpha, 
            n_estimators=self.n_estimators,
            learning_rate=self.lr,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state
        )

    def validate_input(
        self, 
        X, 
        y=None
    ) -> tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2: 
            raise ValueError(f"X must be 2d, got {X.shape}")

        if y is not None: 
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            if X.shape[0] != y.shape[0]: 
                raise ValueError("X/y row mismatch.")
        return X, y 

    def check_fitted(self): 
        if not getattr(self, "is_fitted_", False): 
            raise RuntimeError("call fit() first")

# ---------------------------------------------------------
# Spatial Lag Feature Cache  
# ---------------------------------------------------------

class SpatialLagFeatures: 
    '''
    Caches Haversine distance and row-normalized lag weights + augmented feature dicts. 
    '''

    def __init__(
        self,
        *,
        k: int = 8, 
        bandwidth_k: Optional[int] = 8, 
        self_weight: float = 0.0, 
        eps: float = 1e-9, 
    ): 
        self.k = k 
        self.bandwidth_k = None if bandwidth_k is None else bandwidth_k 
        self.self_weight = self_weight 
        self.eps         = eps 

        self.coords_:     Optional[NDArray[np.float64]] = None 
        self.sample_ids_: Optional[NDArray[np.str_]]    = None
        self.id_to_pos_:  Optional[dict[str, int]]      = None 
        self.dist_km_:    Optional[NDArray[np.float64]] = None 
        self.W_train_:    Optional[NDArray[np.float64]] = None 
        self.cached_:     dict[str, dict[str, NDArray]] = {}

    def fit(self, coords, sample_ids=None) -> "SpatialLagFeatures": 
        c = self.as_coords(coords) 
        self.coords_  = c 
        self.dist_km_ = self.haversine(c)
        self.W_train_ = self.distance_to_weights(self.dist_km_, square=True)

        if sample_ids is not None: 
            ids = self.canon_fips(sample_ids)
            if ids.shape[0] != c.shape[0]: 
                raise ValueError("sample_ids length mismatch.")
            self.sample_ids_ = ids 
            self.id_to_pos_  = {fid: i for i, fid in enumerate(ids)}
        else: 
            self.sample_ids_ = None  
            self.id_to_pos_  = None 

        self.cached_.clear() 
        return self 

    def transform(
        self,
        features: dict[str, NDArray],
        *,
        sample_ids=None, 
    ): 
        self.check_fitted()
        W = self.W_train_ if sample_ids is None else self.subset(sample_ids)
        assert W is not None 

        out:   dict[str, NDArray] = {}
        cache: dict[str, dict[str, NDArray]] = {}
        for name, mat in features.items(): 
            X = np.asarray(mat, dtype=np.float64)
            if X.ndim != 2: 
                raise ValueError(f"features[{name}] must be 2d")
            if X.shape[0] != W.shape[0]: 
                raise ValueError(f"features[{name}] row mismatch: {X.shape[0]} != {W.shape[0]}")
            pack = self.augment(X, W)
            cache[name] = pack
            out[name]   = pack["augmented"]

        self.cached_ = cache 
        return out

    def query(
        self,
        query_features: dict[str, NDArray],
        query_coords, 
        *,
        ref_features: dict[str, NDArray]
    ): 
        self.check_fitted()

        qc   = self.as_coords(query_coords)
        D_qt = self.haversine(qc, self.coords_)
        W_qt = self.distance_to_weights(D_qt, square=False)

        out:   dict[str, NDArray] = {}
        cache: dict[str, dict[str, NDArray]] = {}

        for name, qmat in query_features.items(): 
            if name not in ref_features: 
                raise KeyError(f"missing reference features for key: {name}")

            Xq = np.asarray(qmat, dtype=np.float64)
            Xr = np.asarray(ref_features[name], dtype=np.float64)
            if Xq.ndim != 2 or Xr.ndim != 2: 
                raise ValueError(f"features[{name}] must be 2d")
            if Xq.shape[1] != Xr.shape[1]: 
                raise ValueError(f"features[{name}] col mismatch: "
                                 f"{Xq.shape[1]} != {Xr.shape[1]}")
            if W_qt.shape[0] != Xq.shape[0] or W_qt.shape[1] != Xr.shape[0]: 
                raise ValueError(f"query/reference row mismatch for key: {name}")

            lag  = W_qt @ Xr 
            xlag = Xq * lag 
            aug  = np.concatenate([Xq, lag, xlag], axis=1)

            pack = {
                "base": Xq,
                "lag": lag, 
                "interaction": xlag, 
                "augmented": aug
            }
            cache[name] = pack 
            out[name]   = pack["augmented"]

        self.cached_ = cache 
        return out 

    def augment(self, X: NDArray[np.float64], W: NDArray[np.float64]) -> dict[str, NDArray]: 
        lag  = W @ X 
        xlag = X * lag  
        aug  = np.concatenate([X, lag, xlag], axis=1, dtype=np.float64)
        return {
            "base": X, 
            "lag": lag, 
            "interaction": xlag, 
            "augmented": aug 
        }

    def subset(self, sample_ids) -> NDArray[np.float64]: 
        self.check_fitted()
        if self.id_to_pos_ is None: 
            raise ValueError("fit() must be called with sample_ids.")

        ids = self.canon_fips(sample_ids)
        idx = []
        missing = []
        for fid in ids: 
            j = self.id_to_pos_.get(fid)
            if j is None: 
                missing.append(fid)
            else: 
                idx.append(j)
        if missing: 
            raise ValueError(f"missing ids in fitted cache: {len(missing)}")
        idx_arr = np.asarray(idx, dtype=np.int64)
        return self.W_train_[np.ix_(idx_arr, idx_arr)]

    def check_fitted(self): 
        if self.coords_ is None or self.W_train_ is None: 
            raise RuntimeError("call fit() first")

    def distance_to_weights(
        self, 
        D: NDArray[np.float64], 
        *, 
        square: bool
    ) -> NDArray[np.float64]: 
        if D.ndim != 2: 
            raise ValueError("distance matrix must be 2d.")

        n, m = D.shape 
        if m == 0: 
            raise ValueError("distance matrix has zero columns")

        exclude_self = square and (n == m)
        max_k = m - 1 if exclude_self else m 
        if max_k <= 0: 
            raise ValueError("invalid neighbor count for distance matrix")

        k = min(max(self.k, 1), max_k)
        W = np.zeros((n, m), dtype=np.float64)

        work = np.asarray(D, dtype=np.float64).copy() 
        if exclude_self:
            np.fill_diagonal(work, np.inf)

        nn_idx  = np.argpartition(work, kth=k-1, axis=1)[:, :k]
        nn_dist = np.take_along_axis(work, nn_idx, axis=1)

        if self.bandwidth_k is None: 
            finite = nn_dist[np.isfinite(nn_dist)]
            bw     = float(np.median(finite)) if finite.size else 1.0 
        else: 
            k_bw   = min(max(self.bandwidth_k, 1), k) 
            kth    = np.partition(nn_dist, kth=k_bw - 1, axis=1)[:, k_bw-1]
            finite = kth[np.isfinite(kth)]
            bw     = float(np.median(finite)) if finite.size else 1.0 

        if (not np.isfinite(bw)) or bw <= self.eps: 
            bw = 1.0 

        nn_w = np.exp(-np.square(nn_dist / bw))
        nn_w[~np.isfinite(nn_w)] = 0.0 
        np.put_along_axis(W, nn_idx, nn_w, axis=1)

        if exclude_self and self.self_weight > 0.0: 
            diag = np.arange(n, dtype=np.int64)
            W[diag, diag] = self.self_weight 

        row_sum = np.clip(W.sum(axis=1, keepdims=True), self.eps, None)
        return W / row_sum 

    @staticmethod 
    def haversine(
        src: NDArray[np.float64], 
        dst: Optional[NDArray[np.float64]] = None 
    ) -> NDArray[np.float64]: 

        '''
        Haversine distance out to KM
        '''

        src = SpatialLagFeatures.as_coords(src)
        dst = src if dst is None else SpatialLagFeatures.as_coords(dst)

        lat1 = np.radians(src[:, 0])[:, None]
        lon1 = np.radians(src[:, 1])[:, None]
        lat2 = np.radians(dst[:, 0])[None, :]
        lon2 = np.radians(dst[:, 1])[None, :]

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = np.sin(dlat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon * 0.5) ** 2
        h = np.clip(h, 0.0, 1.0)

        earth_km = 6371.0088
        return (2.0 * earth_km * np.arcsin(np.sqrt(h))).astype(np.float64, copy=False)

    @staticmethod 
    def canon_fips(ids) -> NDArray[np.str_]: 
        arr = np.asarray(ids).reshape(-1)
        out = []
        for v in arr: 
            s = str(v).strip() 
            if s.isdigit(): 
                s = s.zfill(5)
            out.append(s)
        return np.asarray(out, dtype="U5")

    @staticmethod 
    def as_coords(coords) -> NDArray[np.float64]: 
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 2: 
            raise ValueError(f"coords must be (n, 2), got {c.shape}")
        if not np.all(np.isfinite(c)): 
            raise ValueError("coords must be finite")
        return c 

class GatedAttentionPooling(nn.Module): 

    '''
    Gated attention pooling for multi-instance learning (Maximillian et. al)
    Input: 
        tensor of shape (B, N, F)
    Output: 
        embedding of shape (B, F)
    '''

    def __init__(
        self,
        in_dim: int, 
        attn_dim: int = 256, 
        attn_dropout: float = 0.0 
    ): 
        super().__init__()
        self.in_dim   = in_dim 
        self.attn_dim = attn_dim 

        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

        self.tanh    = nn.Tanh() 
        self.sigmoid = nn.Sigmoid() 
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity() 

    def forward(
        self, 
        x: torch.Tensor, 
        batch_indices: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor:

        V = self.tanh(self.V(x))
        U = self.sigmoid(self.U(x))

        A = self.w(V * U)
        A = A.float() 

        A_max  = torch.zeros(batch_size, 1, device=x.device, dtype=torch.float32)
        A_max.scatter_reduce_(0, batch_indices.unsqueeze(1), A, reduce="amax", include_self=False)

        A_exp  = torch.exp(A - A_max[batch_indices])

        A_sum  = torch.zeros(batch_size, 1, device=x.device, dtype=torch.float32)
        A_sum.index_add_(0, batch_indices, A_exp)

        A_soft = A_exp / (A_sum[batch_indices] + 1e-9)
        A_soft = self.dropout(A_soft)

        w = x.float() * A_soft 

        pooled = torch.zeros(batch_size, x.shape[1], device=x.device, dtype=torch.float32)
        pooled.index_add_(0, batch_indices, w)
        return pooled 


class ResNetMIL(nn.Module): 
    '''
    ResNet-18 backbone + gated attention pooled for multi-instance learning 

    Input: 
    - x (B, N, C, H, W) and mask (B, N) 
    Output:
    - pooled embedding (B, 512)
    '''

    def __init__(
        self, 
        in_channels: int = 5, 
        attn_dim: int = 256, 
        attn_dropout: float = 0.0, 
        weights=None
    ): 
        super().__init__()
        self.in_channels = int(in_channels)

        if weights is None: 
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 

        backbone  = tvm.resnet18(weights=weights)
        orig_conv = backbone.conv1
        new_conv  = nn.Conv2d(
            self.in_channels, 64,  kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad(): 
            k = orig_conv.weight 

            n = min(3, self.in_channels)
            new_conv.weight[:, :n].copy_(k[:, :n])

            if self.in_channels > n: 
                filler = k.mean(dim=1, keepdim=True).repeat(1, self.in_channels - n, 1, 1)
                new_conv.weight[:, n:].copy_(filler)

        backbone.conv1 = new_conv 

        backbone.fc = nn.Identity() 
        self.backbone = backbone 

        self.pool   = GatedAttentionPooling(
            in_dim=512,
            attn_dim=attn_dim,
            attn_dropout=attn_dropout
        )
        self.out_dim = 512 

    def forward(self, x: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor: 
        if x.ndim != 4: 
            raise ValueError(f"expected x (T, C, H, W), got {x.shape}")

        t, c, h, w = x.shape 
        if c != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {c}")

        feats = self.backbone(x)
        
        batch_size = int(batch_indices.max().item()) + 1

        pooled = self.pool(feats, batch_indices, batch_size)
        return pooled 


class NonlinearProjector(nn.Module): 
    '''
    Nonlinear residual projector w/ optional bottleneck. Uses pre-norm residual blocks 
    '''

    def __init__(
        self,
        in_dim, 
        out_dim,
        depth=2,
        dropout=0.0 
    ): 

        super().__init__()

        self.in_dim  = in_dim 
        self.out_dim = out_dim 

        if in_dim == out_dim: 
            self.in_proj = nn.Identity() 
        else: 
            self.in_proj = nn.Linear(in_dim, out_dim)

        self.blocks = nn.Sequential(*[
            PreNormResBlock(out_dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x): 
        x = self.in_proj(x)
        return self.blocks(x)


class MILOrdinalHead(nn.Module):
    '''
    Hybrid Ordinal classifier head for MIL embeddings  
    
    Outputs: 
    - emb: (B, fc_dim)
    - logits: (B, n_classes - 1)
    - proj: (B, supcon_dim)
    '''

    def __init__(
        self,
        in_dim: int, 
        fc_dim: int, 
        n_classes: int, 
        dropout: float = 0.15, 
        supcon_dim: int | None = None, # optional for detaching at inference 
        use_logit_scaler: bool = True, # platt/temperature scaling 
        logit_scale_max: float = 30.0, 
        reduce_dim: int | None = None, 
        reduce_depth: int = 2, 
        reduce_dropout: float = 0.0
    ):
        super().__init__()
        if n_classes < 2: 
            raise ValueError("ordinal head requires n_classes >= 2")

        self.reducer = None 
        feat_dim     = in_dim 
        if reduce_dim is not None and reduce_dim != in_dim: 
            self.reducer = NonlinearProjector(
                in_dim=in_dim,
                out_dim=reduce_dim,
                depth=reduce_depth,
                dropout=reduce_dropout
            )
            feat_dim = reduce_dim 

        self.fc     = nn.Linear(feat_dim, fc_dim)
        self.act    = nn.GELU() 
        self.drop   = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.out    = nn.Linear(fc_dim, 1, bias=False)
        if use_logit_scaler: 
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
        else: 
            self.register_parameter("logit_scale", None)
        self.logit_scale_max = logit_scale_max 
        self.cut_anchor = nn.Parameter(torch.tensor(0.0))
        self.cut_deltas = nn.Parameter(torch.ones(n_classes - 2) * 0.35)

        self.proj = None 
        if supcon_dim is not None: 
            self.proj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, supcon_dim)
            )

        self.out_dim = fc_dim 

    def forward(self, feats: torch.Tensor): 
        if self.reducer is not None: 
            feats = self.reducer(feats)

        # logit cut logic 
        deltas = F.softplus(self.cut_deltas)
        cuts   = self.cut_anchor - torch.cat([
            torch.zeros(1, device=deltas.device, dtype=deltas.dtype), torch.cumsum(deltas, dim=0)
        ]) 

        emb    = feats  
        feat_v = self.drop(self.act(self.fc(emb)))
        score  = self.out(feat_v)
        if self.logit_scale is not None: 
            scale = F.softplus(self.logit_scale).clamp(min=1e-6, max=self.logit_scale_max)
            score = score * scale
        logits = score + cuts 
        proj   = self.proj(feats) if self.proj is not None else None 
        return emb, logits, proj 

# ---------------------------------------------------------
# Lightweight CNN backbone for CASM-MIL model  
# ---------------------------------------------------------

class SEBlock(nn.Module): 
    '''
    Squeeze-Excitation block for use in CNN models 
    '''
    def __init__(
        self,
        channels: int, 
        reduction: int = 8, 
        min_hidden: int = 16
    ): 
        super().__init__()
        hidden = max(channels // reduction, min_hidden)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid() 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        b, c, _, _ = x.shape 
        y = self.pool(x).flatten(1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEResBlock(nn.Module): 
    '''
    Combination Squeeze-Excitation and Residual Block for use in CNN models 
    '''

    def __init__(
        self,
        in_ch: int, 
        out_ch: int, 
        *,
        stride: int = 1, 
        se_reduction: int = 16, 
        dropout: float
    ): 
        super().__init__()
        
        if stride != 1 or in_ch != out_ch: 
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_ch)
            )
        else: 
            self.skip = nn.Identity()

        self.act = nn.GELU() 

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False), 
            nn.BatchNorm2d(out_ch),
            nn.ReLU(), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), 
            nn.BatchNorm2d(out_ch),
            SEBlock(out_ch, reduction=se_reduction), 
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id  = self.skip(x)
        out = self.net(x)
        return self.act(out + id)

class TinyDenseSE(nn.Module): 
    '''
    Patch encoder meant for 32x32 patches -> embedding vectors in LightweightBackbone 

    Uses chained SE + Residual blocks
    '''

    def __init__(
        self,
        *,
        in_channels: int, 
        embed_dim: int, 
        base_channels: int = 16, 
        se_reduction: int = 16,
        block_dropout: float = 0.0
    ): 
        super().__init__() 

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 3

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU() 
        )


        self.blocks = nn.Sequential(
            SEResBlock(c1, c1, stride=1, se_reduction=se_reduction, dropout=block_dropout), 
            SEResBlock(c1, c2, stride=2, se_reduction=se_reduction, dropout=block_dropout), 
            SEResBlock(c2, c3, stride=2, se_reduction=se_reduction, dropout=block_dropout), 
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(c3, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

    @staticmethod 
    def init_weights(m): 
        if isinstance(m, nn.Conv2d): 
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d): 
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.trunc_normal_(m.weight, std=0.02) 
            if m.bias is not None: 
                nn.init.zeros_(m.bias)

class LightweightBackbone(nn.Module): 

    '''
    Lightweight CNN model meant to extract features from small patches of images for downstream 
    usage in self-supervised learning. Leverages residual blocks as well as squeeze-excitation
    blocks.
    '''

    def __init__(
        self, 
        in_channels=1,
        embed_dim=64, 
        patch_size=32,
        anchor_stats: list[float] | None = None,
        *,
        base_channels: int = 16, 
        se_reduction: int = 16, 
        block_dropout: float = 0.0 
    ): 
        super().__init__()
        
        self.patch_size = patch_size 
        self.embed_dim  = embed_dim
        self.encoder    = TinyDenseSE(
            in_channels=in_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
            se_reduction=se_reduction,
            block_dropout=block_dropout
        )

        if anchor_stats is not None: 
            mean = torch.as_tensor(anchor_stats[0], dtype=torch.float32) 
            std  = torch.as_tensor(anchor_stats[1], dtype=torch.float32) 
        else: 
            mean, std = (torch.empty(0), torch.empty(0)) 

        self.register_buffer("patch_mean", mean)
        self.register_buffer("patch_std", std)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor: 
        if tiles.ndim != 4: 
            raise ValueError(f"expected (B, C, H, W), got {tuple(tiles.shape)}")

        patches = self.unfold(tiles)
        embs    = self.encoder(patches)
        return embs

    def unfold(self, tiles): 
        B, C, H, W = tiles.shape 
        P = self.patch_size 
        if H % P != 0 or W % P != 0: 
            raise ValueError(f"Tile size must be divisible by patch_size={P}. Got=(){H},{W})")

        patches = F.unfold(tiles, kernel_size=P, stride=P)
        L       = patches.shape[-1]
        return patches.transpose(1, 2).reshape(B * L, C, P, P) 

# ---------------------------------------------------------
# Cascaded Attention Hyper Graph Attention Network 
# Operable with graph generated by the Hypergraph builder 
# ---------------------------------------------------------

class HyperGATStack(nn.Module): 

    def __init__(
        self,
        in_dim, 
        hidden_dim,
        n_layers=1, 
        n_heads=1,
        n_node_types=3,
        n_edge_types=3,
        attn_dropout=0.0,
        dropout=0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList() 
        self.norms  = nn.ModuleList() 
        self.drop   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity() 

        dims = [in_dim] + [hidden_dim] * n_layers

        self.skip_proj = nn.ModuleList() 

        for i in range(n_layers): 
            self.layers.append(
                MultiheadCascadedHyperGAT(
                    in_dim=dims[i],
                    out_dim=hidden_dim,
                    n_heads=n_heads,
                    n_node_types=n_node_types,
                    n_edge_types=n_edge_types,
                    attn_dropout=attn_dropout
                )
            ) 
            self.norms.append(nn.LayerNorm(hidden_dim))
            if dims[i] == hidden_dim:
                self.skip_proj.append(nn.Identity())
            else: 
                self.skip_proj.append(nn.Linear(dims[i], hidden_dim, bias=False))

        self.out_dim = hidden_dim 
        
    def forward(self, x, node_types, edge_type, node_idx, edge_idx): 
        # resnet style stacking on GAT layers 
        h = x 
        for gat, ln, skip in zip(self.layers, self.norms, self.skip_proj): 
            h_update, _ = gat(h, node_types, edge_type, node_idx, edge_idx)
            h = skip(h) + self.drop(h_update)
            h = ln(h)

        return h 

class MultiheadCascadedHyperGAT(nn.Module):

    '''
    Performs cascaded attention on hypergraphs derived from chunked p95 thresholds. 
    Chunks are stratifed and connected based on three types. 
    '''

    def __init__(
        self, 
        in_dim, 
        out_dim, 
        n_heads=1,
        n_node_types=3,
        n_edge_types=3,
        type_query_dim=64,
        attn_dropout=0.0 
    ): 
        super().__init__()

        self.n_heads = n_heads 
        self.out_dim = out_dim 
        head_dim     = out_dim // n_heads 
        if out_dim % n_heads != 0: 
            raise ValueError("out_diim must be divisible by n_heads")

        self.heads   = nn.ModuleList([
            CascadedHyperGAT(
                in_dim=in_dim,
                out_dim=head_dim,
                n_node_types=n_node_types,
                n_edge_types=n_edge_types,
                type_query_dim=type_query_dim
            ) for _ in range(n_heads)
        ])
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0.0 else nn.Identity() 

    def forward(self, x, node_types, edge_type, node_idx, edge_idx): 
        node_outs = []
        edge_outs = []
        for head in self.heads: 
            node_feat, edge_feat = head(x, node_types, edge_type, node_idx, edge_idx)
            node_outs.append(node_feat)
            edge_outs.append(edge_feat)
        node_out = torch.cat(node_outs, dim=1)
        edge_out = torch.cat(edge_outs, dim=1)
        return node_out, edge_out 


class CascadedHyperGAT(nn.Module):

    '''
    Performs cascaded attention on hypergraphs derived from chunked p95 thresholds. 
    Chunks are stratifed and connected based on three types. 
    '''

    def __init__(
        self, 
        in_dim, 
        out_dim, 
        n_node_types=3,
        n_edge_types=3,
        type_query_dim=64
    ): 
        super().__init__()

        self.n_node_types      = n_node_types
        self.n_edge_types      = n_edge_types
        
        # type level attention 
        self.type_query        = nn.Embedding(n_node_types, type_query_dim)
        self.type_attention_fc = nn.Sequential(
            nn.Linear(in_dim + type_query_dim, 32), 
            nn.Tanh(), 
            nn.Linear(32, 1)
        )
        
        # node-level interactions 
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

        # hyperedge context per spatial, semantic, global 
        self.edge_context = nn.Embedding(n_edge_types, out_dim)

    def forward(self, x, node_types, edge_type, node_idx, edge_idx): 
        '''
        Caller Provides:
        - x (N, in_dim),
        - node_types: (N,)
        - edge_type (num_edges,) hyperedge type id (0=spatial,1=semantic,2=global)
        '''
        num_edges             = edge_type.shape[0]

        # type level attention 
        type_sigs   = self.type_query(node_types)
        type_scores = self.type_attention_fc(torch.cat([x, type_sigs], dim=1))
        alpha_type  = torch.sigmoid(type_scores)

        # node level attention 
        Q = self.query(x)
        V = self.value(x)

        # edge context per incidence 
        context = self.edge_context(edge_type)[edge_idx]
        Qi     = Q[node_idx]

        alpha_node = (Qi * context).sum(dim=1)
        alpha_node = F.leaky_relu(alpha_node, 0.2)

        # cascade gate 
        alpha = alpha_node * alpha_type[node_idx].squeeze(1)
        attn  = scatter_softmax(alpha, edge_idx, dim=0)

        edge_feat = scatter_add(attn.unsqueeze(1) * V[node_idx], edge_idx, 
                                dim=0, dim_size=num_edges)
        node_feat = scatter_add(attn.unsqueeze(1) * edge_feat[edge_idx], node_idx,
                                dim=0, dim_size=x.size(0)) 

        return node_feat, edge_feat 

# ---------------------------------------------------------
# Tabular MLP Modules 
# ---------------------------------------------------------

class TransformerProjector(nn.Module): 

    '''
    Transformer Architecture meant to project inputs onto latent space before mixing & passing 
    through Deep Residual MLP 
    '''
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        d_model: int = 64, 
        n_heads: int = 4, 
        n_layers: int = 2, 
        dropout: float = 0.1, 
        attn_dropout: float = 0.1,
        pre_norm: bool = True       # if used with ResidualMLP must stay True for gradients  
    ):
        super().__init__()

        self.in_dim     = in_dim 
        self.out_dim    = out_dim 
        self.d_model    = d_model 
        self.num_tokens = in_dim  

        self.feature_tokenizer = nn.Linear(1, d_model, bias=True)
        self.feature_id_embed  = nn.Parameter(torch.zeros(1, in_dim, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_norm 
        )

        layer.self_attn.dropout = attn_dropout 
        
        self.encoder  = nn.TransformerEncoder(
            layer, 
            num_layers=n_layers,
            enable_nested_tensor=False
        )
        self.out_proj = nn.Linear(d_model, out_dim, bias=True)

        nn.init.trunc_normal_(self.feature_id_embed, std=0.02)

    def forward(self, x): 
        if x.dim() != 2: 
            raise ValueError(f"expecteed (B, F), got {x.shape}")
        if x.size(1) != self.in_dim: 
            raise ValueError(f"feature dim mismatch: got {x.size(1)}, expected {self.in_dim}")

        tokens  = self.feature_tokenizer(x.unsqueeze(-1))
        tokens += self.feature_id_embed

        enc     = self.encoder(tokens)
        pooled  = enc.mean(dim=1)
        return self.out_proj(pooled)


class Mixer(nn.Module): 
    '''
    Mixup logic for taking interpolation of two samples. Acts as an overfitting relaxer 
    to allow for a smoother interpolation of a learned manifold/latent variable. 
    '''

    def __init__(
        self, 
        *,
        alpha: float = 0.2, 
        mix_mult: int = 4, 
        min_lambda: float = 0.0, 
        with_replacement: bool = True 
    ): 
        super().__init__()
        if alpha <= 0: 
            raise ValueError("alpha must be > 0")
        if mix_mult < 1: 
            raise ValueError("mix_mult must be >= 1")
        if min_lambda < 0.0 or min_lambda >= 0.5: 
            raise ValueError("min_lambda must be in [0.0, 0.5)")

        self.alpha            = alpha 
        self.mix_mult         = mix_mult
        self.with_replacement = with_replacement
        self.min_lambda       = min_lambda

        self.idx_a_:      torch.Tensor | None = None 
        self.idx_b_:      torch.Tensor | None = None 
        self.mix_lambda_: torch.Tensor | None = None 
        
    @property 
    def is_fitted(self) -> bool: 
        return (
            self.idx_a_ is not None and self.idx_b_ is not None and self.mix_lambda_ is not None
        )

    def fit(
        self,
        y_bucket: torch.Tensor,
        *,
        generator: torch.Generator | None = None
    ) -> "Mixer": 
        if y_bucket.ndim == 0: 
            raise ValueError("y_bucket must be 1d batch tensor")

        y = y_bucket.reshape(-1)
        b = int(y.numel())
        if b <= 1: 
            raise ValueError(f"batch size must be > 1, got {b}")

        device = y.device 
        n_mix  = max(1, b * self.mix_mult) 
        
        uniform = torch.full((b, ), 1.0 / b, device=device)

        idx_a   = torch.multinomial(
            uniform,
            n_mix, 
            replacement=self.with_replacement,
            generator=generator
        )

        idx_b   = torch.multinomial(
            uniform,
            n_mix,
            replacement=self.with_replacement,
            generator=generator
        )

        mix_lambda = torch.distributions.Beta(
            self.alpha, self.alpha
        ).sample((n_mix, )).to(device)

        if self.min_lambda > 0.0: 
            lo = self.min_lambda 
            hi = 1.0 - self.min_lambda 
            mix_lambda = mix_lambda.clamp(lo, hi)

        self.idx_a_      = idx_a 
        self.idx_b_      = idx_b 
        self.mix_lambda_ = mix_lambda
        return self 

    def transform(self, x: torch.Tensor) -> torch.Tensor: 
        if not self.is_fitted: 
            raise RuntimeError("call fit() before transform()")

        if x.ndim < 2: 
            raise ValueError(f"x must be at least 2d (b, ...), got {tuple(x.shape)}")
        if not torch.is_floating_point(x): 
            raise TypeError("x must be a floating tensor")

        idx_a = self.idx_a_ 
        idx_b = self.idx_b_ 
        lam   = self.mix_lambda_
        assert idx_a is not None and idx_b is not None and lam is not None 
    
        if x.size(0) <= int(torch.max(torch.stack([idx_a, idx_b])).item()): 
            raise ValueError("stored mix indices exceed current batch size")

        idx_a = idx_a.to(device=x.device)
        idx_b = idx_b.to(device=x.device)
        lam   = lam.to(device=x.device, dtype=x.dtype)

        lam_shape = (lam.numel(), ) + (1, ) * (x.ndim - 1) 
        lam = lam.view(*lam_shape)

        x_mix = lam * x[idx_a] + (1.0 - lam) * x[idx_b]
        return x_mix 

    def fit_transform(
        self,
        x: torch.Tensor,
        y_bucket: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.fit(y_bucket, generator=generator)
        x_mix = self.transform(x)

        assert self.idx_a_ is not None and self.idx_b_ is not None 
        assert self.mix_lambda_ is not None

        return x_mix, self.idx_a_, self.idx_b_, self.mix_lambda_

    def plan(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("no mix plan available; call fit() first")

        assert self.idx_a_ is not None and self.idx_b_ is not None 
        assert self.mix_lambda_ is not None

        return self.idx_a_, self.idx_b_, self.mix_lambda_


class PreNormResBlock(nn.Module):
    '''
    X -> LayerNorm -> GELU -> Drop -> Linear(zero-initialized) -> Drop -> x + ...
    Preserves identity path for Gradients, mitigating vanishing gradients for deep networks 
    '''

    def __init__(
        self, 
        dim: int, 
        dropout: float = 0.0
    ): 
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.GELU() 
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x): 
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h 


class ResidualMLP(nn.Module): 
    '''
    Residual Backbone for Tabular data and tabular semantic embeddings  
    '''

    def __init__(
        self,
        in_dim: int, 
        hidden_dim: int = 256, 
        depth: int = 6, 
        dropout: float = 0.1, 
        out_dim: Optional[int] = None, 
        zero_head_init: bool = True, 
    ):
        super().__init__()
        self.in_dim     = in_dim 
        self.hidden_dim = hidden_dim
        self.depth      = depth 
        self.out_dim    = hidden_dim if out_dim is None else out_dim

        self.proj   = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            PreNormResBlock(hidden_dim, dropout=dropout) for _ in range(depth)
        ]) 

        self.norm   = nn.LayerNorm(hidden_dim)
        self.head   = nn.Linear(hidden_dim, self.out_dim)

        if zero_head_init: 
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else: 
            nn.init.xavier_uniform_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x): 
        x = self.proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)

# ---------------------------------------------------------
# Semantic MLP models for SSFE 
# ---------------------------------------------------------

class SemanticMLP(nn.Module): 
    '''
    Shallow semantic projector for SpatialSSFE
    '''

    def __init__(
        self,
        in_dim: int, 
        hidden_dim: int,
        out_dim: int, 
        dropout: float, 
    ): 
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(), 
            nn.Linear(hidden_dim, out_dim)
        )

        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        batch_indices: torch.Tensor, 
        batch_size: int 
    ): 
        if x.ndim != 2: 
            raise ValueError(f"expected (N, d), got {tuple(x.shape)})")

        node  = self.net(x)
        bag   = node.new_zeros((batch_size, node.size(1)))
        count = node.new_zeros((batch_size, 1))
        bag.index_add_(0, batch_indices, node)
        count.index_add_(0, batch_indices, node.new_ones((batch_indices.numel(), 1)))
        bag   = bag / count.clamp_min(1.0)
        return node, bag 

# ---------------------------------------------------------
# Probabilistic Head (ordinal probit estimation) 
# ---------------------------------------------------------

class ProbabilisticRankHead(nn.Module): 
    ''' 
    Head for heteroscedastic rank prediction 
    '''

    def __init__(
        self,
        in_dim: int, 
        hidden_dim: Optional[int], 
        dropout: float, 
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0 
    ): 
        super().__init__()
        self.log_var_min = log_var_min 
        self.log_var_max = log_var_max 

        if hidden_dim is None: 
            self.trunk = nn.Identity() 
            d = in_dim 
        else: 
            self.trunk = nn.Sequential(
                nn.LayerNorm(in_dim), 
                nn.Linear(in_dim, hidden_dim), 
                nn.GELU(), 
                nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
            )
            d = hidden_dim 

        self.mu_head = nn.Linear(d, 1)
        self.lv_head = nn.Linear(d, 1)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.lv_head.weight)
        nn.init.constant_(self.lv_head.bias, 5.0)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        z  = self.trunk(h)
        mu = self.mu_head(z).squeeze(-1)
        log_var = self.lv_head(z).squeeze(-1)
        log_var = log_var.clamp(min=self.log_var_min, max=self.log_var_max)
        return mu, log_var 

# --------------------------------------------------------
# Deep Fusion Model 
# --------------------------------------------------------

class DeepFusionMLP(nn.Module): 

    '''
    Deep Branch for Wide & Deep fusion over expert embeddings
    '''

    def __init__(
        self,
        *,
        expert_dims: dict[str, int], 

        d_model: int, 
        n_heads: int, 
        n_layers: int, 
        ff_mult: int, 
        transformer_dropout: float, 
        transformer_attn_dropout: float, 
        pre_norm: bool = True, 
        gate_floor: float = 0.05, 
        gate_num_tokens: int, 
        gateway_hidden_dim: int, 

        hidden_dim: int, 
        depth: int, 
        dropout: float, 
        trunk_out_dim: Optional[int], 

        head_hidden_dim: Optional[int], 
        head_dropout: float, 
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
    ):
        super().__init__()
        
        self.gate = MoETransformerGate(
            expert_dims=expert_dims, 
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_mult=ff_mult,
            dropout=transformer_dropout,
            attn_dropout=transformer_attn_dropout,
            pre_norm=pre_norm,
            gate_floor=gate_floor,
            num_tokens=gate_num_tokens,
            gateway_hidden_dim=gateway_hidden_dim
        )

        self.trunk = ResidualMLP(
            in_dim=d_model,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            out_dim=trunk_out_dim
        )

        self.head = ProbabilisticRankHead(
            in_dim=self.trunk.out_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            log_var_min=log_var_min,
            log_var_max=log_var_max 
        )

        self.out_dim = self.trunk.out_dim 

    def forward_features(self, experts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: 
        cls_out, token_out, gate = self.gate(experts)
        emb = self.trunk(cls_out)

        return {
            "embedding": emb, 
            "cls": cls_out, 
            "tokens": token_out, 
            "gate": gate 
        }

    def extract(self, experts: dict[str, torch.Tensor]) -> torch.Tensor: 
        return self.forward_features(experts)["embedding"]

    def forward(
        self, 
        experts: dict[str, torch.Tensor],
        return_features: bool = False 
    ) -> dict[str, torch.Tensor]: 
        feats = self.forward_features(experts)
        mu_deep, log_var_deep = self.head(feats["embedding"])

        out = {
            "mu_deep": mu_deep, 
            "log_var_deep": log_var_deep
        }

        if return_features: 
            out.update(feats)
        return out 

# ---------------------------------------------------------
# Wide Model - log-log ridge regressor 
# ---------------------------------------------------------

class WideRidgeRegressor(nn.Module): 
    '''
    
    '''
    def __init__(
        self,
        in_dim: int, 
        *,
        bias: bool = True, 
        l2_alpha: float = 1e-2, 
        init_log_var: float = -1.0, 
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
    ):
        super().__init__()
        
        self.in_dim        = in_dim 
        self.l2_alpha      = l2_alpha
        self.log_var_min   = log_var_min 
        self.log_var_max   = log_var_max 
        self.linear        = nn.Linear(self.in_dim, 1, bias=bias)
        self.log_var_param = nn.Parameter(
            torch.tensor(float(init_log_var), dtype=torch.float32)
        )

        self.reset_parameters() 

    def reset_parameters(self): 
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None: 
            nn.init.zeros_(self.linear.bias)

    def ridge_penalty(self) -> torch.Tensor: 
        return 0.5 * self.l2_alpha * self.linear.weight.pow(2).sum()

    def forward(
        self, 
        x: torch.Tensor,
        *, 
        return_features: bool = False
    ) -> dict[str, torch.Tensor]: 
        mu      = self.linear(x).squeeze(-1)
        lv      = self.log_var_param.clamp(self.log_var_min, self.log_var_max)
        log_var = lv.expand_as(mu)

        out = {
            "mu_wide": mu, 
            "log_var_wide": log_var 
        }
        if return_features:
            out["h_wide"] = x 
        return out  
