#!/usr/bin/env python3 
# 
# loss.py  Andrew Belles  Feb 13th, 2026 
# 
# Implementation of loss functions and summation interface to handle complex weighted terms 
# 
# 

import torch 

import torch.nn            as nn 

import numpy               as np 

import torch.nn.functional as F 

from scipy.spatial         import KDTree

from abc                   import ABC, abstractmethod 

from dataclasses           import dataclass, field 

from typing                import Any, Callable, Mapping 

from numpy.typing          import NDArray

from utils.helpers         import (
    as_1d_f64,
    as_2d_f64,
    to_batch_tensor,
    weighted_mean
)

# --------------------------------------------------------- 
# Class Weighting Methods 
# --------------------------------------------------------- 

@dataclass 
class KESConfig: 
    '''
    Kernel smoothing + Effective number of samples weighting configuration 
    '''
    moran_k: int      = 16 
    moran_clip: float = 0.95 
    ess_floor: float  = 2.0 
    eps: float        = 1e-9 


class KernelEffectiveSamples:
    '''
    Gaussian Kernel Smoothing via silverman bandwidth and empirical determination of 
    effective number of samples weighting for soft rank class labels 

    Fits on training split only. Pipeline: 
    - estimates Moran's I from spatial neighbors to determine beta 
    - fits 1D Gaussian KDE on y_rank 
    - maps dentity to a pseudo-count.
    - computes ENS weights
    '''

    def __init__(self, config: KESConfig | None = None): 
        self.config = config or KESConfig() 

        self.x_ref_:   NDArray | None = None 
        self.h_:       float   | None = None 
        self.beta_:    float   | None = None 
        self.moran_i_: float   | None = None 
        self.ess_:     float   | None = None 

    # -----------------------------------------------------
    # Properties/Getters  
    # -----------------------------------------------------

    @property 
    def beta(self) -> float: 
        if self.beta_ is None: 
            raise RuntimeError("call fit()")
        return self.beta_ 

    @property 
    def moran_i(self) -> float: 
        if self.moran_i_ is None: 
            raise RuntimeError("call fit()")
        return self.moran_i_ 

    @property 
    def ess(self) -> float: 
        if self.ess_ is None: 
            raise RuntimeError("call fit()")
        return self.ess_ 

    @property 
    def bandwidth(self) -> float: 
        if self.h_ is None: 
            raise RuntimeError("call fit()")
        return self.h_ 

    def fit(
        self,
        y_rank_train: NDArray,
        coords_train: NDArray
    ) -> "KernelEffectiveSamples": 
        x = as_1d_f64(y_rank_train)
        c = as_2d_f64(coords_train)
        if x.shape[0] != c.shape[0]: 
            raise ValueError("y_rank_train and coords_train size mismatch.")

        if x.shape[0] < 3: 
            raise ValueError("need >= 3 training samples")

        self.x_ref_   = x
        self.h_       = self.compute_bandwidth(x)
        self.moran_i_ = self.estimate_moran_i(x, c, k=self.config.moran_k)
        self.ess_     = self.moran_to_ess(self.moran_i, n_total=x.shape[0])
        self.beta_    = self.ess_to_beta(self.ess_)
        return self 

    def transform(
        self,
        y_rank: NDArray
    ) -> NDArray: 
        if self.x_ref_ is None or self.h_ is None or self.beta_ is None: 
            raise RuntimeError("call fit() before transform()")

        x_eval = as_1d_f64(y_rank)
        p      = self.kde_density_1d(x_eval, self.x_ref_, self.h_)
        w      = self.density_to_weights(
            p,
            n_ref=self.x_ref_.shape[0],
            h=self.h_,
            beta=self.beta_,
            eps=self.config.eps,
        )
        return w.astype(np.float32, copy=False)

    def fit_transform(self, y_rank_train: NDArray, coords_train: NDArray) -> NDArray: 
        return self.fit(y_rank_train, coords_train).transform(y_rank_train)

    # -----------------------------------------------------
    # Helpers 
    # -----------------------------------------------------

    def compute_bandwidth(self, x: NDArray) -> float: 
        n     = max(int(x.shape[0]), 2)
        std   = float(np.std(x, ddof=1))
        q75, q25 = np.percentile(x, [75.0, 25.0])
        iqr   = float(q75 - q25)
        sigma = min(std, iqr / 1.34) if iqr > 0 else std 
        h     = 0.9 * sigma * np.power(n, -0.2)
        return h
        

    def moran_to_ess(self, p: float, n_total: int) -> float: 
        ess = float(n_total) * (1.0 - p) / (1.0 + p) 
        ess = max(ess, self.config.ess_floor)
        return ess 

    def ess_to_beta(self, ess: float) -> float: 
        beta = 1.0 - (1.0 / ess)
        return float(np.clip(beta, 0.9, 0.9999))

    @staticmethod 
    def estimate_moran_i(y: NDArray, coords: NDArray, k: int) -> float: 
        n     = y.shape[0]
        k_eff = min(max(int(k), 1), max(n - 1, 1)) 

        tree      = KDTree(coords)
        _, nn_idx = tree.query(coords, k=k_eff + 1)
        nn_idx    = nn_idx[:, 1:]

        y_center  = y - y.mean() 
        denom     = np.sum(y_center * y_center)
        if denom <= 0: 
            return 0.0 

        yj  = y_center[nn_idx]
        num = np.sum(y_center[:, None] * yj)
        S0  = float(n * k_eff) 

        I = (n / S0) * (num / denom)
        I = float(np.clip(I, -0.999, 0.999)) # clip unfeasible values 
        return I 

    @staticmethod 
    def kde_density_1d(
        x_eval: NDArray,
        x_ref: NDArray,
        h: float, 
        *,
        chunk_size: int = 4096 
    ): 
        norm  = 1.0 / (h * np.sqrt(2.0 * np.pi))
        out   = np.empty(x_eval.shape[0], dtype=np.float64)

        for s in range(0, x_eval.shape[0], chunk_size): 
            e = min(s + chunk_size, x_eval.shape[0])
            z = (x_eval[s:e, None] - x_ref[None, :]) / h 
            out[s:e] = norm * np.exp(-0.5 * (z**2)).mean(axis=1)
        return out 

    @staticmethod
    def density_to_weights(
        p: NDArray,
        *,
        n_ref: int, 
        h: float,
        beta: float, 
        eps: float, 
    ) -> NDArray:
        n_local = np.clip(p * float(n_ref) * h, 1.0, None)
        ens     = (1.0 - np.power(beta, n_local)) / max(1.0 - beta, eps)
        ens     = np.clip(ens, eps, None) 

        w = 1.0 / ens 
        w = w / max(w.mean(), eps) 
        return w   

# --------------------------------------------------------- 
# Loss Metadata 
# --------------------------------------------------------- 

LossContext = Mapping[str, Any]
WeightSpec  = float | str | Callable[[LossContext], torch.Tensor | float]

@dataclass 
class LossValue: 
    name: str 
    raw: torch.Tensor 
    weighted: torch.Tensor 
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass 
class ComposedLossValues: 
    total: torch.Tensor 
    raw: dict[str, torch.Tensor]
    weighted: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]

# --------------------------------------------------------- 
# Contract for Loss Function & Composition of Loss Functions  
# --------------------------------------------------------- 

class WeightedLossTerm(nn.Module, ABC): 
    '''
    Contract for composable loss terms. 
    - subclasses implement 'compute(context)' and return scalar loss & metrics 
    '''
    required_keys: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        weight: WeightSpec = 1.0, 
        reduction: str = "mean"
    ): 
        super().__init__()
        if reduction not in ("mean", "sum"): 
            raise ValueError(f"invalid reduction={reduction}, expected 'mean'/'sum'")
        self.name      = name 
        self.weight    = weight 
        self.reduction = reduction 

    def validate_context(self, context: LossContext): 
        missing = [k for k in self.required_keys if k not in context]
        if missing: 
            raise KeyError(f"[{self.name}] missing context keys: {missing}")

    def resolve_weight(self, context: LossContext, ref: torch.Tensor) -> torch.Tensor: 
        if callable(self.weight): 
            w = self.weight(context)
        elif isinstance(self.weight, str): 
            if self.weight not in context: 
                raise KeyError(f"[{self.name}] weight key in '{self.weight}' not in context")
            w = context[self.weight]
        else: 
            w = self.weight 

        if torch.is_tensor(w): 
            return w.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(float(w), device=ref.device, dtype=ref.dtype)

    def reduce(self, loss: torch.Tensor) -> torch.Tensor: 
        if loss.ndim == 0: 
            return loss 
        if self.reduction == "sum": 
            return loss.sum() 
        return loss.mean() 

    def forward(self, **context: Any) -> LossValue: 
        self.validate_context(context)
        raw, metrics = self.compute(context)
        if not torch.is_tensor(raw): 
            raise TypeError(f"[{self.name}] compute() must return tensor loss")
        raw = self.reduce(raw)
        w   = self.resolve_weight(context, raw)
        return LossValue(
            name=self.name,
            raw=raw,
            weighted=w * raw, 
            metrics=metrics
        )

    @abstractmethod 
    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: 
        raise NotImplementedError 


class LossComposer(nn.Module): 
    '''
    Sums weighted loss terms and returns raw/weighted breakdown. 
    '''

    def __init__(
        self,
        *terms: WeightedLossTerm
    ): 
        super().__init__()
        self.terms = nn.ModuleList(terms)

    def forward(self, **context: Any) -> ComposedLossValues:
        if len(self.terms) == 0: 
            raise ValueError("LossComposer requires at least one term.")

        total: torch.Tensor | None        = None 
        raw: dict[str, torch.Tensor]      = {}
        weighted: dict[str, torch.Tensor] = {}
        metrics: dict[str, torch.Tensor]  = {}

        for term in self.terms: 
            out = term(**context)

            raw[out.name]      = out.raw 
            weighted[out.name] = out.weighted 
            for mk, mv in out.metrics.items(): 
                metrics[f"{out.name}.{mk}"] = mv 

            total = out.weighted if total is None else (total + out.weighted)

        if total is None: 
            raise ValueError("total loss is invalid.")

        return ComposedLossValues(
            total=total, 
            raw=raw, 
            weighted=weighted,
            metrics=metrics
        )

class MixedLossAdapter(nn.Module): 
    '''
    Mixup wrapper for context based loss functions (WeightedLossTerm)

    Supports: 
    - LossComposer 
    - WeightedLossTerm
    '''

    def __init__(
        self,
        base_loss: nn.Module,
        *,
        target_pairs: Mapping[str, tuple[str, str]], 
    ): 
        super().__init__()

        if not isinstance(base_loss, (LossComposer, WeightedLossTerm)): 
            raise TypeError("base_loss must be from LossComposer or WeightedLossTerm")
        
        self.base_loss    = base_loss 
        self.target_pairs = dict(target_pairs)

    def forward(self, **context: Any) -> ComposedLossValues:
        if "mix_lambda" not in context: 
            raise KeyError("missing mix_lambda in context")

        ref = self.infer_ref_target(context)
        n   = ref.numel() 
        
        device = ref.device 
        dtype  = ref.dtype 

        p = to_batch_tensor(context["mix_lambda"], n=n, device=device, dtype=dtype)
        q = 1.0 - p 

        has_a      = "sample_weight_a" in context 
        has_b      = "sample_weight_b" in context 

        if has_a != has_b: 
            raise KeyError("mixed loss required sample weights for both pairs")

        if has_a: 
            sw_a_base = to_batch_tensor(context["sample_weight_a"], n=n, 
                                        device=device, dtype=dtype)
            sw_b_base = to_batch_tensor(context["sample_weight_b"], n=n, 
                                        device=device, dtype=dtype)
            sw_a = sw_a_base * p 
            sw_b = sw_b_base * q 
        else: 
            sw_a = p 
            sw_b = q 

        context_a = dict(context)
        context_b = dict(context)

        for target_key, (a_key, b_key) in self.target_pairs.items(): 
            if a_key not in context or b_key not in context: 
                raise KeyError(f"missing mix target keys for '{target_key}': ({a_key}, {b_key})")

            context_a[target_key] = context[a_key]
            context_b[target_key] = context[b_key]

        context_a["sample_weight"] = sw_a 
        context_b["sample_weight"] = sw_b 

        out_a = self.call_base(context_a)
        out_b = self.call_base(context_b)

        total    = out_a.total + out_b.total 
        raw      = self.sum_tensor_dicts(out_a.raw, out_b.raw)
        weighted = self.sum_tensor_dicts(out_a.weighted, out_b.weighted)

        metrics: dict[str, torch.Tensor] = {}
        for k, v in out_a.metrics.items(): 
            metrics[f"a.{k}"] = v
        for k, v in out_b.metrics.items(): 
            metrics[f"b.{k}"] = v

        return ComposedLossValues(
            total=total,
            raw=raw,
            weighted=weighted,
            metrics=metrics  
        )

    def infer_ref_target(self, context: LossContext) -> torch.Tensor: 
        for _, (a_key, _) in self.target_pairs.items(): 
            if a_key in context and torch.is_tensor(context[a_key]): 
                return context[a_key].reshape(-1)
        raise ValueError("unabled to infer batch metadata.")

    def call_base(self, context: dict[str, Any]) -> ComposedLossValues:
        out = self.base_loss(**context)
        return self.to_composed(out)

    @staticmethod 
    def to_composed(out: LossValue | ComposedLossValues) -> ComposedLossValues:
        if isinstance(out, ComposedLossValues): 
            return out 

        metrics = {f"{out.name}.{k}": v for k, v in out.metrics.items()}
        return ComposedLossValues(
            total=out.weighted,
            raw={out.name: out.raw},
            weighted={out.name: out.weighted},
            metrics=metrics
        )

    @staticmethod
    def sum_tensor_dicts(
        a: dict[str, torch.Tensor],
        b: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]: 
        keys = set(a.keys()) | set(b.keys()) # union 
        out: dict[str, torch.Tensor] = {}
        for k in keys: 
            if k in a and k in b: 
                out[k] = a[k] + b[k]
            elif k in a: 
                out[k] = a[k]
            else: 
                out[k] = b[k]
        return out 

# --------------------------------------------------------- 
# Self-Supervised Feature Extraction 
# --------------------------------------------------------- 

class ContrastiveLoss(WeightedLossTerm): 
    required_keys = ("sem_bag", "st_bag", "stats_raw", "node_batch_idx")

    def __init__(
        self, 
        *, 
        sem_proj: nn.Module, 
        st_proj: nn.Module,
        temperature: float, 
        active_eps: float = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "contrast"
    ):
        super().__init__(name, weight=weight, reduction="mean")
        self.sem_proj    = sem_proj 
        self.st_proj     = st_proj 
        self.temperature = temperature
        self.active_eps  = active_eps

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sem_bag        = context["sem_bag"]
        st_bag         = context["st_bag"]
        stats_raw      = context["stats_raw"]
        node_batch_idx = context["node_batch_idx"]

        if stats_raw is None: 
            raise ValueError(f"[contrast] stats_raw is required")

        z_sem  = F.normalize(self.sem_proj(sem_bag), dim=1)
        z_st   = F.normalize(self.st_proj(st_bag), dim=1)

        logits = torch.matmul(z_sem, z_st.T) / self.temperature 
        labels = torch.arange(z_sem.size(0), device=z_sem.device)

        loss_a = F.cross_entropy(logits, labels, reduction="none")
        loss_b = F.cross_entropy(logits.T, labels, reduction="none")

        patch_active = (stats_raw.abs().amax(dim=1) > self.active_eps).to(loss_a.dtype)

        if patch_active.numel() == loss_a.numel(): 
            bag_weights = patch_active 
        else: 
            if node_batch_idx is None: 
                raise ValueError("[contrast] node_batch_idx is required.")

            if node_batch_idx.numel() != patch_active.numel(): 
                raise ValueError("")

            idx = node_batch_idx.to(device=loss_a.device, dtype=torch.long)
            bag_weights = loss_a.new_zeros((loss_a.numel(), ))
            bag_counts  = loss_a.new_zeros((loss_a.numel(), ))

            bag_weights.index_add_(0, idx, patch_active)
            bag_counts.index_add_(0, idx, torch.ones_like(patch_active))
            bag_weights = bag_weights / bag_counts.clamp_min(1.0)

        denom = bag_weights.sum().clamp_min(1.0)
        per   = 0.5 * (loss_a + loss_b)
        raw   = (per * bag_weights).sum() / denom 

        metrics = {
            "bag_active_mean": bag_weights.mean().detach(),
            "patch_active_mean": patch_active.mean().detach()
        }
        return raw, metrics 

class SwappedPredictionLoss(WeightedLossTerm):
    '''
    Measures kl-divergence via sinkhorn assignment
    '''
    def __init__(
        self,
        *,
        sem_proj: nn.Module,
        st_proj: nn.Module,
        proto: nn.Module,
        temperature: float, 
        sinkhorn_epsilon: float = 0.05, 
        weight: WeightSpec = 1.0, 
        name: str = "cluster"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")

        self.sem_proj = sem_proj 
        self.st_proj  = st_proj 
        self.proto    = proto 

        self.temperature      = temperature 
        self.sinkhorn_epsilon = sinkhorn_epsilon

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sem_bag = context["sem_bag"]
        st_bag  = context["st_bag"]

        z_sem   = F.normalize(self.sem_proj(sem_bag), dim=1)
        z_st    = F.normalize(self.st_proj(st_bag), dim=1)

        s_sem   = self.proto(z_sem)
        s_st    = self.proto(z_st)

        with torch.no_grad(): 
            q_sem = sinkhorn_assign(s_sem, epsilon=self.sinkhorn_epsilon)
            q_st  = sinkhorn_assign(s_st, epsilon=self.sinkhorn_epsilon)

        l_sem = -(q_st * F.log_softmax(s_sem / self.temperature, dim=1)).sum(dim=1).mean() 
        l_st  = -(q_sem * F.log_softmax(s_st / self.temperature, dim=1)).sum(dim=1).mean()
        raw   = 0.5 * (l_sem + l_st)

        metrics = {
            "proto_entropy_sem": (
                -(q_sem * (q_sem.clamp_min(1e-9).log())).sum(dim=1).mean()
            ).detach(),
            "proto_entropy_st": (
                -(q_st * (q_st.clamp_min(1e-9).log()).sum(dim=1).mean()).detach()
            )
        }
        return raw, metrics 


class ReconstructionLoss(WeightedLossTerm): 

    required_keys = ("prep", "sem_node", "st_node")

    def __init__(
        self,
        *,
        sem_recon: nn.Module,
        st_recon: nn.Module,
        target_attr: str = "repr_target", 
        weight: WeightSpec = 1.0, 
        name: str = "recon"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")
        self.sem_recon   = sem_recon 
        self.st_recon    = st_recon 
        self.target_attr = target_attr  

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prep     = context["prep"]
        sem_node = context["sem_node"]
        st_node  = context["st_node"]

        target   = getattr(prep, self.target_attr)
        rec_sem  = self.sem_recon(sem_node)
        rec_st   = self.st_recon(st_node)

        l_sem = F.mse_loss(rec_sem, target)
        l_st  = F.mse_loss(rec_st, target)
        raw   = 0.5 * (l_sem + l_st)

        metrics = {
            "mse_sem": l_sem.detach(),
            "mse_st": l_st.detach()
        }
        return raw, metrics 

# ---------------------------------------------------------
# Ordinal and Heteroscedastic Regression Loss  
# ---------------------------------------------------------

class FusionGaussianMixin: 

    '''
    Wide and Deep Gaussian Terms for PoE Loss   
    '''

    def __init__(
        self, 
        *,
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
        var_floor:   float = 1e-6 
    ): 
        self.log_var_min = log_var_min 
        self.log_var_max = log_var_max 
        self.var_floor   = var_floor 

    def fuse(self, context): 
        y_ref  = context["y"].reshape(-1) if "y" in context else context["y_bin"].reshape(-1)
        n      = y_ref.numel() 
        device = y_ref.device 
        dtype  = (torch.float32 if y_ref.dtype not in (torch.float32, torch.float64) 
                  else y_ref.dtype) 

        # Deep mean and log variance 
        mu_d   = to_batch_tensor(context["mu_deep"], n=n, device=device, dtype=dtype)
        lv_d   = to_batch_tensor(context["log_var_deep"], n=n, device=device, dtype=dtype)
        
        # Wide 
        mu_w   = to_batch_tensor(context["mu_wide"], n=n, device=device, dtype=dtype)
        lv_w   = to_batch_tensor(context["log_var_wide"], n=n, device=device, dtype=dtype)

        lv_d   = lv_d.clamp(self.log_var_min, self.log_var_max) 
        lv_w   = lv_w.clamp(self.log_var_min, self.log_var_max) 

        tau_d  = torch.exp(-lv_d).clamp_min(self.var_floor)
        tau_w  = torch.exp(-lv_w).clamp_min(self.var_floor)
        tau    = (tau_d + tau_w).clamp_min(self.var_floor)


        alpha_d = tau_d / tau 
        alpha_w = tau_w / tau 
        mu      = alpha_d * mu_d + alpha_w * mu_w  
        var     = (1.0 / tau).clamp_min(self.var_floor)
        log_var = torch.log(var) 

        return mu, log_var, alpha_d, alpha_w

class OrderedProbitFusionLoss(WeightedLossTerm, FusionGaussianMixin): 

    '''
    PoE fusion loss term that leverages ordinal-like nature of log-population problem
    to penalize with uncertainty (for explainability)
    '''

    required_keys = ("mu_deep", "log_var_deep", "mu_wide", "log_var_wide", "y_rank")

    def __init__(
        self,
        *,
        cut_edges, 
        prob_eps: float = 1e-9, 
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
        var_floor: float = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "ordinal"
    ): 
        WeightedLossTerm.__init__(
            self, 
            name=name, 
            weight=weight, 
            reduction="mean"
        )
        
        FusionGaussianMixin.__init__(
            self, 
            log_var_min=log_var_min, 
            log_var_max=log_var_max, 
            var_floor=var_floor
        )

        edges = torch.as_tensor(cut_edges, dtype=torch.float32).reshape(-1)
        if edges.numel() < 3: 
            raise ValueError("cut_edges must have at least 3 values")
        if torch.any(edges[1:] <= edges[:-1]): 
            raise ValueError("cut_edges must be strictly increasing")
        self.register_buffer("cut_edges", edges)
        self.prob_eps = prob_eps
   
    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_rank = context["y_rank"].reshape(-1).to(dtype=torch.float32)
        mu, log_var, alpha_d, alpha_w = self.fuse({**context, "y": y_rank})

        K = self.cut_edges.numel() - 1 
        max_rank = torch.nextafter(
            torch.tensor(float(K), device=y_rank.device, dtype=y_rank.dtype),
            torch.tensor(0.0, device=y_rank.device, dtype=y_rank.dtype)
        )
        y_rank   = y_rank.clamp(0.0, max_rank)

        k0   = torch.floor(y_rank).long().clamp(0, K - 1)
        k1   = torch.clamp(k0 + 1, max=K - 1)
        frac = (y_rank - k0.to(y_rank.dtype)).clamp(0.0, 1.0)
        
        has_upper = (k0 < (K - 1)).to(frac.dtype)
        frac = frac * has_upper 

        q0   = 1.0 - frac 
        q1   = frac 

        edges = self.cut_edges.to(device=mu.device, dtype=mu.dtype)
        sigma = torch.exp(0.5 * log_var).clamp_min(self.var_floor)

        z_lo  = (edges[:-1].unsqueeze(0) - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        z_hi  = (edges[1:].unsqueeze(0) - mu.unsqueeze(1)) / sigma.unsqueeze(1)

        p = (torch.special.ndtr(z_hi) - torch.special.ndtr(z_lo)).clamp_min(self.prob_eps)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(self.prob_eps)

        p0 = p.gather(1, k0.unsqueeze(1)).squeeze(1).clamp_min(self.prob_eps)
        p1 = p.gather(1, k1.unsqueeze(1)).squeeze(1).clamp_min(self.prob_eps)

        per = -(q0 * torch.log(p0) + q1 * torch.log(p1))

        sw  = context.get("sample_weight", None) 
        if sw is not None: 
            sw = to_batch_tensor(sw, n=per.numel(), device=per.device, dtype=per.dtype)

        raw = weighted_mean(per, sw)
        metrics = {
            "sigma_mean": sigma.mean().detach(), 
            "alpha_deep_mean": alpha_d.mean().detach(),
            "alpha_wide_mean": alpha_w.mean().detach(),
            "frac_interp_mean": q1.mean().detach()
        }
        return raw, metrics 

class GaussianNNLFusionLoss(WeightedLossTerm, FusionGaussianMixin): 
    '''
    GaussianNNL loss over fused wide and deep architecture  
    '''

    required_keys = ("mu_deep", "log_var_deep", "mu_wide", "log_var_wide", "y_rank")

    def __init__(
        self, 
        *,
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
        var_floor: float = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "uncertainty"
    ):
        WeightedLossTerm.__init__(
            self,
            name=name,
            weight=weight,
            reduction="mean"
        )
        
        FusionGaussianMixin.__init__(
            self, 
            log_var_min=log_var_min, 
            log_var_max=log_var_max, 
            var_floor=var_floor
        )

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y_rank = context["y_rank"].reshape(-1).to(dtype=torch.float32)
        mu, log_var, alpha_d, alpha_w = self.fuse({**context, "y": y_rank})

        inv_var = torch.exp(-log_var)
        per     = 0.5 * (torch.pow((y_rank - mu), 2) * inv_var + log_var)

        sw = context.get("sample_weight", None)
        if sw is not None: 
            sw = to_batch_tensor(sw, n=per.numel(), device=per.device, dtype=per.dtype)

        raw  = weighted_mean(per, sw)

        metrics = {
            "sigma_mean": torch.exp(0.5 * log_var).mean().detach(), 
            "alpha_deep_mean": alpha_d.mean().detach(),
            "alpha_wide_mean": alpha_w.mean().detach()
        }
        return raw, metrics 

# ---------------------------------------------------------
# Loss Helpers 
# ---------------------------------------------------------

def sinkhorn_assign(scores: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor: 
    scores = scores - scores.max(dim=1, keepdim=True).values 
    Q = torch.exp(scores / epsilon).t() 
    k, b = Q.shape 
    Q = Q / Q.sum().clamp_min(1e-9)

    Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-9)
    Q = Q / k 
    Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-9)
    Q = Q / b 
    Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-9)

    return Q.t().detach() 

# ---------------------------------------------------------
# Loss Build Functions 
# ---------------------------------------------------------

def build_ssfe_loss(
    *,
        sem_proj: nn.Module, 
        st_proj: nn.Module,
    proto: nn.Module,
    sem_recon: nn.Module,
    st_recon: nn.Module, 
    contrast_temperature: float, 
    cluster_temperature: float, 
    contrast_active_eps: float = 1e-6, 
    sinkhorn_epsilon: float = 0.05
):
    return LossComposer(
        ContrastiveLoss(
            sem_proj=sem_proj,
            st_proj=st_proj,
            temperature=contrast_temperature,
            active_eps=contrast_active_eps,
            weight="w_contrast", 
            name="contrast"
        ),
        SwappedPredictionLoss(
            sem_proj=sem_proj,
            st_proj=st_proj,
            proto=proto,
            temperature=cluster_temperature,
            sinkhorn_epsilon=sinkhorn_epsilon,
            weight="w_cluster", 
            name="cluster"
        ),
        ReconstructionLoss(
            sem_recon=sem_recon,
            st_recon=st_recon,
            weight="w_recon",
            name="recon"
        )
    )

def build_wide_deep_loss(
    *,
    cut_edges,
    log_var_min: float = -9.0, 
    log_var_max: float = 9.0, 
    var_floor: float = 1e-6,
    prob_eps: float = 1e-9, 
): 

    return LossComposer(
        OrderedProbitFusionLoss(
            cut_edges=cut_edges,
            prob_eps=prob_eps,
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            var_floor=var_floor,
            weight="w_ordinal",
            name="ordinal"
        ),
        GaussianNNLFusionLoss(
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            var_floor=var_floor,
            weight="w_uncertainty",
            name="uncertainty"
        )
    )
