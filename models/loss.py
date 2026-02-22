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

from scipy.spatial         import (
    KDTree
) 

from abc                   import (
    ABC, 
    abstractmethod 
) 

from dataclasses           import (
    dataclass, 
    field 
) 

from typing                import (
    Any, 
    Callable, 
    Mapping,
    Optional,
    Sequence,
) 

from numpy.typing          import (
    NDArray
) 

from utils.helpers         import (
    as_1d_f64,
    as_2d_f64,
    to_batch_tensor,
    weighted_mean,
    ordered_expert_tensors,
    normalize_attention
)

# ---------------------------------------------------------
# Loss Constants
# ---------------------------------------------------------

HSIC_SIGMA_GRID_DEFAULT: tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
MULTIVIEW_VAR_REG_EPS: float = 1e-4
MULTIVIEW_RECON_NORM_EPS: float = 1e-6

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

class HSICOrthogonalityLoss(WeightedLossTerm): 
    '''
    Penalizes statistical dependence between embeddings and condition variables via RBF HSIC
    '''

    required_keys = ("deep_bag", "wide_cond")

    def __init__(
        self,
        *,
        sigma: Optional[float | Sequence[float] | torch.Tensor] = None, 
        eps: float = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "hsic"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")
        self.sigma = sigma 
        self.eps   = eps 

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        deep = context["deep_bag"]
        wide = context["wide_cond"]

        if wide is None: 
            raise ValueError("hsic requires wide_cond.")

        if wide.ndim == 1: 
            wide = wide.unsqueeze(1)

        if deep.ndim != 2 or wide.ndim != 2: 
            raise ValueError("wide_cond or deep_bag were an unexpected shape.")

        if wide.shape[0] != deep.shape[0]: 
            raise ValueError("row mismastch "
                             f"deep_bag={deep.shape[0]} != wide_cond={wide.shape[0]}")

        wide = wide.detach() 
        n = int(deep.size(0))
        if n < 3: 
            raw = deep.new_tensor(0.0)
            return raw, {"n": deep.new_tensor(float(n))}

        K = rbf_kernel(deep, sigma=self.sigma, eps=self.eps)
        L = rbf_kernel(wide, sigma=self.sigma, eps=self.eps)

        Kc = center_gram(K)
        Lc = center_gram(L)

        denom = float((n - 1)**2)
        hsic  = (Kc * Lc).sum() / denom 

        k_var = (Kc * Kc).sum() / denom 
        l_var = (Lc * Lc).sum() / denom 
        hsic  = hsic / torch.sqrt(k_var.clamp_min(self.eps) * l_var.clamp_min(self.eps))

        raw   = hsic.abs() 
        metrics = {
            "hsic": raw.detach(), 
            "n": deep.new_tensor(float(n)), 
            "deep_norm": deep.norm(dim=1).mean().detach(), 
            "wide_norm": wide.norm(dim=1).mean().detach(), 
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

        mu      = alpha_d.detach() * mu_d + alpha_w.detach() * mu_w 
        var     = (1.0 / tau).clamp_min(self.var_floor)
        log_var = torch.log(var) 

        return mu, log_var, alpha_d, alpha_w


class FusionCRPSLoss(WeightedLossTerm, FusionGaussianMixin): 
    '''
    Continuous ranked probability score for Gaussian predictive distributions 
    '''

    required_keys = ("mu_deep", "log_var_deep", "mu_wide", "log_var_wide")

    def __init__(
        self,
        *,
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
        var_floor: float   = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "crps"
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
        y_true = context["y"]
        mu, log_var, alpha_d, alpha_w = self.fuse({**context, "y": y_true})

        sigma = torch.exp(0.5 * log_var).clamp_min(self.var_floor)

        z = (y_true - mu) / sigma 

        inv_sqrt_2pi = torch.as_tensor(1.0 / np.sqrt(2.0 * np.pi), device=z.device, dtype=z.dtype)
        inv_sqrt_pi  = torch.as_tensor(1.0 / np.sqrt(np.pi), device=z.device, dtype=z.dtype)  

        cdf = torch.special.ndtr(z)
        pdf = torch.exp(-0.5 * z * z) * inv_sqrt_2pi 
        per = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - inv_sqrt_pi) 
        per = per.clamp_min(0.0)

        sw  = context.get("sample_weight", None)
        if sw is not None: 
            sw = to_batch_tensor(sw, n=per.numel(), device=per.device, dtype=per.dtype)

        raw = weighted_mean(per, sw)
        metrics = {
            "sigma_mean": sigma.mean().detach(), 
            "alpha_deep_mean": alpha_d.mean().detach(),
            "alpha_wide_mean": alpha_w.mean().detach(),
            "z_abs_mean": z.abs().mean().detach() 
        }
        return raw, metrics 

class FusionL1Loss(WeightedLossTerm, FusionGaussianMixin): 
    '''
    Point-estimation alignment using fused mean 
    '''

    required_keys = ("mu_deep", "log_var_deep", "mu_wide", "log_var_wide")

    def __init__(
        self,
        *,
        log_var_min: float = -9.0, 
        log_var_max: float =  9.0, 
        var_floor: float   = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "L1"
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
        y_true = context["y"]
        mu, _, alpha_d, alpha_w = self.fuse({**context, "y": y_true})
        
        per = torch.abs(y_true - mu)
        sw  = context.get("sample_weight", None)
        if sw is not None: 
            sw = to_batch_tensor(sw, n=per.numel(), device=per.device, dtype=per.dtype)

        raw = weighted_mean(per, sw)
        metrics = {
            "mae": raw.detach(),
            "alpha_deep_mean": alpha_d.mean().detach(),
            "alpha_wide_mean": alpha_w.mean().detach(),
        }
        return raw, metrics 

class MultiviewReconstructionLoss(WeightedLossTerm): 

    '''
    For each expert i: 
    - self-reconstruction:  x_i := Dec([S_i, P_i])
    - cross-reconstruction: x_i := Dec([S_global, P_i])
    '''

    required_keys = ("shared_bags", "private_bags", "shared_global", "repr_targets")

    def __init__(
        self,
        *,
        expert_ids: list[str],
        self_recon_heads: nn.ModuleDict,
        cross_recon_heads: nn.ModuleDict,
        cross_scale: float = 1.0, 
        normalize_targets: bool = True,
        norm_eps: float = MULTIVIEW_RECON_NORM_EPS,
        weight: WeightSpec = 1.0, 
        name: str = "recon"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")

        self.expert_ids        = expert_ids
        self.self_recon_heads  = self_recon_heads
        self.cross_recon_heads = cross_recon_heads
        self.cross_scale       = cross_scale
        self.normalize_targets = bool(normalize_targets)
        self.norm_eps          = float(norm_eps)

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s_list = ordered_expert_tensors(
            context["shared_bags"], 
            self.expert_ids, 
            key="shared_bags"
        )
        p_list = ordered_expert_tensors(
            context["private_bags"], 
            self.expert_ids, 
            key="private_bags"
        )
        t_list = ordered_expert_tensors(
            context["repr_targets"], 
            self.expert_ids, 
            key="repr_targets"
        )
        rw_list = None
        if "recon_weights" in context and context["recon_weights"] is not None:
            rw_list = ordered_expert_tensors(
                context["recon_weights"],
                self.expert_ids,
                key="recon_weights"
            )
        s_glob = context["shared_global"]
        rec_vals = []
        metrics: dict[str, torch.Tensor]        = {}

        for i, (eid, si, pi, xi) in enumerate(zip(self.expert_ids, s_list, p_list, t_list)): 
            z_self  = torch.cat([si, pi], dim=1)
            z_cross = torch.cat([s_glob, pi], dim=1)

            x_hat_self  = self.self_recon_heads[eid](z_self)
            x_hat_cross = self.cross_recon_heads[eid](z_cross)
            err_self = (x_hat_self - xi).pow(2)
            err_cross = (x_hat_cross - xi).pow(2)
            if self.normalize_targets:
                var = xi.var(dim=0, unbiased=False).clamp_min(self.norm_eps)
                err_self = err_self / var
                err_cross = err_cross / var
                metrics[f"{eid}_target_var_mean"] = var.mean().detach()

            per_self = err_self.mean(dim=1)
            per_cross = err_cross.mean(dim=1)

            w_i = None
            if rw_list is not None:
                w_i = to_batch_tensor(
                    rw_list[i],
                    n=per_self.numel(),
                    device=per_self.device,
                    dtype=per_self.dtype
                ).clamp_min(0.0)

            l_self  = weighted_mean(per_self, w_i)
            l_cross = weighted_mean(per_cross, w_i)
            l_i     = l_self + self.cross_scale * l_cross 
            rec_vals.append(l_i)

            metrics[f"{eid}_self"]  = l_self.detach()
            metrics[f"{eid}_cross"] = l_cross.detach()
            metrics[f"{eid}_self_raw"] = F.mse_loss(x_hat_self, xi).detach()
            metrics[f"{eid}_cross_raw"] = F.mse_loss(x_hat_cross, xi).detach()
            if w_i is not None:
                metrics[f"{eid}_weight_mean"] = w_i.mean().detach()

        raw = torch.stack(rec_vals).sum() 
        return raw, metrics 

class MultiviewAlignmentLoss(WeightedLossTerm): 

    '''
    Alignment refers to L2 distance for each public embedding against pooled global as well 
    as kl-divergence per sinkhorn assignment against topologically pooled assignment. 
    '''

    required_keys = ("shared_bags", "shared_global", "attention_weights", "sinkhorn_logits")

    def __init__(
        self,
        *,
        expert_ids: list[str], 
        sinkhorn_epsilon: float = 0.05, 
        l2_scale: float = 1.0, 
        kl_scale: float = 1.0, 
        weight: WeightSpec = 1.0, 
        name: str = "align"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")

        self.expert_ids         = expert_ids
        self.sinkhorn_epsilon   = sinkhorn_epsilon
        self.l2_scale           = l2_scale
        self.kl_scale           = kl_scale

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s_list = ordered_expert_tensors(
            context["shared_bags"], 
            self.expert_ids, 
            key="shared_bags"
        )
        l_list = ordered_expert_tensors(
            context["sinkhorn_logits"],
            self.expert_ids,
            key="sinkhorn_logits"
        )
        s_glob = context["shared_global"]

        n_experts = len(self.expert_ids)
        n_bags    = s_glob.shape[0]
        attn      = normalize_attention(
            context["attention_weights"].to(device=s_glob.device, dtype=s_glob.dtype),
            n_bags=n_bags, n_experts=n_experts 
        )

        a_list = [sinkhorn_assign(logits, epsilon=self.sinkhorn_epsilon) for logits in l_list]
        a_glob = a_list[0].new_zeros(a_list[0].shape)
        for i, ai in enumerate(a_list): 
            a_glob = a_glob + attn[:, i:i+1] * ai 
        a_glob = a_glob / a_glob.sum(dim=1, keepdim=True).clamp_min(1e-9)

        l2_vals, kl_vals = [], []
        metrics: dict[str, torch.Tensor] = {}

        for eid, si, ai in zip(self.expert_ids, s_list, a_list): 
            ai = ai.clamp_min(1e-9)
            a_ref = a_glob.clamp_min(1e-9)

            l2_i = F.mse_loss(si, s_glob) 
            kl_i = F.kl_div(a_ref.log(), ai, reduction="batchmean")

            l2_vals.append(l2_i)
            kl_vals.append(kl_i)

            metrics[f"{eid}_l2"] = l2_i.detach()
            metrics[f"{eid}_kl"] = kl_i.detach()

        l2_sum = torch.stack(l2_vals).sum()
        kl_sum = torch.stack(kl_vals).sum()

        raw = self.l2_scale * l2_sum + self.kl_scale * kl_sum 
        return raw, metrics 

class MultiviewPrivacyLoss(WeightedLossTerm): 

    '''
    Privacy/Orthogonality Penalty based on Hilbert-Schmidt Independence Criterion (HSIC) 
    
    Computed via: 
    - HSIC(S_glob, X_wide) + sum(HSIC(P_i, S_glob) + HSIC(P_i, X_wide))
    '''

    required_keys = ("private_bags", "shared_global", "wide_cond")

    def __init__(
        self,
        *,
        expert_ids: list[str], 
        sigma: Optional[float | Sequence[float] | torch.Tensor] = None, 
        eps: float = 1e-6, 
        weight: WeightSpec = 1.0, 
        name: str = "privacy"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")

        self.expert_ids = expert_ids
        self.sigma      = sigma 
        self.eps        = eps 
        
    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p_list = ordered_expert_tensors(
            context["private_bags"], 
            self.expert_ids, 
            key="private_bags"
        )
        s_glob = context["shared_global"]
        wide   = context["wide_cond"]

        if wide.ndim == 1: 
            wide = wide.unsqueeze(1)
        if wide.shape[0] != s_glob.shape[0]: 
            raise ValueError("wide_cond and shared_global row mismatch.")

        wide    = wide.detach() 
        hsic_sw = self.hsic_norm(s_glob, wide)

        terms = [hsic_sw]
        metrics: dict[str, torch.Tensor] = {"hsic_s_wide": hsic_sw.detach()}

        for eid, pi in zip(self.expert_ids, p_list):
            if pi.shape[0] != s_glob.shape[0]: 
                raise ValueError(f"{eid} private_bag rows must match shared_global rows.")

            hsic_ps = self.hsic_norm(pi, s_glob)
            hsic_pw = self.hsic_norm(pi, wide)
            terms.extend([hsic_ps, hsic_pw])

            metrics[f"{eid}_hsic_p_s"] = hsic_ps.detach() 
            metrics[f"{eid}_hsic_p_w"] = hsic_pw.detach() 

        raw = torch.stack(terms).sum() 
        return raw, metrics 

    def hsic_norm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        if x.ndim != 2 or y.ndim != 2: 
            raise ValueError("HSIC inputs must 2d.")
        if x.shape[0] != y.shape[0]: 
            raise ValueError("HSIC inputs row mismatch.")

        n = x.shape[0]
        if n < 3: 
            return x.new_tensor(0.0)

        K  = rbf_kernel(x, sigma=self.sigma, eps=self.eps)
        L  = rbf_kernel(y, sigma=self.sigma, eps=self.eps)

        Kc = center_gram(K)
        Lc = center_gram(L)

        denom = float((n - 1)**2)
        hsic  = (Kc * Lc).sum() / denom 

        k_var = (Kc * Kc).sum() / denom 
        l_var = (Lc * Lc).sum() / denom 
        hsic  = hsic / torch.sqrt(k_var.clamp_min(self.eps) * l_var.clamp_min(self.eps))
        return hsic.abs() 

class VarianceRegularizationLoss(WeightedLossTerm): 
    '''
    Prevent low-rank collapse in private embeddings by penalizing feature std below gamma.
    '''

    required_keys = ("private_bags",)

    def __init__(
        self,
        *,
        expert_ids: list[str], 
        gamma: float = 1.0, 
        eps: float = MULTIVIEW_VAR_REG_EPS, 
        weight: WeightSpec = 1.0, 
        name: str = "var_reg"
    ): 
        super().__init__(name=name, weight=weight, reduction="mean")
        self.expert_ids = expert_ids
        self.gamma      = float(gamma)
        self.eps        = float(eps)

    def compute(self, context: LossContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p_list = ordered_expert_tensors(
            context["private_bags"], 
            self.expert_ids, 
            key="private_bags"
        )

        vals = []
        metrics: dict[str, torch.Tensor] = {}
        for eid, pi in zip(self.expert_ids, p_list):
            if pi.ndim != 2: 
                raise ValueError(f"{eid} private_bag must be 2d")

            if pi.shape[0] < 2:
                std = pi.new_zeros((pi.shape[1],))
                l_var = pi.new_tensor(0.0)
            else:
                std = torch.sqrt(pi.var(dim=0, unbiased=False) + self.eps)
                l_var = F.relu(self.gamma - std).mean()

            vals.append(l_var)
            metrics[f"{eid}_std_mean"] = std.mean().detach()
            metrics[f"{eid}_std_min"]  = std.min().detach()
            metrics[f"{eid}_var_loss"] = l_var.detach()

        raw = torch.stack(vals).sum()
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

def resolve_hsic_sigma_grid(
    sigma: Optional[float | Sequence[float] | torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float
) -> torch.Tensor:
    default_grid = torch.as_tensor(
        HSIC_SIGMA_GRID_DEFAULT,
        device=device,
        dtype=dtype,
    )

    if sigma is None:
        grid = default_grid
    elif torch.is_tensor(sigma):
        grid = sigma.detach().to(device=device, dtype=dtype).view(-1)
    elif isinstance(sigma, (list, tuple, np.ndarray)):
        grid = torch.as_tensor(sigma, device=device, dtype=dtype).view(-1)
    else:
        s = float(sigma)
        if s <= 0.0:
            grid = default_grid
        else:
            # Retain multi-scale behavior while allowing a scalar scale factor.
            grid = default_grid * s

    mask = torch.isfinite(grid) & (grid > 0)
    grid = grid[mask]
    if grid.numel() == 0:
        grid = default_grid

    return grid.clamp_min(float(eps))


def rbf_kernel(
    x: torch.Tensor,
    sigma: Optional[float | Sequence[float] | torch.Tensor],
    eps: float
) -> torch.Tensor: 
    x  = x.reshape(x.size(0), -1)
    d2 = torch.cdist(x, x, p=2).pow(2)

    sigma_grid = resolve_hsic_sigma_grid(
        sigma,
        device=d2.device,
        dtype=d2.dtype,
        eps=eps,
    )
    sigma2 = sigma_grid.pow(2).view(1, 1, -1).clamp_min(float(eps))

    K = torch.exp(-0.5 * d2.unsqueeze(-1) / sigma2)
    # Mean over scales keeps magnitude stable across grid sizes.
    return K.mean(dim=-1)

def center_gram(K: torch.Tensor) -> torch.Tensor: 
    return K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()

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
    hsic_weight: WeightSpec, 
    hsic_sigma: Optional[float] = None, 
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
        ),
        HSICOrthogonalityLoss(
            sigma=hsic_sigma,
            weight=hsic_weight,
            name="hsic"
        )
    )

def build_wide_deep_loss(
    *,
    log_var_min: float = -9.0, 
    log_var_max: float = 9.0, 
    var_floor: float = 1e-6,
): 

    return LossComposer(
        FusionCRPSLoss(
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            var_floor=var_floor,
            weight="w_ordinal",
            name="ordinal"
        ),
        FusionL1Loss(
            log_var_min=log_var_min,
            log_var_max=log_var_max,
            var_floor=var_floor,
            weight="w_uncertainty",
            name="uncertainty"
        )
    )

def build_ssfe_multiview_loss(
    *,
    expert_ids: list[str], 
    self_recon_heads: nn.ModuleDict,
    cross_recon_heads: nn.ModuleDict,
    sinkhorn_epsilon: float = 0.05,
    hsic_sigma: Optional[float] = None, 
    recon_cross_scale: float = 1.0, 
    align_l2_scale: float = 1.0, 
    align_kl_scale: float = 1.0,
    var_reg_gamma: float = 1.0,
    var_reg_eps: float = MULTIVIEW_VAR_REG_EPS,
): 
    return LossComposer(
        MultiviewReconstructionLoss(
            expert_ids=expert_ids,
            self_recon_heads=self_recon_heads,
            cross_recon_heads=cross_recon_heads,
            cross_scale=recon_cross_scale,
            weight="w_recon", 
            name="recon"
        ),
        MultiviewAlignmentLoss(
            expert_ids=expert_ids,
            sinkhorn_epsilon=sinkhorn_epsilon,
            l2_scale=align_l2_scale,
            kl_scale=align_kl_scale,
            weight="w_align",
            name="align"
        ),
        MultiviewPrivacyLoss(
            expert_ids=expert_ids,
            sigma=hsic_sigma,
            weight="w_privacy",
            name="privacy"
        ),
        VarianceRegularizationLoss(
            expert_ids=expert_ids,
            gamma=var_reg_gamma,
            eps=var_reg_eps,
            weight="w_var_reg",
            name="var_reg"
        )
    )
