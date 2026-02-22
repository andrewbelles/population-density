#!/usr/bin/env python3 
# 
# ssfe.py  Andrew Belles  Feb 10th, 2026 
# 
# Self-Supervised Feature Extractors for learning latent representations for deep component of 
# model architecture (wide & deep)
# 

from math import sin
from typing import Any, Mapping, Optional
import torch, time, copy, sys

import torch.nn as nn 

import torch.nn.functional as F 

import numpy as np 

from abc import ABC, abstractmethod 

from contextlib import nullcontext 

from dataclasses import dataclass 

from numpy.typing import NDArray

from sklearn.base import BaseEstimator 

from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors

from models.loss import (
    build_ssfe_loss,
    build_ssfe_multiview_loss,
    sinkhorn_assign
)

from torch.utils.data import (
    DataLoader,
    TensorDataset, 
    Dataset, 
    random_split
)

from models.networks import (
    GatedAttentionPooling,
    HyperGATStack, 
    LightweightBackbone,
    ResidualMLP,
)

from models.graph.construction import (
    SpatialHypergraph,
    TabularHypergraph
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True 

# ---------------------------------------------------------
# Single SSFE Model Contract 
# ---------------------------------------------------------

@dataclass 
class SSFEBatch: 
    '''
    Full information for a single batch at each stage of SSFE model
    '''
    repr_target: torch.Tensor       # target for reconstructive loss 
    semantic_input: torch.Tensor    # inputs to semantic mlp 
    stats_z: torch.Tensor | None 
    stats_raw: torch.Tensor | None  # stats for structural branch 
    batch_idx: torch.Tensor         # node -> bag  
    wide_cond: torch.Tensor 

@dataclass 
class SSFEMultiviewState: 
    '''
    Full information required for a multiview batch. Specifically the private versus public 
    channels that embeddings are separated into. 
    '''
    prep: SSFEBatch 
    repr_target_bag: torch.Tensor 
    recon_weight_bag: torch.Tensor | None
    node_batch_idx: torch.Tensor 
    n_bags: int 

    semantic_node: torch.Tensor
    semantic_bag: torch.Tensor 
    structural_node: torch.Tensor 
    structural_bag: torch.Tensor 

    shared_node: torch.Tensor 
    shared_bag: torch.Tensor 
    private_node: torch.Tensor 
    private_bag: torch.Tensor 

    wide_cond_bag: torch.Tensor 
    shared_sinkhorn_logits: torch.Tensor | None 

class ProjectionHead(nn.Module): 
    def __init__(self, in_dim: int, out_dim: int): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)


class SSFEBase(BaseEstimator, ABC): 
    '''
    Parent self-supervised feature extractor 
    
    Child Class should implement: 
    - build_preprocess() 
    - build_semantic_embed(sample)
    - build_structural_embed(sample)
    - forward_preprocess(batch) -> SSFEBatch 
    - forward_semantic(prep) -> (node_emb, bag_emb)
    - forward_structural(prep) -> (node_emb, bag_emb)
    '''

    def __init__(
        self,
        *,
        anchor_n_samples: int = 500_000, 
        anchor_min_norm: float = 1e-6, 
        swap_noise_prob: float = 0.15, 
        epochs: int = 300, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        batch_size: int = 256, 
        eval_fraction: float = 0.15, 
        early_stopping_rounds: int = 20, 
        min_delta: float = 1e-4, 
        random_state: int = 0, 
        collate_fn=None, 
        w_contrast: float = 1.0, 
        w_cluster: float = 1.0, 
        w_recon: float = 1.0, 
        w_hsic: float = 1.0, 
        contrast_temperature: float, 
        cluster_temperature: float,
        hsic_sigma: Optional[float] = None, 
        hsic_view: str = "concat", 
        shared_view: str = "structural", 
        private_view: str = "semantic", 
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None,
        compile_model: bool = False
    ): 
        self.anchor_n_samples      = anchor_n_samples 
        self.anchor_min_norm       = anchor_min_norm
        self.epochs                = epochs
        self.lr                    = lr
        self.weight_decay          = weight_decay
        self.batch_size            = batch_size
        self.eval_fraction         = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta             = min_delta
        self.random_state          = random_state
        self.collate_fn            = collate_fn

        self.swap_noise_prob       = swap_noise_prob

        self.w_contrast            = w_contrast
        self.w_cluster             = w_cluster
        self.w_recon               = w_recon
        self.w_hsic                = w_hsic

        self.contrast_temperature  = contrast_temperature
        self.cluster_temperature   = cluster_temperature
        self.hsic_sigma            = hsic_sigma
        self.n_prototypes          = n_prototypes
        self.proj_dim              = proj_dim

        if hsic_view not in {"concat", "semantic", "structural"}: 
            raise ValueError(f"unknown embeddings view for hsic loss: {hsic_view}")
        if shared_view not in {"semantic", "structural"}: 
            raise ValueError(f"unknown shared view: {shared_view}")
        if private_view not in {"semantic", "structural"}: 
            raise ValueError(f"unknown private view: {private_view}")
        if private_view == shared_view: 
            raise ValueError("shared_view and private_view must differ")

        self.hsic_view    = hsic_view 
        self.shared_view  = shared_view
        self.private_view = private_view

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.device    = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.node_anchors: list[list[float]] = []
        self.anchor_stats: list[float]       = []

        self.model_          = nn.ModuleDict() 
        self.opt_            = None 
        self.scheduler_      = None 
        self.loss_           = None 
        self.best_val_score_ = np.inf 
        self.is_fitted_      = False 

        self.compiled_       = False 
        self.compile_model   = compile_model
        self.current_epoch_  = 0 

    # -----------------------------------------------------
    # Child Hooks 
    # -----------------------------------------------------

    @abstractmethod 
    def build_preprocess(self) -> nn.Module | None: ...  

    @abstractmethod 
    def build_semantic_embed(self, sample: SSFEBatch) -> nn.Module: ... 

    @abstractmethod 
    def build_structural_embed(self, sample: SSFEBatch) -> nn.Module: ... 

    @abstractmethod 
    def forward_preprocess(self, batch) -> SSFEBatch: ... 

    @abstractmethod 
    def forward_semantic(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]: ... 

    @abstractmethod 
    def forward_structural(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]: ... 

    @abstractmethod 
    def anchor_features_from_batch(self, batch) -> np.ndarray: ...

    @abstractmethod 
    def initialize_structural_state(self, train_loader: DataLoader): 
        _ = train_loader 
        return None 

    # -----------------------------------------------------
    # API  
    # -----------------------------------------------------
    
    def fit(self, X, y=None): 
        _ = y
        print(self)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        loader                   = self.ensure_loader(X, shuffle=True)
        train_loader, val_loader = self.split_loader(loader)

        self.init_fit(train_loader, state_loader=loader)

        best = self.run_phase(
            name="SSFE",
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs
        )

        if best is not None: 
            self.model_.load_state_dict(best)

        self.is_fitted_ = True 
        return self 

    def extract(self, X, *, view: str = "concat", level: Optional[str] = None) -> NDArray: 
        if not self.is_fitted_: 
            raise ValueError("Call fit() before extract().")
        if level not in (None, "county"): 
            raise ValueError("extract only supports county-level output (level=county).")

        loader = self.ensure_loader(X, shuffle=False)
        outs: list[NDArray] = []

        self.model_.eval() 
        with torch.no_grad(): 
            for batch in loader: 

                prep = self.forward_preprocess(batch)
                _, sem_county = self.forward_semantic(prep)
                _, st_county  = self.forward_structural(prep)

                if view == "semantic": 
                    x = sem_county  
                elif view == "structural": 
                    x = st_county  
                elif view == "concat": 
                    x = torch.cat([sem_county, st_county], dim=1)
                else: 
                    raise ValueError(f"unknown view={view}")

                outs.append(x.detach().cpu().numpy())

        return np.vstack(outs) if outs else np.empty((0, 0), dtype=np.float32)

    # -----------------------------------------------------
    # Setup   
    # -----------------------------------------
    
    def init_fit(
        self, 
        train_loader: DataLoader,
        state_loader: DataLoader | None = None 
    ): 
        source_loader = train_loader if state_loader is None else state_loader 

        self.initialize_anchor_state(source_loader)
        self.initialize_structural_state(source_loader)

        sample_batch = next(iter(train_loader))

        preprocess   = self.build_preprocess() 
        if preprocess is None: 
            preprocess = nn.Identity() 

        self.model_["preprocess"] = preprocess.to(self.device)

        with torch.no_grad(): 
            prep = self.forward_preprocess(sample_batch)

        semantic   = self.build_semantic_embed(prep).to(self.device)
        structural = self.build_structural_embed(prep).to(self.device)

        self.model_["semantic"]   = semantic 
        self.model_["structural"] = structural 

        with torch.no_grad(): 
            if isinstance(prep, SpatialPatchBatch): 
                sem_boot   = self.model_["semantic"](prep.semantic_input)
                d_sem_boot = sem_boot.shape[1]

                _, st_tile_boot = self.model_["structural"](prep)
                d_st_boot = st_tile_boot.shape[1]

                d_target_boot = prep.repr_target.shape[1] if prep.repr_target.ndim > 1 else 1 

                self.model_["sem_tile_pool"] = GatedAttentionPooling(
                    in_dim=d_sem_boot, 
                    attn_dim=max(32, d_sem_boot // 2), 
                    attn_dropout=getattr(self, "attn_dropout", 0.0)
                ).to(self.device)
                self.model_["sem_county_pool"] = GatedAttentionPooling(
                    in_dim=d_sem_boot,
                    attn_dim=max(32, d_sem_boot // 2), 
                    attn_dropout=getattr(self, "attn_dropout", 0.0)  
                ).to(self.device)
                self.model_["st_county_pool"] = GatedAttentionPooling(
                    in_dim=d_st_boot,
                    attn_dim=max(32, d_st_boot // 2), 
                    attn_dropout=getattr(self, "attn_dropout", 0.0)  
                ).to(self.device)
                self.model_["target_county_pool"] = CountyMeanPooling().to(self.device)

            sem_node, sem_bag = self.forward_semantic(prep)
            st_node, st_bag   = self.forward_structural(prep)

        d_target   = prep.repr_target.shape[1] if prep.repr_target.ndim > 1 else 1 
        d_sem_node = sem_node.shape[1]
        d_st_node  = st_node.shape[1]
        d_sem_bag  = sem_bag.shape[1]
        d_st_bag   = st_bag.shape[1]

        self.model_["sem_proj"] = ProjectionHead(d_sem_bag, self.proj_dim).to(self.device)
        self.model_["st_proj"]  = ProjectionHead(d_st_bag, self.proj_dim).to(self.device)

        self.model_["proto"]    = nn.Linear(
            self.proj_dim, self.n_prototypes, bias=False
        ).to(self.device)

        self.model_["sem_recon"] = nn.Linear(d_sem_node, d_target).to(self.device)
        self.model_["st_recon"]  = nn.Linear(d_st_node, d_target).to(self.device)

        self.loss_ = build_ssfe_loss(
            sem_proj=self.model_["sem_proj"],
            st_proj=self.model_["st_proj"],
            proto=self.model_["proto"],
            sem_recon=self.model_["sem_recon"],
            st_recon=self.model_["st_recon"], 
            hsic_sigma=self.hsic_sigma,
            hsic_weight=self.w_hsic,
            contrast_temperature=self.contrast_temperature,
            cluster_temperature=self.cluster_temperature,
            contrast_active_eps=1e-6, 
            sinkhorn_epsilon=0.05 
        )

        self.compile_modules()

        self.opt_ = torch.optim.AdamW(
            self.model_.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_, T_max=max(1, self.epochs)
        )

    def resolve_contrast_node_batch_idx(
        self,
        prep: SSFEBatch,
        *,
        bag_count: int 
    ) -> torch.Tensor: 
        node_to_tile = prep.batch_idx.to(
            self.device, dtype=torch.long, non_blocking=True
        ).view(-1)

        tile_to_county = getattr(prep, "tile_batch_idx", None)
        if tile_to_county is None:
            return node_to_tile

        tile_to_county = tile_to_county.to(
            self.device, dtype=torch.long, non_blocking=True
        ).view(-1)

        if node_to_tile.numel() == 0:
            return node_to_tile
        if int(node_to_tile.max().item()) >= tile_to_county.numel():
            raise ValueError("prep.batch_idx contains tile index out of range")

        _, tile_to_county_inv = torch.unique(
            tile_to_county, sorted=True, return_inverse=True
        )
        node_to_county = tile_to_county_inv[node_to_tile]

        if node_to_county.numel() and (int(node_to_county.max().item()) + 1 != int(bag_count)):
            raise ValueError("node->county index mismatch with bag embedding count")
        return node_to_county

    # -----------------------------------------------------
    # Training    
    # -----------------------------------------------------

    def run_phase(
        self,
        *,
        name: str,
        train_loader: DataLoader, 
        val_loader: DataLoader | None, 
        epochs: int 
    ): 
        print(f"[{name}] starting...")

        best_val   = float("inf")
        best_state = None 
        patience   = 0
        t0         = time.perf_counter()

        for ep in range(epochs): 
            self.current_epoch_ = ep 
            train_loss          = self.train_epoch(train_loader)
            val_loss, val_parts = self.validate(val_loader)

            score = train_loss if val_loss is None else val_loss 
            if score < best_val - self.min_delta: 
                best_val   = score 
                best_state = copy.deepcopy(self.model_.state_dict())
                patience   = 0  
                self.best_val_score_ = best_val
            else: 
                patience  += 1 
                if self.early_stopping_rounds > 0 and patience >= self.early_stopping_rounds:
                    break 

            if ep % 5 == 0: 
                dt = (time.perf_counter() - t0) / (ep + 1) 
                if val_loss is None: 
                    print(f"[epoch {ep:3d}] {dt:.2f}s avg | train ssl={train_loss:.4f}",
                          file=sys.stderr)
                else: 
                    print(
                        f"[epoch {ep:3d}] {dt:.2f}s avg | "
                        f"val_ssl={val_loss:.4f} | "
                        f"val_con={val_parts.get('contrast', 0.0):.4f} | "
                        f"val_kl={val_parts.get('cluster', 0.0):.4f} | "
                        f"val_rec={val_parts.get('recon', 0.0):.4f} | " 
                        f"val_hsic={val_parts.get('hsic', 0.0):.4f}", 
                        file=sys.stderr
                    )

        return best_state 

    def train_epoch(self, loader: DataLoader) -> float: 
        if self.opt_ is None: 
            raise RuntimeError("training not initalization.")

        self.model_.train() 
        total = 0.0 
        count = 0 

        for batch in loader: 
            loss, _, bsz = self.process_batch(batch)

            self.opt_.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_.step() 

            total += float(loss.item()) * bsz 
            count += bsz 

        if self.scheduler_ is not None: 
            self.scheduler_.step() 

        return total / max(1, count)

    def validate(self, loader: DataLoader | None): 
        if loader is None: 
            return None, {}

        self.model_.eval()
        sums: dict[str, float] = {}
        count = 0 

        with torch.no_grad(): 
            for batch in loader: 
                _, terms, bsz = self.process_batch(batch)
                for name, val in terms.items(): 
                    sums[name] = sums.get(name, 0.0) + float(val.item()) * bsz
                count += bsz 

        denom = max(1, count)
        parts = {k: (v / denom) for k, v in sums.items()}
        total = float(sum(parts.values()))
        return total, parts 

    # -----------------------------------------------------
    # Batching 
    # -----------------------------------------------------

    def process_batch(self, batch): 
        if self.loss_ is None: 
            raise RuntimeError("loss_ not initializaed. Call init_fit() first.")

        with self.amp_ctx(): 
            multi_state       = self.build_multiview_state(batch)
            prep              = multi_state.prep 
            sem_node, sem_bag = multi_state.semantic_node, multi_state.semantic_bag
            st_node, st_bag   = multi_state.structural_node, multi_state.structural_bag
            node_batch_idx    = multi_state.node_batch_idx 
            hsic_bag          = self.get_view_for_hsic(sem_bag, st_bag)
            wide_bag          = multi_state.wide_cond_bag
            prep.repr_target  = multi_state.prep.repr_target

            pack = self.loss_(
                sem_bag=sem_bag, 
                st_bag=st_bag,
                sem_node=sem_node,
                st_node=st_node,
                prep=prep,
                stats_raw=prep.stats_raw,
                node_batch_idx=node_batch_idx,
                w_contrast=self.w_contrast,
                w_cluster=self.w_cluster,
                w_recon=self.w_recon,
                w_hsic=self.w_hsic,
                deep_bag=hsic_bag,
                wide_cond=wide_bag,
                shared_bag=multi_state.shared_bag,
                private_bag=multi_state.private_bag,
                shared_sinkhorn_logits=multi_state.shared_sinkhorn_logits,
                repr_target_bag=multi_state.repr_target_bag
            )

            loss  = pack.total 
            terms = dict(pack.raw)
            bsz   = sem_bag.shape[0]
        return loss, terms, bsz 

    def get_view_for_hsic(self, sem_bag: torch.Tensor, st_bag: torch.Tensor) -> torch.Tensor: 
        if self.hsic_view == "semantic": 
            return sem_bag 
        elif self.hsic_view == "structural": 
            return st_bag 
        else: 
            return torch.cat([sem_bag, st_bag], dim=1)

    # -----------------------------------------------------
    # Loader Utils    
    # -----------------------------------------------------

    def initialize_anchor_state(self, train_loader: DataLoader):
        total_raw   = 0
        total_valid = 0 
        buffer: list[np.ndarray] = []

        for batch in train_loader: 
            feats = self.anchor_features_from_batch(batch)
            total_raw += feats.shape[0]

            norms = np.linalg.norm(feats, axis=1)
            valid = np.isfinite(norms) & (norms > self.anchor_min_norm) 
            if valid.any(): 
                keep = feats[valid]
                buffer.append(keep)
                total_valid += keep.shape[0]

            if total_valid >= self.anchor_n_samples: 
                break 

        if not buffer: 
            raise ValueError("no valid patch stats found for anchor initialization")

        data = np.vstack(buffer)
        anchors, stats = self.fit_anchors(
            data, 
            n_samples=self.anchor_n_samples, 
            k=3, 
            random_state=self.random_state
        )

        self.node_anchors = anchors.tolist() 
        self.anchor_stats = stats.tolist()

        keep_ratio = float(total_valid) / max(total_raw, 1) 
        print(
            f"[anchors:init] kept {total_valid}/{total_raw} patches ({keep_ratio:.2%}) | " 
            f"anchors={anchors.shape}", file=sys.stderr
        )

    def compile_modules(self): 
        if self.compiled_ or not self.compile_model: 
            return 
        if self.device.type != "cuda" or not hasattr(torch, "compile"): 
            return 

        if "preprocess" in self.model_ and hasattr(self.model_["preprocess"], "encoder"): 
            self.model_["preprocess"].encoder = self.try_compile_module(
                    self.model_["preprocess"].encoder, "preprocess.encoder"
            ) 

        self.compiled_ = True 
        print("[compile] modules compiled", file=sys.stderr)

    def try_compile_module(self, module: nn.Module, name: str) -> nn.Module: 
        try: 
            return torch.compile(
                module, 
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True 
            )
        except Exception as e: 
            print(f"[compile:skip] {name}: {e}", file=sys.stderr)
            return module 

    def ensure_loader(self, X, shuffle: bool = False): 
        if isinstance(X, DataLoader):
            return X 
        return self.make_loader(X, shuffle=shuffle)

    def make_loader(self, dataset, shuffle: bool): 
        return DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        ) 

    def split_loader(self, loader): 
        if self.eval_fraction <= 0.0: 
            return loader, None 

        ds = getattr(loader, "dataset", None)
        if ds is None or len(ds) < 2: 
            return loader, None 

        n_val   = max(1, int(len(ds) * self.eval_fraction))
        n_train = len(ds) - n_val
        if n_train < 1: 
            return loader, None 

        gen = torch.Generator().manual_seed(self.random_state)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
        return self.make_loader(train_ds, shuffle=True), self.make_loader(val_ds, shuffle=False)

    def amp_ctx(self): 
        if self.device.type == "cuda": 
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()

    def compute_shared_sinkhorn_logits(
        self,
        shared_bag: torch.Tensor 
    ) -> torch.Tensor:  
        proj_key = "sem_proj" if self.shared_view == "semantic" else "st_proj" 
        if proj_key not in self.model_ or "proto" not in self.model_: 
            raise ValueError("model does not have required components for OMVAE")

        z = F.normalize(self.model_[proj_key](shared_bag), dim=1)
        return self.model_["proto"](z)

    def build_multiview_state(self, batch) -> SSFEMultiviewState: 
        prep = self.forward_preprocess(batch)
        sem_node, sem_bag = self.forward_semantic(prep)
        st_node, st_bag   = self.forward_structural(prep)

        node_batch_idx    = self.resolve_contrast_node_batch_idx(
            prep, bag_count=sem_bag.shape[0]
        )

        n_bags = sem_bag.shape[0]

        shared_node, shared_bag = self.select_branch(
            view=self.shared_view,
            sem_node=sem_node, sem_bag=sem_bag,
            st_node=st_node, st_bag=st_bag
        )

        private_node, private_bag = self.select_branch(
            view=self.private_view,
            sem_node=sem_node, sem_bag=sem_bag,
            st_node=st_node, st_bag=st_bag
        )

        wide_bag = self.normalize_wide_cond_bag(
            wide_cond=prep.wide_cond,
            n_bags=n_bags,
            device=shared_bag.device,
            dtype=shared_bag.dtype 
        )

        self.assert_county_level(
            sem_bag=sem_bag, st_bag=st_bag, 
            wide_bag=wide_bag, n_bags=n_bags
        )

        repr_target_bag = self.forward_repr_target_bag(
            prep,
            n_bags=n_bags,
            device=shared_bag.device,
            dtype=shared_bag.dtype
        )
        recon_weight_bag = self.forward_recon_weight_bag(
            prep,
            n_bags=n_bags,
            device=shared_bag.device,
            dtype=shared_bag.dtype
        )

        shared_logits = self.compute_shared_sinkhorn_logits(shared_bag)

        return SSFEMultiviewState(
            prep=prep,
            repr_target_bag=repr_target_bag, 
            recon_weight_bag=recon_weight_bag,
            node_batch_idx=node_batch_idx,
            n_bags=n_bags,
            semantic_bag=sem_bag,
            semantic_node=sem_node,
            structural_bag=st_bag,
            structural_node=st_node,
            shared_node=shared_node,
            shared_bag=shared_bag,
            private_node=private_node,
            private_bag=private_bag,
            wide_cond_bag=wide_bag,
            shared_sinkhorn_logits=shared_logits,
        )

    def normalize_wide_cond_bag(
        self,
        *,
        wide_cond: torch.Tensor, 
        n_bags: int, 
        device: torch.device, 
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor: 
        wide = wide_cond.to(device=device, dtype=dtype)
        if wide.ndim == 1: 
            wide = wide.unsqueeze(1)
        if wide.ndim != 2: 
            raise ValueError(f"wide_cond must be 2d, got {wide.shape}")

        if wide.shape[0] == n_bags: 
            return wide 
        else: 
            raise ValueError("wide_cond must be county level")

    def select_branch(
        self,
        *,
        view: str, 
        sem_node: torch.Tensor, 
        sem_bag: torch.Tensor, 
        st_node: torch.Tensor, 
        st_bag: torch.Tensor 
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        if view == "semantic": 
            return sem_node, sem_bag 
        else:  
            return st_node, st_bag # hard checked at init 

    def forward_repr_target_bag(
        self,
        prep: SSFEBatch,
        *,
        n_bags: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor: 
        t = prep.repr_target.to(device=device, dtype=dtype)
        if t.ndim == 1: 
            t = t.unsqueeze(1)
        if t.shape[0] == n_bags: 
            return t 
        raise ValueError("target is not county level")

    def forward_recon_weight_bag(
        self,
        prep: SSFEBatch,
        *,
        n_bags: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.ones((n_bags,), device=device, dtype=dtype)

    @staticmethod 
    def assert_county_level(
        *,
        sem_bag: torch.Tensor, 
        st_bag: torch.Tensor, 
        wide_bag: torch.Tensor, 
        n_bags: int 
    ): 
        if sem_bag.ndim != 2 or st_bag.ndim != 2 or wide_bag.ndim != 2: 
            raise ValueError("county-level tensors must be 2d.")
        if (sem_bag.shape[0] != n_bags or st_bag.shape[0] != n_bags or 
            wide_bag.shape[0] != n_bags):
            raise ValueError("bags are not county-level.")

    @staticmethod 
    def fit_anchors(
        features: np.ndarray,
        *,
        n_samples:  int, 
        k: int, 
        random_state: int, 
    ) -> tuple[np.ndarray, np.ndarray]: 
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] == 0: 
            raise ValueError(f"expected non-empty 2d feats, got {x.shape}")

        mean = x.mean(axis=0)
        std  = np.maximum(x.std(axis=0), 1e-6)
        z    = (x - mean) / std 

        take = min(int(n_samples), z.shape[0])
        rng  = np.random.default_rng(random_state)
        idx  = rng.choice(z.shape[0], size=take, replace=False)
        
        samples = z[idx]

        km = KMeans(n_clusters=int(k), random_state=random_state)
        km.fit(samples)

        anchors = km.cluster_centers_.astype(np.float32, copy=False)
        order   = np.argsort(np.linalg.norm(anchors, axis=1))
        anchors = anchors[order]
        stats   = np.asarray([mean, std], dtype=np.float32)
        return anchors, stats 

    @staticmethod 
    def as_numpy(x, *, dtype=None): 
        if isinstance(x, torch.Tensor): 
            arr = x.detach().cpu().numpy() 
        else: 
            arr = np.asarray(x)
        if dtype is not None: 
            arr = arr.astype(dtype, copy=False)
        return arr 

# ---------------------------------------------------------
# Spatial SSFE Modules 
# ---------------------------------------------------------

@dataclass 
class SpatialPatchBatch(SSFEBatch): 
    tile_batch_idx: torch.Tensor 
    patch_idx: torch.Tensor | None 
    n_tiles: int 
    n_patches_per_tile: int 

class CountyMeanPooling(nn.Module): 
    def forward(
        self, 
        x: torch.Tensor, 
        batch_indices: torch.Tensor, 
        batch_size: int
    ) -> torch.Tensor: 
        if x.ndim != 2: 
            raise ValueError(f"expected (N, d), got {tuple(x.shape)}")
        out   = x.new_zeros((batch_size, x.shape[1]))
        count = x.new_zeros((batch_size, 1))

        out.index_add_(0, batch_indices, x) 
        count.index_add_(0, batch_indices, x.new_ones((x.shape[0], 1)))
        return out / count.clamp_min(1.0)

class SpatialPatchPreprocessor(nn.Module): 
    '''
    Converts flat tile tensors to patched embeddings for use by SSFE 
    '''

    def __init__(
        self, 
        *,
        in_channels: int, 
        tile_size: int = 256, 
        patch_size: int = 32, 
        embed_dim: int, 
        anchor_stats: list[float], 
        swap_noise_prob: float = 0.15 
    ): 
        super().__init__()

        mean = torch.as_tensor(anchor_stats[0], dtype=torch.float32)
        std  = torch.as_tensor(anchor_stats[1], dtype=torch.float32)
        if (mean.ndim != 1 or std.ndim != 1 or 
            mean.numel() != in_channels or std.numel() != in_channels):
            raise ValueError("anchor_stats shape mismatch")

        if torch.any(std <= 0): 
            raise ValueError("negative std from anchor_stats")

        self.patch_size      = patch_size 
        self.swap_noise_prob = swap_noise_prob 

        self.encoder      = LightweightBackbone(
            in_channels=in_channels,
            embed_dim=embed_dim, 
            patch_size=patch_size, 
            anchor_stats=anchor_stats,
        )

        self.num_patches = (tile_size // patch_size)**2

    def forward(
        self,
        tiles: torch.Tensor,
        stats: torch.Tensor, 
        wide_cond: torch.Tensor, 
        tile_batch_idx: torch.Tensor | None = None,
    ) -> SpatialPatchBatch: 
        if tiles.ndim != 4: 
            raise ValueError(f"expected (T, C, H, W), got {tuple(tiles.shape)}")
        if stats.ndim != 3: 
            raise ValueError(f"expected stats (T, L, D), got {tuple(stats.shape)}")

        T, C, _, _ = tiles.shape

        if wide_cond.ndim == 1: 
            wide_cond = wide_cond.unsqueeze(1)

        K    = stats.shape[1]
        P = self.patch_size 
        raw_patches = F.unfold(tiles, kernel_size=P, stride=P).transpose(1, 2).contiguous()
        
        K_unfold = raw_patches.shape[1]
        K_stats  = stats.shape[1]

        if K_stats != K_unfold: 
            raise ValueError 

        K = K_unfold 
        raw_patches = raw_patches.reshape(T * K, -1)
        
        embc = self.encoder(tiles)
        if embc.ndim != 2 or embc.shape[0] != T * K: 
            raise ValueError 
        embc = embc.view(T, K, -1)

        embs = self.maybe_patch_shuffle(embc)
        idx  = None 

        stats_raw    = stats[..., :C].to(
            device=tiles.device, dtype=torch.float32, non_blocking=True
        )

        mean         = self.encoder.patch_mean[:C].view(1, 1, C).to(
            device=tiles.device, dtype=torch.float32)
        std          = self.encoder.patch_std[:C].view(1, 1, C).to(
            device=tiles.device, dtype=torch.float32)
        stats_z      = (stats_raw - mean) / std 

        sem_in       = torch.cat([embs, stats_z], dim=-1) 

        repr_target  = raw_patches.detach().to(dtype=torch.float32).contiguous() 
        sem_in       = sem_in.reshape(T * K, -1)
        stats_z_f    = stats_z.reshape(T * K, -1)
        stats_raw_f  = stats_raw.reshape(T * K, -1)

        patch_batch_idx = torch.arange(T, device=tiles.device).repeat_interleave(K)
        if tile_batch_idx is None: 
            tile_batch_idx = torch.arange(T, device=tiles.device, dtype=torch.long)
        else: 
            tile_batch_idx = tile_batch_idx.to(
                device=tiles.device, dtype=torch.long, non_blocking=True).view(-1)

        if tile_batch_idx.numel() != T: 
            raise ValueError("tile_batch_idx length must match number of tiles.")

        n_bags = int(tile_batch_idx.max().item()) + 1 if tile_batch_idx.numel() else 0 
        if wide_cond.shape[0] != n_bags: 
            raise ValueError("wide_cond must be bag-level with one row per bag")

        wide_cond = wide_cond.to(device=tiles.device, dtype=torch.float32, non_blocking=True)

        return SpatialPatchBatch(
            repr_target=repr_target,
            semantic_input=sem_in,
            stats_z=stats_z_f,
            stats_raw=stats_raw_f,
            batch_idx=patch_batch_idx,
            tile_batch_idx=tile_batch_idx,
            wide_cond=wide_cond,
            patch_idx=idx, 
            n_tiles=T,
            n_patches_per_tile=K 
        )

    def maybe_patch_shuffle(self, embs: torch.Tensor) -> torch.Tensor: 
        p = self.swap_noise_prob 
        if (not self.training) or p <= 0.0: 
            return embs 

        t, k, d = embs.shape 
        if t == 0 or k < 2: 
            return embs 

        sel = torch.nonzero(torch.rand((t, ), device=embs.device) < p, as_tuple=False).squeeze(1)
        if sel.numel() == 0: 
            return embs 

        picked = embs.index_select(0, sel)
        perm = torch.argsort(torch.rand((sel.numel(), k), device=embs.device), dim=1)
        shuffled = torch.gather(picked, dim=1, index=perm.unsqueeze(-1).expand(-1, -1, d)) 

        out = embs.clone() 
        out.index_copy_(0, sel, shuffled)
        return out 


class SpatialStructuralEmbedder(nn.Module):
    '''
    Structural path for SpatialSSFE 
    patch tokens -> hypergraph builder -> hyperGAT -> readout node embeddings 
    '''

    def __init__(
        self,
        *,
        in_dim: int,
        stats_dim: int, 
        node_anchors: list[list[float]],
        tile_size: int = 256, 
        patch_size: int = 32, 
        gnn_dim: int, 
        gnn_layers: int, 
        gnn_heads: int, 
        dropout: float, 
        attn_dropout: float,
        global_active_eps: float = 1e-6, 
        device: str = "cuda"
    ): 
        super().__init__() 

        anchors = torch.tensor(node_anchors, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[0] != 3: 
            raise ValueError("node_anchors must be (3, d_stats)")
        if anchors.shape[1] != stats_dim: 
            raise ValueError("node_anchors dim mismatch.")

        self.graph = SpatialHypergraph(
            anchors=anchors,
            tile_size=tile_size,
            patch_size=patch_size,
            global_active_eps=global_active_eps,
            device=device
        )

        self.gnn   = HyperGATStack(
            in_dim=in_dim,
            hidden_dim=gnn_dim,
            n_layers=gnn_layers,
            n_heads=gnn_heads,
            n_node_types=4,
            n_edge_types=3,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        self.readout_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.node_norm = nn.LayerNorm(gnn_dim)
        self.bag_norm  = nn.LayerNorm(gnn_dim)
        self.out_dim   = gnn_dim 

    def forward(self, prep: SpatialPatchBatch) -> tuple[torch.Tensor, torch.Tensor]: 
        if prep.stats_z is None: 
            raise ValueError("must have transformed stats")

        x_nodes = prep.semantic_input
        N, B    = x_nodes.shape[0], prep.n_tiles 

        readout = self.readout_token.expand(B, -1, -1).reshape(B, -1).to(x_nodes.device)
        x_all   = torch.cat([x_nodes, readout], dim=0)

        meta    = self.graph.build(
            prep.stats_z, 
            batch_size=prep.n_tiles,
            active_stats=prep.stats_raw,
            idx=prep.patch_idx
        )

        node_idx, edge_idx = meta.incidence_index
        h_all   = self.gnn(x_all, meta.node_type, meta.edge_type, node_idx, edge_idx)
        h_node  = self.node_norm(h_all[:N])
        h_bag   = self.bag_norm(h_all[meta.readout_node_ids])
        return h_node, h_bag 

class SpatialSemanticEmbedder(nn.Module): 
    '''
    Residual semantic encoder for SpatialSSFE. 
    '''

    def __init__(
        self,
        *,
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int,
        depth: int, 
        dropout: float 
    ): 
        super().__init__()
        self.backbone = ResidualMLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim, 
            out_dim=out_dim, 
            depth=depth, 
            dropout=dropout, 
            zero_head_init=False 
        )

        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if x.ndim != 2: 
            raise ValueError(f"expected (N, d), got {tuple(x.shape)}")
        return self.norm(self.backbone(x))

# ---------------------------------------------------------
# Spatial Self-Supervised Feature Extractor 
# ---------------------------------------------------------

class SpatialSSFE(SSFEBase): 
    '''
    Spatial self-supervised feature extractor that operates on image, tensor datasets. 
    
    Splits images into patches and induces heterogeneous hypergraphs via kNN on patch stats 
    (specifically p95 of each channel). 

    Semantic path uses a shallow non-linear, pre-norm projector to preserve embeddings and 
    minimizing kl-divergence. 
    '''

    def __init__(
        self,
        *,
        # tensor metadata 
        in_channels: int, 
        tile_size: int = 256, 
        patch_size: int = 32, 
        embed_dim: int, 

        # semantic branch 
        semantic_hidden_dim: int, 
        semantic_out_dim: int, 
        semantic_dropout: float, 
        semantic_depth: int = 2, 

        # structural branch
        gnn_dim: int, 
        gnn_layers: int, 
        gnn_heads: int, 
        dropout: float, 
        attn_dropout: float, 
        global_active_eps: float = 1e-6, 
        swap_noise_prob: float = 0.20, 

        # parent class 
        epochs: int = 400, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0, 
        batch_size: int = 256, 
        eval_fraction: float = 0.15, 
        early_stopping_rounds: int = 20, 
        min_delta: float = 1e-4, 
        random_state: int = 0, 
        collate_fn=None, 
        w_contrast: float = 1.0, 
        w_cluster: float = 1.0, 
        w_recon: float = 1.0, 
        w_hsic: float = 1.0, 
        hsic_sigma: Optional[float] = None, 
        hsic_view: str = "concat", 
        shared_view: str = "structural", 
        private_view: str = "semantic", 
        contrast_temperature: float, 
        cluster_temperature: float,
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None
    ): 
        super().__init__(
            epochs=epochs,
            swap_noise_prob=swap_noise_prob,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            eval_fraction=eval_fraction,
            early_stopping_rounds=early_stopping_rounds,
            min_delta=min_delta,
            random_state=random_state,
            collate_fn=collate_fn,
            w_contrast=w_contrast,
            w_cluster=w_cluster,
            w_recon=w_recon,
            w_hsic=w_hsic,
            hsic_sigma=hsic_sigma,
            hsic_view=hsic_view,
            shared_view=shared_view,
            private_view=private_view, 
            contrast_temperature=contrast_temperature,
            cluster_temperature=cluster_temperature,
            n_prototypes=n_prototypes,
            proj_dim=proj_dim,
            device=device,
        )

        self.in_channels         = in_channels
        self.tile_size           = tile_size
        self.patch_size          = patch_size
        self.embed_dim           = embed_dim

        self.semantic_hidden_dim = semantic_hidden_dim
        self.semantic_out_dim    = semantic_out_dim
        self.semantic_dropout    = semantic_dropout
        self.semantic_depth      = semantic_depth 

        self.gnn_dim             = gnn_dim
        self.gnn_layers          = gnn_layers
        self.gnn_heads           = gnn_heads
        self.dropout             = dropout
        self.attn_dropout        = attn_dropout
        self.global_active_eps   = global_active_eps

        self.cache_graph_once    = True 
        self.graph_initialized_  = False 

    def build_preprocess(self) -> nn.Module | None:
        if self.anchor_stats is None: 
            raise ValueError("must call initialize_anchor_state()")

        return SpatialPatchPreprocessor(
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            anchor_stats=self.anchor_stats,
            swap_noise_prob=self.swap_noise_prob
        )

    def build_semantic_embed(self, sample: SSFEBatch) -> nn.Module:
        d_in = sample.semantic_input.shape[1]

        return SpatialSemanticEmbedder(
            in_dim=d_in,
            hidden_dim=self.semantic_hidden_dim,
            out_dim=self.semantic_out_dim,
            dropout=self.semantic_dropout,
            depth=self.semantic_depth 
        )

    def build_structural_embed(self, sample: SSFEBatch) -> nn.Module:
        if self.node_anchors is None: 
            raise ValueError("must call initialize_anchor_state()")

        d_in = sample.semantic_input.shape[1]
        if sample.stats_z is None: 
            raise ValueError("stats_z is required for structural embedder")
        d_stats = sample.stats_z.shape[1]

        return SpatialStructuralEmbedder(
            in_dim=d_in,
            stats_dim=d_stats,
            node_anchors=self.node_anchors,
            tile_size=self.tile_size,
            patch_size=self.patch_size,
            gnn_dim=self.gnn_dim,
            gnn_layers=self.gnn_layers,
            gnn_heads=self.gnn_heads,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            global_active_eps=self.global_active_eps,
            device=str(self.device)
        )

    def forward_preprocess(self, batch) -> SSFEBatch:
        if not isinstance(batch, (list, tuple)) or len(batch) < 5: 
            raise ValueError("expected spatial batch as "
                             "(tiles, labels, tile_batch_idx, tile_stats, wide_cond)")

        tiles          = batch[0].to(self.device, non_blocking=True, 
                                     memory_format=torch.channels_last)
        tile_batch_idx = batch[2].to(self.device, non_blocking=True)
        stats          = batch[3].to(self.device, non_blocking=True)
        wide_cond      = batch[4].to(self.device, non_blocking=True)

        return self.model_["preprocess"](
            tiles, 
            stats=stats, 
            wide_cond=wide_cond, 
            tile_batch_idx=tile_batch_idx,
        )

    def forward_semantic(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch in semantic path")
        if "sem_tile_pool" not in self.model_ or "sem_county_pool" not in self.model_: 
            raise RuntimeError("semantic pooling modules are not initialized")

        sem_node = self.model_["semantic"](prep.semantic_input)

        sem_tile = self.model_["sem_tile_pool"](
            sem_node, 
            prep.batch_idx.to(self.device, dtype=torch.long), 
            prep.n_tiles 
        )

        inv, n_county = self.resolve_county_index(prep.tile_batch_idx, device=self.device)
        sem_county = self.model_["sem_county_pool"](sem_tile, inv, n_county)
        return sem_node, sem_county 

    def forward_structural(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch in structural path")
        st_node, st_tile = self.model_["structural"](prep)

        if "st_county_pool" not in self.model_: 
            return st_node, st_tile

        inv, n_county = self.resolve_county_index(prep.tile_batch_idx, device=self.device)
        st_county = self.model_["st_county_pool"](st_tile, inv, n_county)
        return st_node, st_county 

    def spatial_patch_active_mask(
        self,
        prep: SSFEBatch,
        *,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch for active mask")
        if prep.stats_raw is None:
            raise ValueError("stats_raw required for active mask")

        stats = prep.stats_raw.to(device=device, dtype=dtype)
        if stats.ndim != 2:
            raise ValueError(f"expected patch stats shape (N, D), got {tuple(stats.shape)}")

        eps = max(float(self.global_active_eps), 1e-12)
        return (stats.abs().amax(dim=1) > eps).to(dtype)

    def forward_repr_target_bag(
        self, 
        prep: SSFEBatch, 
        *, 
        n_bags: int, 
        device: torch.device, 
        dtype: torch.dtype
    ) -> torch.Tensor:

        if not isinstance(prep, SpatialPatchBatch): 
            raise TypeError("SpatialSSFE expects SpatialPatchBatch for repr target pooling")
        if prep.stats_z is None:
            raise ValueError("stats_z required for active-weighted repr target pooling")
        target = prep.stats_z.to(device=device, dtype=dtype)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.shape[0] == n_bags:
            return target.detach()

        patch_to_tile = prep.batch_idx.to(device=device, dtype=torch.long).view(-1)
        tile_to_cty, n_county = self.resolve_county_index(prep.tile_batch_idx, device=device)
        if n_county != n_bags: 
            raise ValueError(f"county count mismatch: {n_county} != {n_bags}")
        if target.shape[0] != patch_to_tile.numel():
            raise ValueError("stats_z / patch index length mismatch for repr target pooling")

        patch_to_cty = tile_to_cty[patch_to_tile]
        patch_active = self.spatial_patch_active_mask(
            prep,
            device=device,
            dtype=dtype
        ).unsqueeze(1)
        if patch_active.shape[0] != target.shape[0]:
            raise ValueError("active mask / target length mismatch")

        target_masked = target * patch_active

        pooled = torch.zeros((n_bags, target.shape[1]), device=device, dtype=dtype)
        count  = torch.zeros((n_bags, 1), device=device, dtype=dtype)
        pooled.index_add_(0, patch_to_cty, target_masked)
        count.index_add_(0, patch_to_cty, patch_active)

        out = pooled / count.clamp_min(1.0)

        # Fallback for empty-active counties keeps target numerically stable.
        zero_active = (count.squeeze(1) <= 0)
        if bool(torch.any(zero_active)):
            pooled_all = torch.zeros_like(pooled)
            count_all  = torch.zeros_like(count)
            pooled_all.index_add_(0, patch_to_cty, target)
            count_all.index_add_(0, patch_to_cty, torch.ones_like(patch_active))
            out_all = pooled_all / count_all.clamp_min(1.0)
            out[zero_active] = out_all[zero_active]

        return out.detach()

    def forward_recon_weight_bag(
        self,
        prep: SSFEBatch,
        *,
        n_bags: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch for recon weighting")
        if prep.stats_raw is None:
            return torch.ones((n_bags,), device=device, dtype=dtype)

        patch_to_tile = prep.batch_idx.to(device=device, dtype=torch.long).view(-1)
        tile_to_cty, n_county = self.resolve_county_index(prep.tile_batch_idx, device=device)
        if n_county != n_bags:
            raise ValueError(f"county count mismatch: {n_county} != {n_bags}")
        if prep.stats_raw.shape[0] != patch_to_tile.numel():
            raise ValueError("stats_raw/patch index length mismatch for recon weighting")

        patch_to_cty = tile_to_cty[patch_to_tile]
        patch_active = self.spatial_patch_active_mask(
            prep,
            device=device,
            dtype=dtype
        )

        bag_sum = torch.zeros((n_bags,), device=device, dtype=dtype)
        bag_cnt = torch.zeros((n_bags,), device=device, dtype=dtype)
        bag_sum.index_add_(0, patch_to_cty, patch_active)
        bag_cnt.index_add_(0, patch_to_cty, torch.ones_like(patch_active))
        return bag_sum / bag_cnt.clamp_min(1.0)

    def anchor_features_from_batch(self, batch) -> np.ndarray: 
        if not isinstance(batch, (list, tuple)) or len(batch) < 4:
            raise ValueError("expected batch as (tiles, labels, tile_batch_idx, tile_stats)")

        stats = batch[3]
        if isinstance(stats, torch.Tensor):
            s = stats.detach().cpu().numpy()
        else:
            s = np.asarray(stats)

        if s.ndim != 3:
            raise ValueError(f"expected tile_stats shape (T,L,D), got {s.shape}")
        if s.shape[2] < self.in_channels:
            raise ValueError(
                f"tile_stats D={s.shape[2]} smaller than in_channels={self.in_channels}"
            )

        return np.asarray(
            s[..., :self.in_channels], 
            dtype=np.float32
        ).reshape(-1, self.in_channels)

    def initialize_structural_state(self, train_loader: DataLoader):
        return super().initialize_structural_state(train_loader)

    @staticmethod 
    def resolve_county_index(
        tile_batch_idx: torch.Tensor, 
        *, 
        device: torch.device 
    ) -> tuple[torch.Tensor, int]: 

        tidx = tile_batch_idx.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if tidx.numel() == 0: 
            return tidx, 0 
        _, inv = torch.unique(tidx, sorted=True, return_inverse=True)
        return inv, int(inv.max().item()) + 1

# --------------------------------------------------------- 
# Tabular SSFE Modules 
# --------------------------------------------------------- 

@dataclass 
class TabularBatch(SSFEBatch): 
    n_nodes: int 
    n_bags: int 
    graph_group_idx: torch.Tensor 
    node_ids: torch.Tensor | None = None 


class TabularPreprocessor(nn.Module): 

    def __init__(
        self,
        *,
        in_dim: int,
        embed_dim: int,
        anchor_stats: list[float], 
        swap_noise_prob: float = 0.15, 
    ): 
        super().__init__()

        mean = torch.as_tensor(anchor_stats[0], dtype=torch.float32)
        std  = torch.as_tensor(anchor_stats[1], dtype=torch.float32)

        if mean.ndim != 1 or std.ndim != 1 or mean.numel() != in_dim or std.numel() != in_dim: 
            raise ValueError("anchor_stats shape mismatch")
        if torch.any(std <= 0): 
            raise ValueError("invalid anchor std")

        self.swap_noise_prob = swap_noise_prob

        self.in_dim  = in_dim 
        self.encoder = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        ) 

        self.register_buffer("feat_mean", mean)
        self.register_buffer("feat_std", std)

    def forward(
        self,
        x: torch.Tensor, 
        wide_cond: torch.Tensor, 
        bag_idx: torch.Tensor | None = None, 
        graph_group_idx: torch.Tensor | None = None,
        node_ids: torch.Tensor | None = None 
    ): 
        if x.ndim != 2: 
            raise ValueError(f"expected (N, F), got {tuple(x.shape)}")

        xc = x.to(dtype=torch.float32)
        n  = x.shape[0]

        if wide_cond.ndim == 1: 
            wide_cond = wide_cond.unsqueeze(1)

        if bag_idx is None: 
            bag_idx = torch.arange(n, device=x.device, dtype=torch.long)
        else: 
            bag_idx = self.normalize_index(bag_idx, n, x.device)

        n_bags = int(bag_idx.max().item()) + 1 if n > 0 else 0 

        if wide_cond.shape[0] != n_bags: 
            raise ValueError("wide_cond must be bag-level")

        if graph_group_idx is None: 
            graph_group_idx = torch.zeros(n, device=x.device, dtype=torch.long)
        else: 
            graph_group_idx = self.normalize_index(graph_group_idx, n, x.device)

        if node_ids is None: 
            raise ValueError("node_ids are required.")
        if isinstance(node_ids, torch.Tensor): 
            node_ids = node_ids.to(
                device=xc.device, dtype=torch.long, non_blocking=True
            ).view(-1)
        if node_ids.numel() != n: 
            raise ValueError("node_ids length mismatch.")

        mean = self.feat_mean.view(1, -1) 
        std  = self.feat_std.view(1, -1)

        stats_raw_c = xc 
        stats_z_c   = (stats_raw_c - mean) / std 

        x = self.maybe_swap_noise(xc)

        stats_raw = x 
        stats_z   = (stats_raw - mean) / std 

        embs      = self.encoder(x)
        sem_in    = torch.cat([embs, stats_z], dim=1)

        return TabularBatch(
            repr_target=stats_z_c,
            semantic_input=sem_in,
            stats_z=stats_z,
            stats_raw=stats_raw,
            batch_idx=bag_idx,
            n_nodes=n,
            n_bags=n_bags,
            graph_group_idx=graph_group_idx,
            wide_cond=wide_cond,
            node_ids=node_ids 
        )

    def maybe_swap_noise(self, x: torch.Tensor): 
        p = self.swap_noise_prob 
        if (not self.training) or p <= 0.0: 
            return x 

        n, f = x.shape 
        if n < 2: 
            return x 

        mask = torch.rand((n, f), device=x.device) < p 
        if not bool(mask.any()): 
            return x 

        row = torch.arange(n, device=x.device).view(n, 1).expand(n, f)
        src = torch.randint(0, n, (n, f), device=x.device)
        src = torch.where(src == row, (src + 1) % n, src)
        col = torch.arange(f, device=x.device).view(1, f).expand(n, f)

        swapped = x[src, col]
        return torch.where(mask, swapped, x)

    @staticmethod
    def normalize_index(idx: torch.Tensor, n: int, device: torch.device) -> torch.Tensor: 
        idx = idx.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if idx.numel() != n: 
            raise ValueError(f"index length mismatch: {idx.numel()} != {n}")
        _, inv = torch.unique(idx, sorted=True, return_inverse=True)
        return inv 

class TabularSemanticEmbedder(nn.Module): 
    
    def __init__(
        self,
        *,
        in_dim: int, 
        out_dim: int, 
        proj_dim: int, 
        refine_hidden_dim: int, 
        refine_depth: int, 
        dropout: float, 
    ): 
        super().__init__() 

        self.projector = nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, proj_dim),
            nn.GELU(), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        )

        self.refine    = ResidualMLP(
            in_dim=proj_dim,
            hidden_dim=refine_hidden_dim,
            depth=refine_depth,
            dropout=dropout,
            out_dim=out_dim,
            zero_head_init=False 
        )

        self.node_norm = nn.LayerNorm(out_dim)
        self.bag_norm  = nn.LayerNorm(out_dim)
        self.out_dim   = out_dim 

    def forward(self, prep: TabularBatch) -> tuple[torch.Tensor, torch.Tensor]: 
        if prep.batch_idx.numel() != prep.semantic_input.size(0): 
            raise ValueError("batch_idx/feature length mismatch.")

        x     = prep.semantic_input 
        proj  = self.projector(x)
        node  = self.node_norm(self.refine(proj))

        bag   = node.new_zeros((prep.n_bags, node.size(1)))
        count = node.new_zeros((prep.n_bags, 1))

        bag.index_add_(0, prep.batch_idx, node)
        count.index_add_(0, prep.batch_idx, node.new_ones((node.size(0), 1)))

        bag   = self.bag_norm(bag / count.clamp_min(1.0))
        return node, bag 


class TabularStructuralEmbedder(nn.Module): 
    
    def __init__(
        self,
        *,
        in_dim: int, 
        stats_dim: int, 
        node_anchors: list[list[float]], 
        knn: int, 
        gnn_dim: int, 
        gnn_layers: int, 
        gnn_heads: int, 
        dropout: float, 
        attn_dropout: float, 
        global_active_eps: float = 1e-6, 
        device: str = "cuda"
    ): 
        super().__init__()

        anchors = torch.as_tensor(node_anchors, dtype=torch.float32)
        if anchors.ndim != 2 or anchors.shape[0] != 3: 
            raise ValueError("node_anchors must be (3, d_stats)")
        if anchors.shape[1] != stats_dim: 
            raise ValueError("node_anchors dim mismatch")

        self.knn   = max(1, knn) 
        self.graph = TabularHypergraph(
            anchors=anchors,
            knn=self.knn,
            global_active_eps=global_active_eps,
            device=device
        )

        self.gnn   = HyperGATStack(
            in_dim=in_dim,
            hidden_dim=gnn_dim,
            n_layers=gnn_layers,
            n_heads=gnn_heads,
            n_node_types=4,
            n_edge_types=3,
            dropout=dropout,
            attn_dropout=attn_dropout
        )

        self.readout_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.trunc_normal_(self.readout_token, std=0.02)

        self.node_norm = nn.LayerNorm(gnn_dim)
        self.bag_norm  = nn.LayerNorm(gnn_dim)
        self.out_dim   = gnn_dim 

    def forward(
        self, 
        prep: TabularBatch, 
        *,
        neighbors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        if prep.stats_raw is None or prep.stats_z is None: 
            raise ValueError("Tabular semantic path requires prep.stats_raw and prep.stats_z")

        x_nodes = prep.semantic_input 
        n = x_nodes.shape[0]
        if n == 0: 
            empty = x_nodes.new_zeros((0, self.out_dim))
            return empty, empty 

        g_graph = prep.graph_group_idx.to(device=x_nodes.device, dtype=torch.long).view(-1)
        if g_graph.numel() != n: 
            raise ValueError("graph_group_idx length mismatch")
        _, g_graph = torch.unique(g_graph, sorted=True, return_inverse=True)
        b_graph    = int(g_graph.max().item()) + 1 

        neighbors  = neighbors.to(device=x_nodes.device, dtype=torch.long, non_blocking=True)
        if neighbors.ndim != 2 or neighbors.shape[0] != n: 
            raise ValueError(f"neighbors shape mismatch: got {tuple(neighbors.shape)}")

        readout = (self.readout_token.expand(b_graph, -1, -1).reshape(b_graph, -1)
                   .to(x_nodes.device))
        x_all   = torch.cat([x_nodes, readout], dim=0)

        meta    = self.graph.build(
            prep.stats_z, 
            batch_size=b_graph,
            active_stats=prep.stats_raw,
            group_ids=g_graph,
            neighbors=neighbors
        )

        node_idx, edge_idx = meta.incidence_index 
        h_all  = self.gnn(x_all, meta.node_type, meta.edge_type, node_idx, edge_idx)
        h_node = self.node_norm(h_all[:n])

        bag_idx = prep.batch_idx.to(device=x_nodes.device, dtype=torch.long)
        h_bag   = h_node.new_zeros((prep.n_bags, h_node.size(1)))
        count   = h_node.new_zeros((prep.n_bags, 1))

        h_bag.index_add_(0, bag_idx, h_node)
        count.index_add_(0, bag_idx, h_node.new_ones((n, 1)))
        h_bag   = self.bag_norm(h_bag / count.clamp_min(1.0))
        return h_node, h_bag 

# ---------------------------------------------------------
# SSFE model for Tabular Data 
# ---------------------------------------------------------

class TabularSSFE(SSFEBase): 

    def __init__(
        self, 
        *, 
        # tabular metadata 
        in_dim: int, 
        embed_dim: int, 
        swap_noise_prob: float = 0.20, 

        # semantic branch 
        semantic_out_dim: int, 
        semantic_proj_dim: int, 
        semantic_hidden_dim: int, 
        semantic_depth: int = 2, 
        semantic_dropout: float, 

        # structural branch 
        tabular_knn: int, 
        gnn_dim: int, 
        gnn_heads: int, 
        gnn_layers: int, 
        dropout: float, 
        attn_dropout: float, 
        global_active_eps: float = 1e-6, 

        # base model 
        epochs: int = 300, 
        lr: float = 0.001, 
        weight_decay: float = 0, 
        batch_size: int = 256, 
        eval_fraction: float = 0.15, 
        early_stopping_rounds: int = 20, 
        min_delta: float = 0.0001, 
        random_state: int = 0, 
        collate_fn=None, 
        w_contrast: float = 1.0, 
        w_cluster: float = 1.0, 
        w_recon: float = 1.0, 
        w_hsic: float = 1.0, 
        hsic_sigma: Optional[float] = None, 
        hsic_view: str = "concat", 
        shared_view: str = "structural", 
        private_view: str = "semantic", 
        contrast_temperature: float, 
        cluster_temperature: float, 
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None, 
        compile_model: bool = True
    ):
        super().__init__(
            epochs=epochs, 
            swap_noise_prob=swap_noise_prob,
            lr=lr, 
            weight_decay=weight_decay, 
            batch_size=batch_size, 
            eval_fraction=eval_fraction, 
            early_stopping_rounds=early_stopping_rounds, 
            min_delta=min_delta, 
            random_state=random_state, 
            collate_fn=collate_fn, 
            w_contrast=w_contrast, 
            w_cluster=w_cluster, 
            w_recon=w_recon, 
            w_hsic=w_hsic,
            hsic_sigma=hsic_sigma,
            hsic_view=hsic_view,
            shared_view=shared_view,
            private_view=private_view,
            contrast_temperature=contrast_temperature, 
            cluster_temperature=cluster_temperature, 
            n_prototypes=n_prototypes, 
            proj_dim=proj_dim, 
            device=device, 
            compile_model=compile_model
        )

        self.in_dim                   = in_dim
        self.embed_dim                = embed_dim

        self.semantic_out_dim         = semantic_out_dim
        self.semantic_proj_dim        = semantic_proj_dim
        self.semantic_hidden_dim      = semantic_hidden_dim
        self.semantic_depth           = semantic_depth
        self.semantic_dropout         = semantic_dropout

        self.tabular_knn              = tabular_knn
        self.gnn_dim                  = gnn_dim
        self.gnn_layers               = gnn_layers
        self.gnn_heads                = gnn_heads
        self.dropout                  = dropout
        self.attn_dropout             = attn_dropout
        self.global_active_eps        = global_active_eps

        # nearest neighbor cache 
        self.neighbor_cache_computed_               = False 
        self.node_ids_global_: torch.Tensor | None  = None 
        self.neighbors_global_: torch.Tensor | None = None 
        self.node_ids_sorted_: torch.Tensor | None  = None 
        self.row_from_sorted_: torch.Tensor | None  = None 

    def build_preprocess(self) -> nn.Module | None:
        if self.anchor_stats is None: 
            raise ValueError("must call initialize_anchor_state()")
        return TabularPreprocessor(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            anchor_stats=self.anchor_stats,
            swap_noise_prob=self.swap_noise_prob
        )

    def build_semantic_embed(self, sample: SSFEBatch) -> nn.Module:
        d_in = sample.semantic_input.shape[1]
        return TabularSemanticEmbedder(
            in_dim=d_in ,
            out_dim=self.semantic_out_dim,
            proj_dim=self.semantic_proj_dim,
            refine_hidden_dim=self.semantic_hidden_dim,
            refine_depth=self.semantic_depth,
            dropout=self.semantic_dropout
        )

    def build_structural_embed(self, sample: SSFEBatch) -> nn.Module:
        if self.node_anchors is None or sample.stats_z is None: 
            raise ValueError("must call initialize_anchor_state() and stats_z must be present")

        return TabularStructuralEmbedder(
            in_dim=sample.semantic_input.shape[1],
            stats_dim=sample.stats_z.shape[1],
            node_anchors=self.node_anchors,
            knn=self.tabular_knn,
            gnn_dim=self.gnn_dim,
            gnn_heads=self.gnn_heads,
            gnn_layers=self.gnn_layers,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            global_active_eps=self.global_active_eps,
            device=str(self.device)
        )

    def forward_preprocess(self, batch) -> SSFEBatch:
        bag_idx         = None 
        graph_group_idx = None 
        node_ids        = None 
        wide_cond       = None 

        if isinstance(batch, (list, tuple)): 
            if len(batch) < 1: 
                raise ValueError("empty tabular batch")
            x = batch[0]
            if (len(batch) == 3 and self.is_index_like(batch[1]) and 
                not self.is_index_like(batch[2])): 
                node_ids  = batch[1]
                wide_cond = batch[2]
            else: 
                if len(batch) >= 3:
                    bag_idx = batch[2]
                if len(batch) >= 4:
                    graph_group_idx = batch[3]
                if len(batch) >= 5:
                    node_ids = batch[4]
                elif len(batch) >= 2 and self.is_index_like(batch[1]):
                    node_ids = batch[1]
                if len(batch) >= 6:
                    wide_cond = batch[5]
        else: 
            x = batch 

        if isinstance(x, torch.Tensor): 
            x = x.to(self.device, dtype=torch.float32, non_blocking=True)
        else: 
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if bag_idx is not None: 
            if isinstance(bag_idx, torch.Tensor): 
                bag_idx = bag_idx.to(self.device, non_blocking=True)
            else: 
                bag_idx = torch.as_tensor(bag_idx, dtype=torch.long, device=self.device)

        if graph_group_idx is not None: 
            if isinstance(graph_group_idx, torch.Tensor): 
                graph_group_idx = graph_group_idx.to(self.device, non_blocking=True)
            else: 
                graph_group_idx = torch.as_tensor(graph_group_idx, dtype=torch.long, 
                                                  device=self.device)

        if node_ids is None: 
            raise ValueError("tabular batch missing node_ids, required for cached neighbors")
        if isinstance(node_ids, torch.Tensor): 
            node_ids = node_ids.to(self.device, dtype=torch.long, non_blocking=True).view(-1)
        else: 
            node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.device).view(-1)

        if wide_cond is None: 
            raise ValueError("tabular batch missing wide_cond")
        if isinstance(wide_cond, torch.Tensor): 
            wide_cond = wide_cond.to(self.device, dtype=torch.float32, non_blocking=True) 
        else: 
            wide_cond = torch.as_tensor(wide_cond, dtype=torch.float32, device=self.device)

        return self.model_["preprocess"](
            x, 
            bag_idx=bag_idx,
            graph_group_idx=graph_group_idx,
            wide_cond=wide_cond,
            node_ids=node_ids
        )

    def forward_semantic(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, TabularBatch): 
            raise TypeError("TabularSSFE expects TabularBatch in semantic path")
        return self.model_["semantic"](prep)

    def forward_structural(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if self.node_anchors is None or prep.stats_z is None: 
            raise ValueError("must call initialize_anchor_state() and stats_z must be present")
        if not isinstance(prep, TabularBatch) or prep.node_ids is None: 
            raise TypeError("TabularSSFE expects TabularBatch in structural path")
        if not self.neighbor_cache_computed_: 
            raise RuntimeError("global neighbors not initialized")

        neighbors = self.induce_local_neighbors(prep.node_ids)
        return self.model_["structural"](prep, neighbors=neighbors)

    def forward_repr_target_bag(
        self,
        prep: SSFEBatch, 
        *, 
        n_bags: int, 
        device: torch.device, 
        dtype: torch.dtype 
    ) -> torch.Tensor: 
        if not isinstance(prep, TabularBatch): 
            raise TypeError("TabularSSFE expects TabularBatch for repr target pooling.")
        target = prep.repr_target.to(device=device, dtype=dtype)
        if target.ndim == 1: 
            target = target.unsqueeze(1)
        if target.shape[0] == n_bags: 
            return target 
        raise ValueError("expects county-level.")

    def anchor_features_from_batch(self, batch) -> np.ndarray:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch 

        arr = self.as_numpy(x, dtype=np.float32)
        if arr.ndim != 2: 
            raise ValueError(f"expected tabular batch shape (N, F), got {arr.shape}")
        elif arr.shape[1] != self.in_dim: 
            raise ValueError(f"feature dim {arr.shape[1]} != in_dim {self.in_dim}")
        return arr  

    def extract_cache_fields(self, batch): 
        if not isinstance(batch, (tuple, list)) or len(batch) < 1: 
            raise ValueError("expected tabular batch with x in position 0")

        x = self.as_numpy(batch[0], dtype=np.float32)
        if x.ndim == 1: 
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.in_dim: 
            raise ValueError(f"expected (N, {self.in_dim}), got {x.shape}")

        graph = None 
        if len(batch) >= 4 and batch[3] is not None: 
            graph = self.as_numpy(batch[3], dtype=np.int64).reshape(-1)
        if graph is None: 
            graph = np.zeros((x.shape[0], ), dtype=np.int64)

        node_ids = None 
        if len(batch) >= 5 and batch[4] is not None: 
            node_ids = self.as_numpy(batch[4], dtype=np.int64).reshape(-1)
        elif len(batch) >= 2 and self.is_index_like(batch[1]): 
            node_ids = self.as_numpy(batch[1], dtype=np.int64).reshape(-1)

        if node_ids is None: 
            raise ValueError("node_ids are required to precomptue global tabular neighbors")
        if graph.shape[0] != x.shape[0]: 
            raise ValueError("graph_group_idx length mismatch")

        return x, graph, node_ids 

    def initialize_structural_state(self, train_loader: DataLoader):
        if self.anchor_stats is None: 
            raise ValueError("anchor_stats must be initialized before neighbor cache")

        xs, gs, ids = [], [], []

        for batch in train_loader: 
            x, g, node_ids = self.extract_cache_fields(batch)
            xs.append(x)
            gs.append(g)
            ids.append(node_ids)

        X = np.vstack(xs).astype(np.float32, copy=False)
        G = np.concatenate(gs).astype(np.int64, copy=False)
        I = np.concatenate(ids).astype(np.int64, copy=False)

        if np.unique(I).size != I.size: 
            raise ValueError("node_ids must be unique for global neighbor cache")

        mean = np.asarray(self.anchor_stats[0], dtype=np.float32).reshape(1, -1)
        std  = np.maximum(
            np.asarray(self.anchor_stats[1], dtype=np.float32).reshape(1, -1), 1e-6
        )
        Z = (X - mean) / std 

        nbr_ids = self.build_global_neighbor_ids(Z, I, G, self.tabular_knn)

        self.node_ids_global_  = torch.from_numpy(I).to(self.device, dtype=torch.long)
        self.neighbors_global_ = torch.from_numpy(nbr_ids).to(self.device, dtype=torch.long)

        self.node_ids_sorted_, sort_idx = torch.sort(self.node_ids_global_) 
        self.row_from_sorted_ = sort_idx 
        
        self.neighbor_cache_computed_ = True 

        print(f"[neighbors:init] cached {I.size} nodes | k={self.tabular_knn}", file=sys.stderr)

    def induce_local_neighbors(self, node_ids: torch.Tensor) -> torch.Tensor: 
        if (not self.neighbor_cache_computed_ or self.neighbors_global_ is None or 
            self.node_ids_sorted_ is None or self.row_from_sorted_ is None): 
            raise RuntimeError("global neighbor cache not initialized")

        ids = node_ids.detach().to(self.device, dtype=torch.long, non_blocking=True).view(-1) 
        n   = int(ids.numel())
        k   = int(self.tabular_knn)
        if n == 0: 
            return torch.empty((0, k), device=self.device, dtype=torch.long)
        
        pos = torch.searchsorted(self.node_ids_sorted_, ids)
        in_range = pos < self.node_ids_sorted_.numel()
        pos_c = pos.clamp(max=self.node_ids_sorted_.numel() - 1)
        found = in_range & (self.node_ids_sorted_[pos_c] == ids)
        if (~found).any():
            raise ValueError("batch contains node_ids missing from global neighbor cache")

        rows = self.row_from_sorted_[pos_c]
        nbr_global_ids = self.neighbors_global_[rows]

        ids_sorted, order = torch.sort(ids)
        local_pos_sorted  = torch.empty_like(order)
        local_pos_sorted[order] = torch.arange(n, device=self.device, dtype=torch.long)

        flat = nbr_global_ids.reshape(-1)

        lookup    = torch.searchsorted(ids_sorted, flat)
        in_range2 = lookup < n 
        lookup_c  = lookup.clamp(max=n - 1)
        hit       = (flat >= 0) & in_range2 & (ids_sorted[lookup_c] == flat)

        local_flat = torch.full_like(flat, -1)
        local_flat[hit] = local_pos_sorted[lookup_c[hit]]
        return local_flat.view(n, k)

    def make_loader(self, dataset, shuffle: bool):
        if isinstance(dataset, dict) and "features" in dataset:
            x = torch.from_numpy(np.asarray(dataset["features"], dtype=np.float32))
            ids_np = dataset.get("node_ids", np.arange(x.shape[0], dtype=np.int64))
            ids = torch.from_numpy(np.asarray(ids_np, dtype=np.int64))

            w_np = dataset.get("wide_cond", dataset["features"])
            w = torch.from_numpy(np.asarray(w_np, dtype=np.float32))
            if w.shape[0] != x.shape[0]: 
                raise ValueError("wide_cond row count must match features")
            dataset = TensorDataset(x, ids, w)

        if isinstance(dataset, np.ndarray): 
            x   = torch.from_numpy(np.asarray(dataset, dtype=np.float32))
            ids = torch.arange(x.shape[0], dtype=torch.long)
            dataset = TensorDataset(x, ids, x.clone()) 

        elif isinstance(dataset, torch.Tensor): 
            x   = dataset.detach().cpu().to(torch.float32)
            ids = torch.arange(x.shape[0], dtype=torch.long)
            dataset = TensorDataset(x, ids, x.clone())

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

    @staticmethod 
    def is_index_like(x) -> bool: 
        if isinstance(x, torch.Tensor): 
            return x.ndim == 1 and x.dtype in (torch.int32, torch.int64)
        arr = np.asarray(x)
        return arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)

    @staticmethod 
    def build_global_neighbor_ids(
        stats_z: np.ndarray,
        node_ids: np.ndarray,
        group_ids: np.ndarray, 
        k: int 
    ) -> np.ndarray: 

        n   = node_ids.shape[0]
        out = np.full((n, k), -1, dtype=np.int64)
        if n == 0: 
            return out 

        for g in np.unique(group_ids): 
            idx = np.flatnonzero(group_ids == g)
            m   = idx.size 
            if m <= 1: 
                continue 

            kk = min(k, m - 1)
            nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
            nn.fit(stats_z[idx])

            nbr_local = nn.kneighbors(stats_z[idx], return_distance=False)[:, 1:]
            out[idx, :kk] = node_ids[idx[nbr_local]]

        return out

# ---------------------------------------------------------
# Orthogonal (via HSIC) Multi-view hypergraph autoencoders 
# TengQi et. al, 2016
# ---------------------------------------------------------

class MultiviewManagerSSFE(BaseEstimator): 

    def __init__(
        self,
        *,
        experts: Mapping[str, SSFEBase], 
        global_dim: int, 
        gate_floor: float = 0., 

        epochs: int = 500, 
        lr: float, 
        weight_decay: float, 

        early_stopping_rounds: int = 40, 
        min_delta: float = 1e-4, 
        w_recon: float = 1.0, 
        w_align: float = 1.0, 
        w_privacy: float = 1.0,
        w_var_reg: float = 1.0,
        schedule_loss_weights: bool = True,
        align_start_pct: float = 0.10, 
        align_end_pct: float = 0.40, 
        privacy_start_pct: float = 0.00, 
        privacy_end_pct: float = 0.50, 
        privacy_ramp_power: float = 2.0, 
        sinkhorn_epsilon: float = 0.05, 
        hsic_sigma: Optional[float] = None,
        recon_cross_scale: float = 1.0, 
        align_l2_scale: float = 1.0, 
        align_kl_scale: float = 1.0, 
        var_reg_gamma: float = 1.0,
        var_reg_eps: float = 1e-4,
        random_state: int = 0, 
        device: str = "cuda"
    ):
        if len(experts) < 2: 
            raise ValueError("MultiviewManagerSSFE requires at least 2 classes")

        self.experts               = experts
        self.expert_ids            = list(self.experts.keys())

        self.global_dim            = global_dim
        self.epochs                = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state          = random_state
        self.hsic_sigma            = hsic_sigma
        self.gate_floor            = gate_floor
        self.lr                    = lr
        self.weight_decay          = weight_decay
        self.min_delta             = min_delta

        self.recon_cross_scale     = recon_cross_scale  

        self.w_recon_target        = w_recon 
        self.w_align_target        = w_align 
        self.w_privacy_target      = w_privacy 
        self.w_var_reg_target      = w_var_reg

        self.w_recon               = self.w_recon_target
        self.w_align               = self.w_align_target
        self.w_privacy             = self.w_privacy_target
        self.w_var_reg             = self.w_var_reg_target
        self.schedule_loss_weights = bool(schedule_loss_weights)
        
        self.align_start_pct       = align_start_pct
        self.align_end_pct         = align_end_pct 
        self.privacy_start_pct     = privacy_start_pct 
        self.privacy_end_pct       = privacy_end_pct 
        self.privacy_ramp_power    = privacy_ramp_power 

        self.sinkhorn_epsilon      = sinkhorn_epsilon
        self.recon_cross_scale     = recon_cross_scale
        self.align_l2_scale        = align_l2_scale
        self.align_kl_scale        = align_kl_scale
        self.var_reg_gamma         = var_reg_gamma
        self.var_reg_eps           = var_reg_eps

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_ = nn.ModuleDict({
            "shared_proj": nn.ModuleDict(),
            "gate_head": nn.ModuleDict(),
            "self_recon": nn.ModuleDict(),
            "cross_recon": nn.ModuleDict(),
        })

        self.loss_: nn.Module | None = None
        self.opt_: torch.optim.Optimizer | None = None
        self.scheduler_: Any | None = None
        self.best_val_score_: float = float("inf")
        self.is_fitted_: bool = False
        self.initialized_: bool = False

    # -----------------------------------------------------
    # Model Fit 
    # -----------------------------------------------------

    def fit(self, train_loader, val_loader=None): 
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.init_fit(train_loader)
        best_manager, best_expert, best_val = self.run_phase(
            name="OMDAE-HSIC", 
            train_loader=train_loader, 
            val_loader=val_loader
        )

        if best_manager is not None: 
            self.model_.load_state_dict(best_manager) 
        if best_expert is not None: 
            for eid, ex in self.experts.items(): 
                ex.model_.load_state_dict(best_expert[eid])

        self.best_val_score_ = best_val 
        self.is_fitted_      = True 
        return self 

    # ----------------------------------------------------- 
    # Per Expert Extraction 
    # ----------------------------------------------------- 

    def extract(self, X: Mapping[str, Any]) -> dict[str, NDArray]: 

        if not self.is_fitted_ or not self.initialized_: 
            raise RuntimeError("call fit() before extract()")

        batch_map = self.assert_batch_contract(X)
        loaders   = {
            eid: self.experts[eid].ensure_loader(batch_map[eid], shuffle=False)
            for eid in self.expert_ids 
        }

        for ex in self.experts.values(): 
            ex.model_.eval() 
        self.model_.eval() 
        if self.loss_ is not None: 
            self.loss_.eval() 

        per_expert: dict[str, list[NDArray]] = {eid: [] for eid in self.expert_ids}
        global_out: list[NDArray] = []

        with torch.no_grad(): 
            iters = [iter(loaders[eid]) for eid in self.expert_ids]
            for batches in zip(*iters): 
                cur = {eid: b for eid, b in zip(self.expert_ids, batches)}
                states: dict[str, SSFEMultiviewState] = {}
                shared_proj: dict[str, torch.Tensor]  = {}

                for eid in self.expert_ids: 
                    st = self.experts[eid].build_multiview_state(cur[eid])
                    states[eid] = st 
                    shared_proj[eid] = self.model_["shared_proj"][eid](st.shared_bag) 

                _, s_global = self.consensus(shared_proj)

                for eid in self.expert_ids: 
                    st = states[eid] 
                    per_expert[eid].append(st.private_bag.detach().cpu().numpy())

                global_out.append(s_global.detach().cpu().numpy())

        out: dict[str, NDArray] = {}
        for eid in self.expert_ids: 
            arrs = per_expert[eid] 
            out[eid] = (np.vstack(arrs).astype(np.float32, copy=False) if arrs else
                        np.empty((0, 0), dtype=np.float32))
        out["shared_global"] = (np.vstack(global_out).astype(np.float32, copy=False) 
                                if global_out else np.empty((0, 0), dtype=np.float32)) 
        return out 
        
    # -----------------------------------------------------
    # Fit Helpers  
    # -----------------------------------------------------

    def run_phase(
        self,
        *,
        name: str, 
        train_loader, 
        val_loader 
    ): 
        if self.opt_ is None: 
            raise RuntimeError("opt_ not initialized.")
        
        print(f"[{name}] starting...", file=sys.stderr)

        best_val     = float("inf")
        best_manager = None 
        best_expert  = None 
        patience     = 0 
        t0           = time.perf_counter()

        for ep in range(self.epochs): 
            self.update_loss_weights(ep)
            train_loss          = self.train_epoch(train_loader)
            val_loss, val_parts = self.validate(val_loader)

            score = train_loss if val_loss is None else val_loss 

            if score < best_val - self.min_delta: 
                best_val     = score 
                best_manager = copy.deepcopy(self.model_.state_dict()) 
                best_expert  = {
                    eid: copy.deepcopy(ex.model_.state_dict()) 
                    for eid, ex in self.experts.items() 
                }
                patience  = 0 
            else: 
                patience += 1
                if self.early_stopping_rounds > 0 and patience >= self.early_stopping_rounds: 
                    break 

            if ep % 5 == 0: 
                dt = (time.perf_counter() - t0) / (ep + 1)
                if val_loss is None:
                    print(
                        f"[mv epoch {ep:3d}] {dt:.2f}s avg | train={train_loss:.4f}",
                        file=sys.stderr
                    )
                else:
                    print(
                        f"[mv epoch {ep:3d}] {dt:.2f}s avg | "
                        f"val_total={val_loss:.4f} | "
                        f"val_rec={val_parts.get('recon', 0.0):.4f} | "
                        f"val_align={val_parts.get('align', 0.0):.4f} | "
                        f"val_priv={val_parts.get('privacy', 0.0):.4f} | "
                        f"val_var={val_parts.get('var_reg', 0.0):.4f}",
                        file=sys.stderr
                    )
                    if ep % 10 == 0:
                        rec_terms = []
                        hsic_terms = []
                        for eid in self.expert_ids:
                            rec_total = val_parts.get(f"recon.{eid}_self", 0.0) + \
                                        self.recon_cross_scale * val_parts.get(f"recon.{eid}_cross", 0.0)
                            hsic_total = val_parts.get(f"privacy.{eid}_hsic_p_s", 0.0) + \
                                         val_parts.get(f"privacy.{eid}_hsic_p_w", 0.0)
                            rec_terms.append(f"{eid}={rec_total:.4f}")
                            hsic_terms.append(f"{eid}={hsic_total:.4f}")

                        hsic_sw = val_parts.get("privacy.hsic_s_wide", 0.0)
                        print(
                            f"[mv epoch {ep:3d}] detail | "
                            f"recon_expert[{', '.join(rec_terms)}] | "
                            f"hsic_expert[{', '.join(hsic_terms)}] | "
                            f"hsic_s_wide={hsic_sw:.4f}",
                            file=sys.stderr
                        )

        return best_manager, best_expert, best_val

    def train_epoch(self, loader) -> float: 
        if self.opt_ is None: 
            raise RuntimeError("opt_ not initialized")

        # Ensure in training context for gradients 
        for ex in self.experts.values(): 
            ex.model_.train() 
        self.model_.train() 
        if self.loss_ is not None: 
            self.loss_.train() 

        total = 0.0 
        count = 0 
        for batch in loader: 
            loss, _, bsz = self.process_batch(batch)

            self.opt_.zero_grad(set_to_none=True)
            loss.backward() 
            self.opt_.step() 

            total += float(loss.item()) * bsz 
            count += bsz 

        if self.scheduler_ is not None: 
            self.scheduler_.step() 

        return total / max(1, count) 

    def validate(self, loader): 
        if loader is None: 
            return None, {}

        # Ensure in evalute context for gradients 
        for ex in self.experts.values(): 
            ex.model_.eval() 
        self.model_.eval() 
        if self.loss_ is not None: 
            self.loss_.eval() 

        sums: dict[str, float] = {}
        count = 0 
        with torch.no_grad(): 
            for batch in loader: 
                _, terms, bsz = self.process_batch(batch)
                for k, v in terms.items(): 
                    sums[k] = sums.get(k, 0.0) + float(v.item()) * bsz 
                count += bsz 

        denom = max(1, count)
        parts = {k: (v / denom) for k, v in sums.items()}
        total = float(
            parts.get("recon", 0.0) +
            parts.get("align", 0.0) +
            parts.get("privacy", 0.0) +
            parts.get("var_reg", 0.0)
        ) 
        return total, parts 

    def process_batch(self, raw_batch): 
        batch = self.assert_batch_contract(raw_batch)
        return self.forward_batch(batch)

    def forward_batch(self, batch_map: Mapping[str, Any]): 
        if not self.initialized_ or self.loss_ is None: 
            raise RuntimeError("loss not initialized.")

        states: dict[str, SSFEMultiviewState]    = {}
        shared_proj: dict[str, torch.Tensor]     = {}
        private_bags: dict[str, torch.Tensor]    = {}
        repr_targets: dict[str, torch.Tensor]    = {}
        recon_weights: dict[str, torch.Tensor]   = {}
        sinkhorn_logits: dict[str, torch.Tensor] = {}

        for eid in self.expert_ids: 
            ex = self.experts[eid]
            st = ex.build_multiview_state(batch_map[eid])
            states[eid] = st 

            s = self.model_["shared_proj"][eid](st.shared_bag)
            shared_proj[eid]  = s
            private_bags[eid] = st.private_bag 
            repr_targets[eid] = st.repr_target_bag
            if st.recon_weight_bag is None:
                recon_weights[eid] = torch.ones(
                    (st.n_bags,), device=st.shared_bag.device, dtype=st.shared_bag.dtype
                )
            else:
                recon_weights[eid] = st.recon_weight_bag

            if st.shared_sinkhorn_logits is None: 
                raise ValueError(f"{eid} missing shared_sinkhorn_logits")
            sinkhorn_logits[eid] = st.shared_sinkhorn_logits 

        wide = states[self.expert_ids[0]].wide_cond_bag
        for eid in self.expert_ids[1:]:
            if states[eid].wide_cond_bag.shape[0] != wide.shape[0]:
                raise ValueError(f"bag-count mismatch for expert={eid}")

        alpha, s_global = self.consensus(shared_proj)

        pack = self.loss_(
            shared_bags=shared_proj,
            private_bags=private_bags,
            repr_targets=repr_targets,
            recon_weights=recon_weights,
            shared_global=s_global,
            sinkhorn_logits=sinkhorn_logits,
            attention_weights=alpha,
            wide_cond=wide,
            w_recon=self.w_recon,
            w_align=self.w_align,
            w_privacy=self.w_privacy,
            w_var_reg=self.w_var_reg,
        )

        terms = dict(pack.raw)
        terms.update(pack.metrics)
        bsz   = s_global.shape[0]
        return pack.total, terms, bsz 

    def consensus(
        self, 
        shared_proj: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Compute attention and take consensus as the average sum weighted via attention scores. 
        Returns both attention and global public embeddings so alignment can use attention score.
        '''

        scores = torch.cat(
            [self.model_["gate_head"][eid](shared_proj[eid]) 
            for eid in self.expert_ids], dim=1
        ) 
        alpha  = F.softmax(scores, dim=1)

        if self.gate_floor > 0.0: 
            e     = len(self.expert_ids)
            floor = min(max(self.gate_floor, 0.0), 1.0 / float(e))
            alpha = alpha * (1.0 - floor * e) + floor 
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-9)

        s_global = torch.zeros_like(shared_proj[self.expert_ids[0]])
        for i, eid in enumerate(self.expert_ids): 
            s_global = s_global + alpha[:, i:i+1] * shared_proj[eid]
        return alpha, s_global 

    def update_loss_weights(self, epoch: int): 
        self.w_recon = self.w_recon_target 
        if not self.schedule_loss_weights:
            self.w_align = self.w_align_target
            self.w_privacy = self.w_privacy_target
            self.w_var_reg = self.w_var_reg_target
            return

        t = epoch / max(1, self.epochs - 1)

        if t < self.align_start_pct: 
            self.w_align = 0.0 
        elif t < self.align_end_pct: 
            r = (t - self.align_start_pct) / max(1e-9, self.align_end_pct - self.align_start_pct)
            self.w_align = self.w_align_target * r 
        else: 
            self.w_align = self.w_align_target 

        if t < self.privacy_start_pct: 
            self.w_privacy = 0.0
            self.w_var_reg = 0.0
        elif t < self.privacy_end_pct: 
            r = (t - self.privacy_start_pct) / max(1e-9, self.privacy_end_pct - self.privacy_start_pct) 
            self.w_privacy = self.w_privacy_target * (r ** self.privacy_ramp_power)
            self.w_var_reg = self.w_var_reg_target * (r ** self.privacy_ramp_power)
        else: 
            self.w_privacy = self.w_privacy_target
            self.w_var_reg = self.w_var_reg_target

    # -----------------------------------------------------
    # Initialization and Assertions   
    # -----------------------------------------------------

    def init_fit(self, train_loader): 
        self.assert_experts_ready() 
        sample = self.assert_batch_contract(next(iter(train_loader)))
        self.init(sample)

    def init(self, sample_batch: Mapping[str, Any]): 
        states = {}
        with torch.no_grad(): 
            for eid in self.expert_ids: 
                ex = self.experts[eid]
                ex.model_.eval() 
                states[eid] = ex.build_multiview_state(sample_batch[eid])

        n_bags = None 
        for eid in self.expert_ids: 
            nb = int(states[eid].n_bags) 
            n_bags = nb if n_bags is None else n_bags 
            if nb != n_bags: 
                raise ValueError(f"n_bags mismatch: {eid} has {nb}, expected {n_bags}")

        for eid in self.expert_ids: 
            s_dim = states[eid].shared_bag.shape[1]
            p_dim = states[eid].private_bag.shape[1]
            t_dim = states[eid].repr_target_bag.shape[1]

            self.model_["shared_proj"][eid] = nn.Sequential(
                nn.LayerNorm(s_dim),
                nn.Linear(s_dim, self.global_dim),
                nn.GELU(), 
                nn.LayerNorm(self.global_dim)
            ).to(self.device)

            self.model_["gate_head"][eid] = nn.Linear(self.global_dim, 1).to(self.device)

            in_dim = self.global_dim + p_dim 
            self.model_["self_recon"][eid] = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(), 
                nn.Linear(in_dim, t_dim)
            ).to(self.device)

            self.model_["cross_recon"][eid] = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(), 
                nn.Linear(in_dim, t_dim)
            ).to(self.device)

        self.loss_ = build_ssfe_multiview_loss(
            expert_ids=self.expert_ids,
            self_recon_heads=self.model_["self_recon"], 
            cross_recon_heads=self.model_["cross_recon"], 
            sinkhorn_epsilon=self.sinkhorn_epsilon,
            hsic_sigma=self.hsic_sigma,
            recon_cross_scale=self.recon_cross_scale,
            align_l2_scale=self.align_l2_scale,
            align_kl_scale=self.align_kl_scale,
            var_reg_gamma=self.var_reg_gamma,
            var_reg_eps=self.var_reg_eps,
        ).to(self.device)

        expert_params = []
        for eid in self.expert_ids: 
            expert_params.extend(self.experts[eid].model_.parameters())
        manager_params = self.model_.parameters()
        loss_params    = self.loss_.parameters() 
        params = self.concat_params(expert_params, manager_params, loss_params)

        self.opt_       = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_, T_max=self.epochs
        )
        self.initialized_ = True 
            

    def assert_batch_contract(self, batch: Any) -> Mapping[str, Any]: 
        if not isinstance(batch, Mapping): 
            raise TypeError("Multiview batch must be Mapping[str, Any]")
        missing = [eid for eid in self.expert_ids if eid not in batch]
        extra   = [k for k in batch.keys() if k not in self.expert_ids]
        if missing: 
            raise KeyError(f"batch missing experts: {missing}")
        if extra: 
            raise KeyError(f"batch has unexpected experts: {extra}")
        return batch 

    def assert_experts_ready(self): 
        for eid, ex in self.experts.items(): 
            if not isinstance(ex, SSFEBase): 
                raise TypeError(f"{eid} is not SSFEBase")
            if "semantic" not in ex.model_ or "structural" not in ex.model_: 
                raise RuntimeError(f"{eid} is not initialized.")

    @staticmethod 
    def concat_params(*param_groups): 
        out  = []
        seen = set() 
        for group in param_groups: 
            for p in group: 
                if not p.requires_grad:
                    continue 
                pid = id(p)
                if pid in seen: 
                    continue 
                seen.add(pid)
                out.append(p)
        return out 
