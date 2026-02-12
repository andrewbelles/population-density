#!/usr/bin/env python3 
# 
# ssfe.py  Andrew Belles  Feb 10th, 2026 
# 
# Self-Supervised Feature Extractors for learning latent representations for deep component of 
# model architecture (wide & deep)
# 

import torch, time, copy, sys, hashlib 

import torch.nn as nn 

import torch.nn.functional as F 

import numpy as np 

from abc import ABC, abstractmethod 

from contextlib import nullcontext 

from dataclasses import dataclass 

from numpy.typing import NDArray

from sklearn.base import BaseEstimator 

from sklearn.cluster import KMeans

from typing import Literal 

from collections import OrderedDict

from torch.utils.data import (
    DataLoader, 
    random_split
)
from torch_sparse import SparseTensor

from models.networks import (
    HyperGATStack, 
    LightweightBackbone,
    SemanticMLP
)

from models.graph.construction import (
    Hypergraph,
    HypergraphMetadata,
    SpatialHypergraph
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True 

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
        contrast_temperature: float, 
        cluster_temperature: float,
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None
    ): 
        self.epochs                = epochs
        self.lr                    = lr
        self.weight_decay          = weight_decay
        self.batch_size            = batch_size
        self.eval_fraction         = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta             = min_delta
        self.random_state          = random_state
        self.collate_fn            = collate_fn

        self.w_contrast            = w_contrast
        self.w_cluster             = w_cluster
        self.w_recon               = w_recon

        self.contrast_temperature  = contrast_temperature
        self.cluster_temperature   = cluster_temperature
        self.n_prototypes          = n_prototypes
        self.proj_dim              = proj_dim

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.device    = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_          = nn.ModuleDict() 
        self.opt_            = None 
        self.scheduler_      = None 
        self.best_val_score_ = np.inf 
        self.is_fitted_      = False 

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
    def initialize_anchor_state(self, train_loader: DataLoader): ... 

    # -----------------------------------------------------
    # API  
    # -----------------------------------------------------
    
    def fit(self, X, y=None): 
        print(self)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        loader                   = self.ensure_loader(X, shuffle=True)
        train_loader, val_loader = self.split_loader(loader)

        self.init_fit(train_loader)

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

    def extract(self, X, *, view: str = "structural", level: str = "bag") -> NDArray: 
        if not self.is_fitted_: 
            raise ValueError("Call fit() before extract().")

        loader = self.ensure_loader(X, shuffle=False)
        outs: list[NDArray] = []

        self.model_.eval() 
        with torch.no_grad(): 
            for batch in loader: 
                prep = self.forward_preprocess(batch)
                sem_node, sem_bag = self.forward_semantic(prep)
                st_node, st_bag   = self.forward_structural(prep)

                if view == "semantic": 
                    x = sem_bag if level == "bag" else sem_node 
                elif view == "structural": 
                    x = st_bag  if level == "bag" else st_node 
                elif view == "concat": 
                    if level == "bag": 
                        x = torch.cat([sem_bag, st_bag], dim=1)
                    else: 
                        x = torch.cat([sem_node, st_node], dim=1)
                else: 
                    raise ValueError(f"unknown view={view}")

                outs.append(x.detach().cpu().numpy())

        return np.vstack(outs)

    # -----------------------------------------------------
    # Setup   
    # -----------------------------------------
    
    def init_fit(self, train_loader: DataLoader): 
        self.initialize_anchor_state(train_loader)

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
            sem_node, sem_bag = self.forward_semantic(prep)
            st_node, st_bag   = self.forward_structural(prep)

        d_target   = prep.repr_target.shape[1]
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

        self.opt_ = torch.optim.AdamW(
            self.model_.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_, T_max=max(1, self.epochs)
        )

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
                        f"val_con={val_parts['contrast']:.4f} | "
                        f"val_kl={val_parts['cluster']:.4f} | "
                        f"val_rec={val_parts['recon']:.4f}", 
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
            loss, _, _, _, bsz = self.process_batch(batch)

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
        con   = 0.0 
        clu   = 0.0 
        rec   = 0.0 
        count = 0 

        with torch.no_grad(): 
            for batch in loader: 
                _, l_con, l_kl, l_rec, bsz = self.process_batch(batch)

                con   += float(l_con.item()) * bsz 
                clu   += float(l_kl.item())  * bsz 
                rec   += float(l_rec.item()) * bsz
                count += bsz 

        denom = max(1, count)
        total = (con + clu + rec) / denom  

        return total, {
            "contrast": con / denom, 
            "cluster": clu / denom, 
            "recon": rec / denom
        }

    # -----------------------------------------------------
    # Batching 
    # -----------------------------------------------------

    def process_batch(self, batch): 
        with self.amp_ctx(): 
            prep = self.forward_preprocess(batch)

            sem_node, sem_bag = self.forward_semantic(prep)
            st_node, st_bag   = self.forward_structural(prep)

            l_con = self.contrastive_loss(sem_bag, st_bag)
            l_kl  = self.swapped_prediction_loss(sem_bag, st_bag)
            l_rec = self.two_view_reconstruction_loss(prep, sem_node, st_node)
            loss  = (self.w_contrast * l_con + self.w_cluster * l_kl + self.w_recon * l_rec)
            bsz   = sem_bag.shape[0]
        return loss, l_con, l_kl, l_rec, bsz 

    def contrastive_loss(
        self,
        sem_bag: torch.Tensor, 
        st_bag: torch.Tensor,
    ) -> torch.Tensor:
        z_sem = F.normalize(self.model_["sem_proj"](sem_bag), dim=1) 

        z_st  = F.normalize(self.model_["st_proj"](st_bag), dim=1)

        if z_sem.shape[0] < 2: 
            return torch.zeros((), device=z_sem.device)

        logits = torch.matmul(z_sem, z_st.T) / self.contrast_temperature 
        labels = torch.arange(z_sem.shape[0], device=z_sem.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def swapped_prediction_loss(
        self,
        sem_bag: torch.Tensor, 
        st_bag: torch.Tensor,
        *,
        epsilon=0.05,
    ) -> torch.Tensor:
        tau = self.cluster_temperature

        z_sem = F.normalize(self.model_["sem_proj"](sem_bag), dim=1) 
        z_st  = F.normalize(self.model_["st_proj"](st_bag), dim=1)

        s_sem = self.model_["proto"](z_sem) 
        s_st  = self.model_["proto"](z_st)

        with torch.no_grad(): 
            q_sem = self.sinkhorn_assign(s_sem, epsilon=epsilon)
            q_st  = self.sinkhorn_assign(s_st, epsilon=epsilon)

        l_sem = -(q_st * F.log_softmax(s_sem / tau, dim=1)).sum(dim=1).mean() 
        l_st  = -(q_sem * F.log_softmax(s_st / tau, dim=1)).sum(dim=1).mean() 
        return 0.5 * (l_sem + l_st)

    def two_view_reconstruction_loss(
        self,
        prep: SSFEBatch, 
        sem_node: torch.Tensor,
        st_node: torch.Tensor, 
    ) -> torch.Tensor: 
        target  = prep.repr_target 

        rec_sem = self.model_["sem_recon"](sem_node)

        rec_st  = self.model_["st_recon"](st_node)
        return 0.5 * (F.mse_loss(rec_sem, target) + F.mse_loss(rec_st, target))

    @staticmethod
    def sinkhorn_assign(scores, epsilon=0.05):  
        scores = scores - scores.max(dim=1, keepdim=True).values 
        Q = torch.exp(scores / epsilon).t() 
        K, B = Q.shape 
        Q = Q / Q.sum().clamp_min(1e-9)

        # single pass 
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-9) 
        Q = Q / K 
        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-9) 
        Q = Q / B  

        Q = Q / Q.sum(dim=0, keepdim=True).clamp_min(1e-9)
        return Q.t().detach()
    
    # -----------------------------------------------------
    # Loader Utils    
    # -----------------------------------------------------

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
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
            prefetch_factor=4, 
            persistent_workers=True
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

    @staticmethod 
    def fit_anchors(
        features: np.ndarray,
        *,
        n_samples:  int, 
        k: int, 
        random_state: int, 
    ) -> tuple[np.ndarray, np.ndarray]: 
        x = np.asarray(features, dtype=np.float64)
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

        anchors = km.cluster_centers_.astype(np.float64, copy=False)
        order   = np.argsort(np.linalg.norm(anchors, axis=1))
        anchors = anchors[order]
        stats   = np.asarray([mean, std], dtype=np.float64)
        return anchors, stats 

# ---------------------------------------------------------
# Spatial SSFE Modules 
# ---------------------------------------------------------

@dataclass 
class SpatialPatchBatch(SSFEBatch): 
    tile_batch_idx: torch.Tensor 
    patch_idx: torch.Tensor | None 
    n_tiles: int 
    n_patches_per_tile: int 


class SpatialPatchPreprocessor(nn.Module): 
    '''
    Converts flat tile tensors to patched embeddings for use by SFE 
    '''

    def __init__(
        self, 
        *,
        in_channels: int, 
        tile_size: int = 256, 
        patch_size: int = 32, 
        embed_dim: int, 
        anchor_stats: list[float], 
    ): 
        super().__init__()

        mean = torch.as_tensor(anchor_stats[0], dtype=torch.float32)
        std  = torch.as_tensor(anchor_stats[1], dtype=torch.float32)
        if (mean.ndim != 1 or std.ndim != 1 or 
            mean.numel() != in_channels or std.numel() != in_channels):
            raise ValueError("anchor_stats shape mismatch")

        if torch.any(std <= 0): 
            raise ValueError("negative std from anchor_stats")

        self.patch_size   = patch_size 

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
        tile_batch_idx: torch.Tensor | None = None,
    ) -> SpatialPatchBatch: 
        if tiles.ndim != 4: 
            raise ValueError(f"expected (T, C, H, W), got {tuple(tiles.shape)}")
        if stats.ndim != 3: 
            raise ValueError(f"expected stats (T, L, D), got {tuple(stats.shape)}")

        T, C, _, _ = tiles.shape
        P = self.patch_size 

        patches = self.encoder.unfold(tiles)
        L = patches.shape[0] // T 
        patches = patches.view(T, L, C, P, P)

        stats_raw    = stats[..., :C].to(
            device=tiles.device, dtype=torch.float32, non_blocking=True
        )

        K   = L 
        idx = None 

        patches_flat = patches.reshape(T * K, C, P, P)
        feat_maps    = self.encoder.encoder(patches_flat)
        embs         = self.encoder.projector(feat_maps).view(T, K, -1)

        mean         = self.encoder.patch_mean[:C].view(1, 1, C).to(
            device=tiles.device, dtype=torch.float32)
        std          = self.encoder.patch_std[:C].view(1, 1, C).to(
            device=tiles.device, dtype=torch.float32)
        stats_z      = (stats_raw - mean) / std 

        sem_in       = torch.cat([embs, stats_z], dim=-1) 

        repr_target  = embs.reshape(T * K, -1)
        sem_in       = sem_in.reshape(T * K, -1)
        stats_z_f    = stats_z.reshape(T * K, -1)
        stats_raw_f  = stats_raw.reshape(T * K, -1)

        patch_batch_idx = torch.arange(T, device=tiles.device).repeat_interleave(K)
        if tile_batch_idx is None: 
            tile_batch_idx = torch.arange(T, device=tiles.device)
        else: 
            tile_batch_idx = tile_batch_idx.to(
                device=tiles.device, dtype=torch.long, non_blocking=True)

        return SpatialPatchBatch(
            repr_target=repr_target,
            semantic_input=sem_in,
            stats_z=stats_z_f,
            stats_raw=stats_raw_f,
            batch_idx=patch_batch_idx,
            tile_batch_idx=tile_batch_idx,
            patch_idx=idx, 
            n_tiles=T,
            n_patches_per_tile=K 
        )


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

        h_all   = self.gnn(x_all, meta.node_type, meta.H, meta.edge_type)
        h_node  = self.node_norm(h_all[:N])
        h_bag   = self.bag_norm(h_all[meta.readout_node_ids])
        return h_node, h_bag 

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
        anchor_n_samples: int = 500_000, 
        anchor_min_norm: float = 1e-6, 
        embed_dim: int, 

        # semantic branch 
        semantic_hidden_dim: int, 
        semantic_out_dim: int, 
        semantic_dropout: float, 

        # structural branch
        gnn_dim: int, 
        gnn_layers: int, 
        gnn_heads: int, 
        dropout: float, 
        attn_dropout: float, 
        global_active_eps: float = 1e-6, 

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
        contrast_temperature: float, 
        cluster_temperature: float,
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None
    ): 
        super().__init__(
            epochs=epochs,
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
            contrast_temperature=contrast_temperature,
            cluster_temperature=cluster_temperature,
            n_prototypes=n_prototypes,
            proj_dim=proj_dim,
            device=device,
        )

        self.in_channels         = in_channels
        self.anchor_n_samples    = anchor_n_samples 
        self.anchor_min_norm     = anchor_min_norm
        self.tile_size           = tile_size
        self.patch_size          = patch_size
        self.embed_dim           = embed_dim

        self.semantic_hidden_dim = semantic_hidden_dim
        self.semantic_out_dim    = semantic_out_dim
        self.semantic_dropout    = semantic_dropout

        self.gnn_dim             = gnn_dim
        self.gnn_layers          = gnn_layers
        self.gnn_heads           = gnn_heads
        self.dropout             = dropout
        self.attn_dropout        = attn_dropout
        self.global_active_eps   = global_active_eps

        self.cache_graph_once    = True 
        self.graph_initialized_  = False 

        self.node_anchors: list[list[float]] | None = None 
        self.anchor_stats: list[float] | None       = None 

    def build_preprocess(self) -> nn.Module | None:
        if self.anchor_stats is None: 
            raise ValueError("must call initialize_anchor_state()")

        return SpatialPatchPreprocessor(
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            anchor_stats=self.anchor_stats
        )

    def build_semantic_embed(self, sample: SSFEBatch) -> nn.Module:
        d_in = sample.semantic_input.shape[1]

        return SemanticMLP(
            in_dim=d_in,
            hidden_dim=self.semantic_hidden_dim,
            out_dim=self.semantic_out_dim,
            dropout=self.semantic_dropout
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
        if not isinstance(batch, (list, tuple)) or len(batch) < 4: 
            raise ValueError("expected spatial batch as (tiles, labels, tile_batch_idx)")

        tiles          = batch[0].to(self.device, non_blocking=True)
        tile_batch_idx = batch[2].to(self.device, non_blocking=True)
        stats          = batch[3].to(self.device, non_blocking=True)

        return self.model_["preprocess"](
            tiles, 
            stats=stats, 
            tile_batch_idx=tile_batch_idx,
        )

    def forward_semantic(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch in semantic path")
        return self.model_["semantic"](
            prep.semantic_input,
            prep.batch_idx,
            batch_size=prep.n_tiles
        )

    def forward_structural(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch in structural path")
        return self.model_["structural"](prep)

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
