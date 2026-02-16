#!/usr/bin/env python3 
# 
# ssfe.py  Andrew Belles  Feb 10th, 2026 
# 
# Self-Supervised Feature Extractors for learning latent representations for deep component of 
# model architecture (wide & deep)
# 

from typing import Optional
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
    build_ssfe_loss
)

from torch.utils.data import (
    DataLoader,
    TensorDataset, 
    random_split
)

from models.networks import (
    GatedAttentionPooling,
    HyperGATStack, 
    LightweightBackbone,
    ResidualMLP,
    SemanticMLP,
    TransformerProjector
)

from models.graph.construction import (
    SpatialHypergraph,
    TabularHypergraph
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
        contrast_temperature: float, 
        cluster_temperature: float,
        n_prototypes: int, 
        proj_dim: int, 
        device: str | None = None,
        compile_model: bool = True
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

        self.contrast_temperature  = contrast_temperature
        self.cluster_temperature   = cluster_temperature
        self.n_prototypes          = n_prototypes
        self.proj_dim              = proj_dim

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
            sem_node, sem_bag = self.forward_semantic(prep)
            st_node, st_bag   = self.forward_structural(prep)

        d_target   = prep.repr_target.shape[1]
        d_sem_node = sem_node.shape[1]
        d_st_node  = st_node.shape[1]
        d_sem_bag  = sem_bag.shape[1]
        d_st_bag   = st_bag.shape[1]

        if hasattr(prep, "tile_batch_idx"): 
            self.model_["sem_county_pool"] = GatedAttentionPooling(
                in_dim=d_sem_bag,
                attn_dim=max(32, d_sem_bag // 2), 
                attn_dropout=getattr(self, "attn_dropout", 0.0)  
            ).to(self.device)
            self.model_["st_county_pool"] = GatedAttentionPooling(
                in_dim=d_st_bag,
                attn_dim=max(32, d_st_bag // 2), 
                attn_dropout=getattr(self, "attn_dropout", 0.0)  
            ).to(self.device)

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
        if self.loss_ is None: 
            raise RuntimeError("loss_ not initializaed. Call init_fit() first.")

        with self.amp_ctx(): 
            prep = self.forward_preprocess(batch)

            sem_node, sem_bag = self.forward_semantic(prep)
            st_node, st_bag   = self.forward_structural(prep)
            node_batch_idx    = self.resolve_contrast_node_batch_idx(
                prep, bag_count=sem_bag.shape[0]
            )

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
                w_recon=self.w_recon
            )

            loss  = pack.total 
            l_con = pack.raw["contrast"]
            l_kl  = pack.raw["cluster"]
            l_rec = pack.raw["recon"]

            bsz   = sem_bag.shape[0]
        return loss, l_con, l_kl, l_rec, bsz 

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
        tile_batch_idx: torch.Tensor | None = None,
    ) -> SpatialPatchBatch: 
        if tiles.ndim != 4: 
            raise ValueError(f"expected (T, C, H, W), got {tuple(tiles.shape)}")
        if stats.ndim != 3: 
            raise ValueError(f"expected stats (T, L, D), got {tuple(stats.shape)}")

        T, C, _, _ = tiles.shape
        
        embc = self.encoder(tiles)
        K    = stats.shape[1]
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

        repr_target  = embc.reshape(T * K, -1)
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

        # structural branch
        gnn_dim: int, 
        gnn_layers: int, 
        gnn_heads: int, 
        dropout: float, 
        attn_dropout: float, 
        global_active_eps: float = 1e-6, 
        swap_noise_prob: float = 0.15, 

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

        tiles          = batch[0].to(self.device, non_blocking=True, 
                                     memory_format=torch.channels_last)
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
        sem_node, sem_tile = self.model_["semantic"](
            prep.semantic_input,
            prep.batch_idx,
            batch_size=prep.n_tiles
        )

        if "sem_county_pool" not in self.model_: 
            return sem_node, sem_tile

        tidx = prep.tile_batch_idx.to(self.device, dtype=torch.long).view(-1)
        _, inv = torch.unique(tidx, sorted=True, return_inverse=True)
        n_county = int(inv.max().item()) + 1 if inv.numel() else 0 
        sem_county = self.model_["sem_county_pool"](sem_tile, inv, n_county)
        return sem_node, sem_county 

    def forward_structural(self, prep: SSFEBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(prep, SpatialPatchBatch):
            raise TypeError("SpatialSSFE expects SpatialPatchBatch in structural path")
        st_node, st_tile = self.model_["structural"](prep)

        if "st_county_pool" not in self.model_: 
            return st_node, st_tile

        tidx = prep.tile_batch_idx.to(self.device, dtype=torch.long).view(-1)
        _, inv = torch.unique(tidx, sorted=True, return_inverse=True)
        n_county = int(inv.max().item()) + 1 if inv.numel() else 0 
        st_county = self.model_["st_county_pool"](st_tile, inv, n_county)
        return st_node, st_county 

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
        bag_idx: torch.Tensor | None = None, 
        graph_group_idx: torch.Tensor | None = None,
        node_ids: torch.Tensor | None = None 
    ): 
        if x.ndim != 2: 
            raise ValueError(f"expected (N, F), got {tuple(x.shape)}")

        xc = x.to(dtype=torch.float32)
        n  = x.shape[0]


        if bag_idx is None: 
            bag_idx = torch.arange(n, device=x.device, dtype=torch.long)
        else: 
            bag_idx = self.normalize_index(bag_idx, n, x.device)

        n_bags = int(bag_idx.max().item()) + 1 if n > 0 else 0 

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
        
        # transformer 
        transformer_dim: int, 
        transformer_heads: int, 
        transformer_layers: int, 
        transformer_attn_dropout: float, 

        # projector 
        proj_dim: int, 

        # residual mlp 
        refine_hidden_dim: int, 
        refine_depth: int, 
        dropout: float, 
    ): 
        super().__init__() 

        self.tokenizer = TransformerProjector(
            in_dim=in_dim, 
            out_dim=transformer_dim, 
            d_model=transformer_dim,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            dropout=dropout, 
            attn_dropout=transformer_attn_dropout,
            pre_norm=True 
        )

        self.projector = nn.Sequential(
            nn.LayerNorm(transformer_dim), 
            nn.Linear(transformer_dim, proj_dim),
            nn.GELU(), 
            nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        )

        self.refine    = ResidualMLP(
            in_dim=proj_dim,
            hidden_dim=refine_hidden_dim,
            depth=refine_depth,
            dropout=dropout,
            out_dim=out_dim
        )

        self.node_norm = nn.LayerNorm(out_dim)
        self.bag_norm  = nn.LayerNorm(out_dim)
        self.out_dim   = out_dim 

    def forward(self, prep: TabularBatch) -> tuple[torch.Tensor, torch.Tensor]: 
        if prep.stats_raw is None: 
            raise ValueError("Tabular semantic path requires prep.stats_raw")
        if prep.batch_idx.numel() != prep.stats_raw.size(0): 
            raise ValueError("batch_idx/feature length mismatch")

        x    = prep.stats_raw 
        tok  = self.tokenizer(x)
        proj = self.projector(tok)
        node = self.node_norm(self.refine(proj))

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
        swap_noise_prob: float = 0.15, 

        # semantic branch 
        semantic_out_dim: int, 
        transformer_dim: int, 
        transformer_heads: int, 
        transformer_layers: int, 
        transformer_attn_dropout: float, 
        semantic_proj_dim: int, 
        semantic_hidden_dim: int, 
        semantic_depth: int, 
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
        w_contrast: float = 1, 
        w_cluster: float = 1, 
        w_recon: float = 1, 
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
        self.transformer_dim          = transformer_dim
        self.transformer_heads        = transformer_heads
        self.transformer_layers       = transformer_layers
        self.transformer_attn_dropout = transformer_attn_dropout
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
        _ = sample 
        return TabularSemanticEmbedder(
            in_dim=self.in_dim,
            out_dim=self.semantic_out_dim,
            transformer_dim=self.transformer_dim,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
            transformer_attn_dropout=self.transformer_attn_dropout,
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

        if isinstance(batch, (list, tuple)): 
            if len(batch) < 1: 
                raise ValueError("empty tabular batch")
            x = batch[0]
            if len(batch) >= 3: 
                bag_idx = batch[2]
            if len(batch) >= 4: 
                graph_group_idx = batch[3]
            if len(batch) >= 5: 
                node_ids = batch[4]
            elif len(batch) >= 2 and self.is_index_like(batch[1]): 
                node_ids = batch[1]
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

        return self.model_["preprocess"](
            x, 
            bag_idx=bag_idx,
            graph_group_idx=graph_group_idx,
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
        if isinstance(dataset, np.ndarray): 
            x   = torch.from_numpy(np.asarray(dataset, dtype=np.float32))
            ids = torch.arange(x.shape[0], dtype=torch.long)
            dataset = TensorDataset(x, ids) 

        elif isinstance(dataset, torch.Tensor): 
            x   = dataset.detach().cpu().to(torch.float32)
            ids = torch.arange(x.shape[0], dtype=torch.long)
            dataset = TensorDataset(x, ids)

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
