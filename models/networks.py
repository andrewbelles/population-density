#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

'''
Thresholds for Hypergraph Heterogeneity (derived from multi-otsu thresholds): 
    - L: 0.465 
    - U: 1.855 
These bounds enforce (32x32 chunks): 
    - Type 0: ~2,750,000  
    - Type 1: ~68,500
    - Type 2: ~21,450 
'''

from typing import Optional
import torch 

import torch.nn.functional as F 

from torch import nn, reshape 

import torchvision.models as tvm

from torchvision.models.mobilenet import MobileNet_V3_Small_Weights 

from torch_scatter import (
    scatter_add, 
    scatter_softmax, 
    scatter_mean 
)

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
        out_dim: int | None = None 
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

        nn.init.zeros_(self.head.weight)
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
# Transfomer Gate for Mixture of Experts 
# ---------------------------------------------------------

class MoETransformerGate(nn.Module): 
    '''
    Flexible expert transformer gate with intrinsic explainability: 
    - per-expert projection into shared latent space 
    - per-expert type embedding 
    - per-expert learnable reliability gate 
    - CLS-token aggregation 
    '''

    def __init__(
        self,
        *,
        expert_dims: dict[str, int], 
        d_model: int, 
        n_heads: int, 
        n_layers: int, 
        ff_mult: int, 
        dropout: float, 
        attn_dropout: float, 
        pre_norm: bool = True, 
        gate_floor: float = 0.0 
    ): 
        super().__init__()
        if not expert_dims: 
            raise ValueError("expert_dims must be non-empty.")
        if d_model <= 0: 
            raise ValueError("d_model must be > 0.")

        self.expert_names = tuple(expert_dims.keys())
        self.n_experts    = len(self.expert_names)
        self.d_model      = d_model 
        self.gate_floor   = gate_floor

        self.adapters     = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(int(dim)),
                nn.Linear(int(dim), self.d_model)
            )
            for name, dim in expert_dims.items() 
        })

        self.type_embed   = nn.Parameter(torch.zeros(self.n_experts, self.d_model))
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # reliability gate 
        self.gate_mlp     = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=pre_norm
        )
        enc_layer.self_attn.dropout = attn_dropout 

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=n_layers,
            enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(self.d_model)

        self.reset_parameters() 

    def forward(
        self, 
        experts: dict[str, torch.Tensor],
    ):
        tokens = self.stack_expert_tokens(experts)
        gate   = self.compute_gates(tokens)

        # reliability gating 
        gated_tokens = tokens * gate.unsqueeze(-1)

        # pass through transformer block 
        bsz = gated_tokens.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        seq = torch.cat([cls, gated_tokens], dim=1)
        enc = self.encoder(seq)
        enc = self.out_norm(enc)

        cls_out   = enc[:, 0, :]
        token_out = enc[:, 1:, :]
        return cls_out, token_out, gate

    def reset_parameters(self): 
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for mod in self.adapters.values(): 
            for m in mod.modules(): 
                if isinstance(m, nn.Linear): 
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        for m in self.gate_mlp.modules(): 
            if isinstance(m, nn.Linear): 
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def stack_expert_tokens(self, experts: dict[str, torch.Tensor]) -> torch.Tensor: 
        missing = [k for k in self.expert_names if k not in experts]
        extra   = [k for k in experts.keys() if k not in self.adapters]
        if missing: 
            raise KeyError(f"missing experts: {missing}")
        if extra: 
            raise KeyError(f"missing experts: {extra}")

        tokens = []
        bsz    = None 
        device = None 
        dtype  = None 

        for idx, name in enumerate(self.expert_names): 
            x = experts[name]
            if x.ndim != 2: 
                raise ValueError(f"expert {name} must be shape (B, d), got {tuple(x.shape)}")

            if bsz is None: 
                bsz    = x.size(0)
                device = x.device 
                dtype  = x.dtype
            else: 
                if x.size(0) != bsz or x.device != device or x.dtype != dtype: 
                    raise ValueError("expert mismatch.")

            t = self.adapters[name](x)
            t = t + self.type_embed[idx].unsqueeze(0)
            tokens.append(t)

        return torch.stack(tokens, dim=1)
    
    def compute_gates(self, tokens: torch.Tensor) -> torch.Tensor: 
        g_logits = self.gate_mlp(tokens).squeeze(-1)
        g        = torch.sigmoid(g_logits)
        if self.gate_floor > 0.0: 
            g = self.gate_floor + (1.0 - self.gate_floor) * g 
        return g 

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
            gate_floor=gate_floor 
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]: 
        mu      = self.linear(x).squeeze(-1)
        lv      = self.log_var_param.clamp(self.log_var_min, self.log_var_max)
        log_var = lv.expand_as(mu)

        return {
            "mu_wide": mu, 
            "log_var_wide": log_var 
        }
