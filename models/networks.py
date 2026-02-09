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

import torch 

import torch.nn.functional as F 

from torch import nn 

import torchvision.models as tvm

from torchvision.models.mobilenet import MobileNet_V3_Small_Weights 

from torch_scatter import scatter_add, scatter_softmax 

from models.graph.construction import (
    HypergraphBuilder,
    LOGRADIANCE_GATE_LOW,
    LOGRADIANCE_GATE_HIGH,
    MultichannelHypergraphBuilder
)

from utils.loss_funcs import LogitScaler

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

        self.cut_anchor = nn.Parameter(torch.tensor(0.0))
        self.cut_deltas = nn.Parameter(torch.ones(n_classes - 2) * 1.0)

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
            score = score * F.softplus(self.logit_scale)
        logits = score + cuts 
        proj   = self.proj(feats) if self.proj is not None else None 
        return emb, logits, proj 

# ---------------------------------------------------------
# Lightweight CNN backbone for CASM-MIL model  
# ---------------------------------------------------------

class LightweightBackbone(nn.Module): 

    '''
    Lightweight adaptation of ResNet architecture with 4 residual layers. 
    Imports MobileNet_V3 weights adapted for grayscale as weight initialization 
    '''

    def __init__(
        self, 
        in_channels=1,
        embed_dim=64, 
        patch_size=32,
        stat="viirs",
        patch_quantile=0.95,
        anchor_stats: list[float] | None = None,
        flat: bool = False, 
    ): 
        super().__init__()
        
        self.patch_size = patch_size 
        self.embed_dim  = embed_dim

        self.grayscale  = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        with torch.no_grad(): 
            self.grayscale.weight.fill_(1.0 / 3.0)

        if flat:
            self.encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(in_channels, 64, kernel_size=1), 
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(128, embed_dim, kernel_size=1)
            )
            self.projector = nn.Flatten()
        else: 
            mobilenet = tvm.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            encoder   = mobilenet.features 

            orig_conv = encoder[0][0]
            new_conv  = nn.Conv2d(
                in_channels=in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride, 
                padding=orig_conv.padding, 
                bias=False 
            )

            with torch.no_grad(): 
                new_conv.weight.copy_(orig_conv.weight.sum(dim=1, keepdim=True))
            encoder[0][0] = new_conv 

            self.encoder   = encoder[:4] # truncate encoder to 4 blocks 

            self.projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(24, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU() 
            ) 

        self.stat           = stat 
        self.patch_quantile = patch_quantile

        if anchor_stats is not None: 
            mean = torch.as_tensor(anchor_stats[0], dtype=torch.float32) 
            std  = torch.as_tensor(anchor_stats[1], dtype=torch.float32) 
        else: 
            mean, std = (torch.empty(0), torch.empty(0)) 

        self.register_buffer("patch_mean", mean)
        self.register_buffer("patch_std", std)

    def forward(self, tiles): 
        patches = self.unfold(tiles)
        score   = self.patch_stat(patches, stat=self.stat, q=self.patch_quantile)

        # x      = self.grayscale(patches)
        feats  = self.encoder(patches)
        embs   = self.projector(feats)

        out    = torch.cat([embs, score], dim=1)
        return out 

    def patch_stat(self, patches: torch.Tensor, *, stat: str = "viirs", q: float = 0.95): 

        N, C, _, _ = patches.shape 
        flat       = patches.view(N, C, -1)

        if stat == "viirs" or stat == "usps": 
            feats = torch.quantile(flat, q, dim=2) 
        elif stat == "p95": 
            feats = torch.quantile(flat, q, dim=1)
        elif stat == "max": 
            feats = flat.max(dim=1).values
        else:
            raise ValueError(f"unknown patch stat: {stat}")

        # z-score using precomputed anchor_stats
        if self.patch_mean.numel() > 0: 
            mean = self.patch_mean.to(feats.device)
            std  = self.patch_std.to(feats.device).clamp_min(1e-6)
            feats = (feats - mean) / std 

        return feats 

    @staticmethod 
    def p95_patch(patches): 
        flat = patches.flatten(start_dim=1)
        return torch.quantile(flat, q=0.95, dim=1)

    def unfold(self, tiles): 
        B, C, H, W = tiles.shape 
        P          = self.patch_size 

        if H % P != 0 or W % P != 0: 
            raise ValueError(f"Tile size must be divisible by patch_size={P}. Got=(){H},{W})")

        unfold  = nn.Unfold(kernel_size=P, stride=P)
        patches = unfold(tiles)
        L       = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(B * L, C, P, P)
        return patches 

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

        self.out_dim = in_dim + (hidden_dim * n_layers)
        
    def forward(self, x, node_types, H, edge_type): 

        stats = [x]

        for gat, ln in zip(self.layers, self.norms): 
            h_in = stats[-1] 
            h, _ = gat(h_in, node_types, H, edge_type)

            if h.shape == h_in.shape:
                h = h + h_in 

            h = ln(self.drop(h))
            stats.append(h)

        return torch.cat(stats, dim=-1) 

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

    def forward(self, x, node_types, H, edge_type): 
        node_outs = []
        edge_outs = []
        for head in self.heads: 
            node_feat, edge_feat = head(x, node_types, H, edge_type)
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

    def forward(self, x, node_types, H, edge_type): 
        '''
        Caller Provides:
        - x (N, in_dim),
        - node_types: (N,)
        - H (SparseTensor for Hypergraph incidence matrix)
        - edge_type (num_edges,) hyperedge type id (0=spatial,1=semantic,2=global)
        '''
        node_idx, edge_idx, _ = H.coo() 
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
# Attached Backbone connecting HyperGAT to LightweightBackbone 
# ---------------------------------------------------------

class HypergraphBackbone(nn.Module): 

    def __init__(
        self, 
        in_channels=1, 
        tile_size=256,
        patch_size=32,
        embed_dim=64,
        gnn_dim=128,
        gnn_layers=1,
        gnn_heads=1,
        max_bag_frac=0.75, 
        attn_dropout=0.0,
        dropout=0.0,
        patch_stat="p95",
        patch_quantile=0.95,
        thresh_low=LOGRADIANCE_GATE_LOW,
        thresh_high=LOGRADIANCE_GATE_HIGH,             # ignored for > 1 channels 
        node_anchors: list[list[float]] | None = None, # 1 channel -> None, 1 > must pass 
        anchor_stats: list[float] | None = None        # [mean, std] used to z-score anchors 
    ): 
        super().__init__()
       
        is_flat = True if patch_stat == "usps" else False 

        self.encoder = LightweightBackbone(
            in_channels=in_channels,
            embed_dim=embed_dim, 
            patch_size=patch_size,
            stat=patch_stat,
            patch_quantile=patch_quantile, 
            anchor_stats=anchor_stats,
            flat=is_flat 
        )

        if node_anchors is not None: 
            anchors = torch.tensor(node_anchors, dtype=torch.float32)
            if anchors.shape[0] != 3: 
                raise ValueError("Must provide exactly three anchors per channel")
            if anchors.shape[1] != in_channels: 
                raise ValueError(f"Anchors dim {anchors.shape[1]} != in_channels {in_channels}")

            self.builder = MultichannelHypergraphBuilder(
                anchors=anchors,
                tile_size=tile_size,
                patch_size=patch_size
            )
        else: 
            self.builder = HypergraphBuilder(
                tile_size=tile_size,
                patch_size=patch_size,
                thresh_low=thresh_low,
                thresh_high=thresh_high
            )

        readout_dim = embed_dim + in_channels # token per channel 

        self.gnn     = HyperGATStack(
            in_dim=readout_dim, 
            hidden_dim=gnn_dim, 
            n_layers=gnn_layers,
            n_heads=gnn_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            n_node_types=4   # type 0,1,2 & global 
        )

        self.out_dim        = self.gnn.out_dim 
        self.max_bag_frac   = max_bag_frac
        self.patch_stat     = patch_stat
        self.patch_quantile = patch_quantile

        # position embeddings 
        self.num_patches  = (tile_size // patch_size)**2 
        self.pos_embed    = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.global_token = nn.Parameter(torch.randn(1, 1, readout_dim)) 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def forward(self, tiles): 
        B = tiles.shape[0]
        P = self.encoder.patch_size 

        patches = self.encoder.unfold(tiles)
        L = patches.shape[0] // B 
        C = patches.shape[1]
        patches = patches.view(B, L, C, P, P)

        if self.training and self.max_bag_frac < 1.0: 
            K = max(1, int(L * self.max_bag_frac))
            idx = torch.stack(
                [torch.randperm(L, device=tiles.device)[:K] for _ in range(B)],
                dim=0
            )
        else: 
            idx = None 

        if idx is not None: 
            patches = patches.gather(
                1, idx[:, :, None, None, None].expand(-1, -1, C, P, P)
            )
            K = patches.shape[1]
            patches = patches.view(B * K, C, P, P)
        else: 
            K = L  
            patches = patches.view(B * L, C, P, P)

        feats = self.encoder.encoder(patches)
        embs  = self.encoder.projector(feats).view(B, K, -1)

        score = self.encoder.patch_stat(patches, stat=self.patch_stat, q=self.patch_quantile)
        score = score.view(B, K, -1)

        if idx is None: 
            pos = self.pos_embed
        else: 
            pos = self.pos_embed.squeeze(0)[idx]

        embs = embs + pos 

        patch_feats = torch.cat([embs, score], dim=2)
        patch_feats = patch_feats.view(B * K, -1)

        readout_tokens = self.global_token.expand(B, -1, -1).reshape(B, -1)

        all_feats = torch.cat([patch_feats, readout_tokens], dim=0)

        node_type, _, edge_type, _, H = self.builder.build(
            score.view(-1, score.shape[-1]), 
            batch_size=B, 
            idx=idx
        )

        node_out = self.gnn(all_feats, node_type, H, edge_type)

        readout_out = node_out[-B:]
        return readout_out 

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
        num_tokens: int = 8, 
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
        self.num_tokens = num_tokens

        self.tokenizer  = nn.Linear(in_dim, num_tokens * d_model, bias=True)
        self.token_proj = nn.Linear(in_dim, d_model, bias=True)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))

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

        nn.init.trunc_normal_(self.tokenizer.weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x): 
        if x.dim() == 3: 
            tokens = self.token_proj(x)
        else: 
            B = x.size(0)
            tokens = self.tokenizer(x).view(B, self.num_tokens, self.d_model)

        B      = tokens.size(0)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        
        if tokens.size(1) > self.pos_embed.size(1): 
            raise ValueError(f"token length {tokens.size(1)} exceeds pos_embed length " 
                             f"{self.pos_embed.size(1)}")
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        enc    = self.encoder(tokens)
        pooled = enc[:, 0]
        return self.out_proj(pooled)


class Mixer(nn.Module): 
    '''
    Mixes Hard Labels using beta distributed ratios for TF-Residual MLP 
    '''

    def __init__(
        self, 
        class_weights: torch.Tensor, 
        *,
        alpha: float = 0.2, 
        mix_mult: int = 4, 
        max_mix: int | None = None, 
        anchor_power: float = 1.0, 
        with_replacement: bool = True 
    ): 
        super().__init__()
        self.register_buffer("class_weights", class_weights.float())
        self.alpha            = alpha 
        self.mix_mult         = mix_mult
        self.max_mix          = max_mix 
        self.anchor_power     = anchor_power 
        self.with_replacement = with_replacement
        
    def forward(self, x, y, generator=None, return_indices: bool = False): 

        B      = y.numel() 
        device = y.device 

        n_mix  = B * self.mix_mult 
        if self.max_mix is not None: 
            n_mix = min(n_mix, self.max_mix)

        w = self.class_weights.to(device)
        anchor_w  = w.gather(0, y).float().pow(self.anchor_power)
        anchor_w  = anchor_w / anchor_w.sum()  

        partner_w = torch.ones_like(anchor_w) / anchor_w.numel()  

        idx_a = torch.multinomial(
            anchor_w, n_mix, replacement=self.with_replacement, generator=generator
        )
        idx_b = torch.multinomial(
            partner_w, n_mix, replacement=self.with_replacement, generator=generator
        )

        mix_lambda = mix_lambda = torch.distributions.Beta(
            self.alpha, self.alpha).sample((n_mix,)).to(device)
        lam_shape = (n_mix,) + (1,) * (x.ndim - 1)
        lam = mix_lambda.view(*lam_shape)

        x_mix = lam * x[idx_a] + (1 - lam) * x[idx_b]
        y_a   = y[idx_a]
        y_b   = y[idx_b]
        
        if return_indices:
            return x_mix, idx_a, idx_b, mix_lambda
        return x_mix, y_a, y_b, mix_lambda 


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
    Residual Backbone for Tabular data 

    Uses PreNorm Residual Blocks 
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
