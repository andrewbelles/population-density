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
