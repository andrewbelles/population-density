#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import torch 

from torch import nn 

import torchvision.models as tvm 

from utils.loss_funcs import LearnableLogitScaling

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

        in_dtype = x.dtype 

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
        return pooled.to(dtype=in_dtype) 


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
        use_logit_scaler: bool = True  # platt/temperature scaling 
    ):
        super().__init__()
        if n_classes < 2: 
            raise ValueError("ordinal head requires n_classes >= 2")

        self.fc   = nn.Linear(in_dim, fc_dim)
        self.act  = nn.GELU() 
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity() 
        self.out  = nn.Linear(fc_dim, n_classes - 1)

        self.scaler = LearnableLogitScaling(1.0) if use_logit_scaler else nn.Identity() 

        self.proj = None 
        if supcon_dim is not None: 
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.GELU(),
                nn.Linear(in_dim, supcon_dim)
            )

        self.out_dim = fc_dim 

    def forward(self, feats: torch.Tensor): 
        emb    = self.act(self.fc(feats))
        logits = self.out(self.drop(emb))
        logits = self.scaler(logits)
        proj   = self.proj(feats) if self.proj is not None else None 
        return emb, logits, proj 
