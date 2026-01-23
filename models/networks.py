#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import torch, sys, warnings 

from torch import dtype, nn 

import torch.nn.functional as F 

import torchvision.ops as tops 

import torch.utils.checkpoint as cp


def best_num_groups(ch: int, max_groups: int = 32) -> int: 
    for g in (32, 16, 8, 4, 2, 1): 
        if g <= max_groups and ch % g == 0: 
            return g 
    return 1 


class MaskedGroupNorm(nn.Module): 
    '''
    Uses GroupNorm but Masks out padding to ensure zeros do not pollute normalization 
    '''

    def __init__(
        self, 
        num_groups,
        num_channels,
        eps: float = 1e-6, 
        affine: bool = True 
    ): 
        super().__init__()
        self.num_groups   = num_groups 
        self.num_channels = num_channels
        self.eps          = eps 

        if affine: 
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))
        else: 
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x, mask=None): 
        if mask is None: 
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

        if mask.ndim == 3: 
            mask = mask.unsqueeze(1)

        if mask.shape[-2:] != x.shape[-2:]:
            mask = mask.to(device=x.device, dtype=x.dtype)
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        else: 
            mask = mask.to(device=x.device, dtype=x.dtype)

        n, c, h, w = x.shape
        g  = self.num_groups 
        xg = x.view(n, g, c // g, h, w) 
        mg = mask.view(n, 1, 1, h, w)

        sum_m = mg.sum((2, 3, 4))
        denom = sum_m.clamp(min=1.0)
        mean  = (xg * mg).sum((2, 3, 4)) / denom 
        mean  = mean.view(n, g, 1, 1, 1)

        var   = ((xg - mean)**2 * mg).sum((2, 3, 4)) / denom 
        var   = var.view(n, g, 1, 1, 1)
        x     = (xg - mean) / torch.sqrt(var + self.eps)

        x = x.view(n, c, h, w)
        if self.weight is not None: 
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x 


class MaskedConvBlock(nn.Module): 
    '''
    A single convolutional block using MaskedGroupNorm
    '''

    def __init__(
        self, 
        in_ch,  
        out_ch,
        kernel_size=3,
        dilation=1, 
        use_norm=True 
    ): 
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, 
            out_ch,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=dilation,
            bias=not use_norm 
        )
        self.norm = None 
        if use_norm: 
            num_groups = best_num_groups(out_ch) 
            self.norm  = MaskedGroupNorm(num_groups, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None): 
        out = self.conv(x)

        if self.norm is not None: 
            norm_mask = mask 
            if norm_mask is not None: 
                if norm_mask.ndim == 3: 
                    norm_mask = norm_mask.unsqueeze(1)

                if norm_mask.shape[-2:] != out.shape[-2:]:
                    norm_mask = F.interpolate(norm_mask, size=out.shape[-2:], mode="nearest")

            out = self.norm(out, norm_mask)

        out = self.relu(out)
        return out, mask 


class DilatedResidualBlock(nn.Module): 
    '''
    Residual dilated block implemented with masked normalization
    '''
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        dilation=1,
        num_convs: int = 2, 
        use_norm=True
    ):
        super().__init__()
        if num_convs < 2: 
            raise ValueError("num_convs must be >= 2")
        self.num_convs = num_convs

        self.convs = nn.ModuleList() 
        self.norms = nn.ModuleList() 

        for i in range(num_convs):
            d = dilation if i == 0 else 1 
            p = d 
            in_i = in_ch if i == 0 else out_ch 
            conv = nn.Conv2d(
                in_i,
                out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=d,
                padding=p,
                bias=not use_norm
            )
            self.convs.append(conv)
            if use_norm:
                num_groups = best_num_groups(out_ch) 
                self.norms.append(MaskedGroupNorm(num_groups, out_ch))
            else: 
                self.norms.append(None)

        self.relu = nn.ReLU(inplace=True)

        self.proj = None 
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)

    def forward(self, x, mask=None):
        out = x 
        for i, conv in enumerate(self.convs):
            out = conv(out)
            if self.norms[i] is not None: 
                norm_mask = mask 
                if norm_mask is not None and norm_mask.ndim == 3: 
                    norm_mask = norm_mask.unsqueeze(1)
                out = self.norms[i](out, norm_mask)
            if i < self.num_convs - 1: 
                out = self.relu(out)

        res = x if self.proj is None else self.proj(x)
        out = out + res 
        return out, mask 


class SpatialBackbone(nn.Module): 
    '''
    CNN Backbone tailored for masked, spatial satellite data for NLCD/VIIRS  
    
    Leverages Region of Interest pooling on packed samples.

    Manually downsamples masks to preserve them through model.
    '''

    def __init__(
        self,
        in_channels: int, 
        categorical_input: bool = False, 
        categorical_embed_dim: int = 4,
        categorical_cardinality: int | None = None, 
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3, 
        use_bn: bool = True, 
        roi_output_size: int | tuple[int, int] = 7, 
        sampling_ratio: int = 2, 
        aligned: bool = False 
    ): 
        super().__init__()

        self.model_in_channels = in_channels 
        self.out_dim           = conv_channels[-1] * 2 
        self.spatial_scale     = 1.0 
        self.roi_output_size   = roi_output_size
        self.sampling_ratio    = sampling_ratio 
        self.aligned           = aligned 

        # Categorical information for NLCD
        self.categorical_input       = categorical_input
        self.categorical_embed_dim   = categorical_embed_dim
        self.categorical_cardinality = (categorical_cardinality or 
                                        (in_channels if in_channels > 1 else 8))
        
        if self.categorical_input: 
            self.embedding = nn.Embedding(self.categorical_cardinality + 1,
                                          self.categorical_embed_dim,
                                          padding_idx=0)
        layers    = []
        ch        = self.categorical_embed_dim if self.categorical_input else in_channels
        dilations = [2**i for i in range(len(conv_channels))] 
        for dilation, out_ch in zip(dilations, conv_channels): 
            layers.append(DilatedResidualBlock(
                ch,
                out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                num_convs=2,
                use_norm=use_bn
            ))
            ch = out_ch 

        self.net = nn.ModuleList(layers)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None,
        rois: list[tuple[int, int, int, int, int]] | None = None 
    ) -> torch.Tensor: 

        if mask is not None: 
            if mask.ndim == 3: 
                mask = mask.unsqueeze(1)
            mask = mask.to(device=x.device, dtype=x.dtype)
            if mask.shape[-2:] != x.shape[-2:]: 
                mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")

        MICRO_BATCH_SIZE = 16 
        n_tiles = x.shape[0]

        if rois is None: 
            pooled_chunks = []
            for i in range(0, n_tiles, MICRO_BATCH_SIZE): 

                x_chunk = x[i:i+MICRO_BATCH_SIZE] 
                m_chunk = mask[i:i+MICRO_BATCH_SIZE] if mask is not None else None 

                feat_chunk = self._run_chunk(x_chunk, m_chunk) 

                if not isinstance(feat_chunk, torch.Tensor): 
                    raise TypeError 
                pooled = self._global_pool_chunk(feat_chunk, m_chunk, x_chunk)
                pooled_chunks.append(pooled)
            
            if not pooled_chunks: 
                return x.new_zeros((0, self.out_dim))
            return torch.cat(pooled_chunks, dim=0)

        rois_t = self.rois_to_tensor(rois, x.device)
        if rois_t.numel() == 0: 
            return x.new_zeros((0, self.out_dim))

        b_idx   = rois_t[:, 0].long() 
        outputs = x.new_zeros((rois_t.size(0), self.out_dim))

        for i in range(0, n_tiles, MICRO_BATCH_SIZE): 
            x_chunk = x[i:i+MICRO_BATCH_SIZE] 
            m_chunk = mask[i:i+MICRO_BATCH_SIZE] if mask is not None else None 

            sel = (b_idx >= i) & (b_idx < i + MICRO_BATCH_SIZE)
            if not sel.any(): 
                continue 

            rois_chunk = rois_t[sel].clone() 
            rois_chunk[:, 0] -= i 

            feat_chunk = self._run_chunk(x_chunk, m_chunk) 

            if not isinstance(feat_chunk, torch.Tensor): 
                raise TypeError 
            pooled = self._roi_pool_chunk(feat_chunk, m_chunk, rois_chunk, x_chunk)
            outputs[sel] = pooled 
    
        return outputs

    def global_pool(
        self, 
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        x: torch.Tensor
    ) -> torch.Tensor: 
        m = self.prep_mask(x, mask).to(dtype=feats.dtype)
        m = F.interpolate(m, size=feats.shape[-2:], mode="nearest")
        return self.masked_avgmax(feats, m)

    def _roi_pool_chunk(
        self, 
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        rois: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor: 
        if rois.numel() == 0: 
            return feats.new_zeros((0, self.out_dim))

        spatial_scale = self.spatial_scale 

        m = self.prep_mask(x, mask)
        m = m.to(dtype=feats.dtype)

        chunk_size    = 16 
        pooled_chunks = []
        mask_chunks   = []
    
        for i in range(0, rois.size(0), chunk_size): 
            rois_i   = rois[i:i+chunk_size]
            pooled_i = tops.roi_align(
                feats, rois_i, output_size=self.roi_output_size,
                spatial_scale=spatial_scale, 
                sampling_ratio=self.sampling_ratio, aligned=self.aligned
            )
            mask_i = tops.roi_align(
                m, rois_i, output_size=self.roi_output_size,
                spatial_scale=spatial_scale,
                sampling_ratio=self.sampling_ratio, aligned=self.aligned
            )

            pooled_chunks.append(pooled_i)
            mask_chunks.append(mask_i)

        pooled      = torch.cat(pooled_chunks, dim=0)
        mask_pooled = torch.cat(mask_chunks, dim=0)
        mask_pooled = (mask_pooled > 0).to(mask_pooled.dtype)
        mask_pooled = mask_pooled.to(dtype=pooled.dtype)

        return self.masked_avgmax(pooled, mask_pooled)

    def _run_chunk(self, chunk, mask=None): 
        if self.categorical_input: 
            x_idx = chunk.squeeze(1).long().clamp(0, self.categorical_cardinality)

            if mask is None: 
                mask = (x_idx > 0).unsqueeze(1).to(device=chunk.device)
            else: 
                if mask.ndim == 3: 
                    mask = mask.unsqueeze(1)
                mask = mask.to(device=chunk.device)
                if mask.shape[-2:] != x_idx.shape[-2:]:
                    mask = F.interpolate(
                        mask.to(dtype=chunk.dtype), 
                        size=x_idx.shape[-2:], 
                        mode="nearest"
                    )
                    mask = mask > 0
                mask = mask & (x_idx > 0).unsqueeze(1)

            emb   = self.embedding(x_idx)
            chunk = emb.permute(0, 3, 1, 2).contiguous() 
            if mask is not None: 
                chunk = chunk * mask 

        else: 
            if mask is not None: 
                mask = self.prep_mask(chunk, mask)

        def _checkpoint(x, block, mask):
            return block(x, mask)[0]

        def _run_block(block, x, mask, training): 
            if training: 
                x = cp.checkpoint(_checkpoint, x, block, mask, use_reentrant=True)
                return x, mask 
            return block(x, mask)

        for block in self.net: 
            chunk, mask = _run_block(block, chunk, mask, self.training)

        return chunk

    def _global_pool_chunk(
        self,
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        x: torch.Tensor 
    ) -> torch.Tensor: 
        m = self.prep_mask(x, mask).to(dtype=feats.dtype)
        m = F.interpolate(m, size=feats.shape[-2:], mode="nearest")
        return self.masked_avgmax(feats, m)

    @staticmethod 
    def prep_mask(
        x: torch.Tensor, 
        mask: torch.Tensor | None
    ) -> torch.Tensor: 
        if mask is None: 
            mask = torch.ones(
                (x.shape[0], 1, x.shape[2], x.shape[3]), 
                device=x.device,
                dtype=torch.bool
            )
        elif mask.ndim == 3: 
            mask = mask.unsqueeze(1)
        mask = mask.to(device=x.device)
        if mask.dtype != torch.bool: 
            mask = mask > 0
        return mask

    @staticmethod 
    def rois_to_tensor(rois, device) -> torch.Tensor: 
        if len(rois) == 0: 
            return torch.zeros((0, 5), device=device, dtype=torch.float32)
        return torch.tensor(
            [(b, x0, y0, x1, y1) for (b, y0, x0, y1, x1) in rois],
            device=device,
            dtype=torch.float32
        )

    @staticmethod
    def masked_avgmax(
        x: torch.Tensor, 
        mask: torch.Tensor, 
        eps: float = 1e-6
    ) -> torch.Tensor: 
        mask     = mask.to(device=x.device, dtype=x.dtype)
        sum_x    = (x * mask).flatten(2).sum(2)
        sum_m    = mask.flatten(2).sum(2)
        avg      = sum_x / (sum_m + eps)  

        neg_inf  = torch.finfo(x.dtype).min 
        mx       = torch.where(
            mask > 0, 
            x, 
            torch.tensor(neg_inf, device=x.device, dtype=x.dtype)
        )
        mx       = mx.flatten(2).max(2).values 

        has_mask = sum_m > 0 
        avg      = torch.where(has_mask, avg, torch.zeros_like(avg))
        mx       = torch.where(has_mask, mx, torch.zeros_like(mx))

        return torch.cat([avg, mx], dim=1)
