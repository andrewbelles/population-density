#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import torch, sys 

from torch import nn 

import torch.nn.functional as F 

import torchvision.ops as tops 

import torch.utils.checkpoint as cp


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
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")

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
        pool_size=2, 
        use_norm=True 
    ): 
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, 
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2, 
            bias=not use_norm 
        )
        self.norm = None 
        if use_norm: 
            num_groups = 8 if out_ch < 32 else 32 
            self.norm  = MaskedGroupNorm(num_groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

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
        out = self.pool(out)

        next_mask = mask 
        if next_mask is not None: 
            if next_mask.ndim ==3: 
                next_mask = next_mask.unsqueeze(1)

            next_mask = self.pool(next_mask)
            next_mask = (next_mask > 0).to(next_mask.dtype)

        return out, next_mask


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
        pool_size: int = 2, 
        use_bn: bool = True, 
        roi_output_size: int | tuple[int, int] = 7, 
        sampling_ratio: int = 2, 
        aligned: bool = False 
    ): 
        super().__init__()

        self.model_in_channels = in_channels 
        self.out_dim           = conv_channels[-1] * 2 
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
        layers = []
        ch     = self.categorical_embed_dim if self.categorical_input else in_channels
        for out_ch in conv_channels: 
            layers.append(MaskedConvBlock(ch, out_ch, kernel_size, pool_size, use_bn))
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
        m = self.prep_mask(x, mask)
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

        scale_h = feats.shape[-2] / x.shape[-2] 
        scale_w = feats.shape[-1] / x.shape[-1] 
        if abs(scale_h - scale_w) > 1e-6: 
            raise ValueError("non-uniform spatial scale not supported for roi_align")
        spatial_scale = scale_h 
    
        pooled = tops.roi_align(
            feats, 
            rois, 
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )

        m = self.prep_mask(x, mask)
        mask_pooled = tops.roi_align(
            m,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )

        mask_pooled = (mask_pooled > 0).to(mask_pooled.dtype)
        return self.masked_avgmax(pooled, mask_pooled)

    def _run_chunk(self, chunk, mask=None): 
        if self.categorical_input: 
            x_idx = chunk.squeeze(1).long().clamp(0, self.categorical_cardinality)

            if mask is None: 
                mask = (x_idx > 0).unsqueeze(1).to(dtype=chunk.dtype, device=chunk.device)
            else: 
                if mask.ndim == 3: 
                    mask = mask.unsqueeze(1)
                mask = mask.to(device=chunk.device, dtype=chunk.dtype)
                if mask.shape[-2:] != x_idx.shape[-2:]:
                    mask = F.interpolate(mask, size=x_idx.shape[-2:], mode="nearest")
                mask = mask * (x_idx > 0).unsqueeze(1).to(mask.dtype)

            emb   = self.embedding(x_idx)
            chunk = emb.permute(0, 3, 1, 2).contiguous() 
            if mask is not None: 
                chunk = chunk * mask 

        else: 
            if mask is not None: 
                mask = self.prep_mask(chunk, mask)

        for block in self.net: 
            chunk, mask = block(chunk, mask)

        return chunk

    def _global_pool_chunk(
        self,
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        x: torch.Tensor 
    ) -> torch.Tensor: 
        m = self.prep_mask(x, mask)
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
                dtype=x.dtype 
            )
        elif mask.ndim == 3: 
            mask = mask.unsqueeze(1)
        return mask.to(device=x.device, dtype=x.dtype)

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
