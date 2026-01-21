#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import torch 

from torch import nn 

import torch.nn.functional as F 

import torchvision.ops as tops 

def _conv_block(in_ch, out_ch, kernel_size=3, pool_size=2, use_bn=True): 
    layers = [
        nn.Conv2d(
            in_ch, 
            out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2, 
            bias=not use_bn 
        )
    ]
    
    if use_bn: 
        num_groups = 8 if out_ch < 32 else 32 
        layers.append(nn.GroupNorm(num_groups, out_ch))
    layers += [
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=pool_size)
    ]
    return nn.Sequential(*layers)


class SpatialBackbone(nn.Module): 
    '''
    CNN Backbone tailored for masked, spatial satellite data for NLCD/VIIRS  
    
    Leverages Region of Interest pooling on packed samples.

    Manually downsamples masks to preserve them through model.
    '''

    def __init__(
        self,
        in_channels: int, 
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3, 
        pool_size: int = 2, 
        use_bn: bool = True, 
        roi_output_size: int | tuple[int, int] = 7, 
        sampling_ratio: int = 2, 
        aligned: bool = False 
    ): 
        super().__init__()
        layers = []
        ch     = in_channels 
        for out_ch in conv_channels: 
            layers.append(_conv_block(ch, out_ch, kernel_size, pool_size, use_bn))
            ch = out_ch 

        self.net             = nn.Sequential(*layers)
        self.out_dim         = conv_channels[-1] * 2 # avg + max concat  
        self.roi_output_size = roi_output_size
        self.sampling_ratio  = sampling_ratio 
        self.aligned         = aligned 


    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor | None = None,
        rois: list[tuple[int, int, int, int, int]] | None = None 
    ) -> torch.Tensor: 
        if x.shape[1] == 1 and self.net[0][0].in_channels > 1: 
            x_idx    = x.squeeze(1).long() 
            x_onehot = F.one_hot(x_idx, num_classes=self.net[0][0].in_channels + 1)
            x        = x_onehot[..., 1:].permute(0, 3, 1, 2).float() 

        feats = self.net(x)
        if rois is None: 
            return self.global_pool(feats, mask, x)
        return self.roi_pool(feats, mask, rois, x)

    def global_pool(
        self, 
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        x: torch.Tensor
    ) -> torch.Tensor: 
        m = self.prep_mask(x, mask)
        m = F.interpolate(m, size=feats.shape[-2:], mode="nearest")
        return self.masked_avgmax(feats, m)

    def roi_pool(
        self, 
        feats: torch.Tensor, 
        mask: torch.Tensor | None, 
        rois, 
        x: torch.Tensor
    ) -> torch.Tensor: 
        rois_t = self.rois_to_tensor(rois, feats.device)
        if rois_t.numel() == 0: 
            return feats.new_zeros((0, self.out_dim))


        scale_h = feats.shape[-2] / x.shape[-2] 
        scale_w = feats.shape[-1] / x.shape[-1] 
        if abs(scale_h - scale_w) > 1e-6: 
            raise ValueError("non-uniform spatial scale not supported for roi_align")
        spatial_scale = scale_h 
    
        pooled = tops.roi_align(
            feats, 
            rois_t, 
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )

        m = self.prep_mask(x, mask)
        mask_pooled = tops.roi_align(
            m,
            rois_t,
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned
        )
        mask_pooled = (mask_pooled > 0).to(mask_pooled.dtype)

        return self.masked_avgmax(pooled, mask_pooled)

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
