#!/usr/bin/env python3 
# 
# networks.py  Andrew Belles  Jan 14th, 2026 
# 
# Neural Network, etc. instantiation implementation 
# 
# 

import torch 
from torch import nn 


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


class ConvBackbone(nn.Module): 
    '''
    Generic CNN backbone outputting a feature vector per image 
    '''

    def __init__(
        self,
        in_channels: int, 
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3, 
        pool_size: int = 2, 
        pool_mode: str = "avg", 
        use_bn: bool = True 
    ): 
        super().__init__()
        layers = []
        ch     = in_channels 
        for out_ch in conv_channels: 
            layers.append(_conv_block(ch, out_ch, kernel_size, pool_size, use_bn))
            ch = out_ch 
        self._set_pool(conv_channels, pool_mode)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.net(x)
        if self.pool_mode == "avgmax": 
            avg = nn.functional.adaptive_avg_pool2d(x, 1)
            mx  = nn.functional.adaptive_max_pool2d(x, 1)
            x   = torch.cat([avg, mx], dim=1)
        else: 
            x   = self.pool(x)
        return x.flatten(1)

    def _set_pool(self, conv_channels, mode: str = "avg"): 
        self.pool_mode = mode 
        if mode == "avg": 
            self.pool    = nn.AdaptiveAvgPool2d(1)
            self.out_dim = conv_channels[-1]
        elif mode == "max": 
            self.pool    = nn.AdaptiveMaxPool2d(1)
            self.out_dim = conv_channels[-1]
        elif mode == "avgmax": 
            self.pool    = nn.Identity() 
            self.out_dim = conv_channels[-1] * 2 
        else: 
            raise ValueError("pool_mode must be avg/max/avgmax")

class ConvClassifier(nn.Module):
    '''
    Generic CNN classifier (backbone + MLP head)
    '''

    def __init__(
        self,
        in_channels: int, 
        n_classes: int, 
        conv_channels: tuple[int, ...] = (32, 64, 128), 
        kernel_size: int = 3, 
        pool_size: int = 2, 
        fc_dim: int = 128, 
        dropout: float = 0.2, 
        use_bn: bool = True 
    ): 
        super().__init__() 
        self.backbone = ConvBackbone(
            in_channels=in_channels,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            pool_size=pool_size,
            use_bn=use_bn
        )
        self.head     = nn.Sequential(
            nn.Linear(self.backbone.out_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.head(self.backbone(x))
