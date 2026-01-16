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
        layers.append(nn.BatchNorm2d(out_ch))
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
        use_bn: bool = True 
    ): 
        super().__init__()
        layers = []
        ch     = in_channels 
        for out_ch in conv_channels: 
            layers.append(_conv_block(ch, out_ch, kernel_size, pool_size, use_bn))
            ch = out_ch 
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net     = nn.Sequential(*layers)
        self.out_dim = conv_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x).flatten(1)


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
