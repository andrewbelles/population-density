#!/usr/bin/env python3 
# 
# loss_funcs.py  Andrew Belles  Jan 13th, 2026 
# 
# helper functions for custom implementations of loss functions 
# 
# 

import torch 

import torch.nn as nn 


class WassersteinLoss(nn.Module): 
    '''
    Efficient and Stable loss for ordinal classification 
    '''

    def __init__(self, n_classes: int = 6, class_weights=None): 
        super().__init__() 
        self.n_classes = n_classes 
        if class_weights is not None: 
            self.register_buffer(
                "class_weights", 
                torch.as_tensor(class_weights, dtype=torch.float32)
            )
        else: 
            self.class_weights = None 

    def forward(self, logits, y): 
        probs    = torch.softmax(logits, dim=1)
        prob_cum = torch.cumsum(probs, dim=1)

        y_onehot = nn.functional.one_hot(y, num_classes=self.n_classes)
        y_onehot = y_onehot.to(device=logits.device, dtype=logits.dtype)
        y_cum    = torch.cumsum(y_onehot, dim=1)

        loss     = torch.sum((prob_cum - y_cum) ** 2, dim=1)
        if self.class_weights is not None: 
            weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss    = loss * weights.gather(0, y)
        return loss.mean()


class CornLoss(nn.Module):
    '''
    Conditional Ordinal Regression loss. More stable & efficient for ordinal regression than 
    Wasserstein 
    '''

    def __init__(
        self, 
        n_classes: int, 
        class_weights=None, 
        eps: float = 1e-6
    ): 
        super().__init__()
        if n_classes < 2: 
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes
        self.eps       = eps 
        if class_weights is not None: 
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            if w.numel() != n_classes: 
                raise ValueError(f"class_weights size {w.numel()} != n_classes {n_classes}")
            self.register_buffer("class_weights", w) 
        else: 
            self.class_weights = None 

    def forward(self, logits, y): 
        if logits.ndim != 2: 
            raise ValueError(f"logits must be 2d, got shape {logits.shape}")
        if logits.size(1) != self.n_classes - 1: 
            raise ValueError(f"logits second dim {logits.size(1)} != n_classes - 1 "
                             f"{self.n_classes -1}")

        yb      = y.to(device=logits.device)
        targets = self._ordinal_targets(yb, self.n_classes).to(dtype=logits.dtype)

        loss    = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        if self.class_weights is not None: 
            w    = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = loss * w.gather(0, yb).unsqueeze(1)

        return loss.mean()

    @staticmethod 
    def _ordinal_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor: 
        thresholds = torch.arange(n_classes - 1, device=y.device).view(1, -1)
        return (y.view(-1, 1) > thresholds).to(dtype=torch.float32)


class SupConLoss(nn.Module): 
    '''
    Supervised Contrastive Loss meant to augment/regularize CORN Loss (Khosla et. al)
    '''

    def __init__(self, temperature=0.07): 
        super().__init__() 
        self.temperature = temperature 

    def forward(self, features, labels): 

        device     = features.device 
        batch_size = features.shape[0]

        labels     = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size: 
            raise ValueError("num labels must match num features")

        mask = torch.eq(labels, labels.T).float().to(device)

        # Similarity matrix via cosine similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits        = anchor_dot_contrast - logits_max.detach() 

        logits_mask = torch.scatter(
            torch.ones_like(mask), 0, 
            torch.arange(batch_size).view(-1, 1).to(device), 0 
        )

        mask = mask * logits_mask 

        exp_logits = torch.exp(logits) * logits_mask 
        log_prob   = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        loss = -mean_log_prob.mean() 
        return loss
