#!/usr/bin/env python3 
# 
# loss_funcs.py  Andrew Belles  Jan 13th, 2026 
# 
# helper functions for custom implementations of loss functions 
# 
# 

from typing import Optional, Required
import torch 

import torch.nn as nn 

import torch.nn.functional as F 

import numpy as np 

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

    def forward(self, logits, y, reduction="mean"): 
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

        per_sample = loss.mean(dim=1)

        if reduction == "mean": 
            return per_sample.mean() 
        elif reduction == "none": 
            return per_sample
        else: 
            raise ValueError(f"unknown reduction: {reduction}")

    @staticmethod 
    def _ordinal_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor: 
        thresholds = torch.arange(n_classes - 1, device=y.device).view(1, -1)
        return (y.view(-1, 1) > thresholds).to(dtype=torch.float32)


class SupConLoss(nn.Module): 
    '''
    Combination of (Zhu et. al), (Abbasian et. al), and (Khosla et. al) for weighted,  
    ordinal supervised contrastive loss. 
    '''

    def __init__(self, temperature=0.07, class_weights=None): 
        super().__init__() 
        self.temperature = temperature 

        if class_weights is not None: 
            self.register_buffer(
                "class_weights", torch.as_tensor(class_weights, dtype=torch.float32)
            )
        else: 
            self.class_weights = None 

    def forward(self, features, labels, reduction="mean"): 

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

        # Get ordinal weight from discrete labels 
        label_dist     = torch.abs(labels - labels.T).float() 
        max_dist       = label_dist.max().clamp(min=1.0)
        ordinal_weight = label_dist / max_dist  

        exp_logits   = torch.exp(logits) * logits_mask 
        denom_weight = mask + (1.0 - mask) * ordinal_weight 

        log_prob     = logits - torch.log(
            (exp_logits * denom_weight).sum(1, keepdims=True) + 1e-9
        )
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        per_sample    = -mean_log_prob

        if self.class_weights is not None: 
            anchor_weights = self.class_weights.to(device).gather(0, labels.view(-1))
            per_sample = per_sample * anchor_weights

        if reduction == "none":
            return per_sample 
        elif reduction == "mean": 
            if self.class_weights is not None: 
                w = self.class_weights.to(device).gather(0, labels.view(-1))
                return per_sample.sum() / (w.sum() + 1e-9)
            else: 
                return per_sample.mean() 
        else: 
            raise ValueError(f"unknown reduction: {reduction}")



class LogitScaler(nn.Module): 

    def __init__(self, initial_value: float = 1.0): 
        super().__init__() 
        self.scale = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))

    def forward(self, x): 
        return x * self.scale.clamp(min=1e-4, max=50.0)


class HybridOrdinalLoss(nn.Module): 
    def __init__(
        self, 
        n_classes: int, 
        class_weights: torch.Tensor | None = None, 
        alpha_rps: float = 1.0, 
        beta_supcon: float = 0.5, 
        temperature: float = 0.1, 
        reduction="mean"            # if none then returns per sample loss values 
    ): 

        super().__init__()
        self.alpha      = alpha_rps 
        self.beta       = beta_supcon
        self.corn_fn_   = CornLoss(n_classes, class_weights=class_weights)
        self.supcon_fn_ = SupConLoss(temperature, class_weights=class_weights)
        self.n_classes_ = n_classes

        if class_weights is not None: 
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weights)
        else: 
            self.class_weights = None 

        self.reduction  = reduction

    def forward(self, logits, embeddings, labels, sample_weight: Optional[torch.Tensor] = None): 

        class_w = None 
        if self.class_weights is not None: 
            class_w = self.class_weights.to(logits.device).gather(0, labels)

        sw = sample_weight.view(-1) if sample_weight is not None else None 

        probs_exceed = torch.sigmoid(logits)
        targets      = self.corn_fn_._ordinal_targets(labels, self.n_classes_)
        rps_per      = torch.sum((probs_exceed - targets)**2, dim=1)
        K            = max(self.n_classes_ - 1, 1)
        rps_per     /= float(K) 
 
        corn_per     = self.corn_fn_(logits, labels, reduction="none")

        if self.beta > 0 and embeddings is not None: 
            norm_emb = F.normalize(embeddings, dim=1) 
            sup_per  = self.supcon_fn_(norm_emb, labels, reduction="none") 
        else: 
            sup_per  = torch.zeros_like(rps_per)

        weights = None 
        if class_w is not None and sw is not None: 
            weights = class_w * sw 
        elif class_w is not None: 
            weights = class_w 
        elif sw is not None: 
            weights = sw 

        corn = self.reduce(corn_per, weight=weights, normalize=False)
        rps  = self.reduce(rps_per, weight=weights, normalize=True)
        sup  = self.reduce(sup_per, weight=weights, normalize=True)

        weighted_rps = self.alpha * rps 
        weighted_sup = self.beta * sup

        loss = corn + weighted_rps + weighted_sup
        return loss, corn, weighted_rps, weighted_sup 

    def reduce(self, per_sample, weight=None, normalize=False): 
        if weight is not None: 
            x = per_sample * weight 
        else: 
            x = per_sample

        if self.reduction == "none": 
            return x 

        if normalize and weight is not None: 
            return x.sum() / (weight.sum() + 1e-9)
        else: 
            return x.mean() 

class MixedLoss(nn.Module):

    '''
    Computes loss using both hard labels on mixed tabular data. Returns soft loss as linear 
    ratio of the two hard loss values. 
    '''

    def __init__(
        self, 
        base_loss: HybridOrdinalLoss
    ): 
        super().__init__() 
        base_loss.reduction = "none"    # ensure we don't take mean across samples 
        self.loss_fn_       = base_loss  

    def forward(self, logits, proj, y_a, y_b, mix_lambda): 
        mix_lambda = mix_lambda.view(-1)

        loss_a, corn_a, rps_a, sup_a = self.loss_fn_(
            logits, proj, y_a, sample_weight=mix_lambda
        )
        loss_b, corn_b, rps_b, sup_b = self.loss_fn_(
            logits, proj, y_b, sample_weight=(1 - mix_lambda)
        )

        # values returned are already weighted. 
        loss = loss_a + loss_b
        corn = corn_a + corn_b
        rps  = rps_a  + rps_b
        sup  = sup_a  + sup_b

        return loss.mean(), corn.mean(), rps.mean(), sup.mean() 


def compute_ens_weights(class_counts, beta=0.999):
    '''
    Computes Effective Number of Samples weights proposed by cui et al.
    '''
    class_counts = np.asarray(class_counts)
    
    class_counts = np.clip(class_counts, 1, None) 
    
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)
