#!/usr/bin/env python3 
# 
# loss_funcs.py  Andrew Belles  Jan 13th, 2026 
# 
# helper functions for custom implementations of loss functions 
# 
# 

import torch 

import torch.nn as nn 

import torch.nn.functional as F 

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

        if self.class_weights is not None: 
            anchor_weights = self.class_weights.to(device).gather(0, labels.view(-1))
            loss = - (mean_log_prob * anchor_weights).sum() / anchor_weights.sum() 
        else:
            loss = -mean_log_prob.mean() 
        return loss


class LearnableLogitScaling(nn.Module): 

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
        temperature: float = 0.1 
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

    def forward(self, logits, embeddings, labels): 
        corn_loss = self.corn_fn_(logits, labels)

        log_cond_probs = F.logsigmoid(logits)
        log_cdf        = torch.cumsum(log_cond_probs, dim=1)
        cdf = torch.exp(log_cdf)

        targets  = self.corn_fn_._ordinal_targets(labels, self.n_classes_)
        per_sample_rps = torch.sum((cdf - targets)**2, dim=1)

        if self.class_weights is not None: 
            sample_weights = self.class_weights.to(logits.device).gather(0, labels)
            loss_rps = (per_sample_rps * sample_weights).sum() / (sample_weights.sum() + 1e-9)
        else: 
            loss_rps = per_sample_rps.mean() 


        if self.beta > 0 and embeddings is not None: 
            norm_emb = F.normalize(embeddings, dim=1)
            loss_supcon = self.supcon_fn_(norm_emb, labels)
        else: 
            loss_supcon = torch.tensor(0.0, device=logits.device)

        weighted_rps    = self.alpha * loss_rps 
        weighted_supcon = self.beta * loss_supcon 

        total_loss = corn_loss + weighted_rps + weighted_supcon
        return total_loss, corn_loss, weighted_rps, weighted_supcon 
