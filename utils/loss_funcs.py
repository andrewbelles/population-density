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


class WassersteinDiceLoss(nn.Module):
    '''
    Distance-aware + imbalance-aware setup for Wasserstein-Dice loss (Fidon et. al 2017)
    '''

    def __init__(
        self,
        n_classes: int = 6, 
        class_counts=None,
        class_weights=None,
        distance_matrix=None,
        distance_power: float = 1.0, 
        beta: float = 0.99, 
        eps: float = 1e-6
    ): 
        super().__init__()
        self.n_classes      = n_classes 
        self.distance_power = distance_power 
        self.beta           = beta 
        self.eps            = eps 

        if distance_matrix is None: 
            distance_matrix = self._build_distance_matrix(n_classes, power=distance_power)
        else: 
            distance_matrix = torch.as_tensor(distance_matrix, dtype=torch.float32)
            if distance_matrix.shape != (n_classes, n_classes): 
                raise ValueError(f"distance matrix shape {distance_matrix.shape} != "
                                 f"({n_classes}, {n_classes})")

        self.register_buffer("distance_matrix", distance_matrix)

        if class_weights is not None: 
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights.numel() != n_classes: 
                raise ValueError(f"class_weights size {weights.numel()} != n_classes {n_classes}")

            self.register_buffer("class_weights", weights)
        elif class_counts is not None: 
            weights = self._build_class_weights(class_counts, distance_matrix)
            self.register_buffer("class_weights", weights)
        else: 
            self.class_weights = None 

    def forward(self, logits, y): 
        probs = torch.softmax(logits, dim=1)

        if y.ndim == probs.ndim: 
            y_onehot = y 
        else: 
            y_onehot = self._one_hot(y, probs)

        y_onehot   = y_onehot.to(device=logits.device, dtype=logits.dtype)
        n, c       = probs.shape[:2]
        probs_flat = probs.reshape(n, c, -1)
        y_flat     = y_onehot.reshape(n, c, -1)

        M    = self.distance_matrix.to(device=logits.device, dtype=logits.dtype)
        dist = torch.einsum("lc,ncP->nlP", M, y_flat)
        W    = (probs_flat * dist).sum(dim=1)

        alpha = self._resolve_class_weights(device=logits.device, dtype=logits.dtype)

        if y.ndim == probs.ndim: 
            y_idx = y_onehot.reshape(n, c, -1).argmax(dim=1)
        else: 
            y_idx = y.reshape(n, -1)

        alpha_true = alpha.gather(0, y_idx.reshape(-1)).reshape_as(W)
        
        diff = torch.clamp(alpha_true - W, min=0.0)
        tp   = (alpha_true * diff).sum() 
        num  = 2.0 * tp 
        den  = num + W.sum() 

        dice = num / (den + self.eps)
        return 1.0 - dice 

    def set_class_counts(self, class_counts): 
        weights = self._build_class_weights(class_counts, self.distance_matrix)
        self.class_weights = weights 

    def _resolve_class_weights(self, device, dtype): 
        if self.class_weights is None: 
            alpha = self._distance_aware_weights(self.distance_matrix)
        else: 
            alpha = self.class_weights 
        return alpha.to(device=device, dtype=dtype)

    def _one_hot(self, y, probs): 
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.n_classes)
        if y_onehot.ndim >= 2: 
            perm     = [0, y_onehot.ndim - 1] + list(range(1, y_onehot.ndim - 1))
            y_onehot = y_onehot.permute(*perm)
        if y_onehot.ndim != probs.ndim: 
            raise ValueError(f"y_onehot ndim {y_onehot.ndim} != probs ndim {probs.ndim}")
        return y_onehot 

    def _build_class_weights(self, class_counts, distance_matrix: torch.Tensor): 
        counts = torch.as_tensor(class_counts, dtype=torch.float32, device=distance_matrix.device)
        if counts.numel() != self.n_classes: 
            raise ValueError(f"class_counts size {counts.numel()} != n_classes {self.n_classes}")

        imbalance = self._effective_number_weights(counts, self.beta, self.eps)
        distance  = self._distance_aware_weights(distance_matrix).to(counts.device)
        weights   = distance * imbalance 
        weights   = weights / torch.clamp(weights.mean(), min=self.eps)
        return weights 

    @staticmethod 
    def _build_distance_matrix(n_classes: int, power: float = 1.0, dtype=torch.float32): 
        idx = torch.arange(n_classes, dtype=dtype)
        m   = torch.abs(idx[:, None] - idx[None, :])
        if power != 1.0: 
            m = m**power 
        return m 

    @staticmethod
    def _effective_number_weights(counts, beta: float, eps: float): 
        counts  = torch.as_tensor(counts, dtype=torch.float32)
        denom   = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / torch.clamp(denom, min=eps)
        return weights 

    @staticmethod
    def _distance_aware_weights(distance_matrix: torch.Tensor): 
        return distance_matrix.max(dim=1).values
