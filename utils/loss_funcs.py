#!/usr/bin/env python3 
# 
# loss_funcs.py  Andrew Belles  Jan 13th, 2026 
# 
# helper functions for custom implementations of loss functions 
# 
# 

import numpy as np
from scipy.special import softmax 

import torch 

import torch.nn as nn 

def _resolve_n_classes(preds, y, n_classes): 
    if preds.size % n_classes == 0: 
        return n_classes 
    if y is not None and y.size > 0 and preds.size % y.size == 0: 
        inferred = preds.size // y.size 
        return int(inferred)
    raise ValueError(
        f"preds size {preds.size} not compatible with n_classes={n_classes}"
    )

def _split_labels_preds(first, second): 
    if hasattr(second, "get_label"):
        return second.get_label().astype(int), first, True  
    return np.asarray(first, dtype=int), second, False 

def _normalize_alpha(alpha, n_classes): 
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 0: 
        alpha = np.full(n_classes, alpha)
    if alpha.size != n_classes: 
        raise ValueError(f"alpha size {alpha.size} != n_classes {n_classes}")
    return alpha / np.mean(alpha) 

def focal_loss_obj(n_classes, alpha, gamma: float = 0.5):
    alpha_vec = _normalize_alpha(alpha, n_classes)
    eps = 1e-6
    min_hess = 1e-6

    def _focal_loss(first, second):
        y, preds, _ = _split_labels_preds(first, second)

        k = _resolve_n_classes(preds, y, n_classes)
        preds = preds.reshape(-1, k)
        preds = np.nan_to_num(preds, nan=0.0, posinf=20.0, neginf=-20.0)
        preds = preds - np.max(preds, axis=1, keepdims=True)

        p = softmax(preds, axis=1)
        p = np.clip(p, eps, 1.0 - eps)

        y_one = np.eye(k, dtype=np.float64)[y]
        pt = (p * y_one).sum(axis=1)
        pt = np.clip(pt, eps, 1.0 - eps)

        w = (alpha_vec[y] * (1.0 - pt) ** gamma).reshape(-1, 1)

        grad = w * (p - y_one)
        hess = w * p * (1.0 - p)
        hess = np.maximum(hess, min_hess)

        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        hess = np.nan_to_num(hess, nan=min_hess, posinf=min_hess, neginf=min_hess)
        return grad, hess

    return _focal_loss

def focal_mlogloss_eval(n_classes): 
    def _eval(first, second): 
        y, preds, is_dmatrix  = _split_labels_preds(first, second) 
        k         = _resolve_n_classes(preds, y, n_classes)
        preds     = preds.reshape(-1, k)
        preds     = np.nan_to_num(preds, nan=0.0, posinf=20.0, neginf=-20.0)
        preds     = preds - np.max(preds, axis=1, keepdims=True)
        exp       = np.exp(preds)
        
        p  = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-4, None)
        p  = np.clip(p, 1e-4, 1.0 - 1e-4)

        ll = -np.log(p[np.arange(y.size), y]).mean()
        return ("mlogloss", float(ll)) if is_dmatrix else float(ll) 
    return _eval


class WassersteinLoss(nn.Module): 
    '''
    Efficient and Stable loss for ordinal classification with imbalance 
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
