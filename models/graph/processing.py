#!/usr/bin/env python3 
# 
# processing.py  Andrew Belles  Dec 25th, 2025 
# 
# Modules that aim to post process probabilities output by Classifiers  
# 

import numpy as np 
import torch 

from scipy import sparse 

from typing import Optional 
from numpy.typing import NDArray

from torch_geometric.nn.models import CorrectAndSmooth as PyGCS 

class CorrectAndSmooth: 

    def __init__(
        self, 
        *, 
        correction: tuple[int, float] = (3, 0.01), 
        smoothing: tuple[int, float] = (8, 0.15), 
        autoscale: bool = True, 
        device: Optional[str] = None,
        class_labels: Optional[NDArray] = None 
    ): 
        self.class_labels = class_labels
        self.device = self._resolve_device(device)
        self.model_ = PyGCS(
            num_correction_layers=correction[0], 
            correction_alpha=correction[1],
            num_smoothing_layers=smoothing[0],
            smoothing_alpha=smoothing[1],
            autoscale=autoscale
        ).to(self.device)

        self._last_probs = None 

    def __call__(
        self, 
        y_soft: NDArray, 
        y_true: NDArray, 
        train_mask: NDArray,
        adj: sparse.spmatrix | tuple[NDArray | torch.Tensor, Optional[NDArray | torch.Tensor]]
    ) -> NDArray: 

        y_soft_t, y_true_t, train_mask_t = self._prepare_labels(
            y_soft, y_true, train_mask
        )
        edge_index, edge_weight = self._prepare_edges(adj)

        y_train = y_true_t[train_mask_t]

        y_corr = self.model_.correct(
            y_soft_t, y_train, train_mask_t, edge_index, edge_weight 
        )

        y_smooth = self.model_.smooth(
            y_corr, y_train, train_mask_t, edge_index, edge_weight 
        )

        self._last_probs = y_smooth.detach().cpu().numpy()
        return self._last_probs

    def _resolve_device(self, device: Optional[str]) -> torch.device: 
        if device is None: 
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available(): 
            device = "cpu"
        return torch.device(device)

    def predict(
        self, 
        probs: Optional[NDArray] = None, 
        *, 
        threshold: float = 0.5
    ):
        if probs is None: 
            if self._last_probs is None: 
                raise ValueError("no cached predictions; call the model first or pass probs")
            probs = self._last_probs 

        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1): 
            hard = (probs.ravel() >= threshold).astype(int)
        else: 
            idx = np.argmax(probs, axis=1) 
            if self.class_labels is not None: 
                return np.asarray(self.class_labels)[idx]
            hard = idx 
        return hard 

    def _prepare_labels(self, y_soft, y_true, train_mask): 
        y_soft = np.asarray(y_soft, dtype=np.float32)
        if y_soft.ndim == 1: 
            y_soft = y_soft.reshape(-1, 1)
        if y_soft.ndim != 2: 
            raise ValueError(f"y_soft must be 2d (N, C), got {y_soft.shape}")

        n = y_soft.shape[0]
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] != n: 
            raise ValueError("train_mask length mismatch")

        y_true = np.asarray(y_true)
        if y_true.ndim == 2: 
            if y_true.shape[0] != n: 
                raise ValueError("y_true length mismatch")
            y_idx = np.argmax(y_true, axis=1).astype(np.int64)
        elif y_true.ndim == 1: 
            if y_true.shape[0] != n: 
                raise ValueError("y_true length mismatch")
            y_idx = y_true.astype(np.int64) 
        else: 
            raise ValueError("y_true must be 1d labels or 2d one-hot")

        y_soft_t = torch.as_tensor(y_soft, device=self.device, dtype=torch.float32)
        y_true_t = torch.as_tensor(y_idx, device=self.device, dtype=torch.long)
        train_mask_t = torch.as_tensor(train_mask, device=self.device, dtype=torch.bool)
        return y_soft_t, y_true_t, train_mask_t 

    def _prepare_edges(
        self, 
        adj: sparse.spmatrix | tuple[NDArray | torch.Tensor, Optional[NDArray | torch.Tensor]]
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]: 

        if sparse.isspmatrix(adj): 
            coo = adj.tocoo() 
            edge_index = np.vstack([coo.row, coo.col])
            edge_weight = coo.data 
        elif isinstance(adj, tuple) and len(adj) == 2: 
            edge_index, edge_weight = adj 
        else: 
            raise TypeError("adj must be scipy parse matrix or (edge_index, edge_weight)")

        if isinstance(edge_index, torch.Tensor): 
            edge_index_t = edge_index.to(self.device, dtype=torch.long)
        else: 
            edge_index = np.asarray(edge_index)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2: 
                raise ValueError("edge_index must have shape (2, E)")
            edge_index_t = torch.as_tensor(
                edge_index, device=self.device, dtype=torch.long
            )

        if edge_weight is None: 
            edge_weight_t = None 
        elif isinstance(edge_weight, torch.Tensor): 
            edge_weight_t = edge_weight.to(self.device, dtype=torch.float32)
        else: 
            edge_weight_t = torch.as_tensor(
                edge_weight, device=self.device, dtype=torch.float32
            )

        return edge_index_t, edge_weight_t
