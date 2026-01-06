#!/usr/bin/env python3 
# 
# metric.py  Andrew Belles  Dec 28th, 2025 
# 
# Defines the general interface for metric based methods 
# as well as specific implementations that are operable 
# with the projects pipeline 
# 

import numpy as np 
import torch 

from scipy import sparse 

from numpy.typing import NDArray
from typing import Callable, Iterable, Optional 

from torch.utils.data import DataLoader, TensorDataset 

from sklearn.preprocessing import StandardScaler

from models.graph.construction import normalize_adjacency

def _edge_pairs_from_adj(
    adj: sparse.spmatrix, 
    *, 
    upper_only: bool = True
) -> tuple[NDArray, NDArray]: 
    if not sparse.isspmatrix(adj): 
        raise TypeError("adj must be scipy sparse matrix")
    A = adj.tocsr() 
    if upper_only: 
        A = sparse.triu(A, k=1).tocsr() 
    row, col = A.nonzero()
    return row.astype(np.int64, copy=False), col.astype(np.int64, copy=False)

class EdgeNetwork(torch.nn.Module): 
    '''
    Graph Attention Network base that learns whether edges from some adjacency matrix 
    assumed a priori should be kept in a final adjacency matrix 
    '''

    def __init__(
        self, 
        feature_dim: int, 
        hidden_dims: Iterable[int] = (64, 64), 
        dropout: float = 0.0,
        activation: Callable = torch.nn.ReLU, 
        batch_norm: bool = False, 
        residual: bool = False 
    ): 
        super().__init__() 
        hidden_dims = list(hidden_dims)
        if not hidden_dims: 
            raise ValueError("hidden_dim must have at least one layer")

        input_dim = (2 * feature_dim) + 1 
        dims = [input_dim] + hidden_dims + [1]

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.norms  = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(dims[i + 1]) if batch_norm and i < len(dims) - 2 else 
             torch.nn.Identity() for i in range(len(dims) - 1)]
        )
        self.proj   = torch.nn.ModuleList(
            [torch.nn.Identity() if dims[i] == dims[i + 1] 
             else torch.nn.Linear(dims[i], dims[i + 1], bias=False) 
             for i in range(len(dims) - 1)]
        )
        self.act      = activation() 
        self.drop     = torch.nn.Dropout(dropout)
        self.residual = residual 

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor: 
        diff = torch.abs(x_i - x_j)
        cos_sim = torch.nn.functional.cosine_similarity(x_i, x_j, dim=1, eps=1e-12).unsqueeze(1)
        h = torch.cat([diff, diff**2, cos_sim], dim=1)

        for i, layer in enumerate(self.layers): 
            out = layer(h) 
            if i < len(self.layers) - 1: 
                out = self.norms[i](out)
                out = self.act(out)
                out = self.drop(out)
                if self.residual: 
                    out = out + self.proj[i](h)
            h = out 
        return h.squeeze(1)


class EdgeLearner: 
    def __init__(
        self, 
        *,
        hidden_dims: Iterable[int] = (64, 64), 
        activation: Callable = torch.nn.ReLU,
        dropout: float = 0.0, 
        batch_norm: bool = False,
        residual: bool = False, 
        net_factory = None, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-4, 
        epochs: int = 1500, 
        batch_size: Optional[int] = None, 
        standardize: bool = True, 
        device: Optional[str] = "cuda", 
        seed: int = 0, 
        verbose: bool = False 
    ): 
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.net_factory = net_factory
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.standardize = standardize
        self.device = self._resolve_device(device)
        self.seed = seed
        self.verbose = verbose

        self.model_: Optional[torch.nn.Module] = None
        self.scaler_: Optional[StandardScaler] = StandardScaler() if standardize else None

    def fit(
        self, 
        X, 
        y,
        adj: sparse.spmatrix, 
        train_mask: Optional[NDArray] = None 
    ): # self 

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if y.ndim == 2: 
            y = np.argmax(y, axis=1)
        if y.ndim != 1: 
            raise ValueError("y must be 1d labels or 2d one-hot")

        n = X.shape[0]
        if train_mask is None: 
            train_mask = np.ones(n, dtype=bool)
        else: 
            train_mask = np.asarray(train_mask, dtype=bool)
            if train_mask.shape[0] != n: 
                raise ValueError("train_mask length mismatch")

        if self.scaler_ is not None: 
            self.scaler_.fit(X[train_mask])
            X = self.scaler_.transform(X)

        row, col  = _edge_pairs_from_adj(adj, upper_only=True)
        edge_mask = train_mask[row] & train_mask[col]
        row, col  = row[edge_mask], col[edge_mask]
        if row.size == 0: 
            raise ValueError("no trainable edges")

        y_edge = (y[row] == y[col]).astype(np.float32)

        torch.manual_seed(self.seed)
        if self.model_ is None: 
            self.model_ = self._build_model(feature_dim=X.shape[1])

        x_i = torch.as_tensor(X[row], device=self.device)
        x_j = torch.as_tensor(X[col], device=self.device)
        y_edge_t = torch.as_tensor(y_edge, device=self.device)

        pos = float(y_edge_t.sum().item())
        neg = float(y_edge_t.numel() - pos)
        pos_weight = neg / max(pos, 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.batch_size is None: 
            loader = [(x_i, x_j, y_edge_t)]
        else: 
            dataset = TensorDataset(x_i, x_j, y_edge_t)
            loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train() 
        for epoch in range(self.epochs): 
            epoch_loss = 0.0 
            # iterate per batch 
            for b_xi, b_xj, b_y in loader: 
                optimizer.zero_grad() 
                logits = self.model_(b_xi, b_xj)
                loss   = loss_fn(logits, b_y)
                loss.backward() 
                optimizer.step() 
                epoch_loss += loss.item() 
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0): 
                print(f"[{epoch + 1}/{self.epochs}] loss={epoch_loss:.4f}")

        self.model_.eval()
        return self 

    @torch.no_grad() 
    def edge_weights(
        self, 
        X, 
        adj: sparse.spmatrix 
    ) -> tuple[NDArray, NDArray, NDArray]: 
        if self.model_ is None: 
            raise ValueError("model not fit")

        X = np.asarray(X, dtype=np.float32)
        if self.scaler_ is not None: 
            X = self.scaler_.transform(X)

        row, col = _edge_pairs_from_adj(adj, upper_only=True)
        x_i = torch.as_tensor(X[row], device=self.device)
        x_j = torch.as_tensor(X[col], device=self.device)

        logits = self.model_(x_i, x_j)
        probs  = torch.sigmoid(logits).cpu().numpy().astype(np.float64, copy=False)
        return row, col, probs

    @torch.no_grad() 
    def build_graph(
        self, 
        X, 
        adj: sparse.spmatrix, 
        *, 
        symmetrize: bool = True, 
        normalize: bool = True 
    ) -> sparse.csr_matrix: 
        row, col, w = self.edge_weights(X, adj)
        n = X.shape[0]
        W = sparse.csr_matrix((w, (row, col)), shape=(n, n))
        if symmetrize: 
            W = W + W.T 
        if normalize: 
            W = normalize_adjacency(W, binarize=False)
        return W 

    def __call__(
        self, 
        X,
        adj: sparse.spmatrix, 
        **kwargs
    ) -> sparse.csr_matrix: 
        return self.build_graph(X, adj, **kwargs)

    def _build_model(self, feature_dim: int) -> torch.nn.Module: 
        if self.net_factory is not None: 
            return self.net_factory(feature_dim).to(self.device)
        return EdgeNetwork(
            feature_dim=feature_dim, 
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            residual=self.residual 
        ).to(self.device)

    def _resolve_device(self, device: Optional[str]) -> torch.device: 
        if device is None: 
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available(): 
            device = "cpu"
        return torch.device(device)
