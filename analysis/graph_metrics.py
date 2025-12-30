#!/usr/bin/env python3 
# 
# homophily.py  Andrew Belles  Dec 28th, 2025 
# 
# Analysis of graphs for their ability to identify edges between 
# nodes of equivalent class labels (homophily)
# 

import numpy as np 

import scipy.sparse as sp 
from scipy.sparse.csgraph import connected_components

from typing import Dict
from numpy.typing import NDArray 

class MetricAnalyzer: 

    @staticmethod 
    def compute_metrics(
        adj: sp.csr_matrix,
        probs: NDArray | None = None, 
        train_mask: NDArray | None = None, 
        verbose: bool = False 
    ) -> Dict[str, float]: 

        results = {}

        results["avg_degree"] = MetricAnalyzer.avg_degree(adj)
        results["largest_component_ratio"] = MetricAnalyzer.largest_component_ratio(adj)

        if probs is not None: 
            results["prob_edge_l2"] = MetricAnalyzer.prob_edge_l2(adj, probs)
            results["pred_edge_homophily"] = MetricAnalyzer.pred_edge_homophily(adj, probs)
            results["confidence_edge_diff"] = MetricAnalyzer.confidence_edge_diff(adj, probs)

        if train_mask is not None: 
            results["train_neighbor_coverage"] = MetricAnalyzer.train_neighbor_coverage(
                adj, 
                train_mask
            )
            results["avg_train_neighbors"] = MetricAnalyzer.avg_train_neighbors(
                adj,
                train_mask
            )

        if verbose: 
            MetricAnalyzer._print_report(results)

        return results 

    @staticmethod
    def edge_homophily(
        adj: sp.csr_matrix, 
        y: NDArray
    ) -> float: 

        y = np.asarray(y).reshape(-1)
        adj_binary = (adj > 0).astype(int)
        adj_binary.setdiag(0)
        adj_binary.eliminate_zeros()

        # Check for empty matrix 
        if adj_binary.nnz == 0: 
            return 0.0 

        src, dst = adj_binary.nonzero() 
        matches  = (y[src] == y[dst]).sum() 

        return float(matches / src.shape[0])

    @staticmethod 
    def adjusted_homophily(
        adj: sp.csr_matrix,
        y: NDArray,
        h_edge: float | None = None 
    ) -> float: 

        y = np.asarray(y).reshape(-1)
        if h_edge is None: 
            h_edge = MetricAnalyzer.edge_homophily(adj, y)

        degrees = np.asarray(adj.sum(axis=1)).flatten() 
        total_vol = degrees.sum()

        # Check for empty matrix 
        if total_vol == 0: 
            return 0.0 

        classes  = np.unique(y)
        p_sq_sum = 0.0 

        for c in classes: 
            vol_c    = degrees[y == c].sum() 
            prob_c   = vol_c / total_vol 
            p_sq_sum = prob_c ** 2 

        if p_sq_sum >= 1.0 - 1e-9: 
            return 0.0 
        
        h_adj = (h_edge - p_sq_sum) / (1.0 - p_sq_sum)
        return float(h_adj)

    @staticmethod 
    def _print_report(results: Dict[str, float]): 
        rows = [
            ("avg_degree", "Avg Degree", "{:2f}"),
            ("largest_component_ratio", "Largest Component", "{:.3f}"),
            ("edge_homophily", "Edge Homophily", "{:.4f}"),
            ("adjusted_homophily", "Adjusted Homophily", "{:+.4f}"),
            ("pred_edge_homophily", "Pred Edge Homophily", "{:.4f}"),
            ("prob_edge_l2", "Prob Edge L2", "{:.4f}"),
            ("confidence_edge_diff", "Confidence Edge Diff", "{:.4f}"),
            ("train_neighbor_coverage", "Train Neighbor Coverage", "{:.3f}"),
            ("avg_train_neighbors", "Avg Train Neighbors", "{:.2f}"),
        ]

        print(f"{'Metric':<28} | {'Value':<10}")
        print("=" * 41)

        printed = False 
        for key, label, fmt in rows: 
            if key not in results: 
                continue 
            val = results[key]
            out = "nan" if np.isnan(val) else fmt.format(val) 
            print(f"{label:<28} | {out:<10}")
            printed = True 

        if not printed: 
            print("No metrics to report")

    @staticmethod 
    def _binary_adj(adj: sp.csr_matrix) -> sp.csr_matrix: 
        A = (adj > 0 ).astype(int) 
        A.setdiag(0) 
        A.eliminate_zeros()
        return A 

    @staticmethod 
    def avg_degree(adj: sp.csr_matrix) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.shape[0] == 0: 
            return 0.0 
        deg = np.asarray(A.sum(axis=1)).ravel() 
        return float(deg.mean())

    @staticmethod 
    def largest_component_ratio(adj: sp.csr_matrix) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        n = A.shape[0]
        if n == 0 or A.nnz == 0: 
            return 0.0 
        
        _, labels = connected_components(A, directed=False, connection="weak")
        counts = np.bincount(labels)
        return float(counts.max() / n)

    @staticmethod 
    def prob_edge_l2(adj: sp.csr_matrix, probs: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0: 
            return 0.0 
        P = np.asarray(probs, dtype=np.float64)
        if P.ndim == 1:
            P = P.reshape(-1, 1)

        src, dst = A.nonzero() 
        diff = P[src] - P[dst] 
        l2 = np.sqrt((diff**2).sum(axis=1))
        return float(l2.mean())

    @staticmethod 
    def pred_edge_homophily(adj: sp.csr_matrix, probs: NDArray) -> float: 
        P = np.asarray(probs, dtype=np.float64)
        if P.ndim == 1: 
            preds = (P.ravel() >= 0.5).astype(int)
        else: 
            preds = np.argmax(P, axis=1)
        return MetricAnalyzer.edge_homophily(adj, preds)

    @staticmethod
    def confidence_edge_diff(adj: sp.csr_matrix, probs: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0: 
            return 0.0 
        P = np.asarray(probs, dtype=np.float64)
        if P.ndim == 1: 
            conf = np.abs(P.ravel())
        else: 
            conf = P.max(axis=1)
        src, dst = A.nonzero()
        return float(np.abs(conf[src] - conf[dst]).mean())

    @staticmethod 
    def train_neighbor_coverage(adj: sp.csr_matrix, train_mask: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape[0] != A.shape[0]: 
            raise ValueError("train_mask size does not match adjacency")
        test_mask = ~mask 
        if test_mask.sum() == 0: 
            return 0.0 
        to_train = np.asarray(A[:, mask].sum(axis=1)).ravel() > 0 
        return float(to_train[test_mask].mean())

    @staticmethod 
    def avg_train_neighbors(adj: sp.csr_matrix, train_mask: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        mask = np.asarray(train_mask, dtype=bool)
        if mask.shape[0] != A.shape[0]: 
            raise ValueError("train_mask size does not match adjacency")
        test_mask = ~mask 
        if test_mask.sum() == 0: 
            return 0.0 
        counts = np.asarray(A[:, mask].sum(axis=1)).ravel() 
        return float(counts[test_mask].mean())
