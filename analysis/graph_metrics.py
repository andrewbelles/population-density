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
        y_true: NDArray | None = None, 
        train_mask: NDArray | None = None, 
        coords: NDArray | None = None, 
        verbose: bool = False 
    ) -> Dict[str, float]: 

        results = {}

        results["avg_degree"] = MetricAnalyzer.avg_degree(adj)

        if probs is not None: 
            results["prob_edge_l2"] = MetricAnalyzer.prob_edge_l2(adj, probs)
            results["pred_edge_homophily"] = MetricAnalyzer.pred_edge_homophily(adj, probs)
            results["confidence_edge_diff"] = MetricAnalyzer.confidence_edge_diff(adj, probs)

            if y_true is not None: 
                results["corrective_edge_ratio"] = MetricAnalyzer.corrective_edge_ratio(
                    adj, y_true, probs
                )
                results["recoverable_error_rate"] = MetricAnalyzer.recoverable_error_rate(
                    adj, y_true, probs
                )
                results["smoothness_gap"] = MetricAnalyzer.smoothness_gap(
                    adj, y_true, probs 
                )
                if coords is not None: 
                    results["distance_weighted_rer"] = (
                        MetricAnalyzer.distance_weighted_recoverable_error_rate(
                            adj, y_true, probs, coords 
                        )
                    )
        if coords is not None: 
            results["locality_ratio"] = MetricAnalyzer.locality_ratio(adj, coords)

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
    def _print_report(results: Dict[str, float]): 
        rows = [
            ("avg_degree", "Avg Degree", "{:2f}"),
            ("largest_component_ratio", "Largest Component", "{:.3f}"),
            ("edge_homophily", "Edge Homophily", "{:.4f}"),
            ("adjusted_homophily", "Adjusted Homophily", "{:+.4f}"),
            ("pred_edge_homophily", "Pred Edge Homophily", "{:.4f}"),
            ("corrective_edge_ratio", "Corrective Edge Ratio", "{:.4f}"),
            ("recoverable_error_rate", "Recoverable Error Rate", "{:.4f}"),
            ("distance_weighted_rer", "Dist-Weighted RER", "{:.4f}"),
            ("locality_ratio", "Locality Ratio", "{:.4f}"),
            ("smoothness_gap", "Smoothness Gap", "{:+.4f}"),
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
    def _preb_labels(probs: NDArray) -> NDArray:
        P = np.asarray(probs, dtype=np.float64)
        if P.ndim == 1: 
            return (P.ravel() >= 0.5).astype(int)
        return np.argmax(P, axis=1)

    @staticmethod 
    def _haversine(p1: NDArray, p2: NDArray) -> NDArray:
        lat1 = np.radians(p1[:, 0])
        lon1 = np.radians(p1[:, 1])
        lat2 = np.radians(p2[:, 0])
        lon2 = np.radians(p2[:, 1])
        dlat = lat2 - lat1 
        dlon = lon2 - lon1 
        a = np.sin(0.5 * dlat)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * dlon)**2 
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return 6371.0 * c 

    @staticmethod
    def _edge_distances(adj: sp.csr_matrix, coords: NDArray) -> NDArray: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0: 
            return np.array([], dtype=np.float64)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2: 
            raise ValueError("coords must be (n,2)")
        src, dst = A.nonzero() 
        return MetricAnalyzer._haversine(coords[src], coords[dst])

    @staticmethod 
    def _edge_mean_sq(adj: sp.csr_matrix, X: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0: 
            return 0.0 
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: 
            X = X.reshape(-1, 1)
        src, dst = A.nonzero() 
        diff = X[src] - X[dst]
        return float((diff**2).sum(axis=1).mean())

    @staticmethod 
    def recoverable_error_rate(adj: sp.csr_matrix, y_true: NDArray, probs: NDArray) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0:
            return 0.0 
        y_true = np.asarray(y_true).reshape(-1)
        if y_true.shape[0] != A.shape[0]: 
            raise ValueError("y_true size does not match adjacency")

        preds = MetricAnalyzer._preb_labels(probs)
        err_mask = preds != y_true 
        if err_mask.sum() == 0: 
            return 0.0 

        err_idx = np.where(err_mask)[0] 
        hits = 0 
        for i in err_idx: 
            start, end = A.indptr[i], A.indptr[i + 1]
            neigh = A.indices[start:end]
            if neigh.size and np.any(y_true[neigh] == y_true[i]):
                hits += 1 

        return float(hits / err_idx.size)

    @staticmethod 
    def distance_weighted_recoverable_error_rate(
        adj: sp.csr_matrix,
        y_true: NDArray,
        probs: NDArray,
        coords: NDArray,
        sigma: float | None = None 
    ) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0:
            return 0.0 
        y_true  = np.asarray(y_true).reshape(-1)
        preds   = MetricAnalyzer._preb_labels(probs)
        err_idx = np.where(preds != y_true)[0]
        if err_idx.size == 0: 
            return 0.0 

        dists_all = MetricAnalyzer._edge_distances(adj, coords)
        if dists_all.size == 0: 
            return 0.0 
        if sigma is None: 
            sigma = float(np.median(dists_all))
        if sigma <= 0: 
            sigma = 1.0 

        scores = []
        coords = np.asarray(coords, dtype=np.float64)
        for i in err_idx: 
            start, end = A.indptr[i], A.indptr[i + 1]
            neigh = A.indices[start:end]
            if neigh.size == 0: 
                scores.append(0.0)
                continue 
            same = neigh[y_true[neigh] == y_true[i]]
            if same.size == 0: 
                scores.append(0.0)
                continue 
            d = MetricAnalyzer._haversine(
                np.repeat(coords[i][None, :], same.size, axis=0),
                coords[same]
            )
            scores.append(float(np.exp(-d / sigma).sum()))
        return float(np.mean(scores))

    @staticmethod 
    def locality_ratio(
        adj: sp.csr_matrix,
        coords: NDArray, 
        dist_threshold: float | None = 75 
    ) -> float: 
        dists = MetricAnalyzer._edge_distances(adj, coords)
        if dists.size == 0: 
            return 0.0 
        if dist_threshold is None: 
            dist_threshold = float(np.median(dists))
        if dist_threshold <= 0: 
            return 0.0 
        return float((dists <= dist_threshold).mean())

    @staticmethod 
    def smoothness_gap(adj: sp.csr_matrix, y_true: NDArray, probs: NDArray) -> float: 
        y_true = np.asarray(y_true).reshape(-1)
        P = np.asarray(probs, dtype=np.float64)

        if P.ndim == 1: 
            pred_smooth = MetricAnalyzer._edge_mean_sq(adj, P.ravel())
            true_smooth = MetricAnalyzer._edge_mean_sq(adj, y_true.astype(np.float64))
        else: 
            n_classes = P.shape[1]
            Y = np.zeros((y_true.size, n_classes), dtype=np.float64)
            Y[np.arange(y_true.size), y_true] = 1.0 
            pred_smooth = MetricAnalyzer._edge_mean_sq(adj, P)
            true_smooth = MetricAnalyzer._edge_mean_sq(adj, Y)

        return float(pred_smooth - true_smooth)

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
    def corrective_edge_ratio(
        adj: sp.csr_matrix,
        y_true: NDArray,
        probs: NDArray
    ) -> float: 
        A = MetricAnalyzer._binary_adj(adj)
        if A.nnz == 0: 
            return 0.0 

        y_true = np.asarray(y_true).reshape(-1)
        if y_true.shape[0] != A.shape[0]:
            raise ValueError("y_true size does not match adjacency")

        P = np.asarray(probs, dtype=np.float64)
        if P.ndim == 1: 
            preds = (P.ravel() >= 0.5).astype(int)
        else: 
            preds = np.argmax(P, axis=1)

        src, dst = A.nonzero()
        pred_diff = preds[src] != preds[dst]
        if pred_diff.sum() == 0: 
            return 0.0 

        true_same = y_true[src] == y_true[dst]
        return float((pred_diff & true_same).sum() / pred_diff.sum())

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
