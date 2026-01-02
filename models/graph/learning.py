#!/usr/bin/env python3 
# 
# metric.py  Andrew Belles  Dec 28th, 2025 
# 
# Defines the general interface for metric based methods 
# as well as specific implementations that are operable 
# with the projects pipeline 
# 

import numpy as np 
import scipy.sparse as sp 
from abc import ABC, abstractmethod 

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassifierMixin,
)

from sklearn.neighbors import (
    NeighborhoodComponentsAnalysis,
    kneighbors_graph
)

from sklearn.semi_supervised import LabelSpreading 

from sklearn.preprocessing import StandardScaler

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted 

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from models.graph_utils import normalize_adjacency

class MetricLearner(BaseEstimator, TransformerMixin, ABC): 

    @abstractmethod 
    def get_graph(self, X) -> sp.csr_matrix: 
        '''
        Returns sparse adjacency matrix for input X constructed in 
        the learned metric space 
        '''
        pass 


class IDMLGraphLearner(MetricLearner, ClassifierMixin): 

    def __init__(
        self, 
        *,
        n_neighbors: int = 10, 
        n_components: int = 32, 
        max_iter: int = 5, 
        confidence_threshold: float = 0.9, 
        label_spreading_alpha: float = 0.2, 
        random_state: int = 0,
        n_jobs: int = -1 
    ): 

        self.n_neighbors           = n_neighbors
        self.n_components          = n_components 
        self.max_iter              = max_iter 
        self.confidence_threshold  = confidence_threshold 
        self.label_spreading_alpha = label_spreading_alpha 
        self.random_state          = random_state 
        self.n_jobs                = n_jobs

        self.metric_learner_    = None 
        self.scaler_            = StandardScaler() 
        self.final_label_model_ = None 
        self.classes_           = None

    def fit(self, X, y, train_mask=None): 

        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = np.unique(y)

        if train_mask is None: 
            train_mask = np.ones(X.shape[0], dtype=bool)
        else: 
            train_mask = np.asarray(train_mask, dtype=bool)
            if train_mask.shape[0] != X.shape[0]: 
                raise ValueError("train_mask size does not match X")

        X_scaled = self.scaler_.fit_transform(X)
        current_mask = train_mask.copy() 
        current_y    = y.copy() 

        for i in range(self.max_iter): 
            X_subset = X_scaled[current_mask]
            y_subset = current_y[current_mask]

            if len(np.unique(y_subset)) < 2: 
                break 

            nca = NeighborhoodComponentsAnalysis(
                n_components=self.n_components,
                random_state=self.random_state,
                init='pca' if i == 0 else 'identity'
            )
            nca.fit(X_subset, y_subset)
            self.metric_learner_ = nca 

            if np.all(current_mask) or i == self.max_iter - 1: 
                break 

            X_trans = nca.transform(X_scaled)

            y_prop = current_y.copy() 
            y_prop[~current_mask] = -1 

            ls = LabelSpreading(
                kernel='knn', 
                n_neighbors=self.n_neighbors,
                alpha=self.label_spreading_alpha,
                max_iter=30,
                n_jobs=self.n_jobs 
            )

            ls.fit(X_trans, y_prop)

            probs = ls.predict_proba(X_trans)
            max_probs = np.max(probs, axis=1)
            preds = ls.predict(X_trans)

            new_indices = (max_probs > self.confidence_threshold) & (~current_mask)

            # Test if converged 
            if not np.any(new_indices): 
                break 

            current_mask[new_indices] = True 
            current_y[new_indices] = preds[new_indices]

        if self.metric_learner_ is None: 
            raise RuntimeError("IDML failed to initialize metric learner")
    
        X_final = self.metric_learner_.transform(X_scaled)
        self.final_label_model_ = LabelSpreading(
            kernel='knn',
            n_neighbors=self.n_neighbors,
            alpha=self.label_spreading_alpha,
            max_iter=30,
            n_jobs=self.n_jobs
        )

        y_final_fit = current_y.copy() 
        if not np.all(current_mask): 
            y_final_fit[~current_mask] = -1 

        self.final_label_model_.fit(X_final, y_final_fit)
        return self 

    def transform(self, X): 
        check_is_fitted(self, ['metric_learner_', 'scaler_'])
        X = check_array(X, accept_sparse=False)
        return self.metric_learner_.transform(self.scaler_.transform(X))

    def get_graph(self, X) -> sp.csr_matrix:
        X_trans = self.transform(X)
        return kneighbors_graph(
            X_trans, 
            self.n_neighbors,
            mode='connectivity', 
            include_self=False, 
            n_jobs=self.n_jobs
        )

    def predict(self, X): 
        check_is_fitted(self, ['final_label_model_'])
        X_trans = self.transform(X)
        return self.final_label_model_.predict(X_trans)

    def predict_proba(self, X): 
        check_is_fitted(self, ['final_label_model_'])
        X_trans = self.transform(X)
        return self.final_label_model_.predict_proba(X_trans)


class GradientBoostingMetricLearner(MetricLearner, ClassifierMixin):
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        n_negatives=5, 
        hard_mining_ratio=0.5,
        n_pos_per_anchor=1,
        anchors_per_class=200,
        n_neighbors=16,
        candidate_k=50,
        random_state=0,
        n_jobs=-1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_negatives = n_negatives
        self.hard_mining_ratio = hard_mining_ratio
        self.n_pos_per_anchor = n_pos_per_anchor
        self.anchors_per_class = anchors_per_class
        self.candidate_k = candidate_k
        self.random_state = random_state
        self.n_neighbors = n_neighbors 
        self.n_jobs = n_jobs
        self.model_ = None
        self.scaler_ = StandardScaler()

    def fit(self, X, y, train_mask=None):
        X, y = check_X_y(X, y, accept_sparse=False)

        if train_mask is not None: 
            train_mask = np.asarray(train_mask, dtype=bool)
            if train_mask.shape[0] != X.shape[0]: 
                raise ValueError("train_mask size does not match X")
        else: 
            train_mask = np.ones(X.shape[0], dtype=bool)

        self.scaler_.fit(X[train_mask])
        X_scaled = self.scaler_.transform(X)

        X_pairs, y_pairs = self._generate_pairwise_dataset(
            X_scaled[train_mask], 
            y[train_mask]
        )
        
        self.model_ = XGBClassifier(
            objective='binary:logistic',
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )
        self.model_.fit(X_pairs, y_pairs)
        return self

    def get_graph(self, X):
        check_is_fitted(self, ["model_", "scaler_"])
        X_scaled = self.scaler_.transform(X)

        n = X_scaled.shape[0]
        cand_k = min(self.candidate_k, n - 1)
        cand_graph = kneighbors_graph(
            X_scaled, cand_k, mode="distance", include_self=False, n_jobs=self.n_jobs
        )

        rows, cols, data = [], [], []
        for i in range(n): 
            start, end = cand_graph.indptr[i], cand_graph.indptr[i + 1]
            neigh = cand_graph.indices[start:end]
            if neigh.size == 0: 
                continue 

            feats  = self._make_pair_features(X_scaled[i], X_scaled[neigh])
            scores = self.model_.predict_proba(feats)[:, 1]

            if scores.size > self.n_neighbors:
                idx = np.argpartition(scores, -self.n_neighbors)[-self.n_neighbors:]
            else: 
                idx = np.arange(scores.size)

            rows.extend([i] * idx.size)
            cols.extend(neigh[idx])
            data.extend(scores[idx])

        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def _generate_pairwise_dataset(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        n = X.shape[0]
        cand_k = min(self.candidate_k, n - 1)
        cand_graph = kneighbors_graph(
            X, cand_k, mode="distance", include_self=False, n_jobs=self.n_jobs
        )

        X_pairs = []
        y_pairs = []
        classes = np.unique(y)

        for c in classes: 
            idx_c = np.where(y == c)[0]
            if idx_c.size < 2: 
                continue 
            other_idx = np.where(y != c)[0]
            n_anchors = min(self.anchors_per_class, idx_c.size)
            anchors   = rng.choice(idx_c, size=n_anchors, replace=False)

            for i in anchors: 
                pos_pool = idx_c[idx_c != i]
                if pos_pool.size == 0: 
                    continue 
                pos_choice = rng.choice(
                    pos_pool,
                    size=min(self.n_pos_per_anchor, pos_pool.size),
                    replace=pos_pool.size < self.n_pos_per_anchor 
                )
                for j in np.atleast_1d(pos_choice): 
                    X_pairs.append(self._make_pair_features(X[i], X[j])) 
                    y_pairs.append(1)

                start, end = cand_graph.indptr[i], cand_graph.indptr[i + 1]
                neigh = cand_graph.indices[start:end]
                hard_pool = neigh[y[neigh] != y[i]]
                n_hard = int(round(self.n_negatives * self.hard_mining_ratio))
                n_rand = self.n_negatives - n_hard 

                hard_choice = (
                    rng.choice(hard_pool, size=min(n_hard, hard_pool.size), 
                               replace=hard_pool.size < n_hard)
                    if hard_pool.size > 0 and n_hard > 0 else np.array([], dtype=int)
                )
                rand_choice = (
                    rng.choice(other_idx, size=min(n_rand, other_idx.size), 
                               replace=other_idx.size < n_rand)
                    if other_idx.size > 0 and n_rand > 0 else np.array([], dtype=int)
                )

                for j in np.concatenate([hard_choice, rand_choice]): 
                    X_pairs.append(self._make_pair_features(X[i], X[j]))
                    y_pairs.append(0)

        if not X_pairs: 
            raise ValueError("no pairs generated")

        X_pairs = np.vstack(X_pairs)
        y_pairs = np.asarray(y_pairs, dtype=np.int64)
        return X_pairs, y_pairs 
    
    def _make_pair_features(self, A, B):
        A = np.asarray(A)
        B = np.asarray(B)

        if A.ndim == 1: 
            A = A.reshape(1, -1)
        if B.ndim == 1: 
            B = B.reshape(1, -1)

        if A.shape[0] == 1 and B.shape[0] > 1: 
            A = np.repeat(A, B.shape[0], axis=0)
        if B.shape[0] == 1 and A.shape[0] > 1: 
            B = np.repeat(B, A.shape[0], axis=0)

        diff = A - B 
        return np.hstack([np.abs(diff), diff * diff, A, B]) 

    
class QueenGateLearner: 
    def __init__(
        self,
        hidden_layer_size=(64,128,64),
        alpha=1e-4,
        max_iter=1000, 
    ): 
        self.hidden_layer_sizes = hidden_layer_size 
        self.alpha              = alpha 
        self.max_iter           = max_iter 
        self.model_             = None 
        self.adj_               = None 
        self.fips_              = None 
        self.scaler_            = StandardScaler() 

    def _edge_pairs(self, adj): 
        A = (adj > 0).astype(int)
        A = sp.triu(A, k=1)
        return A.nonzero() 

    def _edge_features(self, X, src, dst): 
        Xi = X[src]
        Xj = X[dst]
        return np.hstack([np.abs(Xi - Xj), (Xi - Xj)**2])

    def fit(self, X, y, *, adj, train_mask=None): 
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)
        src, dst = self._edge_pairs(adj)


        if train_mask is not None: 
            mask = np.asarray(train_mask, dtype=bool)
            self.scaler_.fit(X[mask]) 
            X = self.scaler_.transform(X)
            keep = mask[src] & mask[dst]
            src, dst = src[keep], dst[keep]

        X_edges = self._edge_features(X, src, dst)
        y_edges = (y[src] == y[dst]).astype(int)

        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, 
            alpha=self.alpha,
            max_iter=self.max_iter 
        )
        self.model_.fit(X_edges, y_edges)
        self.adj_ = adj 
        return self 

    def get_graph(self, X): 
        if self.adj_ is None or self.model_ is None: 
            raise ValueError("model not fitted")
        X_scaled = self.scaler_.transform(X)
        src, dst = self._edge_pairs(self.adj_)
        X_edges = self._edge_features(np.asarray(X_scaled, dtype=np.float64), src, dst)
        gate = self.model_.predict_proba(X_edges)[:, 1]

        W = self.adj_.copy().astype(np.float64)
        W.data = np.zeros_like(W.data)

        W[src, dst] = gate 
        W[dst, src] = gate 
        return normalize_adjacency(W, binarize=False) 
