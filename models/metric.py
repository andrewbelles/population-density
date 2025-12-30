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
