#!/usr/bin/env python3 
# 
# xgboost_model.py  Andrew Belles  Dec 14th, 2025 
# 
# Exposes XGBoost model to CrossValidator for use 
# with dynamic choice of dataset 
# 

import helpers as h 
import numpy as np 

from numpy.typing import NDArray
from xgboost import XGBRegressor

class XGBoost(h.ModelInterface): 

    def __init__(self, gpu: bool = True, **xgb_params): 
        self.gpu = gpu 
        self.xgb_params = xgb_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]: 
        X_train, X_test = features 
        y_train, _      = labels 
        c_train, c_test = coords 

        val_size = kwargs.get("val_size", 0.3)
        seed     = kwargs.get("seed", 1) 

        inner_train_idx, inner_val_idx = h.split_indices(len(X_train), val_size, seed) 

        X_tr = np.hstack([c_train[inner_train_idx].astype(np.float32), X_train[inner_train_idx].astype(np.float32)])
        X_va = np.hstack([c_train[inner_val_idx].astype(np.float32), X_train[inner_val_idx].astype(np.float32)])
        X_te = np.hstack([c_test.astype(np.float32), X_test.astype(np.float32)])

        y_tr, y_va = y_train[inner_train_idx].astype(np.float32), y_train[inner_val_idx].astype(np.float32)

        xgb_params = {k: h.unwrap_scalar(v) for k, v in self.xgb_params.items()}

        xgb = h.make_xgb_model( 
            XGBRegressor,
            gpu=self.gpu,
            base_params={
                "n_jobs": -1, 
                **xgb_params 
            } 
        )

        xgb.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            verbose=False,
        )

        y_hat = xgb.predict(X_te) 
        return np.asarray(y_hat, dtype=np.float64)
