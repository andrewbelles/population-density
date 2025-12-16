#!/usr/bin/env python3 
# 
# xgboost_model.py  Andrew Belles  Dec 14th, 2025 
# 
# Exposes XGBoost model to CrossValidator for use 
# with dynamic choice of dataset 
# 

import models.helpers as h 
import numpy as np 

from numpy.typing import NDArray
from xgboost import XGBRegressor

class XGBoost(h.ModelInterface): 

    def __init__(self, gpu: bool = True, ignore_coords=False, **xgb_params): 
        self.gpu = gpu
        self.ignore_coords = ignore_coords
        self.xgb_params = xgb_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_test = features
        y_train, _ = labels
        c_train, c_test = coords

        val_size = float(kwargs.get("val_size", 0.3))
        seed = int(kwargs.get("seed", 1))

        n = len(X_train)
        inner_train_idx, inner_val_idx = h.split_indices(n, val_size, seed)

        y_train = np.asarray(y_train, dtype=np.float64)

        if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
            y1 = y_train.reshape(-1)
            return self._fit_single_target(
                X_train=np.asarray(X_train, dtype=np.float64),
                X_test=np.asarray(X_test, dtype=np.float64),
                y_train_1d=y1,
                c_train=np.asarray(c_train, dtype=np.float64),
                c_test=np.asarray(c_test, dtype=np.float64),
                inner_train_idx=inner_train_idx,
                inner_val_idx=inner_val_idx,
            )

        if y_train.ndim != 2:
            raise ValueError(f"y_train must be 1D or 2D; got shape {y_train.shape}")

        k = int(y_train.shape[1])
        preds = np.zeros((len(X_test), k), dtype=np.float64)

        for j in range(k):
            preds[:, j] = self._fit_single_target(
                X_train=np.asarray(X_train, dtype=np.float64),
                X_test=np.asarray(X_test, dtype=np.float64),
                y_train_1d=y_train[:, j].reshape(-1),
                c_train=np.asarray(c_train, dtype=np.float64),
                c_test=np.asarray(c_test, dtype=np.float64),
                inner_train_idx=inner_train_idx,
                inner_val_idx=inner_val_idx,
            )

        return preds

    def _fit_single_target(
        self,
        *,
        X_train: NDArray[np.float64],
        X_test: NDArray[np.float64],
        y_train_1d: NDArray[np.float64],
        c_train: NDArray[np.float64],
        c_test: NDArray[np.float64],
        inner_train_idx: NDArray[np.int64],
        inner_val_idx: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        X_tr, X_va, X_te = self._build_inputs(
            X_train=X_train,
            X_test=X_test,
            c_train=c_train,
            c_test=c_test,
            inner_train_idx=inner_train_idx,
            inner_val_idx=inner_val_idx,
        )

        y_tr = np.asarray(y_train_1d[inner_train_idx], dtype=np.float32).reshape(-1)
        y_va = np.asarray(y_train_1d[inner_val_idx], dtype=np.float32).reshape(-1)

        xgb_params = {k: h.unwrap_scalar(v) for k, v in self.xgb_params.items()}
        xgb = h.make_xgb_model(
            XGBRegressor,
            gpu=self.gpu,
            base_params={"n_jobs": -1, **xgb_params},
        )

        xgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        y_hat = xgb.predict(X_te)
        return np.asarray(y_hat, dtype=np.float64).reshape(-1)

    def _build_inputs(
          self,
          *,
          X_train: NDArray[np.float64],
          X_test: NDArray[np.float64],
          c_train: NDArray[np.float64],
          c_test: NDArray[np.float64],
          inner_train_idx: NDArray[np.int64],
          inner_val_idx: NDArray[np.int64],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.ignore_coords:
            X_tr = np.asarray(X_train[inner_train_idx], dtype=np.float32)
            X_va = np.asarray(X_train[inner_val_idx], dtype=np.float32)
            X_te = np.asarray(X_test, dtype=np.float32)
        else:
            X_tr = np.hstack([
                np.asarray(c_train[inner_train_idx], dtype=np.float32),
                np.asarray(X_train[inner_train_idx], dtype=np.float32),
            ])
            X_va = np.hstack([
                np.asarray(c_train[inner_val_idx], dtype=np.float32),
                np.asarray(X_train[inner_val_idx], dtype=np.float32),
            ])
            X_te = np.hstack([
                np.asarray(c_test, dtype=np.float32),
                np.asarray(X_test, dtype=np.float32),
            ])
        return X_tr, X_va, X_te
