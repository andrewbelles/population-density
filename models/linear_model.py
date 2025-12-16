#!/usr/bin/env python 
# 
# linear_climpop.py  Andrew Belles  Dec 10th, 2025 
# 
# Prediction of population density from climate data 
# using a linear model as a baseline. 
# 
# Provides a Linear interface generic to specific features vs labels 
# 

import numpy as np 
import support.helpers as h 

from numpy.typing import NDArray 

class LinearModel(h.ModelInterface):

    def __init__(self, **linear_params):
        self.linear_params = linear_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_val = features 
        y_train, _     = labels

        _ = coords 
        _ = kwargs 
        _ = self.linear_params 

        X_train_ = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_val_   = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

        w = np.linalg.pinv(X_train_) @ y_train 
        y_hat = X_val_ @ w 
        return np.asarray(y_hat, dtype=np.float64)
