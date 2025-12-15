#!/usr/bin/env python3 
# 
# random_forest_model.py  Andrew Belles  Dec 14th, 2025 
# 
# Exposes RandomForestRegressor to be used in CrossValidator 
# for dynamic modeling 
# 


import models.helpers as h 
import numpy as np 

from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor

class RandomForest(h.ModelInterface): 
    def __init__(self, **rf_params): 
        self.rf_params = rf_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_val = features 
        y_train, _     = labels 

        _ = coords 
        _ = kwargs 

        rf = RandomForestRegressor(n_jobs=-1, **self.rf_params)
        rf.fit(X_train, y_train) 
        y_hat = rf.predict(X_val)
        return np.asarray(y_hat, dtype=np.float64)
