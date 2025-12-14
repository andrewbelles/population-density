#!/usr/bin/env python3 S
# 
# random_forest.py  Andrew Belles  Dec 11th, 2025
# 
# Provides an interface for training a random forest model 
# on .mat file. Also includes XGBoost and Geospatial XGBoost 
# models
# 

from helpers import project_path, split_and_scale

import argparse 

from scipy.io import loadmat  
import numpy as np 

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score  

from xgboost import XGBRegressor 

class RandomForest: 
    '''
    Wrapper Interface over sklearn.ensemble.RandomForestRegressor

    Caller Provides: 
        features and labels as tuples already pre-split for train/test 
        optional flags for verbose output and out-of-bag bootstrapping 
    '''

    def __init__(self, features, labels, oob_score=True, verbose=False): 
        # Get train, test splits 
        self.X_train, self.X_test = features 

        self.y_train, self.y_test = labels

        # Instantiate Random Forest model and train
        self.rf = RandomForestRegressor(
            n_estimators=500, 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            n_jobs=-1, 
            random_state=0, 
            verbose=verbose, 
            oob_score=oob_score, 
            bootstrap=(True if oob_score else False)
        )

        self.rf.fit(self.X_train, self.y_train)
    
    def evaluate(self): 
        y_hat = self.rf.predict(self.X_test)
        return {
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_hat)), 
            "r2": r2_score(self.y_test, y_hat),
            "pred": y_hat
        } 


class XGBoost: 

    def __init__(self, features, labels, coords, gpu=True, verbose=True): 
        # Convert to numpy types, unravel labels  
        X_train, X_test = features 
        c_train, c_test = coords 

        self.X_train = np.hstack([c_train.astype(np.float32), X_train.astype(np.float32)])
        self.X_test  = np.hstack([c_test.astype(np.float32), X_test.astype(np.float32)])

        y_train, y_test = labels 
        self.y_train = y_train.astype(np.float32)
        self.y_test  = y_test.astype(np.float32)

        # Instantiate XGBoost Regressor on GPU  
        self.xgb = XGBRegressor(
            n_estimators=500, 
            max_depth=None, 
            learning_rate=0.01, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            objective="reg:squarederror", 
            tree_method="hist", 
            reg_lambda=1.0, 
            reg_alpha=0.1, 
            random_state=0, 
            n_jobs=-1, 
            early_stopping_rounds=150, 
            device="cuda" if gpu else "cpu",
        )

        self.xgb.fit(self.X_train, self.y_train, 
                     eval_set=[(self.X_test, self.y_test)],
                     verbose=verbose)

    def evaluate(self): 
        y_hat = self.xgb.predict(self.X_test)
        return {
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_hat)), 
            "r2": r2_score(self.y_test, y_hat), 
            "pred": y_hat
        }


def main(): 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--rf", action="store_true", help="train random forest model")
    parser.add_argument("--xgb", action="store_true", help="train xgboost model")
    parser.add_argument("--decade", default="2020", help="specify which decade to train on")
    args = parser.parse_args()

    '''
    Example Usage of each Model: 
        Climate regression against Population Density 
    '''
    
    data    = loadmat(project_path("data", "climate_population.mat"))
    decades = data["decades"]
    coords  = data["coords"]

    # Get decades 
    decade_key  = f"decade_{args.decade}"
    decade_keys = [name for name in decades.dtype.names if name.startswith("decade_")]
    if decade_key not in decade_keys: 
        raise ValueError(f"decade {args.decade} not found. Available: {decade_keys}")
    
    data  = decades[decade_key][0, 0]
    X, y  = data["features"][0, 0], data["labels"][0, 0]

    # Get splits 
    (X_train, X_test), (y_train, y_test), (train_idx, test_idx), _ = split_and_scale(X, y, 0.20)
    c_train, c_test = coords[train_idx], coords[test_idx]

    y_train, y_test = y_train.ravel(), y_test.ravel()  


    if args.rf is True:  
        forest = RandomForest((X_train, X_test), (y_train, y_test), verbose=False)
        result = forest.evaluate()
        
        print("> RandomForest Regression:")
        print(f"    > rmse: {result["rmse"]:.4f}")
        print(f"    > r2: {result["r2"]:.4}")

    if args.xgb is True: 
        boost  = XGBoost((X_train, X_test), (y_train, y_test), (c_train, c_test), 
                         gpu=False, verbose=False)
        result = boost.evaluate()  

        print("> XGBoost Regression:")
        print(f"    > rmse: {result["rmse"]:.4f}")
        print(f"    > r2: {result["r2"]:.4}")


if __name__ == "__main__": 
    main()
