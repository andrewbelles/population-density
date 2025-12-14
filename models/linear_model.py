#!/usr/bin/env python 
# 
# linear_climpop.py  Andrew Belles  Dec 10th, 2025 
# 
# Prediction of population density from climate data 
# using a linear model as a baseline. 
# 
# Provides a Linear interface generic to specific features vs labels 
# 

from helpers import project_path

import torch, argparse
import numpy as np 
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

class LinearModel: 

    '''
    Simple interface for creating a linear model and solving via psuedo-inverse. 
    Assumes of course that pinv is computationally reasonable (m,n << 1e6)

    Coords if provided are attached to features since the model will be working with county data
    '''

    def __init__(self, features, labels, coords, gpu=True): 

        device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

        X_train, X_test = features 
        y_train, y_test = labels 
        c_train, c_test = coords 

        self.X_train = torch.cat(
            [
                torch.ones(X_train.shape[0], 1, device=device),
                torch.tensor(X_train, dtype=torch.float32).to(device), 
                torch.tensor(c_train, dtype=torch.float32).to(device)
            ],
            dim=1
        )

        self.X_test = torch.cat(
            [
                torch.ones(X_test.shape[0], 1, device=device),
                torch.tensor(X_test, dtype=torch.float32).to(device), 
                torch.tensor(c_test, dtype=torch.float32).to(device)
            ],
            dim=1
        )

        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        self.y_test  = y_test 

    def evaluate(self):
        weights = torch.linalg.pinv(self.X_train) @ self.y_train
        y_hat   = self.X_test @ weights 
        
        y_hat = y_hat.cpu().numpy() 

        return {
            "rmse": np.sqrt(mean_squared_error(self.y_test, y_hat)), 
            "r2": r2_score(self.y_test, y_hat),
            "pred": y_hat
        }


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--decade", default="2020")
    args = parser.parse_args()

    '''
    Example Usage using Climate/Population Density Dataset 
    '''

    data    = loadmat(project_path("data", "climate_population.mat"))
    decades = data["decades"]
    coords  = data["coords"]

    # Get decades 
    decade_key  = f"decade_{args.decade}"
    decade_keys = [name for name in decades.dtype.names if name.startswith("decade_")]
    if decade_key not in decade_keys: 
        raise ValueError(f"decade {args.decade} not found. Available: {decade_keys}")
    
    decade_data = decades[decade_key][0, 0]
    X, y  = decade_data["features"][0, 0], decade_data["labels"][0, 0]

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=1)

    # Get splits 
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()
    c_train, c_test = coords[train_idx], coords[test_idx]

    model   = LinearModel((X_train, X_test), (y_train, y_test), (c_train, c_test), gpu=True)
    results = model.evaluate()

    print("> Linear Model: ")
    print(f"    > rmse: {results["rmse"]:.4f}")
    print(f"    > r2: {results["r2"]:.4f}")

if __name__ == "__main__":
    main()
