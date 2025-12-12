#!/usr/bin/env python 
# 
# linear_climpop.py  Andrew Belles  Dec 10th, 2025 
# 
# Prediction of population density from climate data 
# using a linear model as a baseline. 
# 
# Provides a Linear interface generic to specific features vs labels 
# 

import torch
import numpy as np 
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

class LinearModel: 

    def __init__(self, X_train, y_train, X_test, y_test): 
        if X_train.shape[1] != X_test.shape[1] or y_train.shape[1] != y_test.shape[1]:
            raise ValueError("mismatched shapes for training and test data")

        self.rows = X_train.shape[0]
        self.X_train = X_train 
        self.y_train = y_train
        self.X_test  = X_test 
        self.y_test  = y_test 
        

    def regression(self, gpu=True): 
        '''
        Computes Linear Regression from training data using pseudo-inverse. 

        Caller Provides: 
            GPU flag 

        We return: 
            Predicted labels from test features 
            Weight vector from regression 
        '''
        X_train, y_train, X_test = self.X_train, self.y_train, self.X_test 
        device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

        X_train_tensor = torch.cat([torch.ones(self.X_train.shape[0], 1, device=device), 
                                    torch.tensor(X_train, dtype=torch.float32).to(device)], 
                                    dim=1) 
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor  = torch.cat([torch.ones(self.X_test.shape[0], 1, device=device),
                                    torch.tensor(X_test, dtype=torch.float32).to(device)],
                                    dim=1)

        weights = torch.linalg.pinv(X_train_tensor) @ y_train_tensor
        y_hat   = X_test_tensor @ weights 

        return y_hat.cpu().numpy(), weights.cpu().numpy() 


    def evaluate(self, targets, gpu=True):
        y_hat, _  = self.regression(gpu)
        y_true = self.y_test 

        for i, (target) in enumerate(targets):
            mse  = mean_squared_error(y_true[:, i], y_hat[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_hat[:, i])

            print(f"> target: {target}")
            print(f"    > rmse: {rmse:.4f}")
            print(f"    > r2: {r2:.4f}")

        overall_mse  = mean_squared_error(y_true, y_hat)
        overall_rmse = np.sqrt(overall_mse) 
        overall_r2   = r2_score(y_true, y_hat)

        print("> overall: ")
        print(f"    > rmse: {overall_rmse:.4f}")
        print(f"    > r2: {overall_r2:.4f}")


def main():
    '''
    Example Usage using Climate/Population Density Dataset 
    '''
    data = loadmat("../data/climpop.mat") 
    X, y = data["features"], data["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    model = LinearModel(X_train, y_train, X_test, y_test)
    model.evaluate(targets=["density_1960", "density_1980", "density_2020"])

if __name__ == "__main__":
    main()
