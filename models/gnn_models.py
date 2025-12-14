#!/usr/bin/env python3 
# 
# gnn_models.py  Andrew Belles  Dec 12th, 2025 
# 
# 
# 
# 

import torch, argparse
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from torch_geometric.nn import GCNConv 
from scipy.io import loadmat 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score 

from helpers import project_path 
from geospatial import GeospatialModel


class ClimateGNN(nn.Module): 

    def __init__(self, climate_filepath: str, layers: list[int], method="knn", parameter=5.0, decade=2020): 
        '''
        Initializes GNN using passed layer sizes, metricType, and metricParameter 

        layers includes the input_dim 
        '''
        super().__init__()

        self.climate_filepath = climate_filepath

        filepath = project_path("data", "climate_counties.tsv")
        self.graph      = GeospatialModel(filepath=filepath, method=method, parameter=parameter)
        self.decade     = decade 
        self.layers     = layers 
        self.num_layers = len(layers)
        self.input_dim  = layers[0]  
        self.gnn_model  = self.build_gnn_model_() 
        self.features, self.labels = self.load_climate_data(decade)

    def build_gnn_model_(self):
        layers = []
        layers.append(GCNConv(self.input_dim, self.layers[1]))

        for i in range(1, self.num_layers - 2): 
            layers.append(GCNConv(self.layers[i], self.layers[i+1]))

        layers.append(GCNConv(self.layers[-2], 1)) # Regression output 
        return nn.ModuleList(layers) 

    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 

        for conv in self.gnn_model[:-1]: 
            x = conv(x, edge_index)
            x = F.relu(x) 
            x = F.dropout(x, training=self.training)  

        x = self.gnn_model[-1](x, edge_index) 
        return x.squeeze() 

    def load_climate_data(self, decade): 

        data    = loadmat(self.climate_filepath)
        decades = data["decades"]

        decade_key  = f"decade_{decade}"
        decade_keys = [name for name in decades.dtype.names if name.startswith("decade_")]

        if decade_key not in decade_keys: 
            raise ValueError(f"decade {decade} not found. Available: {decade_keys}")

        decade_data = decades[decade_key][0, 0]
        features    = decade_data["features"][0, 0] 
        labels      = decade_data["labels"][0, 0]

        return (torch.tensor(features, dtype=torch.float32), 
                torch.tensor(labels.flatten(), dtype=torch.float32)) 

    def create_graph_data(self): 
        return self.graph.to_pytorch_geometric(self.features)

    def train_model(self, epochs=200, lr=0.01, test_size=0.25, verbose=False): 
        self.graph_data   = self.create_graph_data()
        self.graph_data.y = self.labels 
        
        indices = np.arange(len(self.features))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=1)

        train_mask = torch.zeros(len(self.features), dtype=torch.bool)
        test_mask  = torch.zeros(len(self.features), dtype=torch.bool)
        train_mask[train_idx] = True 
        test_mask[test_idx]   = True 

        optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        criterion = nn.MSELoss() 
        
        self.train() 
        for epoch in range(epochs): 
            optimizer.zero_grad()
            out  = self.forward(self.graph_data) 
            loss = criterion(out[train_mask], self.labels[train_mask])

            loss.backward() 
            optimizer.step() 

            if verbose and epoch % 50 == 0: 
                print(f"epoch {epoch}, loss: {loss.item():.4f}")

        self.eval() 
        with torch.no_grad(): 
            predictions = self.forward(self.graph_data)
            test_pred   = predictions[test_mask].numpy() 
            test_true   = self.labels[test_mask].numpy() 
            test_r2     = r2_score(test_true, test_pred) 
            test_rmse   = np.sqrt(mean_squared_error(test_true, test_pred))

        return {
            "r2": test_r2, 
            "rmse": test_rmse, 
            "pred": test_pred 
        }

    @staticmethod
    def get_input_dim(filepath: str, decade: int): 
        data = loadmat(filepath)
        decade_data = data["decades"][f"decade_{decade}"][0, 0]
        return decade_data["features"][0, 0].shape[1]


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--decade", type=int, default=2020)
    parser.add_argument("--method", default="knn", choices=["knn", "bounded", "standard"])
    parser.add_argument("--parameter", type=float, default=5.0)
    parser.add_argument("--layers", nargs='+', default=[None, 64, 32])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()
    climate_filepath = project_path("data", "climate_population.mat"); 
    input_dim = ClimateGNN.get_input_dim(climate_filepath, args.decade)
    
    layers = [input_dim] + args.layers[1:]

    model = ClimateGNN(
        climate_filepath=climate_filepath, 
        layers=layers,
        method=args.method, 
        parameter=args.parameter, 
        decade=args.decade
    )

    results = model.train_model(epochs=args.epochs, lr=args.lr)

    print("\n> Climate Graph Neural Network:")
    print(f"    > r2: {results["r2"]}")
    print(f"    > rmse: {results["rmse"]}")


if __name__ == "__main__": 
    main() 
