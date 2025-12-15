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
from sklearn.metrics import mean_squared_error, r2_score 

from helpers import project_path, split_and_scale 
from geospatial import GeospatialModel


class ClimateGNN(nn.Module): 

    def __init__(self, climate_filepath: str, layers: list[int], method="knn", parameter=5.0, decade=2020, verbose=False): 
        '''
        Initializes GNN using passed layer sizes, metricType, and metricParameter 

        layers includes the input_dim 
        '''
        super().__init__()

        self.verbose = verbose 
        self.climate_filepath = climate_filepath

        filepath = project_path("data", "climate_counties.tsv")
        self.graph      = GeospatialModel(filepath=filepath, method=method, parameter=parameter)
        self.decade     = decade 
        self.layers     = layers 
        self.num_layers = len(layers)
        self.input_dim  = layers[0]  
        self.gnn_model  = self.build_gnn_model_() 
        self.features, self.labels, fips_codes = self.load_climate_data(decade)

        geoid_to_idx = self.graph.get_county_mapping() 
        for i in (0, len(fips_codes)//2, len(fips_codes) - 1):
            if geoid_to_idx[fips_codes[i]] != i:
                raise ValueError("row/node order mismatch")
        
        X, y = self.features.numpy(), self.labels.numpy()
        (_, _), (_, _), (train_idx, test_idx), (X_scaler, y_scaler) = split_and_scale(X, y, 0.40)
        
        coords = self.graph.get_coordinate_tensor().numpy() 
        coord_mean = coords[train_idx].mean(axis=0, keepdims=True)
        coord_std  = coords[train_idx].std(axis=0, keepdims=True) 
        coords_all = (coords - coord_mean) / coord_std 

        X_all, y_all = X_scaler.transform(X), y_scaler.transform(y.reshape(-1, 1)).reshape(-1) 
        X_node = np.hstack([X_all, coords_all])
        full_data    = self.graph.to_pytorch_geometric(torch.tensor(X_node, dtype=torch.float32))
        full_data.y  = torch.tensor(y_all, dtype=torch.float32) 

        self.train_data = self.graph.induced_subgraph(full_data, train_idx)
        self.test_data  = self.graph.induced_subgraph(full_data, test_idx)


    def build_gnn_model_(self):
        convs = []
        in_dim = self.input_dim 
        for h in self.layers[1:]:
            convs.append(GCNConv(in_dim, h))
            in_dim = h 

        convs.append(GCNConv(in_dim, 1))
        return nn.ModuleList(convs) 

    def forward(self, data): 
        x, edge_index = data.x, data.edge_index 
        
        edge_weight = None 
        if hasattr(data, "edge_attr") and data.edge_attr is not None: 
            d_km = data.edge_attr.view(-1)
            edge_weight = torch.exp(-(d_km / 200.0) ** 2)

        for conv in self.gnn_model[:-1]: 
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x) 
            x = F.dropout(x, p=0.1, training=self.training)  

        x = self.gnn_model[-1](x, edge_index, edge_weight=edge_weight) 
        return x.squeeze(-1) 

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

        fips_codes = decade_data["fips_codes"][0, 0].reshape(-1)
        fips_codes = [str(f).strip() for f in fips_codes]


        return (torch.tensor(features, dtype=torch.float32), 
                torch.tensor(labels, dtype=torch.float32), fips_codes) 

    def train_model(self, epochs=1000, lr=0.001): 
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        criterion = nn.MSELoss() 
        
        self.train() 
        for epoch in range(epochs): 
            optimizer.zero_grad()
            pred = self.forward(self.train_data) 
            loss = criterion(pred, self.train_data.y)

            loss.backward() 
            optimizer.step() 

            if self.verbose and epoch % 25 == 0: 
                print(f"epoch {epoch}, loss: {loss.item():.4f}")

        self.eval() 
        with torch.no_grad(): 
            test_pred = self.forward(self.test_data).detach().cpu().numpy() 
            test_true = self.test_data.y.detach().cpu().numpy() 

        return {
            "r2": r2_score(test_true, test_pred), 
            "rmse": np.sqrt(mean_squared_error(test_true, test_pred)), 
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
    parser.add_argument("--layers", nargs='+', type=int, default=[64, 32])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--v", action="store_true")

    args = parser.parse_args()
    print(args.layers)
    climate_filepath = project_path("data", "climate_population.mat"); 
    input_dim = ClimateGNN.get_input_dim(climate_filepath, args.decade) + 2
    
    layers = [input_dim] + args.layers

    model = ClimateGNN(
        climate_filepath=climate_filepath, 
        layers=layers,
        method=args.method, 
        parameter=args.parameter, 
        decade=args.decade,
        verbose=args.v
    )

    results = model.train_model(epochs=args.epochs, lr=args.lr)

    print("\n> Climate Graph Neural Network:")
    print(f"    > r2: {results['r2']}")
    print(f"    > rmse: {results['rmse']}")


if __name__ == "__main__": 
    main() 
