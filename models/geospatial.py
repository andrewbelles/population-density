#!/usr/bin/env python 
# 
# geospatial.py  Andrew Belles  Dec 12th, 2025 
# 
# Class that enables GeospatialGraph backend to interact with 
# ML libraries and attach features to counties by inheriting 
# the c++ implementation. 
# 

from torch_geometric.utils import subgraph, to_undirected
import geospatial_graph_cpp as cpp 
import torch 

from helpers import project_path
from torch_geometric.data import Data 

class GeospatialModel(cpp.GeospatialGraph): 

    def __init__(self, filepath: str | None, method: str="knn", parameter: float= 5.0): 
        if filepath is None: 
            filepath = project_path("data", "geography", "2020_Gaz_counties_national.txt")
            
        self.method    = method 
        self.parameter = parameter  

        method_map = {
            "knn": cpp.MetricType.KNN, 
            "bounded": cpp.MetricType.BOUNDED, 
            "standard": cpp.MetricType.STANDARD 
        }

        super().__init__(filepath, method_map[self.method], self.parameter)
        
    def to_pytorch_tensors(self) -> tuple[torch.Tensor, torch.Tensor]: 
        edge_pairs, distances = self.get_edge_indices_and_distances() 
        edges     = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(1) 

        return edges, edge_attr

    def to_pytorch_geometric(self, node_features: torch.Tensor | None = None):
        edge_index, edge_attr = self.to_pytorch_tensors() 
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr, reduce="mean")
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        if node_features is not None: 
            if node_features.shape[0] != len(self.counties()): 
                raise ValueError(f"Node features shape {node_features.shape[0]} != counties "
                                 f"shape {len(self.counties())}")
            data.x = node_features 
        else: 
            data.x = self.get_coordinate_tensor()

        return data 

    def get_county_mapping(self) -> dict[str, int]:
        return dict(self.get_geoid_to_index())

    def get_coordinate_tensor(self) -> torch.Tensor: 
        coords = self.get_all_coordinates() 
        return torch.tensor(coords, dtype=torch.float32)

    @staticmethod 
    def induced_subgraph(data: Data, node_idx, relabel_nodes: bool = True) -> Data: 
        node_idx = torch.as_tensor(node_idx, dtype=torch.long) 

        edge_attr  = getattr(data, "edge_attr", None)
        edge_index = getattr(data, "edge_index", None) 

        edge_index_sub, edge_attr_sub = subgraph(
            node_idx, 
            edge_index, 
            edge_attr=edge_attr, 
            relabel_nodes=relabel_nodes, 
            num_nodes=data.num_nodes 
        )

        out = Data(edge_index=edge_index_sub, edge_attr=edge_attr_sub)

        if relabel_nodes:
            out.x = data.x[node_idx]
            if hasattr(data, "y") and data.y is not None: 
                out.y = data.y[node_idx] 
            out.orig_idx = node_idx
        else: 
            out.x = data.x 
            if hasattr(data, "y"):
                out.y = data.y 

        return out 
