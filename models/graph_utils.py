#!/usr/bin/env python 
# 
# geospatial.py  Andrew Belles  Dec 12th, 2025 
# 
# Class that enables GeospatialGraph backend to interact with 
# ML libraries and attach features to counties by inheriting 
# the c++ implementation. 
# 


import numpy as np 
import torch 

from torch_geometric.data import Data 
from torch_geometric.utils import subgraph, to_undirected 
from numpy.typing import NDArray 

import support.helpers as h 
import support.graph_cpp as g 

def build_knn_graph_from_coords(
    coords: NDArray[np.float64], 
    *, 
    k: int = 5, 
    directed: bool = False 
) -> g.Graph: 

    if coords.ndim != 2 or coords.shape[1] != 2: 
        raise ValueError(f"coords must be (n,2), got shape {coords.shape}")

    n = coords.shape[0]
    # Build empty graph 
    if n == 0: 
        return g.build_graph(
            0, 
            np.zeros((2, 0), dtype=np.int64), 
            np.zeros((0,), dtype=np.float64),
            g.BuildOptions()
        )

    # Ensure k is feasible, get Distance Matrix 
    k = max(1, min(k, n - 1)) 
    D = h._haversine_dist(coords, coords)
    np.fill_diagonal(D, np.inf) 

    nn_idx  = np.argpartition(D, kth=k - 1, axis=1)[:, :k] 
    nn_dist = np.take_along_axis(D, nn_idx, axis=1).reshape(-1) 

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = nn_idx.reshape(-1).astype(np.int64, copy=False)
    
    edge_index  = np.vstack([src, dst]).astype(np.int64, copy=False)
    edge_weight = nn_dist.astype(np.float64, copy=False) 

    opt = g.BuildOptions() 
    opt.directed = directed 
    opt.add_reverse_edges = False 
    opt.allow_self_loops  = False 
    opt.sort_by_dst = True 
    opt.dedup = g.DuplicateEdgePolicy.MIN

    return g.build_graph(n, edge_index, edge_weight, opt)


def to_pyg_data(
    graph: g.Graph, 
    *,
    x: NDArray, 
    y: NDArray | None = None, 
    undirect_mean: bool = True 
) -> Data: 

    edge_index, edge_dist = graph.to_coo_numpy()
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_attr  = torch.as_tensor(edge_dist, dtype=torch.float32).unsqueeze(1)

    if undirect_mean: 
        edge_index, edge_attr = to_undirected(
            edge_index,
            edge_attr=edge_attr,
            reduce="mean"
        )

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32) if y is not None else None 
    data.x = x_tensor  
    data.y = y_tensor if y_tensor is not None else None
       
    return data 

def induced_subgraph(
        data: Data, 
        node_idx, 
        *, 
        relabel_nodes: bool = True
) -> Data: 

    node_idx = torch.as_tensor(node_idx, dtype=torch.long) 

    if hasattr(data, "edge_attr"): 
        edge_attr  = getattr(data, "edge_attr")
    else: 
        raise ValueError("data does not have attr edge_attr")

    if hasattr(data, "edge_index"): 
        edge_index = getattr(data, "edge_index")
    else: 
        raise ValueError("data does not have attr edge_index")

    edge_index_sub, edge_attr_sub = subgraph(
        node_idx, 
        edge_index, 
        edge_attr=edge_attr, 
        relabel_nodes=relabel_nodes, 
        num_nodes=data.num_nodes 
    )

    out = Data(edge_index=edge_index_sub, edge_attr=edge_attr_sub)

    if not hasattr(data, "x"): 
        raise ValueError("data is missing field: x")
    elif data.x is None: 
        raise ValueError("data.x is None")

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
