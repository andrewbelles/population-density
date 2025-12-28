#!/usr/bin/env python 
# 
# geospatial.py  Andrew Belles  Dec 12th, 2025 
# 
# Class that enables GeospatialGraph backend to interact with 
# ML libraries and attach features to counties by inheriting 
# the c++ implementation. 
# 


from typing import Callable
import numpy as np 
import geopandas as gpd 
import pandas as pd 
import torch 

from torch_geometric.data import Data 
from torch_geometric.utils import subgraph, to_undirected 
from numpy.typing import NDArray 

import support.helpers as h 
import support.graph_cpp as g 

from libpysal.weights import Queen

from scipy import sparse 
from scipy.io import loadmat

from preprocessing.loaders import (
    load_oof_predictions
)

from support.helpers import _mat_str_vector


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


def build_queen_adjacency(shapefile_path: str, fips_order: list[str]): 
    '''
    Builds A CSR adjacency list using Queen Contiguity from TIGER shapefile and 
    some imposed ordering on FIPS 
    '''

    fips_order = [str(f).zfill(5) for f in fips_order]
    if len(set(fips_order)) != len(fips_order): 
        raise ValueError("fips_order contains duplicates")

    gdf = gpd.read_file(shapefile_path)
    if "GEOID" not in gdf.columns: 
        raise ValueError("shapefile missing GEOD columns")

    gdf["FIPS"] = gdf["GEOID"].astype(str).str.zfill(5)
    if gdf["FIPS"].duplicated().any():
        dup = gdf[gdf["FIPS"].duplicated()]["FIPS"].tolist() 
        raise ValueError(f"duplicate FIPS in shapefile. head -n 10: {dup[:10]}")

    fips_set = set(fips_order)
    gdf = gdf[gdf["FIPS"].isin(fips_set)].copy() 

    missing = [f for f in fips_order if f not in set(gdf["FIPS"])]
    if missing: 
        raise ValueError(f"missing {len(missing)} FIPS in shapefile. head -n 10: {missing[:10]}")
    
    weights = Queen.from_dataframe(gdf, ids=gdf["FIPS"].tolist())
    adj     = weights.sparse.tocsr()

    id_order  = [str(f).zfill(5) for f in weights.id_order]
    idx_map   = {f: i for i, f in enumerate(id_order)}
    order_idx = np.asarray([idx_map[f] for f in fips_order], dtype=np.int64)

    adj = adj[order_idx][:, order_idx]
    return sparse.csr_matrix(adj, dtype=np.float64)


def normalize_adjacency(
    adj, 
    *, 
    self_loops: bool = True, 
    symmetrize: bool = True, 
    binarize: bool = True
): 
    if adj.shape[0] != adj.shape[1]: 
        raise ValueError(f"adj must be square, got {adj.shape}")

    A = adj.tocsr().astype(np.float64)

    if symmetrize: 
        A = A.maximum(A.T) 

    if binarize: 
        A.data[:] = 1.0 

    if self_loops: 
        A = A + sparse.eye(A.shape[0], format="csr")

    A.eliminate_zeros() 

    deg  = np.asarray(A.sum(axis=1)).ravel() 
    mask = deg > 0
    C    = np.zeros_like(deg)
    C[mask] = 1.0 / np.sqrt(deg[mask])

    D_inv_sqrt = sparse.diags(C)
    S = D_inv_sqrt @ A @ D_inv_sqrt
    return S.tocsr() 

AdjacencyFactory = Callable[[list[str]], sparse.csr_matrix]

def make_queen_adjacency_factory(shapefile_path: str | None = None) -> AdjacencyFactory: 
    if shapefile_path is None: 
        shapefile_path = h.project_path("data", "geography", "county_shapefile",
                                        "tl_2020_us_county.shp")

    def _factory(fips_order: list[str]) -> sparse.csr_matrix: 
        return build_queen_adjacency(shapefile_path, fips_order)

    return _factory


def make_mobility_adjacency_factory(mobility_path: str, probs_path: str) -> AdjacencyFactory:

    adj_parent, fips_parent = compute_adaptive_graph(mobility_path, probs_path)

    def _factory(fips_order: list[str]) -> sparse.csr_matrix: 
        src_map = {f: i for i, f in enumerate(fips_parent)} 
        indices = []
        for f in fips_order: 
            if f not in src_map: 
                raise ValueError(f"target FIPS {f} not found in mobility graph")
            indices.append(src_map[f])

        indices = np.array(indices, dtype=np.int64)

        return adj_parent[indices][:, indices]
    return _factory 


def compute_probability_lag_matrix(
    proba_path: str, 
        adj_factory: AdjacencyFactory, 
    *, 
    model_name: str | None = None, 
    agg: str = "mean"
): 
    '''
    Computes Probability Lag Matrix from Classifier Probabilities and 
    County adjacency Matrix. 

    If multiple models exist in oof probabilities file then an 
    optional model name can be passed 
    
    Method of aggregating Probabilities across multiple folds can be specified

    '''

    oof   = load_oof_predictions(proba_path)
    probs = np.asarray(oof["probs"], dtype=np.float64) # shape: (samples, models, classes)
    fips  = np.asarray(oof["fips_codes"]).reshape(-1)

    if probs.ndim != 3: 
        raise ValueError(f"expected probs shape (samples, models, classes), got {probs.shape}")

    if model_name is not None: 
        names = np.asarray(oof["model_names"]).reshape(-1)
        match = np.where(names == model_name)[0]
        if match.size == 0: 
            raise ValueError(f"model_name '{model_name}' not found in {names}")
        P = probs[:, int(match[0]), :]
    else: 
        if probs.shape[1] == 1: 
            P = probs[:, 0, :]
        elif agg == "mean":
            P = probs.mean(axis=1)
        else: 
            raise ValueError("multiple models present, set model_name or agg='mean'")

    adj = adj_factory(list(fips))
    W   = normalize_adjacency(adj)

    P_lag = W @ P 
    return P, P_lag, W, fips

def compute_adaptive_graph(mobility_matrix_path: str, probs_path: str):
    mat = loadmat(mobility_matrix_path)

    if "fips_codes" not in mat: 
        raise ValueError(f"{mobility_matrix_path} missing fips_codes")

    fips_mob   = _mat_str_vector(mat["fips_codes"]).astype("U5")
    edge_index = mat["edge_index"].astype(int)
    edge_attr  = mat["edge_weight_affinity"].flatten() 
    n_nodes    = mat["coords"].shape[0]

    prob_mat   = loadmat(probs_path)
    if "fips_codes" not in prob_mat: 
        raise ValueError(f"{probs_path} missing fips_codes")
    fips_prob  = _mat_str_vector(prob_mat["fips_codes"]).astype("U5")

    P = prob_mat["probs"]
    if P.ndim == 3: 
        P = P.mean(axis=1)

    prob_map   = {f: i for i, f in enumerate(fips_prob)}
    P_aligned  = np.zeros((n_nodes, P.shape[1]), dtype=np.float64)
    valid_mask = np.zeros(n_nodes, dtype=bool)
    for i, f in enumerate(fips_mob): 
        if f in prob_map: 
            P_aligned[i]  = P[prob_map[f]]
            valid_mask[i] = True 

    row, col   = edge_index[0], edge_index[1]
    edge_mask  = valid_mask[row] & valid_mask[col]
    row = row[edge_mask]
    col = col[edge_mask]
    edge_attr = edge_attr[edge_mask]

    p_src = P_aligned[row]
    p_dst = P_aligned[col]
    
    diff     = p_src - p_dst 
    dist_sq  = np.sum(diff**2, axis=1)
    dists    = np.sqrt(dist_sq) 
    sigma    = np.median(dists[dists > 0])
    if sigma < 1e-6: 
        sigma = 1.0 
    gamma    = 1.0 / (2.0 * sigma**2)
    sim_pred = np.exp(-gamma * dist_sq) 

    new_weights = edge_attr * sim_pred 

    adj_refined = sparse.csr_matrix(
        (new_weights, (row, col)), 
        shape=(n_nodes, n_nodes)
    )

    return adj_refined, fips_mob 
