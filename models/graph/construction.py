#!/usr/bin/env python 
# 
# construction.py  Andrew Belles  Dec 12th, 2025 
# 
# Graph Utilities for Adjacency Graph Construction 
# 


from typing import Callable, Optional 

import numpy as np 
from numpy.typing import NDArray 

import geopandas as gpd 

from sklearn.neighbors import kneighbors_graph 
from libpysal.weights import Queen 
from scipy import sparse 
from scipy.io import loadmat 

import utils.helpers as h 
from preprocessing.loaders import load_oof_predictions

from utils.helpers import _mat_str_vector, bind

from utils.resources import ComputeStrategy

EARTH_RADIUS_KM = 6371.0

def build_knn_graph_from_coords(
    coords: NDArray[np.float64], 
    *, 
    k: int = 5, 
    directed: bool = False,
    metric: str = "haversine",
    metric_params: Optional[dict] = None, 
    include_self: bool = False, 
    compute_strategy: ComputeStrategy = ComputeStrategy.create(greedy=False),
) -> sparse.csr_matrix:  

    '''
    Builds a kNN adjacency matrix from (lat,lon) coordinates. 
    '''

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2: 
        raise ValueError(f"coords must be (n,2), got shape {coords.shape}")

    n = coords.shape[0]
    if n == 0: 
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    max_k = n if include_self else n - 1 
    if k <= 0 or max_k <= 0: 
        return sparse.csr_matrix((n, n), dtype=np.float64)

    k = min(k, max_k)

    kwargs = {}
    if metric_params is not None: 
        kwargs["metric_params"] = metric_params 

    coords_rad = None 
    if metric == "haversine": 
        coords_rad = np.radians(coords)

    adj = kneighbors_graph(
        coords_rad if metric == "haversine" else coords, 
        k,
        mode="distance",
        metric=metric,
        include_self=include_self, 
        n_jobs=compute_strategy.n_jobs,
        **kwargs
    ).tocsr() 

    if metric == "haversine": 
        adj.data = adj.data.astype(np.float64, copy=False) * EARTH_RADIUS_KM 
    else: 
        adj.data = adj.data.astype(np.float64, copy=False)

    if not directed: 
        adj = adj.maximum(adj.T) 

    adj.eliminate_zeros() 
    return adj 


def topk_by_column(adj: sparse.csr_matrix, k: int) -> sparse.csr_matrix: 
    if k < 1: 
        raise ValueError("k must be >= 1")
    A = adj.tocsc()
    if A.shape is None: 
        raise ValueError("unwrapped no graph")
    for j in range(A.shape[1]): 
        start, end = A.indptr[j], A.indptr[j + 1]
        if end - start <= k: 
            continue 
        data = A.data[start:end]
        keep = np.argpartition(data, -k)[-k:]
        mask = np.zeros_like(data, dtype=bool)
        mask[keep] = True 
        data[~mask] = 0.0 
    A.eliminate_zeros()
    return sparse.csr_matrix(A.tocsr())


def build_queen_adjacency(shapefile_path: str, fips_order: list[str]): 
    '''
    Builds A CSR adjacency list using Queen Contiguity from TIGER shapefile. 
    Nodes match fips order 
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

    gdf = gdf[gdf["FIPS"].isin(fips_order)].copy() 

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

# ---------------------------------------------------------
# Adjacency Matrix Factories 
# ---------------------------------------------------------

AdjacencyFactory = Callable[[list[str]], sparse.csr_matrix]


def make_queen_adjacency_factory(shapefile_path: str | None = None) -> AdjacencyFactory: 
    if shapefile_path is None: 
        shapefile_path = h.project_path("data", "geography", "county_shapefile",
                                        "tl_2020_us_county.shp")

    return bind(queen_adjacency_factory, shapefile_path=shapefile_path)

def queen_adjacency_factory(fips_order: list[str], *, shapefile_path: str) -> sparse.csr_matrix: 
    return build_queen_adjacency(shapefile_path, fips_order)


def make_mobility_adjacency_factory(
    mobility_path: str, 
    probs_path: str, 
    *, 
    k_neighbors: int | None = None
) -> AdjacencyFactory:
    
    # Heuristically determined to be the best k for C+S 
    if k_neighbors is None: 
        k_neighbors = 19 

    adj_parent, fips_parent = compute_adaptive_graph(
        mobility_path, 
        probs_path, 
        k_neighbors=k_neighbors
    )

    return bind(
        mobility_adjacency_factory,
        adj_parent=adj_parent,
        fips_parent=fips_parent
    )

def mobility_adjacency_factory(
    fips_order: list[str], 
    *, 
    adj_parent, 
    fips_parent
) -> sparse.csr_matrix: 
    src_map = {f: i for i, f in enumerate(fips_parent)} 
    indices = []
    for f in fips_order: 
        if f not in src_map: 
            raise ValueError(f"target FIPS {f} not found in mobility graph")
        indices.append(src_map[f])

    indices = np.array(indices, dtype=np.int64)
    return adj_parent[indices][:, indices]


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

    If multiple models exist in oof probabilities file, pass model_name or agg="mean"
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


def compute_adaptive_graph(
    mobility_matrix_path: str, 
    probs_path: str,
    *,
    k_neighbors: int
):
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

    adj_refined = topk_by_column(adj_refined, k_neighbors)
    return adj_refined, fips_mob 
