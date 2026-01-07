#!/usr/bin/env python3 
# 
# graph.py  Andrew Belles  Jan 7th, 2026 
# 
# Graph Based Helpers for Testbenches 
# 
# 

import numpy as np 

from testbench.utils.paths import ( 
    SHAPEFILE, 
    MOBILITY_PATH
)

from preprocessing.loaders import (
    load_coords_from_mobility
)

from models.graph.construction import (
    make_queen_adjacency_factory,
    make_mobility_adjacency_factory,
    normalize_adjacency
)

from sklearn.neighbors import kneighbors_graph 

from utils.helpers import (
    align_on_fips
)

def check_adj(adj, n: int, name: str): 
    if adj.shape != (n, n): 
        raise ValueError(f"{name} shape mismatch: {adj.shape} != ({n}, {n})")
    if adj.nnz == 0: 
        raise ValueError(f"{name} has no edges")
    if not np.isfinite(adj.data).all(): 
        raise ValueError(f"{name} has non-finite edge weights")

def make_knn_adjacency_factory(coords_path: str, k_neighbors: int):
    data   = load_coords_from_mobility(coords_path)
    fips   = np.asarray(data["sample_ids"], dtype="U5")
    coords = np.asarray(data["coords"], dtype=np.float64)

    A = kneighbors_graph(
        coords, 
        n_neighbors=max(1, int(k_neighbors)),
        mode="connectivity", 
        include_self=False
    )
    A = A.maximum(A.T)

    def _factory(fips_order: list[str]):
        idx = align_on_fips(fips_order, fips)
        return A[idx][:, idx]
    return _factory

def coords_for_fips(coords_path: str, fips_order):
    data = load_coords_from_mobility(coords_path)
    idx  = align_on_fips(fips_order, data["sample_ids"])
    return np.asarray(data["coords"])[idx]

def build_cs_adjacencies(proba_path, fips, normalize: bool = True): 
    queen = make_queen_adjacency_factory(SHAPEFILE)(list(fips))
    mob   = make_mobility_adjacency_factory(MOBILITY_PATH, proba_path)(list(fips))

    if normalize: 
        queen = normalize_adjacency(queen) 
        mob   = normalize_adjacency(mob)

    return {
        "queen": queen,
        "mobility": mob
    }
