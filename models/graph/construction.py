#!/usr/bin/env python 
# 
# construction.py  Andrew Belles  Dec 12th, 2025 
# 
# Graph Utilities for Adjacency Graph Construction 
# 

from abc import ABC, abstractmethod

from dataclasses import dataclass

import torch 

import torch.nn.functional as F 

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

LOGRADIANCE_GATE_LOW   = 0.4654 # Lower partition on log radiance for type 0 nodes 
LOGRADIANCE_GATE_HIGH  = 1.8548 # Upper partition on log radiance for type 2 nodes  

LOGCAPACITY_GATE_LOW   = 4.6894
LOGCAPACITY_GATE_HIGH  = 9.9478

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

# ---------------------------------------------------------
# Hypergraph Construction for Spatial Classification
# ---------------------------------------------------------

@dataclass 
class HypergraphMetadata: 
    node_type: torch.Tensor 
    incidence_index: torch.Tensor 
    edge_type: torch.Tensor 
    edge_batch: torch.Tensor 
    group_ids: torch.Tensor 
    readout_node_ids: torch.Tensor | None 


class Hypergraph(ABC): 
    '''
    Shared API for hypergraph construction for Spatial/Image and Tabular models: 
    - validates stats 
    - computes active mask 
    - builds semantic/global/readout incidences 
    - assembles metadata

    The child class implements: 
    - node typing policy 
    - layout/group mapping 
    - structural incidences 
    '''
    def __init__(
        self,
        *,
        anchors: torch.Tensor, 
        n_semantic_types: int = 3, 
        global_active_eps: float = 1e-6, 
        device: str = "cuda"
    ): 
        self.n_semantic_types  = n_semantic_types
        self.global_active_eps = global_active_eps
        self.device            = torch.device(device if torch.cuda.is_available() else "cpu")

        if anchors.ndim != 2 or anchors.shape[0] != 3: 
            raise ValueError(f"anchors must be (3, D), got {tuple(anchors.shape)}")
        self.anchors = anchors 

    # -----------------------------------------------------
    # Public API   
    # -----------------------------------------------------

    def build(
        self,
        stats: torch.Tensor, 
        *,
        batch_size: int, 
        active_stats: torch.Tensor | None = None, 
        **kwargs,
    ) -> HypergraphMetadata:

        # Setup 
        stats     = self.into_2d(stats)
        layout    = self.resolve_layout(stats, batch_size=batch_size, **kwargs)
        N, B      = layout["N"], layout["B"]
        group_ids = layout["group_ids"]
        if group_ids.shape[0] != N: 
            raise ValueError("group_ids shape mismatch")

        # Begin construction 
        node_type = self.node_types(stats).to(device=self.device, dtype=torch.long).view(-1)
        if node_type.numel() != N: 
            raise ValueError("node_types length mismatch")

        is_active = self.active_mask(stats, active_stats=active_stats)

        struct_nodes, struct_hedges, n_struct_edges, struct_edge_batch = (
            self.build_structural_incidence(stats, layout, **kwargs)
        )

        sem_nodes, sem_hedges   = self.build_semantic_incidence(
            node_type, is_active, group_ids, n_struct_edges
        )
        glob_nodes, glob_hedges = self.build_global_incidence(
            is_active, group_ids, n_struct_edges, B
        )
        readout_nodes, readout_hedges, readout_ids = self.build_readout_incidence(
            N, B, n_struct_edges
        )

        all_nodes  = torch.cat([struct_nodes, sem_nodes, glob_nodes, readout_nodes], dim=0)
        all_hedges = torch.cat([struct_hedges, sem_hedges, glob_hedges, readout_hedges], dim=0)

        n_sem_edges   = B * self.n_semantic_types
        n_glob_edges  = B
        n_edges_total = n_struct_edges + n_sem_edges + n_glob_edges 

        edge_type = torch.empty(n_edges_total, dtype=torch.long, device=self.device)
        edge_type[:n_struct_edges] = 0 
        edge_type[n_struct_edges:n_struct_edges + n_sem_edges] = 1 
        edge_type[n_struct_edges + n_sem_edges:] = 2

        sem_edge_batch  = (torch.arange(B, device=self.device)
                           .repeat_interleave(self.n_semantic_types))
        glob_edge_batch = torch.arange(B, device=self.device)
        edge_batch      = torch.cat([struct_edge_batch, sem_edge_batch, glob_edge_batch], dim=0)

        readout_type_id = self.n_semantic_types
        readout_types   = torch.full((B,), readout_type_id, dtype=torch.long, device=self.device)
        all_node_types  = torch.cat([node_type, readout_types], dim=0)

        incidence_index = torch.stack([all_nodes, all_hedges], dim=0)

        return HypergraphMetadata(
            node_type=all_node_types,
            incidence_index=incidence_index,
            edge_type=edge_type,
            edge_batch=edge_batch,
            group_ids=group_ids,
            readout_node_ids=readout_ids
        )

    # -----------------------------------------------------
    # Child API  
    # -----------------------------------------------------

    @abstractmethod 
    def resolve_layout(
        self,
        stats: torch.Tensor, 
        *, 
        batch_size: int,
        **kwargs 
    ): 
        raise NotImplementedError

    @abstractmethod 
    def build_structural_incidence(
        self, 
        stats: torch.Tensor, 
        layout: dict, 
        **kwargs 
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]: 
        raise NotImplementedError

    # -----------------------------------------------------
    # Parent Incidence Builders 
    # -----------------------------------------------------

    def build_semantic_incidence(
        self, 
        node_type: torch.Tensor, 
        is_active: torch.Tensor, 
        group_ids: torch.Tensor, 
        n_struct_edges: int
    ) -> tuple[torch.Tensor, torch.Tensor]: 
        nodes_all  = []
        hedges_all = []
        
        for t in range(self.n_semantic_types): 
            mask = (node_type == t) & is_active 
            if not mask.any(): 
                continue 

            nodes = torch.nonzero(mask, as_tuple=False).squeeze(1)
            hedge = n_struct_edges + group_ids[nodes] * self.n_semantic_types + t 
            nodes_all.append(nodes)
            hedges_all.append(hedge)

        if not nodes_all:
            return (
                torch.empty((0,), device=self.device, dtype=torch.long),
                torch.empty((0,), device=self.device, dtype=torch.long),
            )
        return torch.cat(nodes_all, dim=0), torch.cat(hedges_all, dim=0)

    def build_global_incidence(
        self, 
        is_active: torch.Tensor,
        group_ids: torch.Tensor, 
        n_struct_edges: int, 
        B: int # group/bag count 
    ): 
        nodes  = torch.nonzero(is_active, as_tuple=False).squeeze(1)
        base   = n_struct_edges + B * self.n_semantic_types 
        hedges = base  + group_ids[nodes]
        return nodes, hedges 

    def build_readout_incidence(
        self,
        N: int, # node count  
        B: int, # bag count 
        n_struct_edges: int 
    ):
        groups    = torch.arange(B, device=self.device)
        sem_base  = n_struct_edges 
        glob_base = n_struct_edges + B * self.n_semantic_types 

        sem_edges     = sem_base + groups[:, None] * self.n_semantic_types + torch.arange(
            self.n_semantic_types, device=self.device
        )[None, :]
        glob_edges    = (glob_base + groups).unsqueeze(1)
        readout_edges = torch.cat([sem_edges, glob_edges], dim=1).reshape(-1)

        readout_ids   = torch.arange(N, N + B, device=self.device)
        readout_nodes = readout_ids.repeat_interleave(self.n_semantic_types + 1)
        return readout_nodes, readout_edges, readout_ids

    # -----------------------------------------------------
    # Helpers  
    # -----------------------------------------------------

    def node_types(self, stats: torch.Tensor) -> torch.Tensor:
        x = self.into_2d(stats).to(self.device)
        if x.shape[1] != self.anchors.shape[1]: 
            raise ValueError(f"stats dim {x.shape[1]} != anchor dim {self.anchors.shape[1]}")
        d = torch.cdist(x, self.anchors, p=2.0)
        return torch.argmin(d, dim=1)

    def active_mask(
        self,
        stats: torch.Tensor, 
        *,
        active_stats: torch.Tensor | None 
    ) -> torch.Tensor: 
        gate = stats if active_stats is None else self.into_2d(active_stats).to(stats.device)
        if gate.shape[0] != stats.shape[0]: 
            raise ValueError("active_stats size mismatch.")
        return gate.abs().amax(dim=1) > self.global_active_eps

    @staticmethod
    def into_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1: 
            return x.view(-1, 1)
        if x.ndim == 2: 
            return x 
        raise ValueError(f"expected 1d/2d tensor, got {tuple(x.shape)}")

# ---------------------------------------------------------
# Spatial Hypergraph Builder 
# ---------------------------------------------------------

class SpatialHypergraph(Hypergraph): 

    '''
    - node typing handles by nearest anchor in (P, C), that is, the patch space per channel 
    - structural hyperedges are in 3x3 neighborhood of patches 
    '''

    def __init__(
        self,
        *,
        anchors: torch.Tensor, 
        tile_size: int = 256, 
        patch_size: int = 32, 
        global_active_eps: float = 1e-6, 
        device: str 
    ): 
        super().__init__(
            anchors=anchors, 
            n_semantic_types=3,
            global_active_eps=global_active_eps,
            device=device
        )

        self.tile_size  = tile_size 
        self.patch_size = patch_size 

        self.grid_h  = self.tile_size // self.patch_size 
        self.grid_w  = self.grid_h # assume square 

        self.L = self.grid_h**2 
        self.spatial_cache_ = {}

    def node_types(self, stats: torch.Tensor) -> torch.Tensor:
        anchors = self.anchors.to(self.device) 
        if stats.shape[1] != anchors.shape[1]:
            raise ValueError(f"stats dim {stats.shape[1]} != anchor dim {anchors.shape[1]}")
        d = torch.cdist(stats, anchors, p=2.0)
        return torch.argmin(d, dim=1)

    def build_structural_incidence(
        self, 
        stats: torch.Tensor, 
        layout: dict,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        _, *_ = stats, kwargs # unused by spatial hypergraph  
        B, K, idx = layout["B"], layout["K"], layout["idx"]

        neighbor, *_ = self.spatial_template(self.device)

        inv = torch.full((B, self.L), -1, device=self.device, dtype=torch.long) 
        inv.scatter_(1, idx, torch.arange(K, device=self.device).expand(B, K))

        neighbors = neighbor[idx]
        valid     = neighbors >= 0 
        neighbors = neighbors.clamp_min(0)

        local = inv.gather(1, neighbors.reshape(B, -1)).reshape(B, K, 9)
        local = torch.where(valid, local, torch.full_like(local, -1))

        mask_local = local >= 0 
        b_idx      = torch.arange(B, device=self.device).view(B, 1, 1).expand(B, K, 9)

        struct_nodes  = (b_idx * K + local)[mask_local]
        counts        = mask_local.sum(dim=2).reshape(-1)
        struct_hedges = torch.arange(B * K, device=self.device).repeat_interleave(counts)
        
        n_struct_edges    = B * K 
        struct_edge_batch = torch.arange(B, device=self.device).repeat_interleave(K)
        return struct_nodes, struct_hedges, n_struct_edges, struct_edge_batch

    def resolve_layout(self, stats: torch.Tensor, *, batch_size: int, idx=None, **kwargs):
        _ = kwargs 
        if idx is None: 
            idx = torch.arange(self.L, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        if idx.ndim != 2: 
            raise ValueError("idx must be (B, K)")
        B, K = idx.shape 
        N    = B * K 
        if stats.shape[0] != N: 
            raise ValueError(f"stats size mismatch, got {stats.shape[0]} expected {N}")

        group_ids = torch.arange(B, device=self.device).repeat_interleave(K)
        return {"N": N, "B": B, "K": K, "idx": idx, "group_ids": group_ids}

    def spatial_template(self, device: torch.device): 
        if device in self.spatial_cache_:
            return self.spatial_cache_[device]

        grid = torch.arange(self.L, device=device).reshape(1, 1, self.grid_h, self.grid_w) + 1 
        neighbor = F.unfold(grid.float(), kernel_size=3, padding=1, stride=1)
        neighbor = neighbor.long()[0].transpose(0, 1) - 1
        mask     = neighbor >= 0 

        base_nodes  = neighbor[mask]
        counts      = mask.sum(dim=1)
        base_hedges = torch.arange(self.L, device=device).repeat_interleave(counts) 
        self.spatial_cache_[device] = (neighbor, mask, base_nodes, base_hedges, counts)
        return self.spatial_cache_[device]

# ---------------------------------------------------------
# Tabular Hypergraph Builder  
# ---------------------------------------------------------

class TabularHypergraph(Hypergraph): 
    '''
    - node typing by nearest anchor in stats space 
    - structural hyperedges from kNN neighborhoods 
    '''

    def __init__(
        self,
        *,
        anchors: torch.Tensor, 
        knn: int = 8, 
        global_active_eps: float = 1e-6, 
        device: str = "cuda"
    ): 
        super().__init__(
            anchors=anchors, 
            n_semantic_types=3,
            global_active_eps=global_active_eps,
            device=device
        )

        self.knn = knn 

    def build_structural_incidence(
        self, 
        stats: torch.Tensor, 
        layout: dict, 
        neighbors: torch.Tensor | None = None, 
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        _ = kwargs 

        N, g = layout["N"], layout["group_ids"]

        if neighbors is None: 
            raise ValueError("must pass precomputed neighbor")

        neighbors = neighbors.to(self.device, dtype=torch.long)

        self_idx = torch.arange(N, device=self.device, dtype=torch.long).view(N, 1)
        members  = torch.cat([self_idx, neighbors], dim=1)

        valid    = (members >= 0) & (members < N)

        # clamp to stay inside group id 
        ms = members.clamp(0, N - 1)
        valid = valid & (g[ms] == g.view(N, 1))

        edge_ids = (torch.arange(N, device=self.device, dtype=torch.long).view(N, 1)
                    .expand_as(members))

        struct_nodes  = members[valid]
        struct_hedges = edge_ids[valid]

        n_struct_edges = N 
        struct_edge_batch = g 
        return struct_nodes, struct_hedges, n_struct_edges, struct_edge_batch

    def resolve_layout(
        self, 
        stats: torch.Tensor, 
        *, 
        batch_size: int, 
        group_ids: torch.Tensor | None = None, 
        **kwargs
    ):
        _ = kwargs 

        x = self.into_2d(stats).to(self.device)
        N = x.shape[0]

        if group_ids is None: 
            g = torch.zeros(N, device=self.device, dtype=torch.long)
            B = 1 
        else: 
            g = group_ids.to(self.device, dtype=torch.long).view(-1)
            if g.numel() != N: 
                raise ValueError(f"group_ids {g.numel()} != N {N}")
            B = int(g.max().item()) + 1 

        if group_ids is not None and batch_size != B: 
            raise ValueError(f"batch_size {batch_size} != inferred group count {B}")

        return {"N": N, "B": B, "group_ids": g}


