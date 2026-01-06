#!/usr/bin/env python3 
# 
# adjacency_tests.py  Andrew Belles  Jan 6th, 2026 
# 
# Dry tests for adjacency builders and analysis of underlying graph metric 
#
# 

import argparse, io, contextlib  
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analysis.graph_metrics import MetricAnalyzer
from analysis.hyperparameter import _load_yaml_config

from models.graph.construction import (
    build_knn_graph_from_coords,
    make_queen_adjacency_factory,
    make_mobility_adjacency_factory
)

from models.graph.learning import EdgeLearner 

from preprocessing.loaders import (
    ConcatSpec,
    load_concat_datasets,
    load_oof_predictions,
    load_coords_from_mobility,
    load_viirs_nchs, 
    load_compact_dataset,
    make_oof_dataset_loader,
    _align_on_fips 
)

from utils.helpers import (
    project_path,
    make_cfg_gap_factory
) 

from testbench.test_utils import (
    _select_specs_psv,
    _override_oof_path,
    _apply_transforms,
    _coords_for_fips
)

# ---------------------------------------------------------
# Global Variables 
# ---------------------------------------------------------

OOF_PATH      = project_path("data", "results", "final_stacked_passthrough.mat")
MOBILITY_PATH = project_path("data", "datasets", "travel_proxy.mat")
SHAPEFILE     = project_path("data", "geography", "county_shapefile", "tl_2020_us_county.shp")
CONFIG_PATH   = project_path("testbench", "model_config.yaml")
LABELS_PATH   = project_path("data", "datasets", "viirs_nchs_2023.mat")

# ---------------------------------------------------------
# Helpers  
# ---------------------------------------------------------

def _write_metrics(buf: io.StringIO, title: str, adj, P, y, coords): 
    '''
    Pretty prints metric output to a buffer so all staged metric results can be 
    printed simultaneously 
    '''
    metrics = MetricAnalyzer.compute_metrics(
        adj,
        probs=P, 
        y_true=y,
        train_mask=None,
        coords=coords,
        verbose=False 
    )
    buf.write(f"\n== {title} ==\n")
    with contextlib.redirect_stdout(buf): 
        MetricAnalyzer._print_report(metrics)

# ---------------------------------------------------------
# Wrapped Loaders
# ---------------------------------------------------------

def _load_probs_labels_fips(): 
    oof   = load_oof_predictions(OOF_PATH) 
    probs = np.asarray(oof["probs"], dtype=np.float64)
    if probs.ndim != 3: 
        raise ValueError(f"expected probs (n, m, c), got {probs.shape}")
    if probs.shape[1] == 1: 
        P = probs[:, 0, :]
    else: 
        P = probs.mean(axis=1)
    y    = np.asarray(oof["labels"]).reshape(-1)
    fips = np.asarray(oof["fips_codes"]).astype("U5")
    return P, y, fips 

def _load_probs_for_fips(fips: NDArray[np.str_]) -> NDArray: 
    P, _, oof_fips = _load_probs_labels_fips()
    idx = _align_on_fips(fips, oof_fips)
    return P[idx]

# ---------------------------------------------------------
# Assertions 
# ---------------------------------------------------------

def check_adj(adj, n: int, name: str): 
    if adj.shape != (n, n): 
        raise ValueError(f"{name} shape mismatch: {adj.shape} != ({n}, {n})")
    if adj.nnz == 0: 
        raise ValueError(f"{name} has no edges")
    if not np.isfinite(adj.data).all(): 
        raise ValueError(f"{name} has non-finite edge weights")

# ---------------------------------------------------------
# Unit Tests  
# ---------------------------------------------------------

def test_queen(fips: NDArray): 
    factory = make_queen_adjacency_factory(SHAPEFILE)
    W = factory(list(fips))
    check_adj(W, len(fips), "queen")
    return W 

def test_mobility(fips: NDArray, k: int = 12): 
    factory = make_mobility_adjacency_factory(
        MOBILITY_PATH,
        OOF_PATH,
        k_neighbors=k
    )
    W = factory(list(fips))
    check_adj(W, len(fips), "mobility")
    return W 

def test_knn(fips: NDArray, k: int = 12): 
    coords = _coords_for_fips(MOBILITY_PATH, fips)
    W = build_knn_graph_from_coords(coords, k=k, directed=False)
    check_adj(W, len(fips), "knn")
    if W.diagonal().sum() != 0: 
        raise ValueError("knn adjacency formed self-loops. Expected no self-loops")
    return W 

def test_queen_metrics(P, y, fips, coords, buf: io.StringIO): 
    W = test_queen(fips)
    _write_metrics(buf, "Queen", W, P, y, coords)

def test_mobility_metrics(P, y, fips, coords, buf: io.StringIO, k: int = 12): 
    W = test_mobility(fips, k)
    _write_metrics(buf, "Mobility", W, P, y, coords)

def test_knn_metrics(P, y, fips, coords, buf: io.StringIO, k: int = 12): 
    W = test_knn(fips, k)
    _write_metrics(buf, "kNN", W, P, y, coords)

def test_edge_gate(
    key: str, 
    params: dict, 
    *, 
    X, 
    y, 
    fips,
    P,
    coords,
    buf: io.StringIO
): 
    base_adj = make_queen_adjacency_factory(SHAPEFILE)(list(fips))

    model = EdgeLearner(**params) 
    model.fit(X, y, base_adj)
    adj = model.build_graph(X, base_adj)
    check_adj(adj, len(fips), f"{key} adjacency")
    _write_metrics(buf, key, adj, P, y, coords)

METRIC_TESTS  = {
    "/EDGE": test_edge_gate 
}

def test_learned_metrics(metric_keys: list[str], buf: io.StringIO): 
    cfg = _load_yaml_config(Path(CONFIG_PATH))
    
    for key in metric_keys: 
        params = cfg.get("models", {}).get(key)
        if params is None: 
            raise ValueError(f"missing model config for key: {key}")
        params = dict(params)

        dataset_key = params.pop("dataset", None)
        if dataset_key is None: 
            raise ValueError(f"model config missing dataset: {key}")

        specs = _select_specs_psv(dataset_key)
        specs = _override_oof_path(specs, OOF_PATH)
        data  = load_concat_datasets(
            specs=specs, 
            labels_path=LABELS_PATH,
            labels_loader=load_viirs_nchs
        )

        X    = data["features"]
        y    = np.asarray(data["labels"]).reshape(-1)
        fips = np.asarray(data["sample_ids"]).astype("U5")        
        feature_names = data.get("feature_names")

        # If cross modal features exist in dataset, correctly compute scaled versions 
        transforms = make_cfg_gap_factory(feature_names)() 
        if transforms: 
            X = _apply_transforms(X, transforms)

        P        = _load_probs_for_fips(fips)
        coords   = _coords_for_fips(MOBILITY_PATH, fips)

        handler = None 
        for suffix, fn in METRIC_TESTS.items(): 
            if key.endswith(suffix): 
                handler = fn 
                break 
        if handler is None: 
            raise ValueError(f"no test handler for metric key: {key}")

        handler(
            key, 
            params, 
            X=X,
            y=y,
            fips=fips,
            P=P,
            coords=coords,
            buf=buf
        )

# ---------------------------------------------------------
# Unittest Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--metric-keys", nargs="*", default=None)
    args   = parser.parse_args()

    buf = io.StringIO() 

    P, y, fips = _load_probs_labels_fips()
    coords     = _coords_for_fips(MOBILITY_PATH, fips)

    test_queen(fips)
    test_mobility(fips, args.k)
    test_knn(fips, args.k)

    test_queen_metrics(P, y, fips, coords, buf)
    test_mobility_metrics(P, y, fips, coords, buf, args.k)
    test_knn_metrics(P, y, fips, coords, buf, args.k)


    if args.metric_keys: 
        test_learned_metrics(args.metric_keys, buf)

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
