#!/usr/bin/env python3 
# 
# adjacency_tests.py  Andrew Belles  Jan 6th, 2026 
# 
# Dry tests for adjacency builders and analysis of underlying graph metric 
#
# 

import argparse, io  
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from models.graph.construction import (
    build_knn_graph_from_coords,
    make_queen_adjacency_factory,
    make_mobility_adjacency_factory
)

from models.graph.learning import EdgeLearner 

from preprocessing.loaders import (
    load_concat_datasets,
    load_viirs_nchs, 
)

from utils.helpers import (
    make_cfg_gap_factory,
    load_yaml_config
) 

from testbench.utils.graph import (
    check_adj
)

from testbench.utils.paths import (
    PROBA_PATH,
    MOBILITY_PATH,
    SHAPEFILE,
    CONFIG_PATH,
    LABELS_PATH
)

from testbench.utils.data import (
    select_specs_psv,
    override_proba_path,
)

from testbench.utils.oof import (
    load_probs_for_fips,
    load_probs_labels_fips
)

from testbench.utils.transforms import apply_transforms 
from testbench.utils.graph import coords_for_fips
from testbench.utils.etc import write_graph_metrics

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
        PROBA_PATH,
        k_neighbors=k
    )
    W = factory(list(fips))
    check_adj(W, len(fips), "mobility")
    return W 

def test_knn(fips: NDArray, k: int = 12): 
    coords = coords_for_fips(MOBILITY_PATH, fips)
    W = build_knn_graph_from_coords(coords, k=k, directed=False)
    check_adj(W, len(fips), "knn")
    if W.diagonal().sum() != 0: 
        raise ValueError("knn adjacency formed self-loops. Expected no self-loops")
    return W 

def test_queen_metrics(P, y, fips, coords, buf: io.StringIO): 
    W = test_queen(fips)
    m = write_graph_metrics(buf, "Queen", W, P, y, coords)
    return {
        "name": "Queen", 
        "metrics": m, 
        "adj": W
    }

def test_mobility_metrics(P, y, fips, coords, buf: io.StringIO, k: int = 12): 
    W = test_mobility(fips, k)
    m = write_graph_metrics(buf, "Mobility", W, P, y, coords)
    return {
        "name": "Mobility", 
        "metrics": m, 
        "adj": W
    }

def test_knn_metrics(P, y, fips, coords, buf: io.StringIO, k: int = 12): 
    W = test_knn(fips, k)
    m = write_graph_metrics(buf, "kNN", W, P, y, coords)
    return {
        "name": "kNN", 
        "metrics": m, 
        "adj": W
    }

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
    m = write_graph_metrics(buf, key, adj, P, y, coords)

    return {
        "name": "EdgeLearner",
        "metrics": m,
        "adj": adj
    }

METRIC_TESTS  = {
    "/EDGE": test_edge_gate 
}

def test_learned_metrics(metric_keys: list[str], buf: io.StringIO): 
    cfg = load_yaml_config(Path(CONFIG_PATH))
    
    results = []
    for key in metric_keys: 
        params = cfg.get("models", {}).get(key)
        if params is None: 
            raise ValueError(f"missing model config for key: {key}")
        params = dict(params)

        dataset_key = params.pop("dataset", None)
        if dataset_key is None: 
            raise ValueError(f"model config missing dataset: {key}")

        specs = select_specs_psv(dataset_key)
        specs = override_proba_path(specs, PROBA_PATH)
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
            X = apply_transforms(X, transforms)

        P        = load_probs_for_fips(fips)
        coords   = coords_for_fips(MOBILITY_PATH, fips)

        handler = None 
        for suffix, fn in METRIC_TESTS.items(): 
            if key.endswith(suffix): 
                handler = fn 
                break 
        if handler is None: 
            raise ValueError(f"no test handler for metric key: {key}")

        results.append(handler(
            key, 
            params, 
            X=X,
            y=y,
            fips=fips,
            P=P,
            coords=coords,
            buf=buf
        ))

    return results 

# ---------------------------------------------------------
# Unittest Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--metric-keys", nargs="*", default=None)
    args   = parser.parse_args()

    buf = io.StringIO() 

    P, y, fips, _ = load_probs_labels_fips()
    coords        = coords_for_fips(MOBILITY_PATH, fips)

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
