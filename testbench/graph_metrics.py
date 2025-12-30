#!/usr/bin/env python3 
# 
# graph_metrics.py  Andrew Belles  Dec 29th, 2025 
# 
# Computes metrics across multiple graph adjacency matrix methods 
# using analysis/graph_metrics.py 
# 

import argparse
import numpy as np 
from pathlib import Path

from analysis.graph_metrics import MetricAnalyzer 
from analysis.hyperparameter import (
    _load_yaml_config,
    _load_probs_for_fips 
)

from models.metric import IDMLGraphLearner

from models.graph_utils import (
    make_queen_adjacency_factory, 
    make_mobility_adjacency_factory 
)

from preprocessing.loaders import (
    ConcatSpec,
    load_concat_datasets,
    make_oof_dataset_loader,
    load_viirs_nchs,
    load_compact_dataset,
    load_coords_from_mobility
)

from support.helpers import project_path 

from testbench.test_utils import (
    _load_model_params,
    _select_specs_psv,
    _map_fracs
)


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", default=project_path("testbench", "model_config.yaml"))
    parser.add_argument("--model-key", default="StackingOOF/IDML")
    parser.add_argument("--labels-path", default=project_path("data", "datasets", 
                                                              "viirs_nchs_2023.mat"))
    parser.add_argument("--oof-path", default=project_path("data", "results", 
                                                           "final_stacked_predictions.mat"))
    parser.add_argument("--mobility-path", default=project_path("data", "datasets", 
                                                                "travel_proxy.mat"))
    parser.add_argument("--shapefile", default=project_path("data", "geography", 
                                                            "county_shapefile", 
                                                            "tl_2020_us_county.shp"))
    args = parser.parse_args()

    params = _load_model_params(args.config, args.model_key)
    dataset_key = params.pop("dataset", None)
    if dataset_key is None: 
        raise ValueError("model config missing dataset key")

    specs = _select_specs_psv(dataset_key)
    data  = load_concat_datasets(
        specs=specs, 
        labels_path=args.labels_path,
        labels_loader=load_viirs_nchs
    )

    X = data["features"]
    y = np.asarray(data["labels"]).reshape(-1)
    fips = np.asarray(data["sample_ids"]).astype("U5")

    P, _ = _load_probs_for_fips(args.oof_path, fips)

    params = _map_fracs(params, X)
    model  = IDMLGraphLearner(**params)
    model.fit(X, y)
    adj_metric = model.get_graph(X)

    queen_factory = make_queen_adjacency_factory(args.shapefile)
    W_queen = queen_factory(list(fips))

    mobility_factory = make_mobility_adjacency_factory(args.mobility_path, args.oof_path)
    W_mob   = mobility_factory(list(fips))

    print("\n== Learned Metric Graph ==")
    MetricAnalyzer.compute_metrics(adj_metric, probs=P, train_mask=None, verbose=True)

    print("\n== Queen Adjacency ==")
    MetricAnalyzer.compute_metrics(W_queen, probs=P, train_mask=None, verbose=True)

    print("\n== Mobility Adaptive ==")
    MetricAnalyzer.compute_metrics(W_mob, probs=P, train_mask=None, verbose=True)

if __name__ == "__main__": 
    main() 
