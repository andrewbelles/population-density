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

from models.metric import (
    IDMLGraphLearner,
    GradientBoostingMetricLearner
)

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

from support.helpers import (
    project_path,
    make_cfg_gap_factory 
)

from testbench.test_utils import (
    _load_model_params,
    _select_specs_psv,
    _map_fracs,
    _override_oof_path,
    _apply_transforms,
    _iter_metric_models,
    _coords_for_fips,
)


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", default=project_path("testbench", "model_config.yaml"))
    parser.add_argument("--model-key", default=None)
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

    cfg = _load_yaml_config(Path(args.config))

    if args.model_key: 
        params = cfg.get("models", {}).args(args.model_key)
        if params is None: 
            raise ValueError(f"missing model config: {args.model_key}")
        targets = [(args.model_key, dict(params))]
    else: 
        targets = list(_iter_metric_models(cfg))
        if not targets: 
            raise ValueError("no learned metric models found in config")

    for i, (model_key, params) in enumerate(targets): 
        dataset_key = params.pop("dataset", None)
        if dataset_key is None: 
            continue 

        specs = _select_specs_psv(dataset_key)
        specs = _override_oof_path(specs, args.oof_path)

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

        transforms = make_cfg_gap_factory(data.get("feature_names"))()
        if transforms: 
            X = _apply_transforms(X, transforms)

        if model_key.endswith("/GBM"):
            model = GradientBoostingMetricLearner(**params)
        else: 
            model = IDMLGraphLearner(**params)

        model.fit(X, y)
        adj_metric = model.get_graph(X)

        coords = _coords_for_fips(args.mobility_path, fips)

        print(f"\n== {model_key} ==")
        MetricAnalyzer.compute_metrics(
            adj_metric, 
            probs=P, 
            y_true=y, 
            train_mask=None, 
            verbose=True,
            coords=coords 
        )

        if i == 0: 
            queen_factory = make_queen_adjacency_factory(args.shapefile)
            W_queen = queen_factory(list(fips))

            mobility_factory = make_mobility_adjacency_factory(args.mobility_path, args.oof_path)
            W_mob   = mobility_factory(list(fips))

            print("\n== Queen Adjacency ==")
            MetricAnalyzer.compute_metrics(
                W_queen, 
                probs=P, 
                y_true=y,
                train_mask=None, 
                verbose=True, 
                coords=coords 
            )

            print("\n== Mobility Adaptive ==")
            MetricAnalyzer.compute_metrics(
                W_mob, 
                probs=P, 
                y_true=y,
                train_mask=None, 
                verbose=True,
                coords=coords 
            )


if __name__ == "__main__": 
    main() 
