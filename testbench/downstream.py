#!/usr/bin/env python3 
# 
# downstream.py  Andrew Belles  Jan 6th, 2026 
# 
# Downstream optimization tests for learned metrics (w/ C+S)
# 
# 

import argparse, io 

from analysis.hyperparameter import (
    MetricCASEvaluator,
    CorrectAndSmoothEvaluator,
    define_gate_space,
    run_optimization,
    _apply_train_test_transforms,
    make_train_mask
)

from models.graph.learning import EdgeLearner 
from models.graph.processing import CorrectAndSmooth
from models.graph.construction import make_queen_adjacency_factory 

from utils.helpers import (
    make_cfg_gap_factory,
    save_model_config,
    align_on_fips
)

from testbench.utils.paths import (
    SHAPEFILE,
    CONFIG_PATH,
    PROBA_PATH,
    PROBA_PASSTHROUGH_PATH
)

from testbench.utils.metrics import (
    metrics_from_probs
)

from testbench.utils.etc import (
    load_metric_params,
    write_model_summary
)

from testbench.utils.data import (
    load_dataset_raw,
    make_dataset_loader 
)

from testbench.utils.oof import load_probs_labels_fips

# ---------------------------------------------------------
# Global Constants 
# ---------------------------------------------------------

BASE_DATASET_KEY   = "VIIRS+TIGER+NLCD+COORDS"
PASSTHROUGH_DS_KEY = BASE_DATASET_KEY + "+PASSTHROUGH"

EDGE_BASE_KEY        = "Stacking/EDGE"
EDGE_PASSTHROUGH_KEY = "StackingPassthrough/EDGE"

CS_EDGE_KEY             = "CorrectAndSmooth/Stacking_EDGE"
CS_EDGE_PASSTHROUGH_KEY = "CorrectAndSmooth/StackingPassthrough_EDGE"

METRIC_TRIALS = 250 
CS_TRIALS     = 150 
EARLY_STOP    = 25
EARLY_STOP_EP = 1e-4 # tolerance to consider a result as improved 
RANDOM_STATE  = 0 
TRAIN_SIZE    = 0.3 

# ---------------------------------------------------------
# Helpers 
# ---------------------------------------------------------

def _optimize_edge_metric(dataset_key: str, proba_path: str, model_key: str, buf: io.StringIO):
    dataset_loaders = make_dataset_loader(dataset_key, proba_path)

    evaluator = MetricCASEvaluator(
        filepath="virtual",
        base_factory_func=EdgeLearner,
        param_space=define_gate_space,
        dataset_loaders=dataset_loaders,
        proba_path=proba_path,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        feature_transform_factory=make_cfg_gap_factory,
        adjacency_factory=make_queen_adjacency_factory(SHAPEFILE)
    )

    best_params, best_value = run_optimization(
        name=model_key,
        evaluator=evaluator,
        n_trials=METRIC_TRIALS,
        direction="maximize",
        early_stopping_rounds=EARLY_STOP,
        early_stopping_delta=EARLY_STOP_EP,
        sampler_type="multivariate-tpe",
        random_state=RANDOM_STATE
    )

    save_model_config(CONFIG_PATH, model_key, best_params)
    write_model_summary(buf, f"{model_key} (metric opt)", best_value)

    X, y, fips, feature_names = load_dataset_raw(dataset_key, proba_path)
    _, _, oof_fips, _ = load_probs_labels_fips(proba_path)
    oof_set = set(oof_fips)
    common  = [f for f in fips if f in oof_set]
    if len(common) != len(fips):
        idx  = align_on_fips(common, fips)
        X    = X[idx]
        y    = y[idx]
        fips = fips[idx]

    train_mask = make_train_mask(
        y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=True
    )

    transforms = make_cfg_gap_factory(feature_names)()
    if transforms:
        X = _apply_train_test_transforms(transforms, X, train_mask)

    base_adj = make_queen_adjacency_factory(SHAPEFILE)(list(fips))

    best_params.pop("dataset")
    model = EdgeLearner(**best_params)
    model.fit(X, y, base_adj, train_mask=train_mask)
    adj = model.build_graph(X, base_adj)

    return {
        "params": best_params,
        "score": best_value,
        "metadata": {
            "dataset": dataset_key,
            "fips": fips,
            "train_mask": train_mask,
            "base_adj": base_adj,
            "adj": adj,
            "edge_keep_ratio": float(adj.nnz) / max(1, base_adj.nnz),
            "edge_weights": adj.data
        }
    }

def _optimize_cs_on_metric(model_key: str, cs_key: str, proba_path: str, buf: io.StringIO): 
    params = load_metric_params(model_key)

    dataset_key = params.pop("dataset", None)
    if dataset_key is None: 
        raise ValueError(f"{model_key} missing dataset key")

    X, y, fips, feature_names        = load_dataset_raw(dataset_key, proba_path)
    P_all, _, oof_fips, class_labels = load_probs_labels_fips(proba_path)
    oof_set = set(oof_fips)
    common  = [f for f in fips if f in oof_set]
    if len(common) != len(fips):
        idx  = align_on_fips(common, fips)
        X    = X[idx]
        y    = y[idx]
        fips = fips[idx]

    idx_oof = align_on_fips(fips, oof_fips)
    P = P_all[idx_oof]

    train_mask = make_train_mask(
        y,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        stratify=True
    )
    test_mask = ~train_mask

    transforms = make_cfg_gap_factory(feature_names)()
    if transforms: 
        X = _apply_train_test_transforms(transforms, X, train_mask)

    base_adj = make_queen_adjacency_factory(SHAPEFILE)(list(fips))

    # REPLACE W/ GENERIC FACTORY FOR OTHER METRICS 
    model = EdgeLearner(**params)
    model.fit(X, y, base_adj, train_mask=train_mask)
    adj = model.build_graph(X, base_adj)

    cs = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name={"metric": adj}, 
        y_train=y,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels
    )

    best_params, best_value = run_optimization(
        name=cs_key, 
        evaluator=cs, 
        n_trials=CS_TRIALS,
        direction="maximize",
        early_stopping_rounds=EARLY_STOP,
        early_stopping_delta=EARLY_STOP_EP,
        sampler_type="multivariate-tpe", 
        random_state=RANDOM_STATE
    )

    save_model_config(CONFIG_PATH, cs_key, best_params)
    write_model_summary(buf, f"{cs_key} (C+S opt)", best_value)

    params = dict(best_params)
    adj_name = params.pop("adjacency")
    cl, ca   = params.pop("correction_layers"), best_params.pop("correction_alpha")
    sl, sa   = params.pop("smoothing_layers"), best_params.pop("smoothing_alpha")
    autoscl  = params.pop("autoscale")
    cs_model = CorrectAndSmooth(
        class_labels=class_labels,
        correction=(cl, ca),
        smoothing=(sl, sa),
        autoscale=autoscl
    )

    P_cs    = cs_model(P, y, train_mask, {"metric": adj}[adj_name])
    metrics = metrics_from_probs(y[test_mask], P_cs[test_mask], class_labels)

    return {
        "params": best_params,
        "score": best_value,
        "metrics": metrics,
        "metadata": {
            "name": f"{cs_key}/{adj_name}",
            "probs": P,
            "probs_corr": P_cs,
            "labels": y,
            "class_labels": class_labels,
            "train_mask": train_mask,
            "test_mask": test_mask,
            "adj": adj,
            "base_adj": base_adj
        }
    }

# ---------------------------------------------------------
# Tests 
# ---------------------------------------------------------

def test_metric(buf: io.StringIO, passthrough: bool = False): 
    dataset_key = PASSTHROUGH_DS_KEY     if passthrough else BASE_DATASET_KEY 
    proba       = PROBA_PASSTHROUGH_PATH if passthrough else PROBA_PATH 
    edge_key    = EDGE_PASSTHROUGH_KEY   if passthrough else EDGE_BASE_KEY 

    return _optimize_edge_metric(
        dataset_key,
        proba,
        edge_key,
        buf
    )

def test_cs(buf: io.StringIO, passthrough: bool = False):
    proba    = PROBA_PASSTHROUGH_PATH  if passthrough else PROBA_PATH 
    edge_key = EDGE_PASSTHROUGH_KEY    if passthrough else EDGE_BASE_KEY 
    cs_key   = CS_EDGE_PASSTHROUGH_KEY if passthrough else CS_EDGE_KEY 

    return _optimize_cs_on_metric(
        edge_key,
        cs_key, 
        proba, 
        buf
    )

TESTS = {
    "metric_base": test_metric,
    "metric_passthrough": test_metric
}

# ---------------------------------------------------------
# Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--cross", choices=["off", "on", "both"], default="off")
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    for name in targets: 
        fn = TESTS.get(name)
        if fn is None: 
            raise ValueError(f"unknown test: {name}")
        if args.cross == "off": 
            fn(buf)
        elif args.cross == "on": 
            fn(buf, True)
        elif args.cross == "both": 
            fn(buf)
            fn(buf, True)
        else: 
            raise ValueError(f"unexpected choice for args.cross: {args.choice}")

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main()
