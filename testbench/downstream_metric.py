#!/usr/bin/env python3 
# 
# downstream_metric.py  Andrew Belles  Dec 28th, 2025 
# 
# Testbench for Graph Based Modules/Implementations 
# to determine optimal graph construction, etc. 
# 

import argparse 
import numpy as np
import pandas as pd

from analysis.hyperparameter import (
    MetricCASEvaluator,
    MobilityEvaluator,
    CorrectAndSmoothEvaluator,
    define_gbm_metric_space, 
    define_idml_space,
    define_qg_space,
    run_optimization,
    _save_model_config,
    _load_yaml_config,
    _load_probs_for_fips
)

from models.graph_utils import make_queen_adjacency_factory
from models.post_processing import make_train_mask

from preprocessing.loaders import (
    load_concat_datasets,
    load_viirs_nchs,
    load_oof_predictions,
    _align_on_fips
)

from support.helpers import (
    make_cfg_gap_factory,
    project_path 
)

from pathlib import Path

from testbench.test_utils import (
    _power_set,
    _map_fracs,
    _select_specs_csv,
    _select_specs_psv,
    _make_idml,
    _make_gb_metric,
    _make_qg,
    make_knn_adjacency_factory
) 

FULL_KEY = "VIIRS+TIGER+NLCD+COORDS+PASSTHROUGH"

def mobility(
    mobility_path: str, 
    proba_path: str
):
    import matplotlib.pyplot as plt 

    evaluator = MobilityEvaluator(
        mobility_path=mobility_path,
        proba_path=proba_path
    )

    rows = []
    for k in range(4, 101): 
        score = evaluator.evaluate({"k_neighbors": k})
        rows.append({"k": k, "score": score})

    df = pd.DataFrame(rows)

    y_min = df["score"].min() 
    y_max = df["score"].max() 
    pad   = max(0.002, 0.1 * (y_max - y_min))

    plt.figure(figsize=(10,7))
    plt.bar(df["k"], df["score"], width=0.9)
    plt.ylim(y_min - pad, y_max + pad)
    plt.xlabel("k neighbors")
    plt.ylabel("Downstream C+S accuracy")
    plt.title(f"Mobility k sweep best_k={np.argmax(df['score'])}")
    plt.tight_layout()
    plt.savefig("mobility_k_sweep.png", dpi=150)
    plt.close()
    print(f"> Saved plot: mobility_k_sweep.png")
    

def optimize_metric(
    dataset_name: str, 
    proba_path: str,
    dataset_loaders, 
    label_path, 
    label_loader,
    n_trials: int = 400, 
    base_factory=_make_idml,
    param_space=define_idml_space,
    metric_tag="idml",
    adjacency_factory=None
): 

    filepath = project_path("virtual")

    full_specs = _select_specs_psv(FULL_KEY)
    dataset_loaders = {
        FULL_KEY: lambda _: load_concat_datasets(
            specs=full_specs,
            labels_path=label_path,
            labels_loader=label_loader
        )
    }

    evaluator = MetricCASEvaluator(
        filepath=filepath,
        base_factory_func=base_factory,
        param_space=param_space,
        proba_path=proba_path,
        dataset_loaders=dataset_loaders,
        feature_transform_factory=None if metric_tag == "QG" else make_cfg_gap_factory,
        adjacency_factory=adjacency_factory
    )

    best_params, best_value = run_optimization(
        name=f"{dataset_name}/{metric_tag}", 
        evaluator=evaluator,
        n_trials=n_trials,
        direction="maximize",
        sampler_type="multivariate-tpe",
        early_stopping_rounds=100,
        early_stopping_delta=1e-4
    )

    print(f"> Max Downstream Accuracy: {best_value:.4f}")

    config_path = project_path("testbench", "model_config.yaml")
    model_key   = f"{dataset_name}/{metric_tag}"
    _save_model_config(str(config_path), model_key, best_params)


def optimize_downstream(
    *,
    metric_key: str, 
    config_path: str, 
    proba_path: str, 
    label_path: str, 
    label_loader,
    base_factory,
    adjacency_factory=None 
): 
    config = _load_yaml_config(Path(config_path))
    params = config.get("models", {}).get(metric_key)
    if params is None: 
        raise ValueError(f"missing model config: {metric_key}")

    params = dict(params)
    params.pop("dataset", None)
    '''
    dataset_key = params.pop("dataset", None)
    if dataset_key is None: 
        raise ValueError("metric config missing 'dataset' key")
    ''' 

    dataset_key = FULL_KEY 
    specs = _select_specs_psv(dataset_key)
    
    data = load_concat_datasets(
        specs=specs,
        labels_path=label_path,
        labels_loader=label_loader 
    )

    X    = data["features"]
    y    = np.asarray(data["labels"]).reshape(-1)
    fips = np.asarray(data["sample_ids"], dtype="U5")
    oof  = load_oof_predictions(proba_path)

    
    oof_fips = np.asarray(oof["fips_codes"]).astype("U5")
    common   = [f for f in fips if f in set(oof_fips)]
    if not common: 
        raise ValueError("no common fips between dataset and OOF probs")

    if len(common) != len(fips): 
        idx = _align_on_fips(common, fips)
        X = X[idx] 
        y = y[idx]
        fips = fips[idx]
    
    P, class_labels = _load_probs_for_fips(proba_path, fips)
    params = _map_fracs(params, X)

    train_mask = make_train_mask(
        y,
        train_size=0.3,
        random_state=0,
        stratify=True
    )

    test_mask = ~train_mask 

    model = base_factory(**params)

    fit_kwargs = {}
    if "train_mask" in model.fit.__code__.co_varnames:
        fit_kwargs["train_mask"] = train_mask 
    if adjacency_factory is not None and "adj" in model.fit.__code__.co_varnames: 
        fit_kwargs["adj"] = adjacency_factory(list(fips))
    model.fit(X, y, **fit_kwargs)
    adj = model.get_graph(X)

    cs = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name={"metric": adj},
        y_train=y,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels
    )

    best_params, best_value = run_optimization(
        name=f"{metric_key}/CorrectAndSmooth",
        evaluator=cs,
        n_trials=100, 
        early_stopping_rounds=40,
        early_stopping_delta=1e-4,
        sampler_type="multivariate-tpe"
    )

    _save_model_config(config_path, f"{metric_key}/CorrectAndSmooth", best_params)
    print(f"> Downstream Correct and Smooth best value: {best_value:.4f}")

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--sources", default="viirs,tiger,nlcd,coord")
    parser.add_argument("--suppress", action="store_true")
    parser.add_argument("--mobility", action="store_true")
    parser.add_argument("--metric", choices=["idml", "gbm", "qg"], default="idml")
    parser.add_argument("--shapefile", default=project_path("data", "geography",
                                                            "county_shapefile",
                                                            "tl_2020_us_county.shp"))
    parser.add_argument("--metric-opt", action="store_true")
    parser.add_argument("--downstream", action="store_true")
    parser.add_argument("--passthrough", action="store_true")
    args = parser.parse_args() 

    mobility_path = project_path("data", "datasets", "travel_proxy.mat")
    
    adj_factory = None 
    if args.metric == "gbm": 
        metric_tag   = "GBM"
        base_factory = _make_gb_metric 
        space_fn     = define_gbm_metric_space 
    elif args.metric == "qg": 
        metric_tag   = "QG"
        base_factory = _make_qg 
        space_fn     = define_qg_space 
        adj_factory  = make_knn_adjacency_factory(mobility_path, 50)
    else: 
        metric_tag   = "IDML"
        base_factory = _make_idml
        space_fn     = define_idml_space

    if args.passthrough:
        args.sources  = args.sources + ",passthrough" 
        metric_prefix = "StackingPassthrough"
    else: 
        metric_prefix = "StackingOOF"

    specs = _select_specs_csv(args.sources)
    label_path   = project_path("data", "datasets", "viirs_nchs_2023.mat")
    label_loader = load_viirs_nchs
    
    if args.passthrough:
        proba_path    = project_path("data", "results", "final_stacked_passthrough.mat")
        dataset_name  = "StackingPassthrough"
    else: 
        proba_path    = project_path("data", "results", "final_stacked_predictions.mat")
        dataset_name  = "StackingOOF"

    dataset_loaders = {}
    for combo in _power_set(specs): 
        key = "+".join(s["name"] for s in combo)
        dataset_loaders[key] = lambda _, combo=combo: load_concat_datasets(
            specs=combo,
            labels_path=label_path,
            labels_loader=label_loader
        )


    if args.mobility: 
        mobility(
            mobility_path=mobility_path,
            proba_path=proba_path
        )

    if args.metric_opt: 
        optimize_metric(
            dataset_name, 
            proba_path, 
            dataset_loaders=dataset_loaders, 
            label_path=label_path, 
            label_loader=label_loader, 
            n_trials=400,
            base_factory=base_factory, 
            param_space=space_fn,
            metric_tag=metric_tag,
            adjacency_factory=adj_factory
        ) 

    if args.downstream: 
        optimize_downstream(
            metric_key=f"{metric_prefix}/{metric_tag}",
            config_path=project_path("testbench", "model_config.yaml"),
            proba_path=proba_path,
            label_path=project_path("data", "datasets", "viirs_nchs_2023.mat"),
            label_loader=load_viirs_nchs,
            base_factory=base_factory,
            adjacency_factory=adj_factory 
        )

if __name__ == "__main__": 
    main()
