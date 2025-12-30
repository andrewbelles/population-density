#!/usr/bin/env python3 
# 
# downstream_metric.py  Andrew Belles  Dec 28th, 2025 
# 
# Testbench for Graph Based Modules/Implementations 
# to determine optimal graph construction, etc. 
# 

import argparse 
import numpy as np 

from analysis.hyperparameter import (
    MetricCASEvaluator,
    CorrectAndSmoothEvaluator, 
    define_idml_space,
    run_optimization,
    _save_model_config,
    _load_yaml_config,
    _load_yaml_config,
    _load_probs_for_fips
)

from models.post_processing import make_train_mask
from preprocessing.loaders import (
    load_concat_datasets,
    load_coords_from_mobility,
    load_viirs_nchs,
    load_oof_predictions,
    _align_on_fips
)

from support.helpers import project_path 

from models.metric import IDMLGraphLearner 

from pathlib import Path

from testbench.test_utils import (
    _power_set,
    _map_fracs,
    _select_specs_csv,
    _select_specs_psv,
    _make_idml
) 

def idml(
    dataset_name: str, 
    proba_path: str,
    dataset_loaders, 
    n_trials: int = 100, 
): 

    filepath = project_path("virtual")

    evaluator = MetricCASEvaluator(
        filepath=filepath,
        base_factory_func=_make_idml,
        param_space=define_idml_space,
        proba_path=proba_path,
        dataset_loaders=dataset_loaders,
    )

    best_params, best_value = run_optimization(
        name=f"{dataset_name}/IDML", 
        evaluator=evaluator,
        n_trials=n_trials,
        direction="maximize",
        sampler_type="multivariate-tpe",
        early_stopping_rounds=25,
        early_stopping_delta=1e-4
    )

    print(f"> Max Downstream Accuracy: {best_value:.4f}")

    config_path = project_path("testbench", "model_config.yaml")
    model_key   = f"{dataset_name}/IDML"
    _save_model_config(str(config_path), model_key, best_params)


def downstream(
    *,
    metric_key: str, 
    config_path: str, 
    proba_path: str, 
    label_path: str, 
    label_loader 
): 
    config = _load_yaml_config(Path(config_path))
    params = config.get("models", {}).get(metric_key)
    if params is None: 
        raise ValueError(f"missing model config: {metric_key}")

    params = dict(params)
    dataset_key = params.pop("dataset", None)
    if dataset_key is None: 
        raise ValueError("metric config missing 'dataset' key")

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

    model = IDMLGraphLearner(**params)
    model.fit(X, y)
    adj = model.get_graph(X)

    train_mask = make_train_mask(
        y,
        train_size=0.3,
        random_state=0,
        stratify=True
    )

    test_mask = ~train_mask 

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
    parser.add_argument("--sources", default="viirs,tiger,nlcd,oof,coords")
    parser.add_argument("--suppress", action="store_true")
    parser.add_argument("--idml", action="store_true")
    parser.add_argument("--downstream", action="store_true")
    args = parser.parse_args() 

    specs = _select_specs_csv(args.sources)
    label_path   = project_path("data", "datasets", "viirs_nchs_2023.mat")
    label_loader = load_viirs_nchs
    
    dataset_loaders = {}
    for combo in _power_set(specs): 
        key = "+".join(s["name"] for s in combo)
        dataset_loaders[key] = lambda _, combo=combo: load_concat_datasets(
            specs=combo,
            labels_path=label_path,
            labels_loader=label_loader
        )

    proba_path = project_path("data", "results", "final_stacked_predictions.mat")

    if args.idml: 
        idml(
            "StackingOOF", 
            proba_path, 
            dataset_loaders=dataset_loaders, 
            n_trials=100
        ) 

    if args.downstream: 
        downstream(
            metric_key="StackingOOF/IDML",
            config_path=project_path("testbench", "model_config.yaml"),
            proba_path=proba_path,
            label_path=project_path("data", "datasets", "viirs_nchs_2023.mat"),
            label_loader=load_viirs_nchs,
        )

if __name__ == "__main__": 
    main()
