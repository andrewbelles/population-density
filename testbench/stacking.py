#!/usr/bin/env python3 
# 
# stacking.py  Andrew Belles  Dec 27th, 2025 
# 
# Comprehensive Test to determine optimal model over Bayesian Optimization of 
# hyperparameters, saves oof, and compares best in class for stacked classifier 
# 
# Saves out of fold predictions for final, best overall classifier 
# 

import os, argparse 
from pathlib import Path
import numpy as np
import optuna

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    CLASSIFICATION
)

from analysis.hyperparameter import (
    StandardEvaluator,
    CorrectAndSmoothEvaluator,
    run_nested_cv,
    _load_yaml_config,
    _save_model_config,
    run_optimization, 
)

from preprocessing.loaders import (
    load_viirs_nchs,
    load_stacking, 
    load_oof_predictions
)

from preprocessing.disagreement import (
    load_pass_through_stacking,
    DisagreementSpec
)

from support.helpers import project_path

from models.post_processing import (
    CorrectAndSmooth, 
    normalized_proba, 
    make_train_mask
) 

from models.graph_utils import (
    make_queen_adjacency_factory,
    make_mobility_adjacency_factory,
    compute_probability_lag_matrix
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from testbench.test_utils import (
    _metrics_from_summary,
    _score_from_summary, 
    _get_cached_params,
    _normalize_params,
    _best_score
)

def evaluate_model(
    filepath, 
    loader_func, 
    model_name, 
    params, 
    *, 
    config, 
    oof_path=None
):
    model = get_factory(model_name)(**params)
    cv = CrossValidator(
        filepath=filepath, 
        loader=loader_func, 
        task=CLASSIFICATION, 
        scale_y=False
    )
    results = cv.run(
        models={"model": model},
        config=config,
        oof=oof_path is not None
    )
    summary = cv.summarize(results)
    if oof_path: 
        cv.save_oof(oof_path)
    return _metrics_from_summary(summary), _score_from_summary(summary)

def evaluate_correct_and_smooth(
    oof_path,
    shapefile,
    mobility_path, 
    params, 
    random_state=0
): 
    params = dict(params)
    adjacency = params.pop("adjacency")

    oof = load_oof_predictions(oof_path)
    y_train = np.asarray(oof["labels"]).reshape(-1)
    class_labels = np.asarray(oof["class_labels"]).reshape(-1)

    train_mask = make_train_mask(
        y_train,
        train_size=0.3,
        random_state=random_state, 
        stratify=True
    )
    test_mask  = ~train_mask 

    queen_factory    = make_queen_adjacency_factory(shapefile)
    mobility_factory = make_mobility_adjacency_factory(mobility_path, oof_path)

    P, _, W_queen, _ = compute_probability_lag_matrix(oof_path, queen_factory)
    _, _, W_mob, _   = compute_probability_lag_matrix(oof_path, mobility_factory)
    W = W_queen if adjacency == "queen" else W_mob 

    cs = CorrectAndSmooth(class_labels=class_labels, **params)

    P_cs = cs.fit(P, y_train, W, train_mask)
    P_cs_norm = normalized_proba(P_cs, test_mask)

    pred_idx    = np.argmax(P_cs_norm, axis=1)
    pred_labels = class_labels[pred_idx]
    y_true      = y_train[test_mask]

    return {
        "accuracy": accuracy_score(y_true, pred_labels), 
        "f1_macro": f1_score(y_true, pred_labels, average="macro"), 
        "roc_auc": roc_auc_score(y_true, P_cs_norm, multi_class="ovr", average="macro")
    }

# ---------------------------------------------------------
# "Enum" into parameter space and model factory 
# ---------------------------------------------------------

def get_param_space(model_type: str): 
    
    from analysis.hyperparameter import (
        define_xgb_space,
        define_rf_space,
        define_svm_space,
        define_logistic_space
    )

    mapping = {
        "XGBoost": define_xgb_space, 
        "RandomForest": define_rf_space, 
        "SVM": define_svm_space, 
        "Logistic": define_logistic_space
    }
    return mapping[model_type]

def get_factory(model_type: str):
    
    from models.estimators import (
        make_xgb_classifier,
        make_rf_classifier,
        make_svm_classifier,
        make_logistic
    )

    mapping = {
        "XGBoost": make_xgb_classifier, 
        "RandomForest": make_rf_classifier, 
        "SVM": make_svm_classifier, 
        "Logistic": make_logistic
    }
    return mapping[model_type]

# ---------------------------------------------------------
# Test Helper Functions 
# ---------------------------------------------------------

def optimize_correct_and_smooth(
    *,
    oof_path: str, 
    shapefile: str,
    mobility_path: str, 
    n_trials: int = 100,
    early_stopping_rounds=40,
    early_stopping_delta=1e-4,
    random_state: int = 0,
    config_key: str = "CorrectAndSmooth", 
    config_path: str | None = None
): 
    oof          = load_oof_predictions(oof_path)
    y_train      = np.asarray(oof["labels"]).reshape(-1)
    class_labels = np.asarray(oof["class_labels"]).reshape(-1)
    train_mask   = make_train_mask(
        y_train, 
        train_size=0.3, 
        random_state=random_state,
        stratify=True
    )
    test_mask    = ~train_mask 

    queen_factory    = make_queen_adjacency_factory(shapefile)
    mobility_factory = make_mobility_adjacency_factory(mobility_path, oof_path)

    P, _, W_queen, _ = compute_probability_lag_matrix(
        proba_path=oof_path, 
        adj_factory=queen_factory
    )

    _, _, W_mob, _   = compute_probability_lag_matrix(
        proba_path=oof_path, 
        adj_factory=mobility_factory
    )

    W_by_name = {
        "queen": W_queen,
        "mobility_adaptive": W_mob 
    }

    evaluator = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name=W_by_name,
        y_train=y_train,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels
    )

    best_params, best_score = run_optimization(
        name="CorrectAndSmooth", 
        evaluator=evaluator,
        n_trials=n_trials,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_delta=early_stopping_delta
    )

    if config_path is not None: 
        _save_model_config(config_path, config_key, best_params)

    return best_params, best_score 

def optimize_dataset(
    dataset_name, 
    filepath, 
    loader_func, 
    n_trials=100, 
    *,
    outer_config=None, 
    inner_config=None,
    resume=False, 
    cache=None,
    config_path=None,
    random_state: int = 0 
): 

    models_list = ["Logistic", "RandomForest", "XGBoost"]
    leaderboard = []

    if outer_config is None: 
        outer_config = CVConfig(
            n_splits=5,
            n_repeats=1,
            stratify=True,
            random_state=random_state
        )
    if inner_config is None: 
        inner_config = CVConfig(
            n_splits=3,
            n_repeats=1,
            stratify=True,
            random_state=random_state
        )

    cache = cache or {}

    for m in models_list:

        sampler_type = "multivariate-tpe"

        config_key = f"{dataset_name}/{m}"
        cached_params = _get_cached_params(cache, config_key) if resume else None 

        if cached_params is not None: 
            evaluator = StandardEvaluator(
                filepath=filepath, 
                loader_func=loader_func, 
                base_factory_func=get_factory(m),
                param_space=get_param_space(m),
                task=CLASSIFICATION,
                config=CVConfig(n_splits=5, n_repeats=1)
            )

            cached_params = _normalize_params(m, cached_params)
            best_val = evaluator.evaluate(cached_params)
            print(f"[{dataset_name}] resume hit {m} (score: {best_val:.4f})")
            leaderboard.append({
                "model": m, 
                "params": cached_params, 
                "score": best_val
            })
            continue 

        evaluator = StandardEvaluator(
            filepath=filepath,
            loader_func=loader_func,
            base_factory_func=get_factory(m), 
            param_space=get_param_space(m),
            task=CLASSIFICATION,
            config=CVConfig(n_splits=5, n_repeats=1)
        )

        mean_score, best_params, _, _ = run_nested_cv(
            name=f"{dataset_name}_{m}", 
            filepath=filepath,
            loader_func=loader_func, 
            model_factory=get_factory(m),
            param_space=get_param_space(m), 
            task=CLASSIFICATION,
            outer_config=outer_config,
            inner_config=inner_config,
            n_trials=n_trials,
            random_state=random_state,
            early_stopping_rounds=25,
            early_stopping_delta=1e-4,
            sampler_type=sampler_type
        )

        best_params = _normalize_params(m, best_params)
        if config_path is not None: 
            _save_model_config(config_path, config_key, best_params)

        leaderboard.append({
            "model": m,
            "params": best_params,
            "score": mean_score
        })

    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    winner = leaderboard[0]

    print(f"\n [{dataset_name}] winner: {winner['model']} "
          f"(score: {winner['score']:.4f})")
    return winner 


def generate_oof(dataset_name, filepath, loader_func, winner, output_path):

    print(f"[{dataset_name}] generating oof using {winner['model']}...")

    factory        = get_factory(winner['model'])
    model_instance = factory(**winner['params'])

    config = CVConfig(
        n_splits=5,
        n_repeats=5,
        stratify=True,
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath,
        loader=loader_func, 
        task=CLASSIFICATION,
        scale_y=False
    )

    cv.run(
        models={"best_model": model_instance}, 
        config=config,
        oof=True 
    )

    cv.save_oof(output_path)
    print(f"[{dataset_name}] oof saved to {output_path}")

# ---------------------------------------------------------
# Main Test Pipeline  
# ---------------------------------------------------------

def stacking(config_path, args):
    passthrough = [
        "viirs_variance", 
        "nlcd_urban_core",
        "tiger_density_deadend",
        "nlcd_edge_dens"
    ]

    cache = _load_yaml_config(Path(config_path)) if args.resume else {}

    datasets = {
        "VIIRS": {
            "path": project_path("data", "datasets", "viirs_nchs_2023.mat"), 
            "loader": load_viirs_nchs
        },
        "TIGER": {
            "path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
            "loader": load_viirs_nchs 
        },
        "NLCD": {
            "path": project_path("data", "datasets", "nlcd_nchs_2023.mat"), 
            "loader": load_viirs_nchs
        }
    }

    oof_files = []
    
    for name, config in datasets.items(): 
        winner = optimize_dataset(
            name, 
            config["path"],
            config["loader"],
            n_trials=250, 
            cache=cache, 
            resume=args.resume, 
            config_path=config_path
        )

        oof_path = project_path("data", "stacking", f"{name.lower()}_optimized_oof.mat")
        generate_oof(name, config["path"], config["loader"], winner, oof_path)
        oof_files.append(oof_path)

    print("[Stacking] starting meta-learner optimization")

    specs: list[DisagreementSpec] = [
        {
            "name": "viirs", 
            "raw_path": project_path("data", "datasets", "viirs_nchs_2023.mat"),
            "raw_loader": load_viirs_nchs,
            "oof_path": oof_files[0], 
            "oof_loader": load_oof_predictions 
        },
        {
            "name": "tiger", 
            "raw_path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
            "raw_loader": load_viirs_nchs,
            "oof_path": oof_files[1],
            "oof_loader": load_oof_predictions
        },
        {
            "name": "nlcd",
            "raw_path": project_path("data", "datasets", "nlcd_nchs_2023.mat"),
            "raw_loader": load_viirs_nchs,
            "oof_path": oof_files[2],
            "oof_loader": load_oof_predictions 
        }
    ]

    if args.passthrough: 
        stack_loader = lambda fp: load_pass_through_stacking(
            specs, 
            label_path=fp, 
            label_loader=load_viirs_nchs,
            passthrough_features=passthrough
        )
    else: 
        stack_loader = lambda _: load_stacking(oof_files)

    dummy  = oof_files[0]
    prefix = "StackingPassthrough" if args.passthrough else "Stacking"

    stack_winner = optimize_dataset(
        prefix,
        dummy,
        stack_loader, 
        n_trials=250,
        config_path=config_path
    )

    if args.passthrough:
        final_output = project_path("data", "results", "final_stacked_passthrough.mat")
    else: 
        final_output = project_path("data", "results", "final_stacked_predictions.mat")
    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    generate_oof("Stacking", dummy, stack_loader, stack_winner, final_output)
    print(f"> stacking predictions saved to {final_output}")


def correct_and_smooth(config_path): 

    stacking_oof_path = project_path("data", "results", 
                                     "final_stacked_predictions.mat")
    shapefile = project_path("data", "geography", "county_shapefile", 
                             "tl_2020_us_county.shp")
    mobility_mat = project_path("data", "datasets", "travel_proxy.mat")

    cs_params, cs_score = optimize_correct_and_smooth(
        oof_path=stacking_oof_path,
        shapefile=shapefile,
        mobility_path=mobility_mat,
        n_trials=150,
        random_state=0,
        config_path=config_path
    )

    print(cs_params, cs_score)


def report(config_path): 

    config_path = project_path("testbench", "model_config.yaml")
    cfg         = _load_yaml_config(Path(config_path))
    models_cfg  = cfg.get("models", {})

    datasets = {
        "VIIRS": project_path("data", "datasets", "viirs_nchs_2023.mat"), 
        "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"), 
        "NLCD": project_path("data", "datasets", "nlcd_nchs_2023.mat")
    }

    rows = []
    oof_files = []

    config = CVConfig(
        n_splits=5,
        n_repeats=1,
        stratify=True, 
        random_state=0,
        verbose=False
    )

    for name, path in datasets.items(): 
        best = None
        for model_name in ("Logistic", "RandomForest", "XGBoost"): 
            key = f"{name}/{model_name}"
            params = models_cfg.get(key)
            if params is None: 
                continue 

            params = _normalize_params(model_name, params)
            metrics, score = evaluate_model(
                path, 
                load_viirs_nchs, 
                model_name, 
                params,
                config=config
            )
            score = _best_score(metrics)
            if best is None or score > best[0]: 
                best = (score, model_name, params, metrics)

        if best is None: 
            continue 

        _, model_name, params, metrics = best 
        oof_path = project_path("data", "stacking", f"{name.lower()}_optimized_oof.mat")
        evaluate_model(
            path, 
            load_viirs_nchs, 
            model_name, 
            params, 
            config=config, 
            oof_path=oof_path
        )
        oof_files.append(oof_path)

        rows.append({
            "Stage": name, 
            "Model": model_name, 
            **metrics 
        })

    stack_loader = lambda _: load_stacking(oof_files) 
    best_stack = None 

    for model_name in ("Logistic", "RandomForest", "XGBoost"):
        key = f"Stacking/{model_name}"
        params = models_cfg.get(key)
        if params is None: 
            continue 

        params = _normalize_params(model_name, params)
        metrics, score = evaluate_model(
            "virtual", 
            stack_loader, 
            model_name, 
            params, 
            config=config
        )
        if best_stack is None or score > best_stack[0]: 
            best_stack = (score, model_name, params, metrics)

    if best_stack is None: 
        raise ValueError

    _, model_name, params, metrics = best_stack 
    stacking_oof = project_path("data", "stacking", "stacking_oof.mat")
    evaluate_model(
        oof_files[0], 
        stack_loader,
        model_name, 
        params, 
        config=config,
        oof_path=stacking_oof
    )
    rows.append({
        "Stage": "Stacking",
        "Model": model_name,
        **metrics
    })

    cs_params  = models_cfg.get("CorrectAndSmooth")
    shapefile  = project_path("data", "geography", "county_shapefile",
                             "tl_2020_us_county.shp")
    mobility   = project_path("data", "datasets", "travel_proxy.mat")
    cs_metrics = evaluate_correct_and_smooth(
        stacking_oof, 
        shapefile,
        mobility,
        cs_params,
        random_state=0
    ) 

    rows.append({
        "Stage": "CorrectAndSmooth",
        "Model": cs_params.get("adjacency"),
        **cs_metrics
    })

    headers = ["Stage", "Model", "accuracy", "f1_macro", "roc_auc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows: 
        lines.append("| " + " | ".join([
            str(r.get("Stage", "")),
            str(r.get("Model", "")),
            f"{r.get('accuracy', float('nan')):.4f}",
            f"{r.get('f1_macro', float('nan')):.4f}",
            f"{r.get('roc_auc', float('nan')):.4f}",
        ]) + " |")
    print("\n".join(lines))


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stacking", action="store_true")
    parser.add_argument("--cs", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--suppress", action="store_true")
    parser.add_argument("--passthrough", action="store_true")
    args = parser.parse_args()

    config_path = project_path("testbench", "model_config.yaml")

    if args.suppress: 
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.stacking: 
        stacking(config_path, args)

    if args.cs: 
        correct_and_smooth(config_path)

    if args.report: 
        report(config_path)


if __name__ == "__main__": 
    main()
