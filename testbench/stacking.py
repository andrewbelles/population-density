#!/usr/bin/env python3 
# 
# stacking.py  Andrew Belles  Jan 6th, 2026 
# 
# Testbench for expert out of fold prediction generation, stacking meta-learner, and 
# Corrent-and-Smooth on stacked predictions. 
# 

import contextlib
import argparse, io
import sys
from typing import Literal 

import numpy as np 

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    CLASSIFICATION,
    FULL_CLASSIF,
)

from models.graph.processing import CorrectAndSmooth

from optimization.engine     import (
    NestedCVConfig,
    run_optimization,
    EngineConfig 
)

from optimization.evaluators import (
    CorrectAndSmoothEvaluator,
    StandardEvaluator 
)

from utils.helpers           import (
    project_path,
    save_model_config,
    make_train_mask
)

from testbench.utils.paths import (
    CONFIG_PATH,
    check_paths_exist,
    expert_prob_files,
    stacking_context
)

from testbench.utils.config import (
    load_model_params,
    normalize_params,
    eval_config
)

from testbench.utils.graph import (
    build_cs_adjacencies
)

from testbench.utils.metrics import (
    metrics_from_probs,
    metrics_from_summary,
    score_from_summary,
    best_score,
    OPT_TASK
)

from testbench.utils.etc import (
    get_factory,
    get_param_space,
    format_metric,
    merge_results,
    run_tests_table,
)

from testbench.utils.data import (
    DATASETS,
    BASE,
    resolve_expert_loader,
    resolve_stacking_loader,
)

from testbench.utils.oof import load_probs_labels_fips, stacking_metadata

from utils.resources import ComputeStrategy 

strategy = ComputeStrategy.from_env()
    
# ---------------------------------------------------------
# Global Variables 
# ---------------------------------------------------------

EXPERT_DATASETS = {
    "VIIRS": project_path("data", "datasets", "viirs_nchs_2023.mat"),
    # "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"),
    "NLCD": project_path("data", "datasets", "nlcd_nchs_2023.mat"),
    "SAIPE": project_path("data", "datasets", "saipe_nchs_2023.mat")
}

EXPERT_PROBA    = {
    "VIIRS": project_path("data", "stacking", "viirs_optimized_probs.mat"),
    # "TIGER": project_path("data", "stacking", "tiger_optimized_probs.mat"),
    "NLCD": project_path("data", "stacking", "nlcd_optimized_probs.mat"),
    "SAIPE": project_path("data", "stacking", "saipe_optimized_probs.mat"),
    "VIIRS_MANIFOLD": project_path("data", "stacking", "viirs_pooled_probs.mat"),
    "VIIRS_MANIFOLD_LOGITS": project_path("data", "stacking", 
                                          "viirs_pooled_with_logits_probs.mat"),
    "SAIPE_MANIFOLD": project_path("data", "stacking", "saipe_pooled_probs.mat")
}

STACKED_BASE_PROBS        = project_path("data", "results", "final_stacked_predictions.mat")
STACKED_PASSTHROUGH_PROBS = project_path("data", "results", "final_stacked_passthrough.mat")

STACKING_BASE_KEY        = "Stacking"
STACKING_PASSTHROUGH_KEY = "StackingPassthrough"

MODELS = ("XGBoost",)

EXPERT_TRIALS   = 250 
STACKING_TRIALS = 250 
CS_TRIALS       = 150 
EARLY_STOP      = 40 
CS_EARLY_STOP   = 100 
EARLY_STOP_EPS  = 1e-4 
RANDOM_STATE    = 0 
TRAIN_SIZE      = 0.3 

# ---------------------------------------------------------
# Helper Functions 
# ---------------------------------------------------------

def _evaluate_model(
    filepath, 
    loader_func, 
    model_name, 
    params, 
    config, 
    *,
    task=CLASSIFICATION,
    proba_path=None
):
    model = get_factory(model_name, strategy=strategy)(**params)

    cv = CrossValidator(
        filepath=filepath,
        loader=loader_func,
        task=task,
        scale_y=False 
    )

    results = cv.run(
        models={"model": model},
        config=config,
        oof=proba_path is not None, 
        collect=True 
    )

    summary = cv.summarize(results)
    metrics = metrics_from_summary(summary)
    score   = score_from_summary(summary)

    if np.isnan(score):
        score = best_score(metrics)

    if proba_path: 
        cv.save_oof(proba_path)

    return metrics, score 

def _aggregate_leaderboard(
    dataset_key, 
    filepath,
    loader_func,
    config, 
    leaderboard 
): 
    rows = []
    for mean_score, model_name, params in leaderboard: 
        metrics, eval_score = _evaluate_model(
            filepath,
            loader_func,
            model_name,
            params,
            config 
        )
        rows.append({
            "model": model_name,
            "mean_score": mean_score,
            "eval_score": eval_score, 
            "metrics": metrics, 
            "params": params
        })

    return {
        "dataset": dataset_key,
        "leaderboard": rows 
    }

def _optimize_dataset(
    dataset_key, 
    filepath, 
    loader_func, 
    n_trials,
    direction: Literal["minimize", "maximize"] = "maximize",
    *,
    config_path: str = CONFIG_PATH,
    parallel_outer: bool = False, 
    devices: list[int] | None = None
): 

    outer_config = eval_config(RANDOM_STATE)

    inner_config = CVConfig(
        n_splits=3,
        n_repeats=1,
        stratify=True, 
        random_state=RANDOM_STATE
    )

    leaderboard = []
    for model_name in MODELS: 
        evaluator = StandardEvaluator(
            filepath=filepath,
            loader_func=loader_func,
            base_factory_func=get_factory(model_name, strategy=strategy),
            param_space=get_param_space(model_name),
            task=OPT_TASK,
            config=inner_config,
            outer_config=outer_config,
            metric="qwk"
        )

        config = EngineConfig(
            n_trials=n_trials,
            direction=direction,
            random_state=RANDOM_STATE, 
            nested=NestedCVConfig(
                inner_n_trials=n_trials,
                inner_sampler_type="multivariate-tpe",
                inner_early_stopping_rounds=EARLY_STOP,
                inner_early_stopping_delta=EARLY_STOP_EPS,
                parallel_outer=parallel_outer 
            )
        )

        with contextlib.redirect_stdout(sys.stderr): 
            best_params, mean_score, _, _ = run_optimization(
                name=f"{dataset_key}_{model_name}",
                evaluator=evaluator,
                config=config
            )

        best_params = normalize_params(model_name, best_params) 
        save_model_config(config_path, f"{dataset_key}/{model_name}", best_params)
        leaderboard.append((mean_score, model_name, best_params))
        
    leaderboard.sort(key=lambda x: x[0], reverse=True)

    return _aggregate_leaderboard( 
        dataset_key,
        filepath,
        loader_func,
        outer_config,
        leaderboard
    )

def _optimize_cs(
    *,
    name: str, 
    proba: str,
    config_path: str = CONFIG_PATH
):

    key   = f"CorrectAndSmooth/{name}"

    P, y, fips, class_labels = load_probs_labels_fips(proba)
    train_mask = make_train_mask(
        y,
        train_size=TRAIN_SIZE, 
        random_state=RANDOM_STATE,
        stratify=True 
    )
    test_mask = ~train_mask 

    W_by_name = build_cs_adjacencies(proba, fips, normalize=True)
    
    evaluator = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name=W_by_name,
        y_train=y,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels,
        compute_strategy=strategy
    )

    config = EngineConfig(
        n_trials=CS_TRIALS,
        direction="maximize",
        sampler_type="multivariate-tpe",
        random_state=RANDOM_STATE,
        early_stopping_rounds=CS_EARLY_STOP,
        early_stopping_delta=EARLY_STOP_EPS
    )

    with contextlib.redirect_stdout(sys.stderr):
        best_params, best_value, _ = run_optimization(
            name=key,
            evaluator=evaluator,
            config=config,
        )

    save_model_config(config_path, key, best_params)

    return best_params, best_value, {
        "key": key,
        "P": P,
        "y": y,
        "fips": fips,
        "class_labels": class_labels,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "W_by_name": W_by_name
    } 

def _evaluate_cs(
    P,
    y,
    train_mask,
    test_mask,
    class_labels,
    W_by_name,
    params 
): 
    params   = dict(params)
    adj_name = params.pop("adjacency")

    cl, ca   = params.pop("correction_layers"), params.pop("correction_alpha") 
    sl, sa   = params.pop("smoothing_layers"), params.pop("smoothing_alpha")
    autoscl  = params.pop("autoscale")
    
    cs = CorrectAndSmooth(
        class_labels=class_labels,
        correction=(cl, ca),
        smoothing=(sl, sa),
        autoscale=autoscl,
        compute_strategy=strategy
    )
    P_cs    = cs(P, y, train_mask, W_by_name[adj_name])
    metrics = metrics_from_probs(y[test_mask], P_cs[test_mask], class_labels)
    return metrics, adj_name, P_cs

def _select_best_model(
    dataset_key, 
    filepath, 
    loader_func, 
    config, 
    *, 
    task=CLASSIFICATION,
    config_path: str = CONFIG_PATH
): 
    leaderboard = []
    for model_name in MODELS: 
        params = load_model_params(config_path, f"{dataset_key}/{model_name}")
        params = normalize_params(model_name, params)
        metrics, score = _evaluate_model(
            filepath,
            loader_func,
            model_name,
            params,
            config,
            task=task
        )
        leaderboard.append((score, model_name, params, metrics))

    best = max(leaderboard, key=lambda x: x[0])
    return best 

def _row_metrics(name: str, metrics: dict): 
    return {
        "Name":    name, 
        "Acc":     format_metric(metrics.get("accuracy")),
        "F1":      format_metric(metrics.get("f1_macro")),
        "ROC_AUC": format_metric(metrics.get("roc_auc")),
        "LogLoss": format_metric(metrics.get("log_loss")),
        "Brier": format_metric(metrics.get("brier")),
        "ECE": format_metric(metrics.get("ece")),
        "QWK": format_metric(metrics.get("qwk")),
        "OrdMAE": format_metric(metrics.get("ord_mae"))
    }

def _row_opt(name: str, model: str, score: float): 
    return {
        "Name":  name, 
        "Model": model, 
        "F1":    format_metric(score)
    }

# ---------------------------------------------------------
# Tests 
# ---------------------------------------------------------

def test_expert_optimize(
    datasets: list[str] | None = None, 
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    **_
):
    rows    = []
    targets = datasets or list(DATASETS) 

    for name in targets:
        if name not in BASE: 
            raise ValueError(f"unknown dataset: {name}")

        base, loader = resolve_expert_loader(name, filter_dir)

        results = _optimize_dataset(
            name, 
            base["path"],
            loader, 
            direction="maximize",
            n_trials=EXPERT_TRIALS,
            config_path=config_path,
            parallel_outer=False, 
            devices=[0]
        )

        best = max(results["leaderboard"], key=lambda r: r["mean_score"])
        rows.append(_row_opt(name, best["model"], best["mean_score"]))
        
    return {
        "header": ["Name", "Model", "F1"],
        "rows": rows 
    }

def test_expert_oof(
    datasets: list[str] | None = None, 
    filter_dir: str | None = None, 
    config_path: str = CONFIG_PATH, 
    **_
):
    rows    = []
    config  = eval_config() 
    targets = datasets or list(DATASETS) 

    for name in targets:
        if name not in BASE: 
            raise ValueError(f"unknown dataset: {name}")
        base, loader = resolve_expert_loader(name, filter_dir)

        _, best_model, best_params, metrics = _select_best_model(
            name, 
            base["path"],
            loader,
            config,
            task=FULL_CLASSIF,
            config_path=config_path
        )

        _evaluate_model(
            base["path"],
            loader,
            best_model,
            best_params,
            config,
            task=FULL_CLASSIF,
            proba_path=EXPERT_PROBA[name]
        )

        rows.append(_row_metrics(f"{name}/{best_model}", metrics))

    return {
        "header": ["Name", "Acc", "F1", "ROC_AUC", "LogLoss", "Brier", "ECE", "QWK", "OrdMAE"],
        "rows": rows,
        "experts": {k: EXPERT_PROBA[k] for k in DATASETS}
    }

def test_stacking_optimize(
    passthrough: bool = False, 
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    extra: list[str] | None = None, 
    **_
):
    prob_files = expert_prob_files(DATASETS, EXPERT_PROBA)
    if extra: 
        prob_files = prob_files + list(extra)
    check_paths_exist(prob_files, "expert OOF files")

    loader       = resolve_stacking_loader(prob_files, passthrough, filter_dir)
    key, _, name = stacking_context(passthrough)

    results = _optimize_dataset(
        key,
        "virtual",
        loader,
        n_trials=STACKING_TRIALS,
        config_path=config_path
    )

    best = max(results["leaderboard"], key=lambda r: r["mean_score"])

    return {
        "header": ["Name", "Model", "F1"],
        "row": _row_opt(name, best["model"], best["mean_score"])
    }

def test_stacking(
    passthrough: bool = False,
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    extra: list[str] | None = None, 
    **_
): 
    prob_files = expert_prob_files(DATASETS, EXPERT_PROBA)
    if extra: 
        prob_files = prob_files + list(extra)
    check_paths_exist(prob_files, "expert OOF files")

    loader           = resolve_stacking_loader(prob_files, passthrough, filter_dir)
    key, proba, name = stacking_context(passthrough)
    config           = eval_config()

    best_score_val, best_model, best_params, metrics = _select_best_model(
        key,
        "virtual",
        loader,
        config,
        config_path=config_path
    )

    _evaluate_model(
        "virtual",
        loader,
        best_model,
        best_params,
        config,
        proba_path=proba
    )

    meta = stacking_metadata(proba)

    return {
        "header": ["Name", "Acc", "F1", "ROC_AUC"],
        "row": _row_metrics(f"{name}/{best_model}", metrics),
        "proba_path": proba, 
        "model": best_model, 
        "score": best_score_val,
        "metadata": {
            "name": f"{name}/{best_model}",
            **meta
        }
    }

def test_cs_opt(
    passthrough: bool = False, 
    config_path: str = CONFIG_PATH,
    **_
): 

    _, proba, name = stacking_context(passthrough)

    best_params, _, context = _optimize_cs(
        name=name,
        proba=proba,
        config_path=config_path
    )

    metrics, adj_name, P_cs = _evaluate_cs(
        context["P"],
        context["y"],
        context["train_mask"],
        context["test_mask"],
        context["class_labels"],
        context["W_by_name"],
        best_params 
    )

    name = f"{context['key']}/{adj_name}"

    return {
        "header": ["Name", "Acc", "F1", "ROC_AUC"],
        "row": _row_metrics(name, metrics), 
        "params": best_params, 
        "metrics": metrics,
        "metadata": {
            "name": name,
            "probs": context["P"], 
            "probs_corr": P_cs, 
            "train_mask": context["train_mask"],
            "labels": context["y"], 
            "fips": context["fips"],
            "class_labels": context["class_labels"]
        }
    }

# ---------------------------------------------------------
# Test Creation and Callers 
# ---------------------------------------------------------

CROSS_TESTS = {"stacking", "stacking_opt", "cs_opt"}
OPT_TESTS   = {"expert_opt", "stacking_opt"} # cs_opt is so inexpensive it might as well not be an opt test 
TESTS = {
    "expert_opt": test_expert_optimize,
    "expert_oof": test_expert_oof,
    "stacking_opt": test_stacking_optimize,
    "stacking": test_stacking,
    "cs_opt": test_cs_opt,
}

def _call_test(fn, name, *, cross, **kwargs): 
    print(f"[{name}] starting...")
    if name in CROSS_TESTS: 
        if cross == "off": 
            return fn(passthrough=False, **kwargs)
        elif cross == "on": 
            return fn(passthrough=True, **kwargs)
        elif cross == "both": 
            r1 = fn(passthrough=False, **kwargs)
            r2 = fn(passthrough=True, **kwargs)
            return merge_results(r1, r2)
        else: 
            raise ValueError(f"unknown choice for cross-modal feature set {cross}")
    return fn(**kwargs)

# ---------------------------------------------------------
# Test Entry 
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--cross", choices=["off", "on", "both"], default="off")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--no-opt", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--extra", nargs="*", default=None)
    args = parser.parse_args()

    if args.filter: 
        config_path = project_path("testbench", "filter_config.yaml")
        filter_dir  = project_path("data", "results", "boruta_splits")
    else: 
        config_path = CONFIG_PATH 
        filter_dir  = None 

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    if args.no_opt: 
        targets = [t for t in targets if t not in OPT_TESTS]

    run_tests_table(
        buf, 
        TESTS,
        targets=targets,
        caller=lambda fn, name, **kw: _call_test(
            fn, name, cross=args.cross, **kw
        ),
        datasets=args.datasets,
        filter_dir=filter_dir,
        config_path=config_path
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main()
