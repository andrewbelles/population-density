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

import numpy as np 

from pathlib import Path 

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    CLASSIFICATION,
    TaskSpec
)

from models.graph.processing import CorrectAndSmooth

from analysis.hyperparameter import (
    CorrectAndSmoothEvaluator,
    run_nested_cv,
    run_optimization,
    make_train_mask 
)

from utils.helpers import (
    project_path,
    save_model_config,
)

from testbench.utils.paths import (
    CONFIG_PATH,
    PROBA_PASSTHROUGH_PATH, 
    PROBA_PATH,
    check_paths_exist
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
    best_score
)

from testbench.utils.etc import (
    get_factory,
    get_param_space,
    write_model_metrics,
    write_model_summary
)

from testbench.utils.data import (
    DATASETS,
    BASE,
    make_filtered_loader,
    passthrough_loader,
    read_feature_list,
    stacking_loader
)

from testbench.utils.oof import load_probs_labels_fips
    
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
    "SAIPE": project_path("data", "stacking", "saipe_optimized_probs.mat")
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

OPT_TASK = TaskSpec("classification", ("f1_macro",))

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
    proba_path=None
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
    buf,
    *,
    config_path: str = CONFIG_PATH
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
        with contextlib.redirect_stdout(sys.stderr): 
            mean_score, best_params, _, _ = run_nested_cv(
                name=f"{dataset_key}_{model_name}",
                filepath=filepath,
                loader_func=loader_func,
                model_factory=get_factory(model_name),
                param_space=get_param_space(model_name),
                task=OPT_TASK,
                outer_config=outer_config,
                inner_config=inner_config,
                n_trials=n_trials,
                random_state=RANDOM_STATE,
                early_stopping_rounds=EARLY_STOP,
                early_stopping_delta=EARLY_STOP_EPS,
                sampler_type="multivariate-tpe"
            )

        best_params = normalize_params(model_name, best_params) 
        save_model_config(config_path, f"{dataset_key}/{model_name}", best_params)
        leaderboard.append((mean_score, model_name, best_params))
        
    leaderboard.sort(key=lambda x: x[0], reverse=True)
    best_score_val, best_model, _ = leaderboard[0]
    buf.write(f"\n== {dataset_key} optimization ==\n")
    buf.write(f"Best Model: {best_model}\n")
    buf.write(f"Best Score: {best_score_val:.4f}\n")
    return _aggregate_leaderboard( 
        dataset_key,
        filepath,
        loader_func,
        outer_config,
        leaderboard
    )

def _optimize_cs(
    buf: io.StringIO,
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
        class_labels=class_labels
    )

    with contextlib.redirect_stdout(buf):
        best_params, best_value = run_optimization(
            name=key,
            evaluator=evaluator,
            n_trials=CS_TRIALS,
            direction="maximize",
            early_stopping_rounds=CS_EARLY_STOP,
            early_stopping_delta=EARLY_STOP_EPS,
            sampler_type="multivariate-tpe",
            random_state=RANDOM_STATE
        )

    save_model_config(config_path, key, best_params)
    write_model_summary(buf, key, best_value)
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
        autoscale=autoscl
    )
    P_cs    = cs(P, y, train_mask, W_by_name[adj_name])
    metrics = metrics_from_probs(y[test_mask], P_cs[test_mask], class_labels)
    return metrics, adj_name, P_cs

def _select_best_model(dataset_key, filepath, loader_func, config, *, config_path: str = CONFIG_PATH): 
    leaderboard = []
    for model_name in MODELS: 
        params = load_model_params(config_path, f"{dataset_key}/{model_name}")
        params = normalize_params(model_name, params)
        metrics, score = _evaluate_model(
            filepath,
            loader_func,
            model_name,
            params,
            config
        )
        leaderboard.append((score, model_name, params, metrics))

    best = max(leaderboard, key=lambda x: x[0])
    return best 

# ---------------------------------------------------------
# Tests 
# ---------------------------------------------------------

def test_expert_optimize(
    buf: io.StringIO, 
    datasets: list[str] | None = None, 
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    **_
):
    targets = datasets or list(DATASETS)
    for name in targets:
        if name not in BASE: 
            raise ValueError(f"unknown dataset: {name}")
        base = BASE[name]

        if filter_dir: 
            keep_path = str(Path(filter_dir) / f"boruta_keep_{name.lower()}.txt")
        else: 
            keep_path = None 
        keep_list = read_feature_list(keep_path)

        loader = make_filtered_loader(base["loader"], keep_list)

        _optimize_dataset(
            name, 
            base["path"],
            loader, 
            n_trials=EXPERT_TRIALS,
            buf=buf,
            config_path=config_path 
        )

def test_expert_oof(
    buf: io.StringIO, 
    filter_dir: str | None = None, 
    config_path: str = CONFIG_PATH, 
    **_
):
    config = eval_config() 
    for name in DATASETS: 
        base = BASE[name]

        if filter_dir: 
            keep_path = str(Path(filter_dir) / f"boruta_keep_{name.lower()}.txt")
        else: 
            keep_path = None 
        keep_list = read_feature_list(keep_path)
        loader    = make_filtered_loader(base["loader"], keep_list)

        _, best_model, best_params, metrics = _select_best_model(
            name, 
            base["path"],
            loader,
            config,
            config_path=config_path
        )

        write_model_metrics(buf, f"{name}/{best_model}", metrics)
        _evaluate_model(
            base["path"],
            loader,
            best_model,
            best_params,
            config,
            proba_path=EXPERT_PROBA[name]
        )

    return {
        "experts": {k: EXPERT_PROBA[k] for k in DATASETS}
    }

def test_stacking_optimize(
    buf: io.StringIO, 
    passthrough: bool = False, 
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    **_
):
    prob_files = [EXPERT_PROBA[n] for n in DATASETS]
    check_paths_exist(prob_files, "expert OOF files")

    if passthrough: 
        if filter_dir: 
            keep_path = str(Path(filter_dir) / "boruta_keep_cross.txt")
            keep_list = read_feature_list(keep_path)
            loader    = passthrough_loader(prob_files, keep_list)
        else: 
            loader = passthrough_loader(prob_files)
    else: 
        loader = stacking_loader(prob_files) 
    
    key    = STACKING_PASSTHROUGH_KEY if passthrough else STACKING_BASE_KEY 

    _optimize_dataset(
        key,
        "virtual",
        loader,
        n_trials=STACKING_TRIALS,
        buf=buf,
        config_path=config_path
    )

def test_stacking(
    buf: io.StringIO, 
    passthrough: bool = False,
    filter_dir: str | None = None,
    config_path: str = CONFIG_PATH,
    **_
): 
    prob_files = [EXPERT_PROBA[n] for n in DATASETS]
    check_paths_exist(prob_files, "expert OOF files")

    key    = STACKING_PASSTHROUGH_KEY if passthrough else STACKING_BASE_KEY 
    proba  = PROBA_PASSTHROUGH_PATH   if passthrough else PROBA_PATH
    name   = "StackingPassthrough"    if passthrough else "Stacking"

    if passthrough: 
        if filter_dir: 
            keep_path = str(Path(filter_dir) / "boruta_keep_cross.txt")
            keep_list = read_feature_list(keep_path)
            loader    = passthrough_loader(prob_files, keep_list)
        else: 
            loader = passthrough_loader(prob_files)
    else: 
        loader = stacking_loader(prob_files) 

    config = eval_config()
    best_score_val, best_model, best_params, metrics = _select_best_model(
        key,
        "virtual",
        loader,
        config,
        config_path=config_path
    )

    write_model_metrics(buf, f"{name}/{best_model}", metrics)

    _evaluate_model(
        "virtual",
        loader,
        best_model,
        best_params,
        config,
        proba_path=proba
    )

    P, y, fips, class_labels = load_probs_labels_fips(proba)

    return {
        "proba_path": proba, 
        "model": best_model, 
        "score": best_score_val,
        "metadata": {
            "name": f"{name}/{best_model}",
            "probs": P, 
            "labels": y, 
            "fips": fips,
            "class_labels": class_labels
        }
    }

def test_cs_opt(buf: io.StringIO, passthrough: bool = False, config_path: str = CONFIG_PATH): 

    name  = "StackingPassthrough" if passthrough else "Stacking"
    proba = PROBA_PASSTHROUGH_PATH if passthrough else PROBA_PATH 

    best_params, _, context = _optimize_cs(
        buf,
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

    return {
        "params": best_params, 
        "metrics": metrics,
        "metadata": {
            "name": f"{context['key']}/{adj_name}",
            "probs": context["P"], 
            "probs_corr": P_cs, 
            "train_mask": context["train_mask"],
            "labels": context["y"], 
            "fips": context["fips"],
            "class_labels": context["class_labels"]
        }
    }

def test_pipeline(
    buf: io.StringIO, 
    passthrough: bool = False,
    datasets: list[str] | None = None, 
    filter_dir: str | None = None, 
    config_path: str = CONFIG_PATH,
    **_
): 
    expert_opt   = test_expert_optimize(
        buf,
        datasets=datasets,
        filter_dir=filter_dir,
        config_path=config_path
    )
    expert_oof   = test_expert_oof(
        buf,
        filter_dir=filter_dir,
        config_path=config_path
    )
    stacking_opt = test_stacking_optimize(
        buf,
        passthrough=passthrough,
        filter_dir=filter_dir,
        config_path=config_path
    )
    stacking     = test_stacking(
        buf,
        passthrough=passthrough,
        filter_dir=filter_dir,
        config_path=config_path
    )
    cs           = test_cs_opt(
        buf,
        passthrough=passthrough,
        config_path=config_path
    )

    return {
        "expert_opt": expert_opt,
        "expert_oof": expert_oof,
        "stacking_opt": stacking_opt,
        "stacking": stacking,
        "cs": cs
    }


CROSS_TESTS = {"stacking", "stacking_opt", "cs_opt", "pipeline"}

TESTS = {
    "expert_opt": test_expert_optimize,
    "expert_oof": test_expert_oof,
    "stacking_opt": test_stacking_optimize,
    "stacking": test_stacking,
    "cs_opt": test_cs_opt,
    "pipeline": test_pipeline
}

def run_test(name: str, buf: io.StringIO, *, cross: str = "off", **kwargs): 

    fn = TESTS.get(name)
    if fn is None: 
        raise ValueError(f"unknown test: {name}")

    if name in CROSS_TESTS: 
        if cross == "off": 
            fn(buf, passthrough=False, **kwargs)
        elif cross == "on": 
            fn(buf, passthrough=True, **kwargs)
        elif cross == "both": 
            fn(buf, passthrough=False, **kwargs)
            fn(buf, passthrough=True, **kwargs)
        else: 
            raise ValueError(f"unknown choice for cross-modal feature set: {cross}")
        return 
    fn(buf, **kwargs)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--cross", choices=["off", "on", "both"], default="off")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--no-opt", action="store_true")
    parser.add_argument("--filter", action="store_true")
    args = parser.parse_args()

    if args.filter: 
        config_path = project_path("testbench", "filter_config.yaml")
        filter_dir  = project_path("data", "results", "boruta_splits")
    else: 
        config_path = CONFIG_PATH 
        filter_dir  = None 

    buf = io.StringIO() 

    kw     = vars(args).copy() 
    kw.pop("tests", None)
    cross  = kw.pop("cross", "off") 
    no_opt = kw.pop("no_opt", False) 

    targets = args.tests or list(TESTS.keys())
    for name in targets: 
        if no_opt and name.endswith("_opt"): 
            continue 
        run_test(name, buf, cross=cross, config_path=config_path, filter_dir=filter_dir, **kw)

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main()
