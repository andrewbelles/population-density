#!/usr/bin/env python3 
# 
# stacking.py  Andrew Belles  Jan 6th, 2026 
# 
# Testbench for expert out of fold prediction generation, stacking meta-learner, and 
# Corrent-and-Smooth on stacked predictions. 
# 

import contextlib
import argparse, io 

import numpy as np 

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    CLASSIFICATION
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
    passthrough_loader,
    stacking_loader
)

from testbench.utils.oof import load_probs_labels_fips
    
# ---------------------------------------------------------
# Global Variables 
# ---------------------------------------------------------

EXPERT_DATASETS = {
    "VIIRS": project_path("data", "datasets", "viirs_nchs_2023.mat"),
    "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"),
    "NLCD": project_path("data", "datasets", "nlcd_nchs_2023.mat")
}

EXPERT_PROBA    = {
    "VIIRS": project_path("data", "stacking", "viirs_optimized_probs.mat"),
    "TIGER": project_path("data", "stacking", "tiger_optimized_probs.mat"),
    "NLCD": project_path("data", "stacking", "nlcd_optimized_probs.mat")
}

STACKED_BASE_PROBS        = project_path("data", "results", "final_stacked_predictions.mat")
STACKED_PASSTHROUGH_PROBS = project_path("data", "results", "final_stacked_passthrough.mat")

STACKING_BASE_KEY        = "Stacking"
STACKING_PASSTHROUGH_KEY = "StackingPassthrough"

MODELS = ("Logistic", "RandomForest", "XGBoost")

EXPERT_TRIALS   = 250 
STACKING_TRIALS = 250 
CS_TRIALS       = 150 
EARLY_STOP      = 25 
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
    buf
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
        with contextlib.redirect_stdout(buf): 
            mean_score, best_params, _, _ = run_nested_cv(
                name=f"{dataset_key}_{model_name}",
                filepath=filepath,
                loader_func=loader_func,
                model_factory=get_factory(model_name),
                param_space=get_param_space(model_name),
                task=CLASSIFICATION,
                outer_config=outer_config,
                inner_config=inner_config,
                n_trials=n_trials,
                random_state=RANDOM_STATE,
                early_stopping_rounds=EARLY_STOP,
                early_stopping_delta=EARLY_STOP_EPS,
                sampler_type="multivariate-tpe"
            )

        best_params = normalize_params(model_name, best_params) 
        save_model_config(CONFIG_PATH, f"{dataset_key}/{model_name}", best_params)
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

def _select_best_model(dataset_key, filepath, loader_func, config): 
    leaderboard = []
    for model_name in MODELS: 
        params = load_model_params(CONFIG_PATH, f"{dataset_key}/{model_name}")
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

def test_expert_optimize(buf: io.StringIO):
    for name in DATASETS:
        base = BASE[name]
        _optimize_dataset(
            name, 
            base["path"],
            base["loader"],
            n_trials=EXPERT_TRIALS,
            buf=buf
        )

def test_expert_oof(buf: io.StringIO):
    config = eval_config() 
    for name in DATASETS: 
        base = BASE[name]
        _, best_model, best_params, metrics = _select_best_model(
            name, 
            base["path"],
            base["loader"],
            config 
        )

        write_model_metrics(buf, f"{name}/{best_model}", metrics)
        _evaluate_model(
            base["path"],
            base["loader"],
            best_model,
            best_params,
            config,
            proba_path=EXPERT_PROBA[name]
        )

    return {
        "experts": {k: EXPERT_PROBA[k] for k in DATASETS}
    }

def test_stacking_optimize(buf: io.StringIO, passthrough: bool = False):
    prob_files = [EXPERT_PROBA[n] for n in DATASETS]
    check_paths_exist(prob_files, "expert OOF files")
    
    key    = STACKING_PASSTHROUGH_KEY if passthrough else STACKING_BASE_KEY 
    loader = passthrough_loader       if passthrough else stacking_loader 

    _optimize_dataset(
        key,
        "virtual",
        loader(prob_files),
        n_trials=STACKING_TRIALS,
        buf=buf
    )

def test_stacking(buf: io.StringIO, passthrough: bool = False): 
    prob_files = [EXPERT_PROBA[n] for n in DATASETS]
    check_paths_exist(prob_files, "expert OOF files")

    key    = STACKING_PASSTHROUGH_KEY if passthrough else STACKING_BASE_KEY 
    loader = passthrough_loader       if passthrough else stacking_loader
    proba  = PROBA_PASSTHROUGH_PATH   if passthrough else PROBA_PATH
    name   = "StackingPassthrough"    if passthrough else "Stacking"

    config = eval_config()
    best_score_val, best_model, best_params, metrics = _select_best_model(
        key,
        "virtual",
        loader(prob_files),
        config
    )

    write_model_metrics(buf, f"{name}/{best_model}", metrics)

    _evaluate_model(
        "virtual",
        loader(prob_files),
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

def test_cs(buf: io.StringIO, passthrough: bool = False): 

    name  = "StackingPassthrough" if passthrough else "Stacking"
    proba = PROBA_PASSTHROUGH_PATH if passthrough else PROBA_PATH

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
            early_stopping_rounds=EARLY_STOP,
            early_stopping_delta=EARLY_STOP_EPS,
            sampler_type="multivariate-tpe",
            random_state=RANDOM_STATE
        )

    save_model_config(CONFIG_PATH, key, best_params)
    write_model_summary(buf, key, best_value)

    metrics, adj_name, P_cs = _evaluate_cs(
        P,
        y,
        train_mask,
        test_mask,
        class_labels,
        W_by_name,
        best_params
    )

    write_model_metrics(buf, f"{key}/{adj_name}", metrics)
    return {
        "params": best_params, 
        "metrics": metrics,
        "metadata": {
            name: f"{key}/{adj_name}",
            "probs": P, 
            "probs_corr": P_cs, 
            "train_mask": train_mask,
            "labels": y, 
            "fips": fips,
            "class_labels": class_labels
        }
    }

TESTS = {
    "expert_opt": test_expert_optimize,
    "expert_oof": test_expert_oof,
    "stacking_opt": test_stacking_optimize,
    "stacking": test_stacking,
    "cs": test_cs
}

def choice_function(name: str, cross: str, buf: io.StringIO): 
    '''
    Runs the appropriate combination of functions depending on passed opt cross 
    '''
    CHOICE_SUBSET = {"stacking", "stacking_opt", "cs"}

    if name not in CHOICE_SUBSET: 
        return False  

    fn = TESTS[name] 

    if cross == "off": 
        fn(buf, False)
    elif cross == "on": 
        fn(buf, True)
    elif cross == "both": 
        fn(buf, False)
        fn(buf, True)
    else: 
        raise ValueError(f"unknown choice for cross-modal feature tests: {cross}")

    return True 

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

        if not choice_function(name, args.cross, buf): 
            fn(buf)

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main()
