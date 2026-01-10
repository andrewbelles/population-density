#!/usr/bin/env python3 
# 
# round_robin.py  Andrew Belles  Jan 7th, 2025 
# 
# One versus Rest Tests per County Label for three major classifier datasets (VIIRS, TIGER, NLCD)
# Outputs clear table showing best performers and their model metrics. Saves best params 
# 

import argparse, contextlib, io, sys
from pathlib import Path

from analysis.cross_validation import (
    CrossValidator,
    CVConfig, 
    CLASSIFICATION,
    TaskSpec 
)

from analysis.hyperparameter import (
    CorrectAndSmoothEvaluator,
    run_optimization
)

from analysis.hyperparameter import run_nested_cv

from testbench.utils.paths import (
    ROUND_ROBIN_PROBA,
    ROUND_EXPERT_PROBA,
    ROUND_ROBIN_CONFIG,
    ROUND_ROBIN_OVR_PROBA,
    CONFIG_PATH,
    check_paths_exist
)

from testbench.utils.config import (
    eval_config,
    load_model_params, 
    normalize_params
) 

from testbench.utils.etc import (
    get_factory,
    get_param_space,
    format_cell,
    render_table,
    write_model_summary
)

from testbench.utils.metrics import (
    metrics_from_summary,
    rank_by_label
)

from testbench.utils.oof import load_probs_labels_fips

from testbench.utils.graph import build_cs_adjacencies

import testbench.stacking as stacking 

from testbench.utils.data import (
    load_dataset,
    make_binary_loader,
    DATASETS,
    BASE,
    stacking_loader 
)

from utils.helpers import (
    load_yaml_config, 
    save_model_config,
    make_train_mask
) 

# Silence-able logging 
def _log(msg, quiet=False): 
    if not quiet: 
        print(msg)

# ---------------------------------------------------------
# Global Variables 
# ---------------------------------------------------------

RR_STACK_KEY   = "RoundRobinStacking"
RR_CS_KEY      = f"CorrectAndSmooth/{RR_STACK_KEY}"
MODELS         = ("Logistic", "RandomForest", "XGBoost")
OPT_TASK       = TaskSpec("classification", ("accuracy",))
TRIALS         = 200 
EARLY_STOP     = 40
EARLY_STOP_EPS = 1e-4 
RANDOM_STATE   = 0
LABELS         = tuple(range(6))

# ---------------------------------------------------------
# Test Helpers  
# ---------------------------------------------------------

def _evaluate_model(
    filepath, 
    loader_func, 
    model_name, 
    params, 
    config,
    proba=None 
): 
    model = get_factory(model_name)(**params)
    cv    = CrossValidator(
        filepath=filepath,
        loader=loader_func,
        task=CLASSIFICATION,
        scale_y=False 
    )
    results = cv.run(models={"model": model}, config=config, oof=proba is not None)
    summary = cv.summarize(results)
    if proba: 
        cv.save_oof(proba)
    return metrics_from_summary(summary)

def _optimize_single(
    dataset_key, 
    data, 
    label, 
    n_trials
): 
    loader_func  = make_binary_loader(data, label)
    outer_config = eval_config(RANDOM_STATE)
    inner_config = CVConfig(
        n_splits=3,
        n_repeats=1,
        stratify=True,
        random_state=RANDOM_STATE 
    )

    best = None 

    for model_name in MODELS: 
        with contextlib.redirect_stdout(sys.stderr): 
            mean_score, best_params, _, _ = run_nested_cv(
                name=f"{dataset_key}_label{label}_{model_name}",
                filepath="virtual",
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
                sampler_type="multivariate-tpe",
                metric="accuracy"
            )

        best_params = normalize_params(model_name, best_params)
        metrics     = _evaluate_model(
            "virtual",
            loader_func, 
            model_name, 
            best_params, 
            outer_config
        )

        candidate = {
            "dataset": dataset_key, 
            "label": label, 
            "model": model_name, 
            "score": metrics.get("accuracy", float("-inf")),
            "metrics": metrics, 
            "params": best_params, 
            "mean_score": mean_score 
        }

        if best is None or candidate["score"] > best["score"]: 
            best = candidate 

    return best 

def _select_best_multiclass(filepath, dataset_key, loader_func, config): 
    best = None 
    for model_name in MODELS: 
        params  = load_model_params(CONFIG_PATH, f"{dataset_key}/{model_name}")
        params  = normalize_params(model_name, params)
        metrics = _evaluate_model(
            filepath,
            loader_func, 
            model_name, 
            params, 
            config
        ) 
        score   = metrics.get("accuracy", float("-inf"))
        if best is None or score > best["score"]: 
            best = {"model": model_name, "params": params, "score": score}
    if best is None: 
        raise ValueError(f"failure to select best multiclass model for {dataset_key}")
    return best 

def _load_round_robin_entries(): 
    cfg    = load_yaml_config(Path(ROUND_ROBIN_CONFIG))
    models = cfg.get("models", {})
    return {
        i: models.get(f"label_{i}")
        for i in range(1,7)
    }

def _compute_stacking_proba_files(quiet: bool = False):
    base_oof_paths = []
    for name in DATASETS: 
        base   = BASE[name]
        config = eval_config(RANDOM_STATE)
        best   = _select_best_multiclass(
            base["path"],
            name, 
            base["loader"], 
            config
        )
        _log(f"[{name}] best model: {best['model']}", quiet=quiet)
        _evaluate_model(
            base["path"],
            base["loader"],
            best["model"],
            best["params"],
            config,
            proba=ROUND_EXPERT_PROBA[name]
        )
        base_oof_paths.append(ROUND_EXPERT_PROBA[name])

    rr_entries    = _load_round_robin_entries()
    ovr_oof_paths = []
    for label_value in range(6): 
        label_id = label_value + 1 
        entry = rr_entries.get(label_id)
        if entry is None: 
            raise ValueError(f"missing round robin entry for label_{label_value + 1}")

        dataset_key = entry["dataset"]
        model_name  = entry["model"]
        params      = normalize_params(model_name, entry["params"])
        data        = load_dataset(dataset_key)
        loader      = make_binary_loader(data, label_value)

        _log(f"[label {label_value + 1}] {dataset_key}/{model_name}", quiet=quiet)
        _evaluate_model(
            "virtual",
            loader, 
            model_name,
            params,
            eval_config(RANDOM_STATE), 
            proba=ROUND_ROBIN_OVR_PROBA[label_id]
        )
        ovr_oof_paths.append(ROUND_ROBIN_OVR_PROBA[label_id])

    return base_oof_paths + ovr_oof_paths

# ---------------------------------------------------------
# Tests   
# ---------------------------------------------------------

def test_round_robin(buf: io.StringIO, n_trials: int, quiet: bool = False): 
    data_cache = {name: load_dataset(name) for name in DATASETS}
    results    = []

    for i, dataset_key in enumerate(DATASETS): 
        data = data_cache[dataset_key]
        for j, label in enumerate(LABELS): 
            _log(f"[{dataset_key}/{label}] starting test {i * 3 + j}...", quiet=quiet)
            results.append(_optimize_single(dataset_key, data, label, n_trials))

    by_label = rank_by_label(results, LABELS)

    header = ["Rank"] + [str(l + 1) for l in LABELS]
    rows   = []
    rank_labels = ("1st", "2nd", "3rd")
    for i, rank in enumerate(rank_labels): 
        row = [rank]
        for label in LABELS: 
            items = by_label[label]
            best  = items[0]
            meta  = {
                "dataset": best["dataset"],
                "model": best["model"],
                "params": best["params"]
            }
            save_model_config(ROUND_ROBIN_CONFIG, f"label_{label + 1}", meta)
            row.append(format_cell(items[i]) if i < len(items) else "-")
        rows.append(row)

    render_table(header, rows)
    return by_label


def test_round_robin_stacking(buf: io.StringIO, n_trials: int, quiet: bool = False): 
    prob_files = (
        [ROUND_EXPERT_PROBA[n] for n in DATASETS] + 
        [ROUND_ROBIN_OVR_PROBA[i] for i in range(1, 7)]
    ) 

    check_paths_exist(prob_files + [ROUND_ROBIN_PROBA], "round robin stacking probs")

    P, y, fips, class_labels = load_probs_labels_fips(ROUND_ROBIN_PROBA)
    return {
        "metadata": {
            "name": RR_STACK_KEY,
            "probs": P,
            "labels": y,
            "fips": fips,
            "class_labels": class_labels
        }
    }


def test_round_robin_stacking_opt(buf: io.StringIO, n_trials: int, quiet: bool = False): 
    prob_files = _compute_stacking_proba_files(quiet)
    loader     = stacking_loader(prob_files)

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
            _, best_params, _, _ = run_nested_cv(
                name=f"{RR_STACK_KEY}_{model_name}",
                filepath="virtual",
                loader_func=loader,
                model_factory=get_factory(model_name),
                param_space=get_param_space(model_name),
                task=OPT_TASK,
                outer_config=outer_config,
                inner_config=inner_config,
                n_trials=n_trials,
                random_state=RANDOM_STATE,
                early_stopping_rounds=EARLY_STOP,
                early_stopping_delta=EARLY_STOP_EPS,
                sampler_type="multivariate-tpe",
                metric="accuracy"
            ) 

        best_params = normalize_params(model_name, best_params)
        metrics = _evaluate_model(
            "virtual",
            loader, 
            model_name,
            best_params, 
            outer_config,
            proba=None 
        )
        leaderboard.append((metrics.get("accuracy"), model_name, best_params, metrics))

    leaderboard.sort(key=lambda x: x[0], reverse=True)
    best_score, best_model, best_params, metrics = leaderboard[0]

    write_model_summary(buf, f"{RR_STACK_KEY}/{best_model}", best_score)
    save_model_config(CONFIG_PATH, f"{RR_STACK_KEY}/{best_model}", best_params)

    _evaluate_model(
        "virtual",
        loader,
        best_model,
        best_params,
        outer_config,
        proba=ROUND_ROBIN_PROBA
    )

    P, y, fips, class_labels = load_probs_labels_fips(ROUND_ROBIN_PROBA)

    return {
        "best_model": best_model,
        "best_score": best_score, 
        "best_params": best_params,
        "metrics": metrics, 
        "oof_inputs": prob_files,
        "metadata": {
            "name": RR_STACK_KEY,
            "probs": P,
            "labels": y,
            "fips": fips,
            "class_labels": class_labels
        }
    }


def test_round_robin_cs(buf: io.StringIO, n_trials: int, quiet: bool = False):
    P, y, fips, class_labels = load_probs_labels_fips(ROUND_ROBIN_PROBA)
    train_mask = make_train_mask(y, train_size=0.3, random_state=RANDOM_STATE, stratify=True)
    test_mask = ~train_mask

    W_by_name = build_cs_adjacencies(ROUND_ROBIN_PROBA, fips, normalize=True)
    evaluator = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name=W_by_name,
        y_train=y,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels
    )

    best_params, best_value = run_optimization(
        name=RR_CS_KEY,
        evaluator=evaluator,
        n_trials=n_trials,
        direction="maximize",
        early_stopping_rounds=EARLY_STOP,
        early_stopping_delta=EARLY_STOP_EPS,
        sampler_type="multivariate-tpe",
        random_state=RANDOM_STATE
    )

    save_model_config(CONFIG_PATH, RR_CS_KEY, best_params)
    write_model_summary(buf, f"{RR_CS_KEY} (C+S opt)", best_value)

    _, adj_name, P_cs = stacking._evaluate_cs(
        P, y, train_mask, test_mask, class_labels, W_by_name, best_params
    )

    return {
        "metadata": {
            "name": f"{RR_CS_KEY}/{adj_name}",
            "probs": P,
            "probs_corr": P_cs,
            "train_mask": train_mask,
            "labels": y,
            "fips": fips,
            "class_labels": class_labels
        }
    }

TESTS = {
    "round_robin_opt": test_round_robin,
    "stacking": test_round_robin_stacking,
    "stacking_opt": test_round_robin_stacking_opt,
    "cs_opt": test_round_robin_cs
}

# ---------------------------------------------------------
# Main Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--no-opt", action="store_true")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    for name in targets: 
        if args.no_opt and name.endswith("_opt"): 
            continue 
        fn = TESTS.get(name)
        if fn is None: 
            raise ValueError(f"unknown test: {name}")
        fn(buf, args.trials, args.quiet)

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
