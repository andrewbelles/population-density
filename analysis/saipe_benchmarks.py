#!/usr/bin/env python
# 
# saipe_benchmark.py  Andrew Belles  Dec 16th, 2025 
# 
# Testbenches for datasets derived from SAIPE Poverty Datasets 
# 
# 

REPEATS=15

import argparse 
import numpy as np

from analysis.cross_validation import (
    # CLASSIFICATION,
    CrossValidator, 
    CVConfig, 
    # TaskSpec,
    REGRESSION,
)

from analysis.optimizer import optimize_hyperparameters

from models.estimators import (
    make_gcn_regressor,
    make_linear,
    make_rf_regressor,
    # make_rf_classifier,
    make_xgb_regressor,
    # make_xgb_classifier,
    # make_logistic
)

import support.helpers as h

from preprocessing.loaders import (
    # DatasetLoader,
    load_saipe_population,
)


def run_full_against_population(): 

    print("\nREGRESSION: SAIPE -> Population (decade: 2020)")

    filepath = h.project_path("data", "datasets", "saipe_population.mat")

    loader = lambda fp: load_saipe_population(
        filepath=fp, 
        decade=2020, 
        groups=["all"]
    )

    models = {
        "Linear": make_linear(),
        # "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor(
            learning_rate=0.01,
            max_depth=6,
            n_estimators=500,
            subsample=0.8
        ), 
        # "GraphNN": make_gcn_regressor(hidden_dims=(64,64))
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    transforms = {
        # "RandomForest": (np.log1p, np.expm1), 
        "XGBoost": (np.log1p, np.expm1), 
        "Linear": (None, None), 
        "GraphNN": (None, None)
    }

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=REGRESSION,
    )

    results = cv.run(
        models=models, 
        config=config, 
        label_transforms=transforms
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_full_optimize_xgboost(): 

    print("\nREGRESSION: SAIPE -> Population (decade: 2020)")

    filepath = h.project_path("data", "datasets", "saipe_population.mat")

    loader = lambda fp: load_saipe_population(fp, decade=2020, groups=["all"])

    xgb_grid = {
        "n_estimators": 50 * [300, 350, 400, 450, 500, 550, 600], 
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10], 
        "learning_rate": [0.01, 0.1], 
        "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }

    summary, best_model_name = optimize_hyperparameters(
        base_name="XGB",
        filepath=filepath, 
        loader_func=loader, 
        base_factory_func=make_xgb_regressor,
        param_grid=xgb_grid,
        task=REGRESSION,
        transform=(np.log1p, np.expm1)
    )

    return summary, best_model_name


def run_all(): 
    print("SAIPE STORES: (Poverty Rate/County, Median Income, Under 18 in Poverty)")

    run_full_against_population() 
    run_full_optimize_xgboost()


def main(): 

    TASKS = [
        "full", 
        "full_optimize_xgboost"
    ] 

    task_dict = {
        "full": run_full_against_population, 
        "full_optimize_xgboost": run_full_optimize_xgboost, 
        "all": run_all 
    }

    parser = argparse.ArgumentParser() 
    parser.add_argument("--task", choices=TASKS, nargs="+", default=["all"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()    

    if args.list: 
        print(f"Available tasks: {task_dict.keys()}")
        return 

    tasks_to_run = args.task if isinstance(args.task, list) else [args.task]

    for task in tasks_to_run: 
        if task == "all": 
            run_all() 
            break 
        else: 
            fn = task_dict.get(task)
            if fn is None: 
                raise KeyError(f"invalid task: {task}")
            fn() 


if __name__ == "__main__": 
    main()
