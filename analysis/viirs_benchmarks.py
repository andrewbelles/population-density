#!/usr/bin/env python3 
# 
# viirs_benchmakrs.py  Andrew Belles  Dec 23rd, 2025 
# 
# Analysis of Classification Problem for Identifying NCHS 
# Urban-Rural Code for a given county using the VIIRS Nighttime 
# Lights Dataset 
# 

import argparse 

import numpy as np

from analysis.cross_validation import (
    CLASSIFICATION,
    CrossValidator, 
    CVConfig, 
)

# from analysis.optimizer import optimize_hyperparameters

from models.estimators import (
    make_rf_classifier,
    make_xgb_classifier,
    make_logistic, 
    make_svm_classifier
)

from preprocessing.loaders import (
    load_viirs_nchs,
)

REPEATS = 10

from support.helpers import project_path

def run_viirs_nchs_full(): 
    
    print("CLASSIFICATION: VIIRS Nighttime Lights classification via NCHS Scheme (2023)")

    filepath = project_path("data", "datasets", "viirs_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_all(): 

    run_viirs_nchs_full() 


def main(): 

    TASKS = [
        "viirs", 
        "all"
    ] 

    task_dict = {
        "viirs": run_viirs_nchs_full, 
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
