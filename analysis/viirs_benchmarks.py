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
    # DatasetLoader,
    load_viirs_nchs,
)


def run_all(): 
    pass 


def main(): 

    TASKS = [
    ] 

    task_dict = {
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
