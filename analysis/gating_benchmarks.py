#!/usr/bin/env python3 
# 
# gating_benchmakrs.py  Andrew Belles  Dec 23rd, 2025 
# 
# Analysis of Classification Problem for Identifying NCHS from 
# various datasets identified as key for gating layer of HMoE model 
# 

import argparse 

import numpy as np

from analysis.cross_validation import (
    CLASSIFICATION,
    CrossValidator, 
    CVConfig, 
)

from analysis.optimizer import optimize_hyperparameters

from models.estimators import (
    make_rf_classifier,
    make_xgb_classifier,
    make_logistic, 
    make_svm_classifier
)

from preprocessing.loaders import (
    load_viirs_nchs, # also loads tiger dataset for now 
    load_stacking 
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
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        )
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


def run_viirs_nchs_smushed(): 
    
    print("CLASSIFICATION: VIIRS Nighttime Lights classification via NCHS Scheme (2023)")
    print("3-Class Scheme")

    filepath = project_path("data", "datasets", "viirs_nchs_2023_smushed.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        )
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


def run_tiger_nchs_full(): 

    print("CLASSIFICATION: (TIGER 2023) Intersection Density and Road Lengths on NCHS")

    filepath = project_path("data", "datasets", "tiger_nchs_2023.mat") 
    
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


def run_tiger_nchs_smushed(): 

    print("CLASSIFICATION: (TIGER 2023) Intersection Density and Road Lengths on NCHS")
    print("3-Class Scheme")

    filepath = project_path("data", "datasets", "tiger_nchs_2023_smushed.mat") 
    
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


def create_oof_datasets(): 
    print("DATASET: Creates OOF datasets for each best performer per dataset.")

    viirs = project_path("data", "datasets", "viirs_nchs_2023.mat")
    tiger = project_path("data", "datasets", "tiger_nchs_2023.mat")

    loader = lambda fp: load_viirs_nchs(filepath=fp)

    viirs_model = {
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        ),
    }

    tiger_model = {
        "Logistic": make_logistic(),
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    viirs_cv = CrossValidator(
        filepath=viirs, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    tiger_cv = CrossValidator(
        filepath=tiger, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False 
    )

    viirs_cv.run(
        models=viirs_model, 
        config=config,
        oof=True 
    )
    splits = viirs_cv.splits_ 

    tiger_cv.run(
        models=tiger_model, 
        config=config, 
        oof=True, 
        splits=splits
    )

    viirs_cv.save_oof(project_path("data", "stacking", "viirs_2023_oof.mat"))
    tiger_cv.save_oof(project_path("data", "stacking", "tiger_2023_oof.mat"))


def run_stacking(): 

    print("CLASSIFICATION: Stacking Classifier from Layer A Classifier's Probabilities")
    
    viirsstack = project_path("data", "stacking", "viirs_2023_oof.mat") 
    tigerstack = project_path("data", "stacking", "tiger_2023_oof.mat")
    files = [
        viirsstack, 
        # tigerstack
    ]

    def stacking_loader(fp: str):
        _ = fp 
        return load_stacking(files)

    models = {
        "Logistic": make_logistic(), 
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
        filepath=viirsstack, 
        loader=stacking_loader, 
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



def optimize_svm_viirs(): 
    print("OPTIMIZATION: SVM For VIIRS multi-classification for NCHS Urban-Rural Labels (2023)")

    filepath = project_path("data", "datasets", "viirs_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    svm_grid = {
        "C": np.linspace(50.0, 150.0, 10), 
        "kernel": ["rbf"], 
        "gamma": np.linspace(0.01, 0.1, 8) 
    }

    _, best_model_name = optimize_hyperparameters(
        base_name="SVM", 
        filepath=filepath,
        loader_func=loader,
        base_factory_func=make_svm_classifier, 
        param_grid=svm_grid,
        task=CLASSIFICATION, 
    )

    return best_model_name


def run_all(): 

    run_viirs_nchs_full() 
    run_viirs_nchs_smushed()
    run_tiger_nchs_full()
    run_tiger_nchs_smushed()
    create_oof_datasets()
    run_stacking() 
    optimize_svm_viirs()


def main(): 

    TASKS = [
        "viirs", 
        "viirs_smushed", 
        "tiger", 
        "tiger_smushed", 
        "oof", 
        "stacking", 
        "optimize_svm", 
        "all"
    ] 

    task_dict = {
        "viirs": run_viirs_nchs_full, 
        "viirs_smushed": run_viirs_nchs_smushed, 
        "tiger": run_tiger_nchs_full,
        "tiger_smushed": run_tiger_nchs_smushed, 
        "oof": create_oof_datasets,
        "stacking": run_stacking, 
        "optimize_svm": optimize_svm_viirs,  
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
