#!/usr/bin/env python
# 
# climate_geospatial_benchmark.py  Andrew Belles  Dec 16th, 2025 
# 
# Flexible testbench for running any models/datasets.  
# 
# 

import argparse 

from analysis.cross_validation import (
    CrossValidator, 
    CVConfig, 
    TaskSpec,
    REGRESSION,
)

from models.estimators import (
    make_linear,
    make_rf_regressor,
    make_rf_classifier,
    make_xgb_regressor,
    make_xgb_classifier,
    make_logistic
)

import support.helpers as h


def run_geospatial_from_climate_regression():

    print("REGRESSION: Climate -> Coordinates")

    filepath = h.project_path("data", "climate_geospatial.mat")

    loader = lambda fp: h.load_climate_geospatial(
        fp, target="all", groups=["degree_days", "palmer_indices"]
    )

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=5, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_climate_from_geospatial_regression(): 

    print("REGRESSION: Coordinates -> Climate")

    filepath = h.project_path("data", "climate_geospatial.mat")

    loader = lambda fp: h.load_geospatial_climate(
        fp, target="all", groups=["lat", "lon"]
    )

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=5, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 



def run_contrastive_raw_climate_representation(): 

    print("CLASSIFICATION: Contrastive Pairs on Raw Climate Representation")

    filepath = h.project_path("data", "climate_norepr_contrastive.mat")

    loader = h.load_contrastive_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=5, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_contrastive_pca_compact_climate_representation(): 

    print("CLASSIFICATION: Contrastive Pairs on Compact PCA Representation (95% threshold)")

    filepath = h.project_path("data", "climate_pca_contrastive.mat")

    loader = h.load_contrastive_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=5, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_contrastive_kernel_pca_compact_climate_representation(): 

    print("CLASSIFICATION: Contrastive Pairs on Compact KernelPCA Representation (95% threshold)")

    filepath = h.project_path("data", "climate_kpca_contrastive.mat")

    loader = h.load_contrastive_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=5, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 

def run_all(): 

    run_climate_from_geospatial_regression() 
    run_geospatial_from_climate_regression()
    run_contrastive_raw_climate_representation() 
    run_contrastive_pca_compact_climate_representation() 
    run_contrastive_kernel_pca_compact_climate_representation()


def main(): 

    TASKS = ["coords_to_climate", "climate_to_coords", "norepr", "pca", "kpca", "all"] 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--task", choices=TASKS, default="all")
    args = parser.parse_args()    

    choice_dict = {
        "coords_to_climate": run_climate_from_geospatial_regression, 
        "climate_to_coords": run_geospatial_from_climate_regression, 
        "norepr": run_contrastive_raw_climate_representation, 
        "pca": run_contrastive_pca_compact_climate_representation, 
        "kpca": run_contrastive_kernel_pca_compact_climate_representation, 
        "all": run_all 
    }

    fn = choice_dict.get(args.task)
    if fn is None: 
        raise KeyError(f"invalid task: {args.task}")
    fn() 


if __name__ == "__main__": 
    main()
