#!/usr/bin/env python3 
# 
# optimizer.py  Andrew Belles  Dec 21st, 2025 
# 
# Grid Search optimization for NON NEURAL NETWORK BASED MODELS 
# For use in dataset benchmark files  
# 

import numpy as np

from sklearn.model_selection import ParameterGrid 

from analysis.cross_validation import (
    CrossValidator, 
    CVConfig,
    TaskSpec, 
)

from typing import Callable

def optimize_hyperparameters(
    base_name: str,
    *, 
    filepath: str, 
    loader_func,
    base_factory_func, 
    param_grid: dict, 
    task: TaskSpec, 
    transform: tuple[Callable, Callable | None] | None = None
): 
    grid = ParameterGrid(param_grid)
    models_to_test = {}
    transforms = {}

    print(f"> Generating {len(grid)} configurations for {base_name}...")

    for i, params in enumerate(grid): 

        param_str = "_".join([f"{k}={v}" for k, v in params.items()])
        model_key = f"{base_name}_{i}|{param_str}"

        models_to_test[model_key] =  base_factory_func(**params)
        transforms[model_key] = transform

    config = CVConfig(
        n_splits=5, 
        n_repeats=3, 
        random_state=0
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader_func, 
        task=task,
    )
    results = cv.run(
        models=models_to_test, 
        config=config,
        label_transforms=transforms
    )

    summary = cv.summarize(results)

    if "r2_mean" in summary.columns: 
        best_row = summary.loc[summary["r2_mean"].idxmax()]
        metric_val  = best_row["r2_mean"]
        metric_name = "r2"
    else: 
        raise ValueError 

    print("\n> Optimization Results:")
    print(f"Best Configuration: {best_row['model']}")
    print(f"Best {metric_name}: {metric_val:.4f}")

    return summary, best_row['model']
