#!/usr/bin/env python3 
# 
# climate_population_benchmark.py  Andrew Belles  Dec 15th, 2025 
# 
# 
# 
# 

import argparse 
import numpy as np 

from analysis.cross_validation import CrossValidator, CVConfig

from support.helpers import load_climate_population, project_path, load_models_from_yaml

def main(): 
    seed = np.random.randint(0, 2**32 - 1)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--decade", type=int, default=2020)
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--models", type=str, default=project_path("analysis", "model_config.yaml"))
    parser.add_argument("--gpu", action="store_true") 

    args = parser.parse_args()

    config = CVConfig(
        n_splits=args.splits, 
        test_size=args.test_size, 
        split_mode=args.mode, 
        base_seed=seed
    )

    filepath = project_path("data", "climate_population.mat") 

    loader = lambda fp: load_climate_population(
        filepath=fp,
        decade=args.decade,  
        groups=["climate", "coords"]
    )

    models = load_models_from_yaml(args.models)

    cv = CrossValidator(filepath=filepath, loader=loader)

    results = cv.run_repeated(
        config=config, 
        models=models, 
        n_repeats=args.repeats
    )

    summary = cv.summarize_repeated(results)
    cv.format_summary(summary)


if __name__ == "__main__": 
    main()
