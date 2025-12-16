#!/usr/bin/env python
# 
# climate_geospatial_benchmark.py  Andrew Belles  Dec 16th, 2025 
# 

import argparse 
import numpy as np 

from analysis.cross_validation import CrossValidator, CVConfig

import support.helpers as sh

def main(): 
    seed = np.random.randint(0, 2**32 - 1)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--mode", type=str, default="random")
    parser.add_argument("--models", type=str, default=sh.project_path("analysis", "model_config.yaml"))
    parser.add_argument("--gpu", action="store_true") 

    args = parser.parse_args()

    config = CVConfig(
        n_splits=args.splits, 
        test_size=args.test_size, 
        split_mode=args.mode, 
        base_seed=seed
    )

    filepath = sh.project_path("data", "climate_geospatial.mat") 

    coord_to_climate_loader = lambda fp: sh.load_geospatial_climate(
        filepath=fp,
        target="all", 
        groups=["lat", "lon"]
    )

    models = sh.load_models_from_yaml(args.models)

    climate_to_coord_loader = lambda fp: sh.load_climate_geospatial(
        filepath=fp, 
        target="all",
        groups=["degree_days", "palmer_indices"]
    )

    climate_to_coord_cv = CrossValidator(
        filepath=filepath, 
        loader=climate_to_coord_loader
    )

    results = climate_to_coord_cv.run_repeated(
        config=config, 
        models=models, 
        n_repeats=args.repeats,
        collect=False, 
    )

    climate_to_coord_cv.format_summary(
        climate_to_coord_cv.summarize_repeated(results)
    )

    coord_to_climate_cv = CrossValidator(filepath=filepath, loader=coord_to_climate_loader)

    results = coord_to_climate_cv.run_repeated(
        config=config, 
        models=models, 
        n_repeats=args.repeats,
        collect=False, 
    )

    coord_to_climate_cv.format_summary(
        coord_to_climate_cv.summarize_repeated(results)
    )



if __name__ == "__main__": 
    main()
