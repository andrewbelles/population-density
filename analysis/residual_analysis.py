#!/usr/bin/env python3 
# 
# 
# 
# 
# 
# 

import argparse 

import models.helpers as h 

from models.xgboost_model import XGBoost 
from models.cross_validation import CVConfig, CrossValidator

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    filepath = h.project_path("data", "climate_geospatial.mat")

    loader = lambda fp: h.load_geospatial_climate(fp, target="all", groups=["lat", "lon"])
    
    cv = CrossValidator(filepath=filepath, loader=loader)

    models = {
        "XGBoost": lambda: XGBoost(
            gpu=False, 
            ignore_coords=True,
            random_state=0, 
            early_stopping_rounds=200 
        )
    }

    config = CVConfig(
        n_splits=args.folds, 
        test_size=0.4, 
        split_mode="kfold", 
        base_seed=0 
    )

    mat = h.loadmat(filepath) 
    output_names = [str(s) for s in h._mat_str_vector(mat["feature_names"]).tolist()]

    _ = cv.run_repeated(
        models=models, 
        config=config, 
        n_repeats=1, 
        collect=True, 
        output_names=output_names
    )

    out_path = args.out 
    if out_path is None: 
        coords_tag = "-".join(["lat", "lon"])
        out_path = h.project_path(
            "data", "models", "raw", f"residuals_from_{coords_tag}_kfold_f{args.folds}.mat"
        )

    cv.save_residuals_dataset(out_path, model="XGBoost", reducer="mean")
    print(f"> Wrote: {out_path}")

    config2 = CVConfig(
        n_splits=args.folds, 
        test_size=0.4, 
        split_mode="random", 
        base_seed=0
    )

    resid_loader = lambda fp: h.load_residual_dataset(fp)
    cv2 = CrossValidator(filepath=out_path, loader=resid_loader)

    resid_models = {
            "XGBoost": lambda: XGBoost(
                gpu=False, 
                ignore_coords=True, 
                random_state=0, 
                early_stopping_rounds=200
            )
    } 

    resid_results = cv2.run_repeated(
        models=resid_models, 
        config=config2, 
        n_repeats=args.repeats, 
        collect=False 
    )

    resid_summary = cv2.summarize_repeated(resid_results)
    cv2.format_summary(resid_summary)

if __name__ == "__main__": 
    main()
