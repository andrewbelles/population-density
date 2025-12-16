#!/usr/bin/env python3 
# 
# cross_validation.py  Andrew Belles  Dec 14th, 2025 
# 
# Interface which models can register with for intelligent folding of test data 
# and flexible number of repeats to gain understanding about models' efficacy and stability 
# 

import argparse  

import numpy as np 
import pandas as pd
import models.helpers as h

from typing import Any, Callable, Iterator, Literal, Mapping  
 
from sklearn.metrics import mean_squared_error, r2_score 

from models.linear_model import LinearModel
from models.random_forest_model import RandomForest 
from models.xgboost_model import XGBoost
from models.gp_xgboost_model import GPBoost

from dataclasses import dataclass 

ModelFactory = Callable[[], h.ModelInterface]

@dataclass(frozen=True)
class CVConfig: 
    n_splits: int = 5
    test_size: float = 0.4
    split_mode: Literal["kfold", "random"] = "kfold"
    val_size: float = 0.3 
    base_seed: int = 0 


def _iter_split(n_samples: int, *, n_splits: int, test_size: float, 
                seed: int, mode: Literal["kfold", "random"]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    
    if mode == "kfold": 
        folds = h.kfold_indices(n_samples, n_folds=n_splits, seed=seed) 
        for split_i, (train_idx, test_idx) in enumerate(folds): 
            yield split_i, train_idx, test_idx 
        return 

    if mode == "random": 
        for split_i in range(n_splits): 
            train_idx, test_idx = h.split_indices(n_samples, test_size, seed + split_i) 
            yield split_i, train_idx, test_idx 
        return 


class CrossValidator: 

    def __init__(self, *, filepath: str, loader: h.DatasetLoader): 

        self.filepath = filepath 
        self.data     = loader(filepath) 

    def run(self, *, models: Mapping[str, ModelFactory], config: CVConfig, seed: int) -> pd.DataFrame: 
        X = np.asarray(self.data["features"], dtype=np.float64) 
        y = np.asarray(self.data["labels"], dtype=np.float64).reshape(-1)
        coords = np.asarray(self.data["coords"])

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")
        if coords.shape[0] != y.shape[0]:
            raise ValueError(f"coords rows ({coords.shape[0]}) != y rows ({y.shape[0]})")

        results: list[dict[str, Any]] = []

        for model_name, make_model in models.items():
            for split_i, train_idx, test_idx in _iter_split(
                len(X), n_splits=config.n_splits, 
                test_size=config.test_size, seed=seed, 
                mode=config.split_mode): 

                X_train_raw, X_test_raw   = X[train_idx], X[test_idx] 
                y_train_raw, y_test_raw   = y[train_idx], y[test_idx] 
                coords_train, coords_test = coords[train_idx], coords[test_idx]

                X_scaler, y_scaler = h.fit_scaler(X_train_raw, y_train_raw) 

                X_train, y_train = h.transform_with_scalers(X_train_raw, y_train_raw, X_scaler, y_scaler)
                X_test, y_test   = h.transform_with_scalers(X_test_raw, y_test_raw, X_scaler, y_scaler)

                model = make_model() 

                try: 
                    y_hat_scaled = model.fit_and_predict(
                        (X_train, X_test), 
                        (y_train, y_test), 
                        (coords_train, coords_test), 
                        seed=seed + split_i, 
                        val_size=config.val_size, 
                        y_scaler=y_scaler 
                    )

                    y_hat  = y_scaler.inverse_transform(np.asarray(y_hat_scaled).reshape(-1, 1)).ravel() 
                    y_true = y_scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1)).ravel()  

                    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
                    r2   = float(r2_score(y_true, y_hat))

                    y_train_mean = float(np.mean(y_train_raw))
                    y_base       = np.full_like(y_true, y_train_mean, dtype=np.float64)

                    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, y_base)))
                    baseline_r2   = float(r2_score(y_true, y_base))

                    win_rmse   = float(rmse < baseline_rmse) 
                    win_r2     = float(r2 > baseline_r2)
                    win_r2_gt0 = float(r2 > 0.0)

                    results.append({
                        "model": model_name, 
                        "fold": split_i, 
                        "r2": r2,
                        "rmse": rmse, 
                        "baseline_r2": baseline_r2, 
                        "baseline_rmse": baseline_rmse, 
                        "win_r2": win_r2, 
                        "win_rmse": win_rmse, 
                        "win_r2_gt0": win_r2_gt0, 
                        "n_train": len(X_train), 
                        "n_test": len(X_test), 
                        "seed": int(seed)
                    })

                except Exception as e: 
                    print(f"    Error in fold {split_i + 1}: {e}")
                    results.append({
                          "model": model_name,
                          "split": split_i,
                          "r2": np.nan,
                          "rmse": np.nan,
                          "baseline_r2": np.nan,
                          "baseline_rmse": np.nan,
                          "win_r2": np.nan,
                          "win_rmse": np.nan,
                          "win_r2_gt0": np.nan,
                          "n_train": int(len(train_idx)),
                          "n_test": int(len(test_idx)),
                          "seed": int(seed),
                          "error": str(e),
                      })

        return pd.DataFrame(results)

    def run_repeated(self, *, models: Mapping[str, ModelFactory], 
                     config: CVConfig, n_repeats: int) -> pd.DataFrame: 

        all_results: list[pd.DataFrame] = []

        for repeat_i in range(n_repeats):
            print(f"\n> Running CV Repeat {repeat_i + 1}/{n_repeats}")
            
            repeat_seed    = config.base_seed + repeat_i * 2
            repeat_results = self.run(models=models, config=config, seed=repeat_seed) 
            repeat_results["repeat"] = repeat_i 

            all_results.append(repeat_results, )
        return pd.concat(all_results, ignore_index=True)

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame: 
        summary = results_df.groupby("model").agg({
            "r2":   ["mean", "median", "std", "min", "max"], 
            "rmse": ["mean", "median", "std", "min", "max"],
            "fold": "count"
        }).round(4)

        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns] 
        summary = summary.reset_index()

        return summary 

    def summarize_repeated(self, results_df: pd.DataFrame) -> pd.DataFrame: 

        repeat_summary = results_df.groupby(["model", "repeat"]).agg({
            "r2": "mean", 
            "rmse": "mean",
            "baseline_r2": "mean", 
            "baseline_rmse": "mean", 
            "win_r2": "mean", 
            "win_rmse": "mean", 
            "win_r2_gt0": "mean" 
        }).reset_index()

        final_summary = repeat_summary.groupby("model").agg({
            "r2":   ["mean", "median", "std", "min", "max"], 
            "rmse": ["mean", "median", "std", "min", "max"],
            "baseline_r2": ["mean", "median", "std", "min", "max"], 
            "baseline_rmse": ["mean", "median", "std", "min", "max"], 
            "win_r2": ["mean", "median", "std", "min", "max"], 
            "win_rmse": ["mean", "median", "std", "min", "max"], 
            "win_r2_gt0": ["mean", "median", "std", "min", "max"] 
        }).round(4)

        final_summary.columns = [f"{metric}_{stat}" for metric, stat in final_summary.columns]
        final_summary = final_summary.reset_index()  

        n_repeats = int(repeat_summary["repeat"].nunique())
        r2_half   = 1.96 * final_summary["r2_std"] / np.sqrt(n_repeats)
        rmse_half = 1.96 * final_summary["rmse_std"] / np.sqrt(n_repeats) 

        final_summary["r2_ci_lower"] = (final_summary["r2_mean"] - r2_half).round(4)
        final_summary["r2_ci_upper"] = (final_summary["r2_mean"] + r2_half).round(4)
        final_summary["rmse_ci_lower"] = (final_summary["rmse_mean"] - rmse_half).round(4)
        final_summary["rmse_ci_upper"] = (final_summary["rmse_mean"] + rmse_half).round(4)
        
        return final_summary

    @staticmethod 
    def format_summary(summary_df: pd.DataFrame): 

        print("===========================================================================================")
        print("================             Repeated Cross-Validation Summary             ================")
        print("===========================================================================================")

        has_win_rmse = "win_rmse_mean" in summary_df.columns
        has_win_r2gt0 = "win_r2_gt0_mean" in summary_df.columns

        print(
            f"{'Model':<15} {'r2 Score':<25} {'RMSE':<30} {'WinRMSE':<10} {'P(R2>0)':<10}"
        )
        print("===========================================================================================") 

        for _, row in summary_df.iterrows():
            model = row["model"]
            r2_mean = row["r2_mean"]
            r2_ci_lower = row["r2_ci_lower"]
            r2_ci_upper = row["r2_ci_upper"]

            rmse_mean = row["rmse_mean"]
            rmse_ci_lower = row["rmse_ci_lower"]
            rmse_ci_upper = row["rmse_ci_upper"]

            r2_str   = f"{r2_mean:+.3f} [{r2_ci_lower:+.3f}, {r2_ci_upper:+.3f}]"
            rmse_str = f"{rmse_mean:+.3f} [{rmse_ci_lower:+.3f}, {rmse_ci_upper:+.3f}]"


            win_rmse_str = ""
            win_r2gt0_str = ""

            if has_win_rmse:
                win_rmse_mean = float(row["win_rmse_mean"])
                win_rmse_str = f"{100.0 * win_rmse_mean:5.1f}%"

            if has_win_r2gt0:
                win_r2gt0_mean = float(row["win_r2_gt0_mean"])
                win_r2gt0_str = f"{100.0 * win_r2gt0_mean:5.1f}%"

            print(
                f"{model:<15} {r2_str:<25} {rmse_str:<30} {win_rmse_str:<10} {win_r2gt0_str:<10}"
            )


def main(): 

    parser = argparse.ArgumentParser() 
    parser.add_argument("--decade", type=int, default=2020) 
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "linear", "gpxgb"])
    parser.add_argument("--gpu", action="store_true")
    
    args = parser.parse_args()

    filepath = h.project_path("data", "climate_population.mat")
    loader   = lambda fp: h.load_climate_population(fp, decade=args.decade, groups=["climate", "coords"])

    cv = CrossValidator(filepath=filepath, loader=loader)  

    models = {}

    if "rf" in args.models: 
        models["RandomForest"] = lambda: RandomForest(n_estimators=500, random_state=args.seed)
    if "xgb" in args.models: 
        models["XGBoost"] = lambda: XGBoost(
            gpu=args.gpu, 
            ignore_coords=False, 
            random_state=args.seed, 
            early_stopping_rounds=200
        )
    if "linear" in args.models: 
        models["Linear"] = lambda: LinearModel(gpu=args.gpu)
    if "gpxgb" in args.models: 
        models["Gaussian-Process + XGBoost"] = lambda: GPBoost(
            gpu=args.gpu, 
            gp_params={
                "n_inducing": 512, 
                "steps": 500, 
                "lr": 0.001, 
                "batch_size": 1024, 
                "include_coords_in_booster": True  
            }, 
            n_estimators=500, 
            max_depth=8, 
            subsample=0.8, 
            colsample_bynode=0.8, 
            random_state=args.seed 
        )

    config     = CVConfig(n_splits=args.folds, test_size=0.4, split_mode="random", base_seed=args.seed)
    results_df = cv.run_repeated(models=models, config=config, n_repeats=args.repeats)
    summary_df = cv.summarize_repeated(results_df)

    CrossValidator.format_summary(summary_df)
    
    raw_dir = h.project_path("data", "models", "raw")
    results_csv = h.project_path(raw_dir, f"cv_results_decade_{args.decade}_f{args.folds}_r{args.repeats}.csv")
    summary_csv = h.project_path(raw_dir, f"cv_summary_decade_{args.decade}_f{args.folds}_r{args.repeats}.csv")

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

if __name__ == "__main__": 
    main() 
