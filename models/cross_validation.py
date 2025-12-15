#!/usr/bin/env python3 
# 
# cross_validation.py  Andrew Belles  Dec 14th, 2025 
# 
# 

import argparse 
import numpy as np 
import pandas as pd
import helpers as h

from scipy.io import loadmat 

from typing import Dict  
from numpy.typing import NDArray 

from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
 
from sklearn.metrics import mean_squared_error, r2_score 

# from gnn_models import ClimateGNN

'''
Providing an Interface for which a model can quickly be wrapped over to allow for cross validation 
'''

def _unwrap_scalar(v): 
    return v[0] if isinstance(v, tuple) and len(v) == 1 else v 

class ModelInterface(ABC):

    @abstractmethod 
    def fit_and_predict(self, 
                        features: tuple[NDArray[np.float64], NDArray[np.float64]], 
                        labels: tuple[NDArray[np.float64], NDArray[np.float64]], 
                        coords: tuple[NDArray[np.float64], NDArray[np.float64]],
                        **kwargs) -> NDArray[np.float64]:
        '''
        Abstract Method for model to be trained and predict some y_hat for the provided 
        features and labels.

        Returns y_hat scaled via scikit-learn StandardScaler()  
        '''
        raise NotImplementedError


class LinearModelCV(ModelInterface):

    def __init__(self, **linear_params):
        self.linear_params = linear_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_val = features 
        y_train, _     = labels

        _ = coords 
        _ = kwargs 
        _ = self.linear_params 

        X_train_ = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        X_val_   = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

        w = np.linalg.pinv(X_train_) @ y_train 
        y_hat = X_val_ @ w 
        return np.asarray(y_hat, dtype=np.float64)


class RandomForestCV(ModelInterface): 
    def __init__(self, **rf_params): 
        self.rf_params = rf_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        X_train, X_val = features 
        y_train, _     = labels 

        _ = coords 
        _ = kwargs 

        rf = RandomForestRegressor(n_jobs=-1, **self.rf_params)
        rf.fit(X_train, y_train) 
        y_hat = rf.predict(X_val)
        return np.asarray(y_hat, dtype=np.float64)


class XGBoostCV(ModelInterface): 

    def __init__(self, **xgb_params): 
        self.xgb_params = xgb_params 

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]: 
        X_train, X_test = features 
        y_train, _      = labels 
        c_train, c_test = coords 

        val_size = kwargs.get("val_size", 0.3)
        seed     = kwargs.get("seed", 1) 

        inner_train_idx, inner_val_idx = h.split_indices(len(X_train), val_size, seed) 

        X_tr = np.hstack([c_train[inner_train_idx].astype(np.float32), X_train[inner_train_idx].astype(np.float32)])
        X_va = np.hstack([c_train[inner_val_idx].astype(np.float32), X_train[inner_val_idx].astype(np.float32)])
        X_te = np.hstack([c_test.astype(np.float32), X_test.astype(np.float32)])

        y_tr, y_va = y_train[inner_train_idx].astype(np.float32), y_train[inner_val_idx].astype(np.float32)

        xgb_params = {k: _unwrap_scalar(v) for k, v in self.xgb_params.items()}

        xgb = XGBRegressor(
            tree_method="hist", 
            n_jobs=-1,
            **xgb_params 
        )
        xgb.fit(
            X_tr, y_tr, 
            eval_set=[(X_va, y_va)], 
            verbose=False,
        )

        y_hat = xgb.predict(X_te) 
        return np.asarray(y_hat, dtype=np.float64)



class GNNCV(ModelInterface): 
    
    def __init__(self, layers, method="knn", parameter=5.0, epochs=1000, lr=0.001, **gnn_params): 
        self.layers     = layers 
        self.method     = method 
        self.parameter  = parameter 
        self.epochs     = epochs 
        self.lr         = lr 
        self.gnn_params = gnn_params

    def fit_and_predict(self, features, labels, coords, **kwargs) -> NDArray[np.float64]:
        _ = features 
        _ = coords 
        _ = kwargs 
        _, y_val = labels 
        return np.zeros_like(y_val, dtype=np.float64)


class CrossValidator: 

    def __init__(self, filepath: str, decade: int = 2020): 

        self.filepath = filepath 
        self.decade   = decade 
        self.data     = self._load_data() 

    def _load_data(self): 

        data    = loadmat(self.filepath)
        decades = data["decades"]
        coords  = data["coords"]

        decade_key  = f"decade_{self.decade}"
        decade_data = decades[decade_key][0, 0]

        X = decade_data["features"][0, 0]
        y = decade_data["labels"][0, 0]

        return {"features": X, "labels": y, "coords": coords}

    def run(self, models: Dict[str, ModelInterface], n_folds: int, test_size: float, seed: int) -> pd.DataFrame: 
        X = np.asarray(self.data["features"], dtype=np.float64) 
        y = np.asarray(self.data["labels"], dtype=np.float64).reshape(-1)
        coords = np.asarray(self.data["coords"])

        cv_idx, _ = h.split_indices(len(X), test_size, seed) 
        
        cv_folds_rel = h.kfold_indices(len(cv_idx), n_folds, seed) 
        results = []

        for model_name, model in models.items():
            print(f"> Running {n_folds}-fold CV for {model_name}...")

            for fold_i, (train_rel, val_rel) in enumerate(cv_folds_rel): 
                print(f"    > Fold {fold_i}...")

                train_idx = cv_idx[train_rel] 
                val_idx   = cv_idx[val_rel] 

                X_train_raw, X_val_raw   = X[train_idx], X[val_idx] 
                y_train_raw, y_val_raw   = y[train_idx], y[val_idx] 
                coords_train, coords_val = coords[train_idx], coords[val_idx]

                X_scaler, y_scaler = h.fit_scaler(X_train_raw, y_train_raw) 

                X_train, y_train = h.transform_with_scalers(X_train_raw, y_train_raw, X_scaler, y_scaler)
                X_val, y_val     = h.transform_with_scalers(X_val_raw, y_val_raw, X_scaler, y_scaler)

                try: 
                    y_hat_scaled = model.fit_and_predict(
                        (X_train, X_val), 
                        (y_train, y_val), 
                        (coords_train, coords_val), 
                        seed=seed + fold_i, 
                        y_scaler=y_scaler 
                    )

                    y_hat  = y_scaler.inverse_transform(np.asarray(y_hat_scaled).reshape(-1, 1)).ravel() 
                    y_true = y_scaler.inverse_transform(np.asarray(y_val).reshape(-1, 1)).ravel()  

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
                        "fold": fold_i, 
                        "r2": r2,
                        "rmse": rmse, 
                        "baseline_r2": baseline_r2, 
                        "baseline_rmse": baseline_rmse, 
                        "win_r2": win_r2, 
                        "win_rmse": win_rmse, 
                        "win_r2_gt0": win_r2_gt0, 
                        "n_train": len(X_train), 
                        "n_test": len(X_val)
                    })

                except Exception as e: 
                    print(f"    Error in fold {fold_i + 1}: {e}")
                    # Ignore win rates for failure 
                    results.append({
                        "model": model_name, 
                        "fold": fold_i, 
                        "r2": np.nan,
                        "rmse": np.nan, 
                        "n_train": len(X_train), 
                        "n_test": len(X_val)
                    })

        return pd.DataFrame(results)

    def run_repeated(self, models: Dict[str, ModelInterface], n_repeats: int, n_folds: int, 
                     test_size: float, base_seed: int) -> pd.DataFrame: 

        all_results = []
        for repeat_i in range(n_repeats):
            print(f"\n> Running CV Repeat {repeat_i + 1}/{n_repeats}")
            
            repeat_seed    = base_seed + repeat_i * 2
            repeat_results = self.run(models, n_folds, test_size, repeat_seed) 
            repeat_results["repeat"] = repeat_i 

            all_results.append(repeat_results)

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
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "linear"])
    
    args = parser.parse_args()

    filepath = h.project_path("data", "climate_population.mat")
    cv = CrossValidator(filepath, decade=args.decade)  

    models = {}

    if "rf" in args.models: 
        models["RandomForest"] = RandomForestCV(n_estimators=500, random_state=args.seed)
    if "xgb" in args.models: 
        models["XGBoost"] = XGBoostCV(random_state=args.seed, early_stopping_rounds=150)
    if "linear" in args.models: 
        models["Linear"] = LinearModelCV(gpu=False)
    if "gnn" in args.models: 
        # models["ClimateGNN"] = ClimateGNN() 
        pass 

    results_df = cv.run_repeated(models, n_repeats=args.repeats, n_folds=args.folds, test_size=0.4, base_seed=args.seed)
    summary_df = cv.summarize_repeated(results_df)

    CrossValidator.format_summary(summary_df)
    
    results_csv = h.project_path("data", "models", f"cv_results_decade_{args.decade}_f{args.folds}_r{args.repeats}.csv")
    summary_csv = h.project_path("data", "models", f"cv_summary_decade_{args.decade}_f{args.folds}_r{args.repeats}.csv")

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

if __name__ == "__main__": 
    main() 
