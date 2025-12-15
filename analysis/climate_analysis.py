#!/usr/bin/env python3 
# 
# climate_analysis.py  Andrew Belles  Dec 13th, 2025 
# 
# Rigorous analysis of ClimateGNN to understand contributions 
# of specific features from sophisticate climate view 
# 

import argparse 

from pathlib import Path
from models.cross_validation import CrossValidator
import models.helpers as h 

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from dataclasses import dataclass 
from typing import Dict, Mapping, Optional, List
from numpy.typing import NDArray 

from models.xgboost_model import XGBoost


class ClimateFeaturesAblation: 

    @dataclass(frozen=True)
    class AblationGroup: 
        features: NDArray[np.float64] 
        coords: NDArray[np.float64]
        using_groups: Optional[List[str]]

    def __init__(self, cv: CrossValidator, feature_groups: List[str]): 
        X = np.asarray(cv.data["features"], dtype=np.float64)
        coords = np.asarray(cv.data["coords"], dtype=np.float64)

        self.cv     = cv 
         
        self.groups = {}

        for group_name in feature_groups: 
            # pull group from cv.data, utilize it and store which group we are using  


            self.groups[group_name] = self.AblationGroup(features=)

        self.groups = {
            "coords_only": self.AblationGroup(features=X, coords=coords, using_groups=groups), 
            "climate_only": self.AblationGroup(features=X, coords=coords, ignore_groups=), 
            "both": self.AblationGroup(features=X, coords=coords, ignore_climate=False, ignore_coords=False)
        }

    def run(self, *, groups: Mapping[str, AblationGroup], 
             models: Dict[str, h.ModelInterface], n_repeats: int, n_folds: int, 
             test_size: float, base_seed: int) -> pd.DataFrame: 
        
        original_data = self.cv.data 
        out: list[pd.DataFrame] = []

        y = np.asarray(original_data["labels"], dtype=np.float64)

        for group_name, group in groups.items(): 

            Xg = np.asarray(group.features, dtype=np.float64) 
            Cg = np.asarray(group.coords, dtype=np.float64)

            if Xg.shape[0] != y.shape[0]: 
                raise ValueError(f"{group_name}: features rows {Xg.shape[0]} != labels rows {y.shape[0]}")
            if Cg.shape[0] != y.shape[0]: 
                raise ValueError(f"{group_name}: coords rows {Cg.shape[0]} != labels rows {y.shape[0]}")

            self.cv.data = {"features": Xg, "labels": y, "coords": Cg} 

            for m in models.values(): 
                if not hasattr(m, "ignore_coords"):
                    continue 

                effective_ignore = bool(group.ignore_coords) or bool(group.ignore_climate)
                setattr(m , "ignore_coords", effective_ignore)

                if bool(group.ignore_climate): 
                    self.cv.data["features"] = Cg 

            df = self.cv.run_repeated(
                models,
                n_repeats=n_repeats, 
                n_folds=n_folds, 
                test_size=test_size, 
                base_seed=base_seed
            )
            df["feature_group"] = group_name 
            df["ignore_coords"] = bool(group.ignore_coords) or bool(group.ignore_climate)
            out.append(df)

        self.cv.data = original_data 
        return pd.concat(out, ignore_index=True)

    def _default_results_path(self, *, n_folds: int, n_repeats: int) -> str: 
        fname = f"climate_ablation_decade{self.cv.decade}_f{n_folds}_r{n_repeats}.csv" 
        return h.project_path("data", "models", "raw", fname)

    def compile(self, results_df: pd.DataFrame, filepath: str): 
        path = Path(filepath)

        self.merged = self._compute_winrates(results_df) 
        self.merged.to_csv(path, index=False)

        summary = self._summary_by_group(self.merged)
        summary.to_csv(path.with_name(path.stem + "_summary.csv"), index=False)

        print(f"> Saved ablation results: {filepath}")
        return summary

    @staticmethod 
    def _compute_winrates(df: pd.DataFrame) -> pd.DataFrame: 
        required = {"model", "rmse", "r2", "baseline_r2", "baseline_rmse"} 
        missing = required - set(df.columns)
        if missing: 
            raise ValueError(f"results_df missing required columns: {missing}")

        out = df.copy() 
        if not isinstance(out, pd.DataFrame): 
            raise TypeError("copy failed type check _compute_winrates")

        out["win_rmse"]   = (out["rmse"] < out["baseline_rmse"]).astype(float)
        out["win_r2"]     = (out["r2"] > out["baseline_r2"]).astype(float)
        out["win_r2_gt0"] = (out["r2"] > 0.0).astype(float) 

        return out 

    @staticmethod 
    def _summary_by_group(df: pd.DataFrame) -> pd.DataFrame: 

        required = {
            "feature_group", "model", "repeat", 
            "r2", "rmse", "baseline_r2", "baseline_rmse",
            "win_r2", "win_rmse", "win_r2_gt0"
        }

        missing = required - set(df.columns) 
        if missing: 
            raise ValueError(f"_summary_by_group missing required columns: {missing}")

        repeat_summary = (
            df.groupby(["feature_group", "model", "repeat"]).agg({
                "r2": "mean", 
                "rmse": "mean", 
                "baseline_r2": "mean", 
                "baseline_rmse": "mean", 
                "win_rmse": "mean", 
                "win_r2": "mean", 
                "win_r2_gt0": "mean", 
            })
        ).reset_index()

        if not isinstance(repeat_summary, pd.DataFrame): 
            raise TypeError("repeat summary failed to aggregate")

        n_repeats = int(repeat_summary["repeat"].nunique())

        final_summary = repeat_summary.groupby(["feature_group", "model"]).agg({
            "r2": ["mean", "median", "std", "min", "max"],
            "rmse": ["mean", "median", "std", "min", "max"], 
            "baseline_r2": ["mean", "median", "std", "min", "max"], 
            "baseline_rmse": ["mean", "median", "std", "min", "max"], 
            "win_r2": ["mean", "median", "std", "min", "max"], 
            "win_rmse": ["mean", "median", "std", "min", "max"], 
            "win_r2_gt0": ["mean", "median", "std", "min", "max"]  
        })

        if not isinstance(final_summary, pd.DataFrame): 
            raise TypeError("aggregatation of ablation summary failed type check")

        final_summary.columns = [f"{metric}_{stat}" for metric, stat in final_summary.columns]
        final_summary = final_summary.reset_index()

        final_summary["r2_ci_half"]    = 1.96 * final_summary["r2_std"] / (np.sqrt(n_repeats))
        final_summary["rmse_ci_half"]  = 1.96 * final_summary["rmse_std"] / (np.sqrt(n_repeats))
        final_summary["r2_ci_lower"]   = final_summary["r2_mean"] - final_summary["r2_ci_half"]
        final_summary["r2_ci_upper"]   = final_summary["r2_mean"] + final_summary["r2_ci_half"]
        final_summary["rmse_ci_lower"] = final_summary["rmse_mean"] - final_summary["rmse_ci_half"]
        final_summary["rmse_ci_upper"] = final_summary["rmse_mean"] + final_summary["rmse_ci_half"]

        return final_summary 

    @staticmethod 
    def plot(results: pd.DataFrame, out_dir: str) -> None: 
        out = Path(out_dir)

        repeats = results.groupby(["feature_group", "model", "repeat"]).agg({
            "win_rmse": "mean", 
            "win_r2": "mean", 
            "win_r2_gt0": "mean"
        }).reset_index()

        if not isinstance(repeats, pd.DataFrame): 
            raise TypeError("win aggregation failed type check")

        wins_long = repeats.melt(
            id_vars=["feature_group", "model", "repeat"],
            value_vars=["win_rmse", "win_r2", "win_r2_gt0"],
            var_name="metric",
            value_name="win_rate",
        )
        if not isinstance(wins_long, pd.DataFrame):
            raise TypeError("win melt failed type check")

        wins_long["win_rate"] = 100.0 * wins_long["win_rate"]
        wins_long["metric"] = wins_long["metric"].replace({
            "win_rmse": "WinRMSE",
            "win_r2": "WinR2",
            "win_r2_gt0": "P(R2>0)",
        })

        group_order = ["coords_only", "both", "climate_only"]
        metric_order = ["WinRMSE", "WinR2", "P(R2>0)"]

        wins_long["feature_group"] = pd.Categorical(
            wins_long["feature_group"], categories=group_order, ordered=True
        )
        wins_long["metric"] = pd.Categorical(
            wins_long["metric"], categories=metric_order, ordered=True
        )

        g = sns.catplot(
            data=wins_long,
            x="feature_group",
            y="win_rate",
            hue="metric",
            col="model",
            kind="box",
            order=group_order,
            hue_order=metric_order,
            sharey=True,
            height=5.6,
            aspect=1.8,
            dodge=True,
        )
        g.set_axis_labels("", "Win rate (%)")
        g.set_titles("{col_name}")
        fig = g.figure
        axes = getattr(g, "axes", None)
        if isinstance(axes, np.ndarray) and axes.size > 0:
            ax0 = axes.flat[0]
            handles, labels = ax0.get_legend_handles_labels()
            legend = getattr(g, "_legend", None)
            if legend is not None:
                legend.remove()
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                frameon=True,
            )
        fig.tight_layout()
        fig.savefig(out / "ablation_winrates_box.png")
        plt.close(fig)

        repeats_metrics = results.groupby(["feature_group", "model", "repeat"]).agg({
            "r2": "mean",
            "rmse": "mean",
        }).reset_index()
        if not isinstance(repeats_metrics, pd.DataFrame):
            raise TypeError("metrics aggregation failed type check")

        metrics_long = repeats_metrics.melt(
            id_vars=["feature_group", "model", "repeat"],
            value_vars=["r2", "rmse"],
            var_name="metric",
            value_name="value",
        )
        if not isinstance(metrics_long, pd.DataFrame):
            raise TypeError("metrics melt failed type check")

        metrics_long["feature_group"] = pd.Categorical(
            metrics_long["feature_group"], categories=group_order, ordered=True
        )
        metrics_long["metric"] = metrics_long["metric"].replace({
            "r2": "R2",
            "rmse": "RMSE",
        })

        g2 = sns.catplot(
            data=metrics_long,
            x="feature_group",
            y="value",
            col="model",
            row="metric",
            kind="box",
            order=group_order,
            sharey=False,
            height=5.6,
            aspect=1.8,
        )
        g2.set_axis_labels("", "")
        g2.set_titles("{row_name} | {col_name}")
        fig2 = g2.figure
        fig2.tight_layout()
        fig2.savefig(out / "ablation_metric_spread_box.png")
        plt.close(fig2)

        print(f"> Wrote plots at {out}")

    def interpret(self, results: pd.DataFrame | str): 

        if isinstance(results, str): 
            path = Path(results) 
            df   = pd.read_csv(path) 
        else: 
            df = results.copy() 

        needed = {
            "win_rmse", 
            "win_r2", 
            "win_r2_gt0"
        }

        missing = needed - set(df.columns)
        if missing: 
            df = self._compute_winrates(df)
            if not isinstance(df, pd.DataFrame): 
                raise TypeError("_compute_winrates failed type check")

        summary = self._summary_by_group(df)
        if not isinstance(summary, pd.DataFrame):
            raise TypeError("_summary_by_group failed type check")

        summary_sorted = summary.sort_values(["model", "feature_group"])
        if not isinstance(summary_sorted, pd.DataFrame):
            raise TypeError("summary sort failed type check")

        for model_name, model_rows in summary_sorted.groupby("model", sort=False):
            print(f"> {model_name}")
            for _, row in model_rows.iterrows():
                group = row["feature_group"]

                r2_str   = f"{row['r2_mean']:+.3f} [{row['r2_ci_lower']:+.3f}, {row['r2_ci_upper']:+.3f}]"
                rmse_str = f"{row['rmse_mean']:+.3f} [{row['rmse_ci_lower']:+.3f}, {row['rmse_ci_upper']:+.3f}]"

                win_rmse   = h._as_float(row.get("win_rmse_mean", np.nan))
                win_r2     = h._as_float(row.get("win_r2_mean", np.nan))
                win_r2_gt0 = h._as_float(row.get("win_r2_gt0_mean", np.nan))

                win_rmse_str   = "" if np.isnan(win_rmse) else f"{100.0 * win_rmse:5.1f}%"
                win_r2_str     = "" if np.isnan(win_r2) else f"{100.0 * win_r2:5.1f}%"
                win_r2_gt0_str = "" if np.isnan(win_r2_gt0) else f"{100.0 * win_r2_gt0:5.1f}%"

                print(
                    f"    > {group}: r2={r2_str} rmse={rmse_str} "
                    f"WinRMSE={win_rmse_str} WinR2={win_r2_str} P(R2>0)={win_r2_gt0_str}"
                )


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    filepath = h.project_path("data", "climate_population.mat")
    cv  = CrossValidator(filepath, decade=2020)
    abl = ClimateFeaturesAblation(cv)  

    models: Dict[str, h.ModelInterface] = {
        "XGBoost": XGBoost(
            gpu=False, 
            random_state=0,
            early_stopping_rounds=200
        )
    }

    results = abl.run(
        groups=abl.groups, 
        models=models, 
        n_repeats=args.repeats, 
        n_folds=args.folds, 
        test_size=0.4,
        base_seed=0
    )

    _ = abl.compile(results, h.project_path("data", "models", "ablation_results.csv"))

    abl.interpret(results)
    abl.plot(abl.merged, h.project_path("data", "models"))


if __name__ == "__main__":
    main() 
