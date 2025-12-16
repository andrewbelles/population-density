#!/usr/bin/env python3 
# 
# climate_analysis.py  Andrew Belles  Dec 13th, 2025 
# 
# Rigorous analysis of ClimateGNN to understand contributions 
# of specific features from sophisticate climate view 
# 

import argparse 

from pathlib import Path
from models.cross_validation import CrossValidator, ModelFactory, CVConfig
import models.helpers as h 

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from dataclasses import dataclass 
from typing import Callable, Mapping, Optional, Any

from models.linear_model import LinearModel
from models.xgboost_model import XGBoost
from models.random_forest_model import RandomForest

DatasetLoader = h.DatasetLoader
LoaderFactory = Callable[[list[str]], DatasetLoader]
ModelPatcher  = Callable[[Mapping[str, ModelFactory]], Mapping[str, ModelFactory]]

def set_model_attrs(factory: ModelFactory, **attrs: Any) -> ModelFactory: 
    def _wrapped() -> h.ModelInterface: 
        m = factory() 
        for k, v in attrs.items(): 
            setattr(m, k, v)
        return m 
    return _wrapped



@dataclass(frozen=True)
class AblationSpec: 
    name: str 
    tags: list[str] 
    patch_models: Optional[ModelPatcher] = None 


class FeatureAblation: 

    def __init__(self, *, filepath: str, loader_factory: LoaderFactory): 
        self.filepath       = filepath 
        self.loader_factory = loader_factory

    @staticmethod 
    def leave_one_out(*, all_tags: list[str], baseline_name: str = "all") -> list[AblationSpec]: 
        specs = [AblationSpec(baseline_name, tags=list(all_tags))]
        for t in all_tags:
            specs.append(AblationSpec(f"minus_{t}", tags=[x for x in all_tags if x != t]))
        return specs 

    def run(self, *, specs: list[AblationSpec], models: Mapping[str, ModelFactory], 
            cv_config: CVConfig, n_repeats: int) -> pd.DataFrame: 
        out: list[pd.DataFrame] = []

        for spec in specs: 
            loader = self.loader_factory(spec.tags)
            cv = CrossValidator(filepath=self.filepath, loader=loader)

            effective_models = dict(models)
            if spec.patch_models is not None: 
                effective_models = dict(spec.patch_models(effective_models))

            df = cv.run_repeated(models=effective_models, config=cv_config, n_repeats=n_repeats)
            df["ablation"] = spec.name 
            df["tags"] = ",".join(spec.tags) 
            out.append(df)

        return pd.concat(out, ignore_index=True)

    @staticmethod
    def _as_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def compile(self, results_df: pd.DataFrame, filepath: str): 
        path = Path(filepath)

        self.merged = self._compute_winrates(results_df) 
        self.merged.to_csv(path, index=False)

        summary = self._summary_by_group(self.merged)
        summary_path = path.with_name(path.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False)
        self.summary = summary

        print(f"> Saved ablation results: {path}")
        print(f"> Saved ablation summary: {summary_path}")
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
            "ablation", "model", "repeat", 
            "r2", "rmse", "baseline_r2", "baseline_rmse",
            "win_r2", "win_rmse", "win_r2_gt0"
        }

        missing = required - set(df.columns) 
        if missing: 
            raise ValueError(f"_summary_by_group missing required columns: {missing}")

        repeat_summary = (
            df.groupby(["ablation", "model", "repeat"]).agg({
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

        final_summary = repeat_summary.groupby(["ablation", "model"]).agg({
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
        out.mkdir(parents=True, exist_ok=True)

        if "win_rmse" not in results.columns or "win_r2" not in results.columns or "win_r2_gt0" not in results.columns:
            results = FeatureAblation._compute_winrates(results)

        group_col = "ablation"
        if group_col not in results.columns:
            raise ValueError(f"plot requires '{group_col}' column in results")

        group_order = list(dict.fromkeys(results[group_col].astype(str).tolist()))

        repeats = results.groupby([group_col, "model", "repeat"]).agg({
            "win_rmse": "mean", 
            "win_r2": "mean", 
            "win_r2_gt0": "mean"
        }).reset_index()

        if not isinstance(repeats, pd.DataFrame): 
            raise TypeError("win aggregation failed type check")

        wins_long = repeats.melt(
            id_vars=[group_col, "model", "repeat"],
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

        metric_order = ["WinRMSE", "WinR2", "P(R2>0)"]

        wins_long[group_col] = pd.Categorical(
            wins_long[group_col], categories=group_order, ordered=True
        )
        wins_long["metric"] = pd.Categorical(
            wins_long["metric"], categories=metric_order, ordered=True
        )

        g = sns.catplot(
            data=wins_long,
            x=group_col,
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

        repeats_metrics = results.groupby([group_col, "model", "repeat"]).agg({
            "r2": "mean",
            "rmse": "mean",
        }).reset_index()
        if not isinstance(repeats_metrics, pd.DataFrame):
            raise TypeError("metrics aggregation failed type check")

        metrics_long = repeats_metrics.melt(
            id_vars=[group_col, "model", "repeat"],
            value_vars=["r2", "rmse"],
            var_name="metric",
            value_name="value",
        )
        if not isinstance(metrics_long, pd.DataFrame):
            raise TypeError("metrics melt failed type check")

        metrics_long[group_col] = pd.Categorical(
            metrics_long[group_col], categories=group_order, ordered=True
        )
        metrics_long["metric"] = metrics_long["metric"].replace({
            "r2": "R2",
            "rmse": "RMSE",
        })

        g2 = sns.catplot(
            data=metrics_long,
            x=group_col,
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

        summary_sorted = summary.sort_values(["model", "ablation"])
        if not isinstance(summary_sorted, pd.DataFrame):
            raise TypeError("summary sort failed type check")

        for model_name, model_rows in summary_sorted.groupby("model", sort=False):
            print(f"> {model_name}")
            for _, row in model_rows.iterrows():
                group = row["ablation"]

                r2_str   = f"{row['r2_mean']:+.3f} [{row['r2_ci_lower']:+.3f}, {row['r2_ci_upper']:+.3f}]"
                rmse_str = f"{row['rmse_mean']:+.3f} [{row['rmse_ci_lower']:+.3f}, {row['rmse_ci_upper']:+.3f}]"

                win_rmse   = self._as_float(row.get("win_rmse_mean", np.nan))
                win_r2     = self._as_float(row.get("win_r2_mean", np.nan))
                win_r2_gt0 = self._as_float(row.get("win_r2_gt0_mean", np.nan))

                win_rmse_str   = "" if np.isnan(win_rmse) else f"{100.0 * win_rmse:5.1f}%"
                win_r2_str     = "" if np.isnan(win_r2) else f"{100.0 * win_r2:5.1f}%"
                win_r2_gt0_str = "" if np.isnan(win_r2_gt0) else f"{100.0 * win_r2_gt0:5.1f}%"

                print(
                    f"    > {group}: r2={r2_str} rmse={rmse_str} "
                    f"WinRMSE={win_rmse_str} WinR2={win_r2_str} P(R2>0)={win_r2_gt0_str}"
                )

        return summary


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target", type=str, default="lat")
    args = parser.parse_args()

    filepath = h.project_path("data", "climate_geospatial.mat")
    loader_factory = lambda tags: (lambda fp: h.load_climate_geospatial(fp, target=args.target, groups=tags))

    abl   = FeatureAblation(filepath=filepath, loader_factory=loader_factory)
    specs = FeatureAblation.leave_one_out(all_tags=["degree_days", "palmer_indices"]) 

    models = {
        "XGBoost": lambda: XGBoost(
            gpu=False, 
            random_state=0,
            early_stopping_rounds=200
        ),
        "Linear": lambda: LinearModel(),
        "RandomForest": lambda: RandomForest(
            n_estimators=500, 
            random_state=0
        )
    }

    cv_config = CVConfig(n_splits=args.folds, test_size=0.4, split_mode="random", base_seed=0)
    df = abl.run(specs=specs, models=models, cv_config=cv_config, n_repeats=args.repeats)

    results_path = h.project_path("data", "models", "raw", "feature_ablation_results.csv")
    _ = abl.compile(df, results_path)
    _ = abl.interpret(abl.merged)

    FeatureAblation.plot(abl.merged, h.project_path("analysis", "images", args.target))


if __name__ == "__main__":
    main() 
