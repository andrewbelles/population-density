#!/usr/bin/env python3 
# 
# ablation.py  Andrew Belles  Dec 13th, 2025 
# 

import argparse 

from pathlib import Path
from analysis.cross_validation import (
    REGRESSION, 
    CrossValidator,  
    CVConfig,
    TaskSpec
)

from support.helpers import (
    project_path, 
    ModelFactory,
    ModelInterface, 
)

from preprocessing.loaders import (
    load_climate_geospatial,
    DatasetLoader
)

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from dataclasses import dataclass 
from typing import Callable, Mapping, Optional, Any

from models.estimators import (
    make_linear, 
    make_rf_regressor, 
    make_xgb_regressor
)

DatasetLoader = DatasetLoader
LoaderFactory = Callable[[list[str]], DatasetLoader]
ModelPatcher  = Callable[[Mapping[str, ModelFactory]], Mapping[str, ModelFactory]]

def set_model_attrs(factory: ModelFactory, **attrs: Any) -> ModelFactory: 
    def _wrapped() -> ModelInterface: 
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

    def run(
        self, 
        *, 
        specs: list[AblationSpec], 
        models: Mapping[str, ModelFactory], 
        config: CVConfig, 
        task: TaskSpec = REGRESSION
    ) -> pd.DataFrame: 
        out: list[pd.DataFrame] = []

        for spec in specs: 
            loader = self.loader_factory(spec.tags)

            cv = CrossValidator(
                filepath=self.filepath, loader=loader, task=task
            )
            
            df = cv.run(models=models, config=config)
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

        '''Add win rate columns to results'''
        
        out = df.copy()

        if "r2" in out.columns and "baseline_r2" in out.columns: 
            out["win_r2"] = (out["r2"] > out["baseline_r2"]).astype(float)
            out["win_r2_gt0"] = (out["r2"] > 0.0).astype(float)

        if "rmse" in out.columns and "baseline_rmse" in out.columns:
               out["win_rmse"] = (out["rmse"] < out["baseline_rmse"]).astype(float)

        if "accuracy" in out.columns and "baseline_accuracy" in out.columns:
            out["win_accuracy"] = (
                out["accuracy"] > out["baseline_accuracy"]
            ).astype(float)

        return out


    @staticmethod 
    def _summary_by_group(df: pd.DataFrame) -> pd.DataFrame: 
    
        '''Aggregate results by ablation and model'''

        metric_cols = [
            c for c in df.columns
            if c in ("r2", "rmse", "accuracy", "f1", "roc_auc")
        ]
        win_cols = [c for c in df.columns if c.startswith("win_")]
        baseline_cols = [c for c in df.columns if c.startswith("baseline_")]

        agg_cols = metric_cols + win_cols + baseline_cols 
        agg_dict = {c: ["mean", "std", "min", "max"] for c in agg_cols if c in df.columns}

        summary = df.groupby(["ablation", "model"]).agg(agg_dict)
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index()

        n_folds = df["fold"].nunique() if "fold" in df.columns else 1 
        for m in metric_cols:
            if f"{m}_std" in summary.columns: 
                ci_half = 1.96 * summary[f"{m}_std"] / np.sqrt(n_folds)
                summary[f"{m}_ci_lower"] = summary[f"{m}_mean"] - ci_half
                summary[f"{m}_ci_upper"] = summary[f"{m}_mean"] + ci_half

        return summary 

    @staticmethod 
    def plot(results: pd.DataFrame, out_dir: str) -> None: 
        
        '''
        Generate ablation visualization plots. 

        Credit Claude Opus 4.5, direct code contribution
        '''

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if "win_rmse" not in results.columns: 
            results = FeatureAblation._compute_winrates(results)

        group_col = "ablation"
        if group_col not in results.columns:
            raise ValueError(f"plot requires '{group_col}' column in results")

        group_order = list(dict.fromkeys(results[group_col].astype(str).tolist()))

        agg_cols = [c for c in results.columns if c.startswith("win_")]
        if not agg_cols: 
            print("> No win columns found, skipping plots")
            return 

        repeats = (
            results.groupby([group_col, "model", "fold"])
            .agg({c: "mean" for c in agg_cols})
            .reset_index()
        )

        wins_long = repeats.melt(
            id_vars=[group_col, "model", "fold"], 
            value_vars=agg_cols,
            var_name="metric",
            value_name="win_rate"
        )
        wins_long["win_rate"] = 100.0 * wins_long["win_rate"]

        metric_map = {
            "win_rmse": "WinRMSE", 
            "win_r2": "$WinR^2$", 
            "win_r2_gt0": "$P(R^2 > 0)$", 
            "win_accuracy": "WinAcc"
        }
        wins_long["metric"] = wins_long["metric"].map(
            lambda x: metric_map.get(x, x)
        )

        g = sns.catplot(
            data=wins_long,
            x=group_col,
            y="win_rate",
            hue="metric",
            col="model",
            kind="box",
            order=group_order,
            hue_order=group_order,
            sharey=True,
            height=5.6,
            aspect=1.8,
            dodge=True,
        )

        g.set_axis_labels("", "Win rate (%)")
        g.set_titles("{col_name}")
        g.figure.tight_layout()
        g.figure.savefig(out / "ablation_winrates_box.png", dpi=150)
        plt.close(g.figure)

        print(f"> Wrote plots to {out}")

    def interpret(self, results: pd.DataFrame | str): 

        '''
        Human readable formatting for results from ablation


        Caller can optionally provide results as a str to the csv 
        '''

        if isinstance(results, str):
            df = pd.read_csv(results)
        else:
            df = results.copy()

        if "win_rmse" not in df.columns:
            df = self._compute_winrates(df)

        summary = self._summary_by_group(df)

        for model_name, model_rows in summary.groupby("model", sort=False):
            print(f"\n> {model_name}")
            for _, row in model_rows.iterrows():
                group = row["ablation"]
                parts = [f"    {group}:"]

                # Add metrics
                for m in ("r2", "rmse", "accuracy"):
                    if f"{m}_mean" in row:
                        mean = row[f"{m}_mean"]
                        ci_lo = row.get(f"{m}_ci_lower", np.nan)
                        ci_hi = row.get(f"{m}_ci_upper", np.nan)
                        parts.append(f"{m}={mean:+.3f}[{ci_lo:+.3f},{ci_hi:+.3f}]")

                # Add win rates
                for w in ("win_rmse", "win_r2", "win_r2_gt0", "win_accuracy"):
                    if f"{w}_mean" in row:
                        val = row[f"{w}_mean"]
                        parts.append(f"{w}={100*val:.1f}%")

                print(" ".join(parts))

        return summary


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--target", type=str, default="lat")
    args = parser.parse_args()

    filepath = project_path("data", "climate_geospatial.mat")

    def loader_factory(tags: list[str]) -> DatasetLoader: 
        return lambda fp: load_climate_geospatial(
            fp, target=args.target, groups=tags 
        )

    abl   = FeatureAblation(filepath=filepath, loader_factory=loader_factory)
    specs = FeatureAblation.leave_one_out(
        all_tags=["degree_days", "palmer_indices"]
    ) 

    models = {
        "XGBoost": make_xgb_regressor(n_estimators=400, early_stopping_rounds=200), 
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(n_estimators=400)
    }

    config = CVConfig(
        n_splits=args.splits,
        n_repeats=args.repeats, 
        random_state=0 
    )

    df = abl.run(
        specs=specs, 
        models=models, 
        config=config, 
        task=REGRESSION
    )

    results_path = project_path("data", "models", "raw", "feature_ablation_results.csv")
    _ = abl.compile(df, results_path)
    _ = abl.interpret(abl.merged)

    FeatureAblation.plot(abl.merged, project_path("analysis", "images", args.target))

if __name__ == "__main__":
    main() 
