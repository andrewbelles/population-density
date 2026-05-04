#!/usr/bin/env python3
#
# final_compare.py  Andrew Belles  May 3rd, 2026
#
# Shared-support final comparison over selected graph/linear optimizer runs.
#

import argparse
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from graph.config import load_config as load_graph_config
from graph.topology import run as run_graph_topology
from analysis.hypothesis import run_hypothesis_tests
from analysis.loaders import AnalysisBundle, load_analysis_config
from analysis.shared import write_frame
from nowcast.censal import StrictResult, evaluate_strict
from nowcast.common import load_county_display_lookup
from nowcast.config import NowcastConfig, load_config as load_nowcast_config
from nowcast.postcensal import run_postcensal


LOGGER = logging.getLogger("analysis.final_compare")


@dataclass(slots=True)
class CandidateRun:
    key: str
    label: str
    graph_best_trial_json: Path
    linear_best_trial_json: Path


@dataclass(slots=True)
class FinalCompareConfig:
    analysis_config_path: Path
    nowcast_config_path: Path
    graph_config_path: Path
    output_root: Path
    baseline_model: str
    treatment_model: str
    materialize_graphs: bool
    postcensal_enabled: bool
    runs: list[CandidateRun]


@dataclass(slots=True)
class CandidateResult:
    run: CandidateRun
    nowcast_config: NowcastConfig
    strict: StrictResult
    postcensal: dict[str, object]


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def _as_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = _as_path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = _as_path(path)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path}: config must be a mapping")
    return dict(raw)


def load_final_compare_config(path: str | Path) -> FinalCompareConfig:
    cfg_path = _as_path(path)
    raw = _read_yaml(cfg_path)
    repo_root = cfg_path.resolve().parent.parent.parent
    runs_raw = list(raw.get("runs", []))
    if not runs_raw:
        raise ValueError("final comparison requires at least one run")
    runs: list[CandidateRun] = []
    seen: set[str] = set()
    for item in runs_raw:
        run_raw = dict(item)
        key = str(run_raw.get("key", "")).strip().lower()
        if not key:
            raise ValueError("each final comparison run requires key")
        if key in seen:
            raise ValueError(f"duplicate final comparison run key={key!r}")
        seen.add(key)
        runs.append(
            CandidateRun(
                key=key,
                label=str(run_raw.get("label", key)).strip() or key,
                graph_best_trial_json=_resolve_path(repo_root, run_raw["graph_best_trial_json"]),
                linear_best_trial_json=_resolve_path(repo_root, run_raw["linear_best_trial_json"]),
            )
        )
    postcensal_raw = dict(raw.get("postcensal", {}))
    return FinalCompareConfig(
        analysis_config_path=_resolve_path(repo_root, raw.get("analysis_config_path", "configs/analysis/hypothesis.yaml")),
        nowcast_config_path=_resolve_path(repo_root, raw.get("nowcast_config_path", "configs/nowcast/nowcast.yaml")),
        graph_config_path=_resolve_path(repo_root, raw.get("graph_config_path", "configs/graph/topology.yaml")),
        output_root=_resolve_path(repo_root, raw.get("output_root", "analysis/artifacts/final_compare")),
        baseline_model=str(raw.get("baseline_model", "pep")).strip().lower(),
        treatment_model=str(raw.get("treatment_model", "embeddings_mem")).strip().lower(),
        materialize_graphs=bool(raw.get("materialize_graphs", True)),
        postcensal_enabled=bool(postcensal_raw.get("enabled", False)),
        runs=runs,
    )


def valid_model_fips(abs_df: pd.DataFrame, *, baseline_model: str, treatment_model: str) -> set[str]:
    work = abs_df.copy()
    work["fips"] = work["fips"].astype(str).str.strip().str.zfill(5)
    work["model"] = work["model"].astype(str).str.strip().str.lower()
    out: set[str] | None = None
    for model in (baseline_model, treatment_model):
        part = work.loc[work["model"] == str(model)].copy()
        keep = (
            np.isfinite(np.asarray(part["y_level"], dtype=np.float64))
            & np.isfinite(np.asarray(part["pred_level"], dtype=np.float64))
            & np.isfinite(np.asarray(part["ape_pop_pct"], dtype=np.float64))
        )
        model_fips = set(part.loc[keep, "fips"].astype(str).tolist())
        out = model_fips if out is None else out.intersection(model_fips)
    return set() if out is None else out


def filter_strict_to_support(result: StrictResult, *, support_fips: set[str]) -> StrictResult:
    support = {str(f).zfill(5) for f in support_fips}
    abs_df = result.abs_df.copy()
    abs_df["fips"] = abs_df["fips"].astype(str).str.strip().str.zfill(5)
    abs_df = abs_df.loc[abs_df["fips"].isin(support)].copy().reset_index(drop=True)
    return StrictResult(
        summary_df=result.summary_df.copy(),
        fold_df=result.fold_df.copy(),
        state_df=result.state_df.copy(),
        pop_df=result.pop_df.copy(),
        pop_compare_df=result.pop_compare_df.copy(),
        coverage_df=result.coverage_df.copy(),
        abs_df=abs_df,
        summary=dict(result.summary),
    )


def empty_postcensal_result() -> dict[str, object]:
    return {
        "trajectory": pd.DataFrame(columns=["fips", "state", "year", "corrected_level", "pep_level", "corrected_log", "pep_log"]),
        "year_metrics": pd.DataFrame(columns=["year", "has_truth", "delta_mape_pct"]),
        "county_summary": pd.DataFrame(columns=["year", "fips", "state"]),
        "summary": {},
    }


def build_bundle_for_candidate(
    *,
    cfg: FinalCompareConfig,
    candidate: CandidateResult,
    strict: StrictResult,
) -> AnalysisBundle:
    analysis_cfg = load_analysis_config(
        cfg.analysis_config_path,
        graph_best_trial_json=candidate.run.graph_best_trial_json,
        linear_best_trial_json=candidate.run.linear_best_trial_json,
    )
    analysis_cfg = replace(
        analysis_cfg,
        comparison=replace(
            analysis_cfg.comparison,
            baseline_model=str(cfg.baseline_model),
            treatment_model=str(cfg.treatment_model),
        ),
    )
    county_lookup = load_county_display_lookup(candidate.nowcast_config.paths.county_shapefile)
    return AnalysisBundle(
        config=analysis_cfg,
        nowcast_config=candidate.nowcast_config,
        county_lookup=county_lookup,
        censal_summary=strict.summary_df.copy(),
        censal_fold_metrics=strict.fold_df.copy(),
        censal_abs_errors=strict.abs_df.copy(),
        county_trajectory=pd.DataFrame(candidate.postcensal.get("trajectory", pd.DataFrame())).copy(),
        year_metrics=pd.DataFrame(candidate.postcensal.get("year_metrics", pd.DataFrame())).copy(),
        county_summary=pd.DataFrame(candidate.postcensal.get("county_summary", pd.DataFrame())).copy(),
        summary_json=dict(candidate.postcensal.get("summary", {})),
    )


def summarize_shared_support(
    *,
    run_key: str,
    run_label: str,
    graph_tag_base: str,
    mem_top_k: int,
    county_pairs: pd.DataFrame,
) -> dict[str, object]:
    base = np.asarray(county_pairs["baseline_ape_pop_pct"], dtype=np.float64)
    treat = np.asarray(county_pairs["treatment_ape_pop_pct"], dtype=np.float64)
    adj = np.asarray(county_pairs["adjusted_treatment_ape_pop_pct"], dtype=np.float64)
    hard_q = float(np.asarray(county_pairs.get("hard_case_quantile", pd.Series([np.nan])), dtype=np.float64)[0]) if "hard_case_quantile" in county_pairs.columns else float("nan")
    return {
        "run_key": str(run_key),
        "label": str(run_label),
        "graph_tag_base": str(graph_tag_base),
        "mem_top_k": int(mem_top_k),
        "n_counties": int(county_pairs["fips"].nunique()),
        "n_states": int(county_pairs["state"].nunique()),
        "baseline_mape_pct": float(np.nanmean(base)),
        "treatment_mape_pct": float(np.nanmean(treat)),
        "adjusted_mape_pct": float(np.nanmean(adj)),
        "adjusted_relative_improvement_pct": float((np.nanmean(base) - np.nanmean(adj)) / max(np.nanmean(base), 1e-9) * 100.0),
        "hard_case_quantile": hard_q,
    }


def add_hard_case_summary(summary: dict[str, object], county_pairs: pd.DataFrame, *, hard_case_quantile: float) -> dict[str, object]:
    vals = np.asarray(county_pairs["baseline_ape_pop_pct"], dtype=np.float64)
    cutoff = float(np.nanquantile(vals[np.isfinite(vals)], float(hard_case_quantile)))
    hard = county_pairs.loc[np.asarray(county_pairs["baseline_ape_pop_pct"], dtype=np.float64) >= cutoff].copy()
    if hard.empty:
        summary.update(
            {
                "hard_case_n": 0,
                "hard_case_adjusted_relative_improvement_pct": float("nan"),
                "hard_case_adjusted_mape_pct": float("nan"),
            }
        )
        return summary
    base = float(np.nanmean(np.asarray(hard["baseline_ape_pop_pct"], dtype=np.float64)))
    adj = float(np.nanmean(np.asarray(hard["adjusted_treatment_ape_pop_pct"], dtype=np.float64)))
    summary.update(
        {
            "hard_case_n": int(hard["fips"].nunique()),
            "hard_case_cutoff_baseline_ape_pct": cutoff,
            "hard_case_adjusted_mape_pct": adj,
            "hard_case_adjusted_relative_improvement_pct": float((base - adj) / max(base, 1e-9) * 100.0),
        }
    )
    return summary


def run_final_compare(cfg: FinalCompareConfig) -> None:
    candidates: list[CandidateResult] = []
    support_sets: dict[str, set[str]] = {}
    for run in cfg.runs:
        LOGGER.info("[final compare] evaluate key=%s label=%s", run.key, run.label)
        nowcast_cfg = load_nowcast_config(
            cfg.nowcast_config_path,
            graph_best_trial_json=run.graph_best_trial_json,
            linear_best_trial_json=run.linear_best_trial_json,
        )
        if cfg.materialize_graphs:
            graph_cfg = load_graph_config(cfg.graph_config_path, best_trial_json=run.graph_best_trial_json)
            run_graph_topology(
                graph_cfg,
                skip_existing=True,
                family_end_year=None if cfg.postcensal_enabled else int(nowcast_cfg.evaluation.strict_year),
                source_year=None if cfg.postcensal_enabled else int(nowcast_cfg.evaluation.strict_year),
            )
        strict = evaluate_strict(nowcast_cfg, model_key=str(nowcast_cfg.downstream.selected))
        postcensal = run_postcensal(nowcast_cfg, model_key=str(nowcast_cfg.downstream.selected)) if cfg.postcensal_enabled else empty_postcensal_result()
        candidates.append(CandidateResult(run=run, nowcast_config=nowcast_cfg, strict=strict, postcensal=postcensal))
        support_sets[run.key] = valid_model_fips(strict.abs_df, baseline_model=cfg.baseline_model, treatment_model=cfg.treatment_model)

    shared_support = set.intersection(*support_sets.values()) if support_sets else set()
    if not shared_support:
        raise ValueError("final comparison produced empty shared support")
    LOGGER.info("[final compare] shared support n=%d", int(len(shared_support)))

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fips": sorted(shared_support)}).to_parquet(output_root / "shared_support_fips.parquet", index=False)

    summary_rows: list[dict[str, object]] = []
    all_hypothesis: list[pd.DataFrame] = []
    all_county_pairs: list[pd.DataFrame] = []
    all_state_pairs: list[pd.DataFrame] = []
    all_year_safety: list[pd.DataFrame] = []

    for candidate in candidates:
        strict_shared = filter_strict_to_support(candidate.strict, support_fips=shared_support)
        run_dir = output_root / str(candidate.run.key)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_frame(strict_shared.abs_df, run_dir / "abs_errors.shared_support.parquet")
        write_frame(candidate.strict.fold_df, run_dir / "fold_metrics.parquet")
        write_frame(candidate.strict.summary_df, run_dir / "summary.original_support.parquet")
        if cfg.postcensal_enabled:
            write_frame(pd.DataFrame(candidate.postcensal["trajectory"]), run_dir / "county_trajectory.parquet")
            write_frame(pd.DataFrame(candidate.postcensal["year_metrics"]), run_dir / "year_metrics.parquet")
            write_frame(pd.DataFrame(candidate.postcensal["county_summary"]), run_dir / "county_summary.parquet")
            with open(run_dir / "postcensal_summary.json", "w", encoding="utf-8") as fh:
                json.dump(dict(candidate.postcensal.get("summary", {})), fh, indent=2)

        bundle = build_bundle_for_candidate(cfg=cfg, candidate=candidate, strict=strict_shared)
        results, county_pairs, state_pairs, year_safety = run_hypothesis_tests(bundle)
        county_pairs.insert(0, "run_key", str(candidate.run.key))
        county_pairs.insert(1, "label", str(candidate.run.label))
        state_pairs.insert(0, "run_key", str(candidate.run.key))
        state_pairs.insert(1, "label", str(candidate.run.label))
        results.insert(0, "run_key", str(candidate.run.key))
        results.insert(1, "label", str(candidate.run.label))
        year_safety.insert(0, "run_key", str(candidate.run.key))
        year_safety.insert(1, "label", str(candidate.run.label))

        hard_q = float(bundle.config.selection.hard_case_quantile)
        summary = summarize_shared_support(
            run_key=candidate.run.key,
            run_label=candidate.run.label,
            graph_tag_base=candidate.nowcast_config.graph.graph_tag_base,
            mem_top_k=int(candidate.nowcast_config.graph.mem_top_k),
            county_pairs=county_pairs,
        )
        summary = add_hard_case_summary(summary, county_pairs, hard_case_quantile=hard_q)
        summary_rows.append(summary)

        all_hypothesis.append(results)
        all_county_pairs.append(county_pairs)
        all_state_pairs.append(state_pairs)
        all_year_safety.append(year_safety)

    summary_df = pd.DataFrame(summary_rows).sort_values("adjusted_mape_pct").reset_index(drop=True)
    hypothesis_df = pd.concat(all_hypothesis, axis=0, ignore_index=True) if all_hypothesis else pd.DataFrame()
    county_pairs_df = pd.concat(all_county_pairs, axis=0, ignore_index=True) if all_county_pairs else pd.DataFrame()
    state_pairs_df = pd.concat(all_state_pairs, axis=0, ignore_index=True) if all_state_pairs else pd.DataFrame()
    year_safety_df = pd.concat(all_year_safety, axis=0, ignore_index=True) if all_year_safety else pd.DataFrame()

    write_frame(summary_df, output_root / "shared_support_summary.parquet")
    write_frame(hypothesis_df, output_root / "hypothesis_results.shared_support.parquet")
    write_frame(county_pairs_df, output_root / "county_pairs.shared_support.parquet")
    write_frame(state_pairs_df, output_root / "state_pairs.shared_support.parquet")
    write_frame(year_safety_df, output_root / "year_safety.shared_support.parquet")
    LOGGER.info("shared-support summary\n%s", summary_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    if not hypothesis_df.empty:
        keep = ["run_key", "hypothesis_id", "subset", "test_name", "estimate", "p_value", "passed", "n_obs", "n_groups"]
        present = [c for c in keep if c in hypothesis_df.columns]
        LOGGER.info("shared-support hypothesis tests\n%s", hypothesis_df.loc[:, present].to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    LOGGER.info("wrote final comparison artifacts to %s", output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final shared-support comparison across selected optimized graph/linear setups.")
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/final_compare.yaml"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_final_compare_config(args.config)
    run_final_compare(cfg)


if __name__ == "__main__":
    main()
