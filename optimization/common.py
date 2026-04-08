#!/usr/bin/env python3
#
# common.py  Andrew Belles  Apr 7th, 2026
#
# Shared Optuna harness helpers and stopping logic for stage-wise optimization.
#

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna


LOGGER = logging.getLogger("optimization.common")


@dataclass(slots=True)
class StudyConfig:
    study_name: str
    direction: str
    n_trials: int
    timeout_sec: int
    sampler_seed: int
    n_startup_trials: int
    gc_after_trial: bool


@dataclass(slots=True)
class GL2Config:
    enabled: bool
    min_trials: int
    patience: int
    max_generalization_loss_pct: float
    min_relative_improvement_pct: float


@dataclass(slots=True)
class UPSweepConfig:
    enabled: bool
    successive_worsening_strips: int
    strip_length: int


def setup_logging(level: str) -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout)


def normalize_weights(values: Sequence[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError("objective weights must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("objective weights must be finite")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("objective weights must sum to a positive value")
    return [float(v / total) for v in arr.tolist()]


def scalarize_values(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(list(values), dtype=np.float64).reshape(-1)
    w = np.asarray(normalize_weights(weights), dtype=np.float64).reshape(-1)
    if vals.size != w.size:
        raise ValueError(f"objective value count ({vals.size}) does not match weight count ({w.size})")
    if not np.isfinite(vals).all():
        return float("-inf")
    return float(np.dot(vals, w))


def signed_tanh_score(value: float, *, scale: float) -> float:
    v = float(value)
    s = float(scale)
    if not np.isfinite(v):
        return float("nan")
    if (not np.isfinite(s)) or s <= 0.0:
        raise ValueError(f"scale must be positive and finite; got {scale!r}")
    return float(np.tanh(v / s))


def centered_unit_interval_score(value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    u = float(np.clip(v, 0.0, 1.0))
    return float(2.0 * u - 1.0)


def generalization_loss_pct(*, best_value: float, current_value: float) -> float:
    best = float(best_value)
    cur = float(current_value)
    if not np.isfinite(best) or not np.isfinite(cur):
        return float("inf")
    scale = max(abs(best), 1e-9)
    return float(max(0.0, (best - cur) / scale) * 100.0)


def up_s_should_stop(history: Sequence[float], *, successive_worsening_strips: int) -> bool:
    vals = [float(v) for v in list(history) if np.isfinite(float(v))]
    s = int(successive_worsening_strips)
    if s <= 0:
        return False
    if len(vals) < s + 1:
        return False
    recent = vals[-(s + 1):]
    return all(float(recent[i + 1]) < float(recent[i]) for i in range(s))


class GL2StudyStopper:
    def __init__(self, cfg: GL2Config):
        self.cfg = cfg
        self.best_scalar = float("-inf")
        self.best_complete_index = -1

    def _trial_scalar(self, trial: optuna.trial.FrozenTrial) -> float:
        if "scalar_objective" in trial.user_attrs:
            return float(trial.user_attrs["scalar_objective"])
        if trial.value is None:
            return float("-inf")
        return float(trial.value)

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (not bool(self.cfg.enabled)) or trial.state != optuna.trial.TrialState.COMPLETE:
            return
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        n_complete = int(len(completed))
        if n_complete < int(self.cfg.min_trials):
            return
        trial_scalar = self._trial_scalar(trial)
        rel_margin = abs(float(self.best_scalar)) * float(self.cfg.min_relative_improvement_pct) / 100.0
        if (not np.isfinite(self.best_scalar)) or trial_scalar > float(self.best_scalar) + float(rel_margin):
            self.best_scalar = float(trial_scalar)
            self.best_complete_index = int(n_complete)
            return
        gl_pct = generalization_loss_pct(best_value=float(self.best_scalar), current_value=float(trial_scalar))
        no_improve_trials = int(max(0, n_complete - self.best_complete_index))
        if no_improve_trials < int(self.cfg.patience):
            return
        if gl_pct < float(self.cfg.max_generalization_loss_pct):
            return
        LOGGER.info(
            "GL_2 stop after %d complete trials: no improvement for %d trials and current generalization loss %.4f%% exceeds %.4f%%",
            int(n_complete),
            int(no_improve_trials),
            float(gl_pct),
            float(self.cfg.max_generalization_loss_pct),
        )
        study.stop()


def create_study(cfg: StudyConfig) -> optuna.study.Study:
    direction = str(cfg.direction).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"unsupported study direction={direction!r}")
    sampler = optuna.samplers.TPESampler(
        seed=int(cfg.sampler_seed),
        n_startup_trials=int(cfg.n_startup_trials),
        multivariate=True,
    )
    return optuna.create_study(
        study_name=str(cfg.study_name),
        direction=direction,
        sampler=sampler,
    )


def suggest_from_space(trial: optuna.trial.Trial, name: str, spec: dict[str, Any]) -> Any:
    if not isinstance(spec, dict):
        raise ValueError(f"search space entry for {name!r} must be a mapping")
    kind = str(spec.get("type", "")).strip().lower()
    if kind == "int":
        return int(
            trial.suggest_int(
                str(name),
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        )
    if kind == "float":
        return float(
            trial.suggest_float(
                str(name),
                float(spec["low"]),
                float(spec["high"]),
                step=None if "step" not in spec else float(spec["step"]),
                log=bool(spec.get("log", False)),
            )
        )
    if kind == "categorical":
        choices = list(spec.get("choices", spec.get("values", [])))
        if not choices:
            raise ValueError(f"categorical search space for {name!r} requires non-empty choices")
        return trial.suggest_categorical(str(name), choices)
    if kind == "bool":
        choices = [False, True]
        if "choices" in spec:
            choices = [bool(x) for x in list(spec["choices"])]
        return bool(trial.suggest_categorical(str(name), choices))
    if kind == "fixed":
        return spec.get("value")
    raise ValueError(f"unsupported search space type={kind!r} for {name!r}")


def trial_payload(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    scalar = float(trial.user_attrs["scalar_objective"]) if "scalar_objective" in trial.user_attrs else float(trial.value) if trial.value is not None else float("-inf")
    return {
        "number": int(trial.number),
        "state": str(trial.state.name),
        "value": None if trial.value is None else float(trial.value),
        "scalar_objective": float(scalar),
        "params": dict(trial.params),
        "user_attrs": {str(k): v for k, v in trial.user_attrs.items()},
    }


def best_completed_trial(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        raise RuntimeError("study completed with no successful trials")
    direction = str(getattr(study.direction, "name", str(study.direction))).strip().lower()
    if direction == "minimize":
        return min(completed, key=lambda t: float(t.value))
    return max(completed, key=lambda t: float(t.value))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
