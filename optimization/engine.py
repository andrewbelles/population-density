#!/usr/bin/env python3 
# 
# engine.py  Andrew Belles  Jan 24th, 2026 
# 
# Multi and Single Process Optuna Optimization Engines 
# 
# 

import optuna, torch, inspect  

import numpy as np

import torch.multiprocessing as mp 

from dataclasses import dataclass 

from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Tuple 

from numpy.typing import NDArray

# ---------------------------------------------------------
# Evaluator Contracts 
# ---------------------------------------------------------

class EvaluatorProtocol(Protocol):
    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]: ... 

    def evaluate(self, params: Dict[str, Any]) -> float: ...


class MultiProcessEvaluatorProtocol(EvaluatorProtocol, Protocol):
    def build_worker_specs(
        self,
        params: Dict[str, Any],
        devices: Optional[List[int]] = None
    ) -> List["WorkerSpec"]: ...

    def reduce_worker_results(self, results: List[float]) -> float: ...

class NestedCVEvaluatorProtocol(EvaluatorProtocol, Protocol):
    def outer_splits(self) -> Iterable[Tuple[NDArray, NDArray]]: ... 

    def inner_evaluator(self, train_idx: NDArray) -> EvaluatorProtocol: ...

    def outer_score(
        self, 
        param: Dict[str, Any], 
        train_idx: NDArray, 
        test_idx: NDArray
    ) -> float: ...

# ---------------------------------------------------------
# Engine Config 
# ---------------------------------------------------------


@dataclass 
class NestedCVConfig: 
    inner_n_trials: int 
    inner_sampler_type: Optional[str] = None 
    
    inner_early_stopping_rounds: Optional[int] = None 
    inner_early_stopping_delta: float = 0.0 

    parallel_outer: bool = False 
    devices: Optional[List[int]] = None 


@dataclass 
class EngineConfig: 
    n_trials: int = 50 
    direction: Literal["maximize", "minimize"] = "maximize" 
    sampler_type: str = "multivariate-tpe"
    random_state: int = 0 

    early_stopping_rounds: Optional[int] = None 
    early_stopping_delta: float = 0.0 

    pruner_type: Optional[str] = None 
    pruner_startup_trials: int = 5 
    pruner_warmup_steps: int = 2 
    pruner_max_resource: int = 125 

    mp_start_method: str = "spawn"
    mp_enabled: bool = False 

    devices: Optional[List[int]] = None 
    enqueue_trials: Optional[List[Dict[str, Any]]] = None 

    nested: Optional[NestedCVConfig] = None 


@dataclass(frozen=True)
class WorkerSpec:
    fn: Callable[..., float]
    kwargs: Dict[str, Any] 
    device_id: Optional[int] = None 

# ---------------------------------------------------------
# Helpers  
# ---------------------------------------------------------

def select_sampler(sampler_type: str, random_state: int): 
    if sampler_type == "cmaes": 
        return optuna.samplers.CmaEsSampler(seed=random_state)
    elif sampler_type == "multivariate-tpe": 
        return optuna.samplers.TPESampler(multivariate=True, seed=random_state) 
    else: 
        return optuna.samplers.TPESampler(multivariate=False, seed=random_state) 

def select_pruner(
    pruner_type: Optional[str],
    n_startup_trials: int = 5, 
    n_warmup_steps: int = 2,
    max_resource: int = 125, 
): 
    if pruner_type is None or pruner_type == "none": 
        return optuna.pruners.NopPruner() 
    elif pruner_type == "hyperband": 
        return optuna.pruners.HyperbandPruner(
            min_resource=n_warmup_steps,
            max_resource=max_resource,
            reduction_factor=3
        )
    elif pruner_type == "median": 
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps
        )
    elif pruner_type == "sha":
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0
        )
    else: 
        raise ValueError(f"unknown pruner_type: {pruner_type}")


def make_early_stop_callabacks(
    early_stopping_rounds: Optional[int],
    early_stopping_delta: float 
) -> List[Callable]: 
    if early_stopping_rounds is None: 
        return []
    
    best_value = None 
    best_trial = None 

    def stagnation_callback(study: optuna.Study, trial: optuna.Trial):
        nonlocal best_value, best_trial 
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return 

        if best_value is None: 
            best_value = study.best_value 
            best_trial = trial.number 
            return 
        
        if study.direction == optuna.study.StudyDirection.MAXIMIZE: 
            improved = study.best_value > best_value + early_stopping_delta 
        else: 
            improved = study.best_value < best_value - early_stopping_delta

        if improved: 
            best_value = study.best_value 
            best_trial = trial.number 
            return 

        if best_trial is not None and trial.number - best_trial >= early_stopping_rounds: 
            print(f"> Early stopping after {early_stopping_rounds} stagnant trials.")
            study.stop()

    return [stagnation_callback]


def worker_entry(fn: Callable[..., float], kwargs: Dict[str, Any], device_id: Optional[int], q): 
    try: 
        if device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        val = fn(**kwargs)
        q.put((True, val, None))
    except Exception as e: 
        q.put((False, None, repr(e)))


def run_workers(
    specs: List[WorkerSpec],
    start_method: str,
    on_result: Optional[Callable[[List[float]], bool]] = None 
) -> List[float]:

    if not specs: 
        raise ValueError("no worker specs provided")

    context = mp.get_context(start_method)
    q       = context.Queue()
    procs   = []

    for spec in specs: 
        p = context.Process(target=worker_entry, args=(spec.fn, spec.kwargs, spec.device_id, q))
        p.start() 
        procs.append(p)

    results = []
    for _ in procs: 
        ok, val, err = q.get() 
        if not ok: 
            for p in procs: 
                if p.is_alive(): 
                    p.terminate() 
            raise RuntimeError(f"worker failed: {err}")

        results.append(val)

        if on_result is not None: 
            try: 
                keep = on_result(results)
            except optuna.TrialPruned:
                for p in procs: 
                    if p.is_alive():
                        p.terminate()
                raise 
            if keep is False: 
                for p in procs:
                    if p.is_alive(): 
                        p.terminate()
                raise optuna.TrialPruned()

    for p in procs: 
        p.join() 

    return results


def run_nested_cv(
    name: str,
    evaluator: NestedCVEvaluatorProtocol,
    base_config: EngineConfig,
    nested_config: NestedCVConfig 
): 

    if nested_config.parallel_outer and hasattr(evaluator, "build_nested_worker_specs"): 
        specs = evaluator.build_nested_worker_specs(name, nested_config, base_config)
        if specs: 
            results = run_workers(specs, base_config.mp_start_method)
            
            if results and isinstance(results[0], tuple):
                scores = [float(r[0]) for r in results]
                fold_params = [r[1] for r in results]
            else: 
                scores = [float(r) for r in results]
                fold_params = None 

            mean_score = float(np.mean(scores))
            best_idx   = (int(np.argmin(scores)) if base_config.direction == "minimize" else
                          int(np.argmax(scores)))
            best_params = fold_params[best_idx] if fold_params else None 
            return best_params, mean_score, scores, best_idx 

    fold_scores = []
    fold_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(evaluator.outer_splits()): 
        inner_config = EngineConfig(
            n_trials=nested_config.inner_n_trials,
            direction=base_config.direction,
            sampler_type=nested_config.inner_sampler_type or base_config.sampler_type,
            random_state=base_config.random_state,
            early_stopping_rounds=nested_config.inner_early_stopping_rounds,
            early_stopping_delta=nested_config.inner_early_stopping_delta,
            mp_enabled=False
        )

        inner_eval        = evaluator.inner_evaluator(train_idx)
        best_params, _, _ = run_optimization(f"{name}_fold{fold_idx}", inner_eval, inner_config)

        score = evaluator.outer_score(best_params, train_idx, test_idx)
        fold_scores.append(float(score))
        fold_params.append(best_params)

    mean_score = float(np.mean(fold_scores))
    if base_config.direction == "maximize": 
        best_idx = int(np.argmax(fold_scores))
    else: 
        best_idx = int(np.argmin(fold_scores))

    return fold_params[best_idx], mean_score, fold_scores, fold_params 


# ---------------------------------------------------------
# Public Interface  
# ---------------------------------------------------------

def run_optimization(
    name: str, 
    evaluator: EvaluatorProtocol,
    config: EngineConfig
):
    if config.nested is not None: 
        if not hasattr(evaluator, "outer_splits"):
            raise TypeError("NestedCV requires evaluator with "
                            "outer_splits/inner_evaluator/outer_score")
        # technically speaking we are assuming has outer_splits => is NestedCV evaluator
        return run_nested_cv(name, evaluator, config, config.nested)

    pruner    = select_pruner(
        config.pruner_type,
        n_startup_trials=config.pruner_startup_trials,
        n_warmup_steps=config.pruner_warmup_steps,
        max_resource=config.pruner_max_resource 
    )

    sampler   = select_sampler(config.sampler_type, config.random_state)
    callbacks = make_early_stop_callabacks(
        config.early_stopping_rounds, 
        config.early_stopping_delta
    )

    study = optuna.create_study(
        study_name=name, 
        direction=config.direction,
        sampler=sampler,
        pruner=pruner
    )

    if config.enqueue_trials:
        for params in config.enqueue_trials:
            study.enqueue_trial(dict(params))

    def objective(trial: optuna.Trial): 
        params = evaluator.suggest_params(trial)

        try: 
            if config.mp_enabled and hasattr(evaluator, "build_worker_specs"):
                devices = config.devices 
                if devices is None and torch.cuda.is_available():
                    devices = list(range(torch.cuda.device_count()))
                specs   = evaluator.build_worker_specs(params, devices=devices)

                def on_result(results): 
                    score = evaluator.reduce_worker_results(results)
                    trial.report(score, step=len(results))
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    return True 

                results = run_workers(specs, config.mp_start_method, on_result=on_result)
                score   = evaluator.reduce_worker_results(results)
            else: 
                if "trial" in inspect.signature(evaluator.evaluate).parameters: 
                    score   = evaluator.evaluate(params, trial=trial)
                else: 
                    score   = evaluator.evaluate(params)
        except optuna.TrialPruned: 
            raise 
        except Exception as e: 
            print(f"trial failure: {e}")
            return float("-inf") if config.direction == "maximize" else float("inf")

        return score 
        
    print(f"OPTIMIZATION: Starting {name} ({config.n_trials} trials)")
    study.optimize(
        objective, 
        n_trials=config.n_trials, 
        callbacks=callbacks,
        n_jobs=len(config.devices)
    )

    print("> Optimization Results:")
    print(f"Best Value: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    return study.best_params, study.best_value, study
