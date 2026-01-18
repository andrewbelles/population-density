#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

import argparse, io 

from testbench.utils.paths     import (
    CONFIG_PATH,
)

from testbench.utils.data      import make_roi_loader 

from analysis.hyperparameter   import (
    run_optimization,
    define_spatial_space,
    SpatialEvaluator
)

from testbench.utils.config    import (
    load_model_params,
    eval_config
)

from testbench.utils.metrics   import OPT_TASK

from testbench.utils.etc       import (
    run_tests_table,
    format_metric 
)

from models.estimators         import make_spatial_ordinal

from analysis.cross_validation import CVConfig 

from utils.helpers             import (
    save_model_config,
    project_path
)

DEFAULT_MODEL_KEY = "Spatial/VIIRS_ROI"

def _row_score(name: str, score: float): 
    return {"Name": name, "F1": format_metric(score)}

def test_spatial_opt(
    *,
    data_path: str = project_path("data", "datasets"),
    model_key: str = DEFAULT_MODEL_KEY,
    canvas_hw: tuple[int, int] = (512, 512), 
    trials: int = 50, 
    folds: int = 1, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH 
): 
    loader  = make_roi_loader(canvas_hw=canvas_hw) 
    factory = make_spatial_ordinal()

    config  = CVConfig(
        n_splits=folds, 
        n_repeats=1, 
        stratify=True, 
        random_state=random_state
    )
    config.verbose = False 

    evaluator = SpatialEvaluator(
        filepath=data_path,
        loader_func=loader,
        model_factory=factory,
        param_space=define_spatial_space,
        task=OPT_TASK,
        config=config
    )

    best_params, best_value = run_optimization(
        name=model_key,
        evaluator=evaluator,
        n_trials=trials,
        direction="maximize",
        random_state=random_state,
        sampler_type="multivariate-tpe"
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "F1"],
        "row": _row_score(model_key, best_value),
        "params": best_params
    }
'''
def test_spatial(
    *,
    data_path: str = project_path("data", "datasets"),
    model_key: str = DEFAULT_MODEL_KEY,
    canvas_hw: tuple[int, int] = (512, 512), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH 
):
    
    loader  = make_roi_loader(canvas_hw=canvas_hw) 
    factory = make_spatial_ordinal()
    config  = eval_config(random_state)

    evaluator = SpatialEvaluator(
        filepath=data_path,
        loader_func=loader,
        model_factory=factory,
        param_space=define_spatial_space,
    )
'''

TESTS = {
    "spatial_opt": test_spatial_opt
}

def _call_test(fn, name, **kwargs): 
    print(f"[{name}] starting...")
    return fn(**kwargs)

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--data-path", default=project_path("data", "datasets"))
    parser.add_argument("--trials", default=50)
    parser.add_argument("--folds", default=2)
    parser.add_argument("--canvas-hw", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--random-state", default=0)
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())

    run_tests_table(
        buf, 
        TESTS,
        targets=targets,
        caller=lambda fn, name, **kw: _call_test(
            fn, name, **kw 
        ),
        data_path=args.data_path,
        trials=args.trials,
        folds=args.folds,
        random_state=args.random_state,
        canvas_hw=tuple(args.canvas_hw)
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
