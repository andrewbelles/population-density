#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

import argparse, io 

import numpy as np

from torch.utils.data import Subset

from scipy.io import savemat 

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

from models.estimators         import SpatialClassifier, make_spatial_sfe 

from analysis.cross_validation import CVConfig 

from utils.helpers             import (
    save_model_config,
    project_path
)

DEFAULT_MODEL_KEY = "Spatial/VIIRS_ROI"

def _row_score(name: str, score: float): 
    return {"Name": name, "Loss": format_metric(score)}

def test_spatial_opt(
    *,
    data_path: str = project_path("data", "datasets"),
    model_key: str = DEFAULT_MODEL_KEY,
    canvas_hw: tuple[int, int] = (512, 512), 
    trials: int = 50, 
    folds: int = 1, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 
    loader  = make_roi_loader(canvas_hw=canvas_hw) 
    factory = make_spatial_sfe()

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
        direction="minimize",
        random_state=random_state,
        sampler_type="multivariate-tpe"
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "Loss"],
        "row": _row_score(model_key, best_value),
        "params": best_params
    }

def test_spatial_extract(
    *,
    data_path: str = project_path("data", "datasets"),
    model_key: str = DEFAULT_MODEL_KEY,
    canvas_hw: tuple[int, int] = (512, 512), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    out_path: str | None = None,
    **_
): 
    loader_data = make_roi_loader(canvas_hw=canvas_hw)(data_path)
    ds          = loader_data["dataset"] 
    labels      = np.asarray(loader_data["labels"], dtype=np.int64).reshape(-1)
    fips        = np.asarray(loader_data["sample_ids"]).astype("U5")
    collate_fn  = loader_data["collate_fn"]
    params      = load_model_params(config_path, model_key)
    
    conv = params.get("conv_channels")
    if isinstance(conv, str): 
        params["conv_channels"] = tuple(int(x) for x in conv.split("-") if x)

    params.setdefault("random_state", random_state)
    params.setdefault("collate_fn", collate_fn)

    config   = eval_config(random_state)
    splitter = config.get_splitter(OPT_TASK) 

    embs = None 
    for train_idx, test_idx in splitter.split(np.arange(len(labels)), labels): 
        model = SpatialClassifier(**params)
        model.fit(Subset(ds, train_idx), labels[train_idx]) 
        emb   = model.extract(Subset(ds, test_idx))

        if embs is None: 
            embs = np.zeros((len(labels), emb.shape[1]), dtype=emb.dtype)
        embs[test_idx] = emb 

    if embs is None: 
        raise ValueError("failed to extract any embeddings")

    feature_names = np.array([f"viirs_emb_{i}" for i in range(embs.shape[1])], dtype="U32")

    if out_path is None: 
        out_path = project_path("data", "datasets", "viirs_pooled_nchs_2023.mat")

    savemat(out_path, {
        "features": embs, 
        "labels": labels.reshape(-1, 1), 
        "fips_codes": fips, 
        "feature_names": feature_names, 
        "n_counties": np.array([len(labels)], dtype=np.int64)
    }) 

    return {
        "header": ["Name", "Path", "Dim"],
        "row": {"Name": model_key, "Path": out_path, "Dim": embs.shape[1]}
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
    "spatial_opt": test_spatial_opt,
    "spatial_extract": test_spatial_extract
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
