#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

from typing_extensions import evaluate_forward_ref
from numpy.typing import NDArray

import argparse, io

import numpy as np

from pathlib                   import Path  

from umap                      import UMAP

from scipy.io                  import savemat 

from sklearn.preprocessing     import StandardScaler

from preprocessing.loaders     import (
    load_spatial_mmap_manifest,
    load_compact_dataset 
) 

from testbench.utils.paths     import (
    CONFIG_PATH,
    PROBA_PATH,
)

from testbench.utils.data      import (
    make_dataset_loader,
    make_mmap_loader, 
    load_embedding_mat 
)

from optimization.evaluators   import (
    TabularEvaluator,
    SSFEEvaluator
)

from optimization.spaces       import (
    define_tabular_space,
    define_spatial_ssfe_space
)

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from models.ssfe               import SpatialSSFE

from testbench.utils.config    import (
    cv_config,
    load_model_params,
    make_residual_tabular,
    load_node_anchors
)

from testbench.utils.metrics   import (
    OPT_TASK,
)

from testbench.utils.etc       import (
    run_tests_table,
    format_metric 
)

from utils.helpers             import (
    save_model_config,
    project_path
)

from utils.resources import ComputeStrategy 

strategy = ComputeStrategy.from_env()

USPS_2013  = project_path("data", "datasets", "usps_scalar_2013.mat")

VIIRS_ROOT = project_path("data", "tensors", "viirs_2013")
NLCD_ROOT  = project_path("data", "tensors", "nlcd") 
USPS_ROOT  = project_path("data", "tensors", "usps_2013")

VIIRS_KEY  = "Manifold/VIIRS"
NLCD_KEY   = "Manifold/NLCD"
USPS_KEY   = "Manifold/USPS"
SAIPE_KEY  = "Manifold/SAIPE"

# Out Paths 
USPS_OUT              = project_path("data", "datasets", "usps_pooled.mat")
VIIRS_OUT             = project_path("data", "datasets", "viirs_pooled.mat")
VIIRS_OUT_WITH_LOGITS = project_path("data", "datasets", "viirs_pooled_with_logits.mat")
NLCD_OUT              = project_path("data", "datasets", "nlcd_pooled.mat")

VIIRS_ANCHORS = project_path("data", "anchors", "viirs.npy")
USPS_ANCHORS  = project_path("data", "anchors", "usps.npy")

# ---------------------------------------------------------
# Test Helpers 
# ---------------------------------------------------------

def _row_score(name: str, score: float): 
    return {"Name": name, "SSFE Loss": format_metric(score)}

def _spatial_opt(
    *,
    root_dir: str, 
    model_key: str, 
    tile_shape: tuple[int, int, int] = (3, 256, 256),
    node_anchors: list[list[float]], 
    anchor_stats: NDArray,  
    param_space=define_spatial_ssfe_space, 
    factory_overrides=None, 
    max_bag_size: int = 64, 
    sample_frac: float | None = None, 
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 

    anchors = np.asarray(node_anchors, dtype=np.float32)
    stats   = np.asarray(anchor_stats, dtype=np.float32)

    spatial = load_spatial_mmap_manifest(
        root_dir, 
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state
    )
    in_channels = int(spatial["in_channels"])

    loader  = make_mmap_loader(
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state
    )

    fixed = dict(factory_overrides or {})
    fixed.update({
        "in_channels": in_channels,
        "tile_size": tile_shape[1],
        "node_anchors": anchors.tolist(), 
        "anchor_stats": stats.tolist() 
    })

    def ssfe_factory(*, compute_strategy=None, collate_fn=None, **params): 
        merged = dict(fixed)
        merged.update(params)
        if collate_fn is not None: 
            merged.setdefault("collate_fn", collate_fn)
        return SpatialSSFE(**merged)

    evaluator = SSFEEvaluator(
        filepath=root_dir,
        loader_func=loader,
        model_factory=ssfe_factory,
        param_space=param_space,
        random_state=random_state,
        n_runs=2,
        compute_strategy=strategy
    )

    prior_params = None 
    try: 
        prior_params = load_model_params(config_path, model_key)
    except Exception: 
        prior_params = None 

    config  = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state,
        sampler_type="multivariate-tpe",
        mp_enabled=False,
        enqueue_trials=[prior_params] if prior_params else None,
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "SSFE Loss"],
        "row": _row_score(model_key, best_value),
        "params": best_params
    }

# ---------------------------------------------------------
# Tests Entry Point 
# ---------------------------------------------------------

def test_saipe_opt(
    *,
    data_path: str = project_path("data", "datasets"),
    dataset_key: str = "SAIPE_2013",
    proba_path: str = PROBA_PATH,
    model_key: str = SAIPE_KEY,
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 
    loader = make_dataset_loader(dataset_key, proba_path)[dataset_key] 
    data   = loader(data_path) if callable(loader) else loader 
    X      = np.asarray(data["features"], dtype=np.float32)
    y      = np.asarray(data["labels"], dtype=np.float32).reshape(-1) 

    evaluator = TabularEvaluator(
        X, y,
        define_tabular_space,
        model_factory=make_residual_tabular(),
        random_state=random_state,
        compute_strategy=strategy
    )

    prior_params = None 
    try: 
        prior_params = load_model_params(config_path, model_key)
    except Exception: 
        prior_params = None 

    config  = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state,
        sampler_type="multivariate-tpe",
        enqueue_trials=[prior_params] if prior_params else None,
        devices=strategy.visible_devices()
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "Corn + RPS"],
        "row": _row_score(model_key, best_value),
        "params": best_params 
    }

def test_reduce_all( 
    *,
    embedding_paths: list[str] | None = None, 
    out_dir: str | Path | None = None, 
    n_components: int = 2,
    n_neighbors: int = 15, 
    min_dist: float = 0.3,
    metric: str = "euclidean",
    **_ 
): 
    if embedding_paths is None: 
        embedding_paths = [
            project_path("data", "datasets", "viirs_2019_pooled.mat"),
            project_path("data", "datasets", "saipe_2019_pooled.mat"),
            project_path("data", "datasets", "usps_2019_pooled.mat")
        ]

    if out_dir is None: 
        out_dir = project_path("testbench", "local")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for path in embedding_paths: 
        p = Path(path)
        if not p.exists(): 
            print(f"[reduce_all] skip missing: {p}")
            continue 

        X, y     = load_embedding_mat(str(p))
        X        = StandardScaler().fit_transform(X)
        reducer  = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            target_metric="l1",
            target_weight=0.3,
            n_jobs=strategy.n_jobs,
        )
        coords   = reducer.fit_transform(X, y)
        name     = p.stem 
        out_path = out_dir / f"{name}_pca2d.mat"

        savemat(out_path, {
            "coords": coords,
            "labels": y.reshape(-1, 1),
            "name": name,
            "method": "umap",
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric
        })

        rows.append({
            "Name": name,
            "Path": str(out_path),
            "Dim": coords.shape[1]
        })

    if not rows: 
        raise ValueError("no embedding files found")

    return {
        "header": ["Name", "Path", "Dim"],
        "rows": rows 
    }

def test_viirs_opt(
    *,
    data_path: str = VIIRS_ROOT, 
    model_key: str = VIIRS_KEY,
    tile_shape: tuple[int, int, int] = (3, 256, 256), 
    viirs_anchors: str = VIIRS_ANCHORS,
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
):

    node_anchors, anchor_stats = load_node_anchors(viirs_anchors)

    return _spatial_opt(
        root_dir=data_path,
        model_key=model_key,
        tile_shape=tile_shape, 
        trials=trials,
        param_space=define_spatial_ssfe_space,
        node_anchors=node_anchors,
        anchor_stats=anchor_stats,
        random_state=random_state,
        config_path=config_path
    )

def test_usps_opt(
    *,
    data_path: str = USPS_2013,
    model_key: str = "Manifold/USPS",
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    **_
):
    data   = load_compact_dataset(data_path)
    X      = np.asarray(data["features"], dtype=np.float32)
    y      = np.asarray(data["labels"], dtype=np.float32).reshape(-1)

    evaluator = TabularEvaluator(
        X, y,
        define_tabular_space,
        model_factory=make_residual_tabular(),
        random_state=random_state,
        compute_strategy=strategy
    )

    prior_params = None
    try:
        prior_params = load_model_params(config_path, model_key)
    except Exception:
        prior_params = None

    config  = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state,
        sampler_type="multivariate-tpe",
        enqueue_trials=[prior_params] if prior_params else None,
        devices=strategy.visible_devices()
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "Corn + RPS"],
        "row": _row_score(model_key, best_value),
        "params": best_params
    }

# ---------------------------------------------------------
# Tests Entry Point 
# ---------------------------------------------------------

TESTS = {
    "viirs-opt": test_viirs_opt, 
    "saipe-opt": test_saipe_opt,
    "usps-opt": test_usps_opt,
    "reduce-all": test_reduce_all,
}

def _call_test(fn, name, **kwargs): 
    print(f"[{name}] starting...")
    return fn(**kwargs)

ablation_groups=[
    ["gem","entropy","var"],
    ["max","entropy","var"],
    ["logsum","entropy","var"],
    ["gem","entropy"]
]

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--random-state", default=0)
    parser.add_argument("--embedding-paths", default=None)
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
        trials=args.trials,
        folds=args.folds,
        embedding_paths=args.embedding_paths,
        random_state=args.random_state,
        ablation_groups=ablation_groups,
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
