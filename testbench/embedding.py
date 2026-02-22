#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

from numpy.typing import NDArray

from typing import Optional

import argparse, io, inspect

import numpy as np

from pathlib                   import Path  

from umap                      import UMAP

from scipy.io                  import savemat 

from sklearn.preprocessing     import StandardScaler

from preprocessing.loaders     import (
    load_spatial_mmap_manifest,
    load_compact_dataset,
    load_ssfe_expert_inputs
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
    SSFEEvaluator,
    MultiviewSSFEEEvaluator
)

from optimization.spaces       import (
    define_multiview_ssfe_space,
    define_spatial_ssfe_space,
    define_tabular_ssfe_space
)

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from models.ssfe               import (
    SpatialSSFE, 
    TabularSSFE,
    MultiviewManagerSSFE
)

from testbench.utils.config    import (
    cv_config,
    load_model_params,
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
MULTI_KEY  = "Manifold/Multiview"

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

def _canon_fips(fips: NDArray) -> NDArray:
    return np.asarray([str(x).strip().zfill(5) for x in np.asarray(fips).reshape(-1)], dtype="U5")

def _spatial_opt(
    *,
    root_dir: str, 
    model_key: str, 
    tile_shape: tuple[int, int, int] = (3, 256, 256),
    param_space=define_spatial_ssfe_space, 
    factory_overrides=None, 
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 
    spatial = load_spatial_mmap_manifest(
        root_dir, 
        tile_shape=tile_shape,
        random_state=random_state
    )
    in_channels = int(spatial["in_channels"])

    loader  = make_mmap_loader(
        tile_shape=tile_shape,
        random_state=random_state
    )

    fixed = dict(factory_overrides or {})
    fixed.update({
        "in_channels": in_channels,
        "tile_size": tile_shape[1],
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

def _load_external_wide_cond(
    *,
    source_fips: NDArray,
    wide_cond_path: str
) -> tuple[NDArray, NDArray]: 
    wide_path = wide_cond_path
    wide_ds   = load_compact_dataset(wide_path)

    wide_x = np.asarray(wide_ds["features"], dtype=np.float64)
    wide_fips = _canon_fips(np.asarray(wide_ds["sample_ids"]))
    if wide_x.ndim != 2 or wide_x.shape[0] != wide_fips.shape[0]: 
        raise ValueError("wide_cond dataset has invalid shape.")

    idx  = {fid: i for i, fid in enumerate(wide_fips)}
    keep = np.asarray([fid in idx for fid in source_fips], dtype=bool)
    if not keep.any(): 
        raise ValueError("no overlapping fips.")

    aligned = np.asarray([wide_x[idx[fid]] for fid in source_fips[keep]])
    dropped = int((~keep).sum())
    print(f"[tabular-ssfe] dropped {dropped} rows missing in wide_cond.")
    return aligned, keep 


def _tabular_ssfe_opt(
    *,
    load_data_fn,
    data_path: str, 
    model_key: str, 
    param_space=define_tabular_ssfe_space,
    factory_overrides=None,
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    **_ 
): 

    data   = load_data_fn(data_path) 
    X      = np.asarray(data["features"], dtype=np.float32)
    W      = np.asarray(data["wide_cond"], dtype=np.float32)
    if W.ndim != 2 or W.shape[0] != X.shape[0]: 
        raise ValueError("shape mismatch.")
    in_dim = int(X.shape[1])

    fixed = dict(factory_overrides or {})
    fixed.update({"in_dim": in_dim})

    def loader_func(_filepath): 
        return {"features": X, "wide_cond": W}

    def ssfe_factory(*, compute_strategy=None, collate_fn=None, **params): 
        _ = compute_strategy 
        merged = dict(fixed)
        merged.update(params)
        if collate_fn is not None: 
            merged.setdefault("collate_fn", collate_fn)
        return TabularSSFE(**merged)

    evaluator = SSFEEvaluator(
        filepath=data_path,
        loader_func=loader_func,
        model_factory=ssfe_factory,
        param_space=param_space,
        random_state=random_state,
        n_runs=1,
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

    def load_data(path): 
        return loader(path) if callable(loader) else loader 

    return _tabular_ssfe_opt(
        load_data_fn=load_data, 
        data_path=data_path,
        model_key=model_key,
        trials=trials,
        random_state=random_state,
        config_path=config_path
    )



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
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
):

    return _spatial_opt(
        root_dir=data_path,
        model_key=model_key,
        tile_shape=tile_shape, 
        trials=trials,
        param_space=define_spatial_ssfe_space,
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
    year: int = 2013, 
    wide_cond_path: Optional[str] = None, 
    **_
):
    if wide_cond_path is None:
        wide_cond_path = project_path("data", "datasets", f"wide_scalar_{year}.mat")

    usps = load_compact_dataset(data_path)
    wide = load_compact_dataset(wide_cond_path)

    X_usps = np.asarray(usps["features"], dtype=np.float32)
    f_usps = _canon_fips(usps["sample_ids"])
    X_wide = np.asarray(wide["features"], dtype=np.float32)
    f_wide = _canon_fips(wide["sample_ids"])

    idx_wide = {f: i for i, f in enumerate(f_wide)}
    keep = np.asarray([f in idx_wide for f in f_usps], dtype=bool)
    if not keep.any():
        raise ValueError(f"no overlapping fips between USPS and wide_cond: {wide_cond_path}")

    X_keep = X_usps[keep]
    f_keep = f_usps[keep]
    W_keep = np.asarray([X_wide[idx_wide[f]] for f in f_keep], dtype=np.float32)

    dropped = int((~keep).sum())
    if dropped:
        print(f"[usps-opt] dropped {dropped} rows missing in wide_cond.")

    aligned = {
        "features": X_keep,
        "wide_cond": W_keep,
    }
    def load_data_fn(_):
        return aligned

    return _tabular_ssfe_opt(
        load_data_fn=load_data_fn, 
        data_path=data_path,
        model_key=model_key,
        trials=trials,
        random_state=random_state,
        config_path=config_path
    )

def test_multiview_opt(
    *,
    admin_path: str = USPS_2013, 
    viirs_root: str = VIIRS_ROOT, 
    model_key: str = MULTI_KEY, 
    viirs_key: str = VIIRS_KEY, 
    admin_key: str = USPS_KEY, 
    tile_shape: tuple[int, int, int] = (3, 256, 256),
    year: int = 2013, 
    trials: int = 30, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    wide_mat_path: Optional[str] = None,
    **_
): 
    viirs_params = load_model_params(config_path, viirs_key)
    admin_params = load_model_params(config_path, admin_key)

    if wide_mat_path is None: 
        wide_mat_path = project_path("data", "datasets", f"wide_scalar_{year}.mat")

    aligned = load_ssfe_expert_inputs(
        admin_path=admin_path,
        viirs_root=viirs_root,
        tile_shape=tile_shape,
        random_state=random_state,
        wide_mat_path=wide_mat_path,
        year=year
    )

    admin_in_dim = np.asarray(aligned["admin"]["features"]).shape[1]
    viirs_in_ch  = aligned["viirs"].get("in_channels", tile_shape[0])

    def loader_func(_):
        return aligned 

    def _filter_kwargs(
        params: dict,
        ctor,
        *,
        deny: tuple[str, ...] = ()
    ) -> dict:
        allowed = set(inspect.signature(ctor.__init__).parameters.keys()) - {"self"}
        deny_set = set(deny)
        return {k: v for k, v in params.items() if k in allowed and k not in deny_set}

    def manager_factory(*, compute_strategy=None, **manager_params):
        _ = compute_strategy
        admin_config = dict(admin_params)
        viirs_config = dict(viirs_params)
        sem_depth    = manager_params.pop("semantic_depth", 2)

        for k in ("in_dim", "random_state", "device"):
            admin_config.pop(k, None)
        for k in ("in_channels", "tile_size", "random_state", "device"):
            viirs_config.pop(k, None)
        admin_config = _filter_kwargs(
            admin_params, TabularSSFE,
            deny=("in_dim", "random_state", "device")
        )
        viirs_config = _filter_kwargs(
            viirs_params, SpatialSSFE,
            deny=("in_channels", "tile_size", "random_state", "device")
        )
        mgr_cfg = _filter_kwargs(
            manager_params, MultiviewManagerSSFE,
            deny=("experts", "random_state", "device")
        )

        admin_config["semantic_depth"] = sem_depth
        viirs_config["semantic_depth"] = sem_depth

        experts = {
            "admin": TabularSSFE(
                in_dim=admin_in_dim,
                **admin_config
            ),
            "viirs": SpatialSSFE(
                in_channels=viirs_in_ch,
                tile_size=tile_shape[1],
                **viirs_config
            )
        }
        return MultiviewManagerSSFE(
            experts=experts,
            random_state=random_state,
            device=str(strategy.device),
            **mgr_cfg
        )

    evaluator = MultiviewSSFEEEvaluator(
        filepath="__virtual__", 
        loader_func=loader_func,
        model_factory=manager_factory,
        param_space=define_multiview_ssfe_space,
        random_state=random_state,
        n_runs=1,
        compute_strategy=strategy
    )

    prior_params = None
    try:
        prior_params = load_model_params(config_path, model_key)
    except Exception:
        prior_params = None

    config = EngineConfig(
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

TESTS = {
    "viirs-opt": test_viirs_opt, 
    "saipe-opt": test_saipe_opt,
    "usps-opt": test_usps_opt,
    "multiview-opt": test_multiview_opt,
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
    parser.add_argument("--year", type=int, default=2013)
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
        year=args.year, 
        random_state=args.random_state,
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
