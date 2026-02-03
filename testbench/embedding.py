#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

import argparse, io, torch, gc

import numpy as np

from pathlib                   import Path  

from torch.utils.data          import Subset

from umap                      import UMAP

from scipy.io                  import savemat 

from functools                 import partial 

from sklearn.preprocessing     import StandardScaler

from sklearn.model_selection   import StratifiedGroupKFold

from sklearn.decomposition     import PCA 

from preprocessing.loaders     import load_spatial_mmap_manifest

from testbench.utils.paths     import (
    CONFIG_PATH,
    PROBA_PATH,
)

from testbench.utils.data      import (
    load_spatial_dataset,
    make_dataset_loader,
    make_mmap_loader, 
    load_embedding_mat 
)

from optimization.evaluators   import (
    ProjectorEvaluator,
    SpatialEvaluator
)

from optimization.spaces       import (
    define_projector_space,
    define_manifold_projector_space,
    define_spatial_space,
    define_hgnn_space
)

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from testbench.utils.config    import (
    cv_config,
    load_model_params,
    eval_config,
    make_spatial_gat,
    normalize_spatial_params,
    with_spatial_channels
)

from testbench.utils.metrics   import (
    OPT_TASK,
)

from testbench.utils.etc       import (
    run_tests_table,
    format_metric 
)

from models.estimators         import (
    EmbeddingProjector,
    SpatialClassifier, 
    make_spatial_sfe,
)

from utils.helpers             import (
    save_model_config,
    project_path
)

from utils.resources import ComputeStrategy 

strategy = ComputeStrategy.from_env()

VIIRS_ROOT = project_path("data", "tensors", "viirs_2013")
NLCD_ROOT  = project_path("data", "tensors", "nlcd") 
USPS_ROOT  = project_path("data", "tensors", "usps")

VIIRS_KEY  = "Spatial/VIIRS_ROI"
NLCD_KEY   = "Spatial/NLCD_ROI"
USPS_KEY   = "Spatial/USPS_TRACTS"

# Out Paths 
VIIRS_OUT             = project_path("data", "datasets", "viirs_pooled.mat")
VIIRS_OUT_WITH_LOGITS = project_path("data", "datasets", "viirs_pooled_with_logits.mat")
NLCD_OUT              = project_path("data", "datasets", "nlcd_pooled.mat")

# ---------------------------------------------------------
# Test Helpers 
# ---------------------------------------------------------

def _row_score(name: str, score: float): 
    return {"Name": name, "RPS": format_metric(score)}

def _projector_fold_factory(name: str, proj_trials: int, random_state: int): 
    cached_params = None 

    def _projector_fold(train_emb, train_y, val_emb, val_y, *, random_state, fold): 
        nonlocal cached_params

        if cached_params is None: 
            config = EngineConfig(
                n_trials=proj_trials, 
                direction="minimize", 
                random_state=random_state
            )

            proj_eval = ProjectorEvaluator(
                train_emb, 
                train_y, 
                define_projector_space, 
                random_state=random_state + fold,
                compute_strategy=strategy
            )

            cached_params, _, _ = run_optimization(
                name="Spatial/VIIRS_ROI_Projector",
                evaluator=proj_eval,
                config=config
            )
            cached_params = dict(cached_params)

        proj = EmbeddingProjector(
            in_dim=train_emb.shape[1],
            **cached_params,
            random_state=random_state + fold,
            device=strategy.device
        )
        proj.fit(train_emb, train_y)
        return proj.transform(val_emb)

    return _projector_fold

def _holdout_embeddings(
    ds,
    labels,
    splitter,
    model_factory,
    extract_fn, 
    postprocess=None, 
    random_state: int = 0, 
    subset_fn=None 
): 
    n   = len(labels)
    out = None 

    if subset_fn is None: 
        subset_fn = lambda data, idx: Subset(data, idx)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.arange(n), labels)): 
        train_data = subset_fn(ds, train_idx)
        val_data   = subset_fn(ds, val_idx)

        model = model_factory() 
        if callable(model) and not hasattr(model, "fit"): 
            model = model() 
        if isinstance(train_data, np.ndarray): 
            scaler     = StandardScaler() 
            train_data = scaler.fit_transform(train_data)
            val_data   = scaler.transform(val_data)

        model.fit(train_data, labels[train_idx]) 

        train_emb = extract_fn(model, train_data) if postprocess else None 
        val_emb   = extract_fn(model, val_data)

        if postprocess:
            val_emb = postprocess(
                train_emb, labels[train_idx],
                val_emb, labels[val_idx],
                random_state=random_state,
                fold=fold_idx 
            )

        if out is None: 
            out = np.zeros((n, val_emb.shape[1]), dtype=val_emb.dtype)
        out[val_idx] = val_emb 

        del model, train_emb, val_emb  
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
        gc.collect() 

    if out is None: 
        raise ValueError("no embeddings returned")

    return out 

def _spatial_opt(
    *,
    root_dir: str, 
    model_key: str, 
    tile_shape: tuple[int, int, int] = (1, 256, 256), 
    max_bag_size: int = 64, 
    sample_frac: float | None = None, 
    trials: int = 50, 
    folds: int = 2, 
    random_state: int = 0, 
    bag_tiles: bool = False,
    config_path: str = CONFIG_PATH,
    **_
): 

    spatial = load_spatial_mmap_manifest(
        root_dir, 
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state
    )

    loader  = make_mmap_loader(
        tile_shape=tile_shape,
        max_bag_size=max_bag_size,
        sample_frac=sample_frac,
        random_state=random_state
    )

    factory = make_spatial_gat(compute_strategy=strategy)
    factory = with_spatial_channels(factory, spatial)

    evaluator = SpatialEvaluator(
        filepath=root_dir,
        loader_func=loader,
        model_factory=factory,
        param_space=define_hgnn_space,
        compute_strategy=strategy,
        task=OPT_TASK,
        config=cv_config(folds, random_state)
    )

    devices = strategy.visible_devices()
    config  = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state,
        sampler_type="multivariate-tpe",
        mp_enabled=False,#(True if devices else False),
        devices=devices,
        pruner_type=None,
        pruner_warmup_steps=5,
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "RPS"],
        "row": _row_score(model_key, best_value),
        "params": best_params
    }

def _spatial_extract(
    *,
    root_dir: str, 
    out_path: str,
    model_key: str,
    canvas_hw: tuple[int, int] = (512, 512), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    proj_trials: int = 50,
    **_
): 
    ds, labels, fips, collate_fn, in_channels = load_spatial_dataset(root_dir, canvas_hw)

    params = load_model_params(config_path, model_key)
    params = normalize_spatial_params(params, random_state=random_state, collate_fn=collate_fn)

    splitter      = eval_config(random_state).get_splitter(OPT_TASK) 
    model_factory = lambda: SpatialClassifier(
        in_channels=in_channels, 
        compute_strategy=strategy, 
        **params
    )

    projector     = _projector_fold_factory(
        f"{model_key}_Projector",
        proj_trials,
        random_state 
    )

    embs = _holdout_embeddings(
        ds, labels, splitter, 
        model_factory=model_factory, 
        extract_fn=lambda m, subset: m.extract(subset), 
        postprocess=projector, 
        random_state=random_state 
    )

    feature_names = np.array(
        [f"{model_key.split('/')[-1].lower()}_emb_{i}" for i in range(embs.shape[1])], 
        dtype="U32"
    )

    savemat(out_path, {
        "features": embs, 
        "labels": labels.reshape(-1, 1), 
        "fips_codes": fips, 
        "feature_names": feature_names, 
        "n_counties": np.array([len(labels)], dtype=np.int64)
    }) 

    return {
        "header": ["Name", "Path", "Dim"],
        "row": {
            "Name": model_key, 
            "Path": out_path, 
            "Dim": embs.shape[1], 
        }
    }

# ---------------------------------------------------------
# Tests Entry Point 
# ---------------------------------------------------------

def test_saipe_opt(
    *,
    data_path: str = project_path("data", "datasets"),
    dataset_key: str = "SAIPE",
    proba_path: str = PROBA_PATH,
    model_key: str = "SAIPE/Manifold",
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 
    loader = make_dataset_loader(dataset_key, proba_path)[dataset_key] 
    data   = loader(data_path) if callable(loader) else loader 
    X      = np.asarray(data["features"], dtype=np.float32)
    y      = np.asarray(data["labels"], dtype=np.int64).reshape(-1) 

    evaluator = ProjectorEvaluator(
        X, y,
        define_manifold_projector_space,
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
        "header": ["Name", "QWK"],
        "row": _row_score(model_key, best_value),
        "params": best_params 
    }


def test_saipe_extract(
    *,
    data_path: str = project_path("data", "datasets"), 
    dataset_key: str = "SAIPE",
    model_key: str = "SAIPE/Manifold",
    out_path: str = project_path("data", "datasets", "saipe_pooled.mat"),
    proba_path: str = PROBA_PATH,
    folds: int = 5,
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    **_ 
): 
    loader = make_dataset_loader(dataset_key, proba_path)[dataset_key] 
    data   = loader(data_path) if callable(loader) else loader 
    X      = np.asarray(data["features"], dtype=np.float32)
    y      = np.asarray(data["labels"], dtype=np.int64).reshape(-1) 
    fips   = np.asarray(data["sample_ids"])

    params = load_model_params(config_path, model_key)
    params = dict(params)

    params["hidden_dims"] = int(params.pop("base_width")),
    params["mode"] = "manifold"

    splitter      = cv_config(folds, random_state).get_splitter(OPT_TASK)
    model_factory = lambda: EmbeddingProjector(
        in_dim=X.shape[1],
        random_state=random_state,
        device=strategy.device,
        **params
    )

    embs = _holdout_embeddings(
        X, y, splitter, 
        model_factory=model_factory,
        extract_fn=lambda m, subset: m.transform(subset),
        subset_fn=lambda data, idx: data[idx]
    )

    feature_names = np.array([f"saipe_manifold_{i}" for i in range(embs.shape[1])],
                             dtype="U32")

    savemat(out_path, {
        "features": embs,
        "labels": y.reshape(-1, 1),
        "fips_codes": fips,
        "feature_names": feature_names,
        "n_counties": np.array([len(y)], dtype=np.int64)
    })

    return {
        "header": ["Name", "Path", "Dim"],
        "row": {
            "Name": model_key,
            "Path": out_path,
            "Dim": embs.shape[1]
        }
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
            project_path("data", "datasets", "viirs_2023_pooled.mat"),
            project_path("data", "datasets", "saipe_2023_pooled.mat")
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
    canvas_hw: tuple[int, int] = (512, 512), 
    tile_hw: tuple[int, int] = (512, 512), 
    bag_tiles: bool = False, 
    trials: int = 50, 
    folds: int = 2, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
):
    return _spatial_opt(
        root_dir=data_path,
        model_key=model_key,
        canvas_hw=canvas_hw,
        tile_hw=tile_hw, 
        bag_tiles=bag_tiles,
        trials=trials,
        folds=folds,
        random_state=random_state,
        config_path=config_path
    )

def test_viirs_extract(
    *,
    data_path: str = VIIRS_ROOT, 
    model_key: str = VIIRS_KEY,
    out_path: str = VIIRS_OUT,
    canvas_hw: tuple[int, int] = (512, 512), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    proj_trials: int = 50,
    **_
):
    return _spatial_extract(
        root_dir=data_path,
        model_key=model_key,
        out_path=out_path,
        canvas_hw=canvas_hw,
        random_state=random_state,
        config_path=config_path,
        proj_trials=proj_trials,
    )

def test_nlcd_opt(
    *,
    data_path: str = NLCD_ROOT, 
    model_key: str = NLCD_KEY,
    canvas_hw: tuple[int, int] = (512, 512), 
    trials: int = 50, 
    folds: int = 2, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
):
    return _spatial_opt(
        root_dir=data_path,
        model_key=model_key,
        canvas_hw=canvas_hw,
        trials=trials,
        folds=folds,
        random_state=random_state,
        config_path=config_path
    )

def test_nlcd_extract(
    *,
    data_path: str = NLCD_ROOT, 
    model_key: str = NLCD_KEY,
    out_path: str = NLCD_OUT,
    canvas_hw: tuple[int, int] = (512, 512), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    proj_trials: int = 50,
    **_
):
    return _spatial_extract(
        root_dir=data_path,
        model_key=model_key,
        out_path=out_path,
        canvas_hw=canvas_hw,
        random_state=random_state,
        config_path=config_path,
        proj_trials=proj_trials,
    )

def test_usps_opt(
    *,
    data_path: str = USPS_ROOT,
    model_key: str = USPS_KEY, 
    canvas_hw: tuple[int, int] = (512, 512), 
    trials: int = 50, 
    folds: int = 2, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    **_ 
): 
    
    return _spatial_opt(
        root_dir=data_path, 
        model_key=model_key,
        canvas_hw=canvas_hw,
        trials=trials, 
        folds=folds,
        random_state=random_state,
        config_path=config_path
    )

# ---------------------------------------------------------
# Tests Entry Point 
# ---------------------------------------------------------

TESTS = {
    "viirs-opt": test_viirs_opt, 
    "viirs-extract": test_viirs_extract,
    "nlcd-opt": test_nlcd_opt,
    "nlcd-extract": test_nlcd_extract,
    "saipe-opt": test_saipe_opt,
    "saipe-extract": test_saipe_extract,
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
    parser.add_argument("--canvas-hw", nargs=2, type=int, default=(512, 512))
    parser.add_argument("--tile-hw", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--bag-tiles", action="store_true")
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
        canvas_hw=tuple(args.canvas_hw),
        tile_hw=tuple(args.tile_hw),
        bag_tiles=bool(args.bag_tiles)
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
