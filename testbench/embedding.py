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

from preprocessing.loaders     import load_spatial_roi_manifest

from testbench.utils.paths     import (
    CONFIG_PATH,
    PROBA_PATH,
)

from testbench.utils.data      import (
    load_spatial_dataset,
    make_dataset_loader, 
    make_roi_loader, 
    load_embedding_mat 
)

from optimization.evaluators   import (
    ProjectorEvaluator,
    SpatialEvaluator
)

from optimization.spaces       import (
    define_projector_space,
    define_manifold_projector_space,
    define_spatial_space
)

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from testbench.utils.config    import (
    cv_config,
    load_model_params,
    eval_config,
    normalize_spatial_params,
    with_spatial_channels
)

from testbench.utils.metrics   import (
    OPT_TASK,
    linear_cka,
    cca_score,
    block_mi,
    distance_correlation
)

from testbench.utils.etc       import (
    run_tests_table,
    format_metric 
)

from testbench.utils.oof       import (
    extract_pooled,
    extract_with_logits,
    fit_without_labels,
    holdout_embeddings,
    make_spatial_classifier,
    subset_by_groups
)

from models.estimators         import (
    EmbeddingProjector,
    SpatialAblation, 
    SpatialClassifier, 
    make_spatial_sfe,
)

from utils.helpers             import (
    save_model_config,
    project_path
)

from utils.resources import ComputeStrategy 

strategy = ComputeStrategy.from_env()

VIIRS_ROOT = project_path("data", "tensors", "viirs_roi")
NLCD_ROOT  = project_path("data", "tensors", "nlcd_roi") 

VIIRS_KEY  = "Spatial/VIIRS_ROI"
NLCD_KEY   = "Spatial/NLCD_ROI"

# Out Paths 
VIIRS_OUT             = project_path("data", "datasets", "viirs_pooled.mat")
VIIRS_OUT_WITH_LOGITS = project_path("data", "datasets", "viirs_pooled_with_logits.mat")
NLCD_OUT              = project_path("data", "datasets", "nlcd_pooled.mat")

# ---------------------------------------------------------
# Test Helpers 
# ---------------------------------------------------------

def _resolve_root(path: str) -> str: 
    p = Path(path)
    return str(p) if p.is_absolute() else project_path(path)

def _row_score(name: str, score: float): 
    return {"Name": name, "Loss": format_metric(score)}

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

def _pca_logits_fold_factory(n_components: int = 11): 
    stats = []

    def _pca_logits_fold(train_emb, train_y, val_emb, val_y, *, random_state, fold): 
        n_classes = len(np.unique(train_y))
        logit_dim = max(1, n_classes - 1) 

        if train_emb.shape[1] <= logit_dim or val_emb.shape[1] <= logit_dim: 
            raise ValueError("expected embeddings concatenated with logits")

        train_feats = train_emb[:, :-logit_dim]
        val_feats   = val_emb[:, :-logit_dim]
        val_logits  = val_emb[:, -logit_dim:]

        pca = PCA(n_components=n_components, random_state=random_state + fold)
        pca.fit(train_feats)
        stats.append(float(np.sum(pca.explained_variance_ratio_)))

        val_pca = pca.transform(val_feats)
        return np.concatenate([val_pca, val_logits], axis=1)
    return _pca_logits_fold, stats

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
    canvas_hw: tuple[int, int] = (512, 512), 
    trials: int = 50, 
    folds: int = 2, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    **_
): 
    spatial = load_spatial_roi_manifest(root_dir, canvas_hw=canvas_hw)
    loader  = make_roi_loader(canvas_hw=canvas_hw) 
    factory = make_spatial_sfe(compute_strategy=strategy)
    factory = with_spatial_channels(factory, spatial)

    evaluator = SpatialEvaluator(
        filepath=root_dir,
        loader_func=loader,
        model_factory=factory,
        param_space=define_spatial_space,
        compute_strategy=strategy,
        task=OPT_TASK,
        config=cv_config(folds, random_state)
    )

    devices = strategy.visible_devices()
    config  = EngineConfig(
        n_trials=trials,
        direction="maximize",
        random_state=random_state,
        sampler_type="multivariate-tpe",
        mp_enabled=(True if devices else False),
        devices=(devices if devices else None)
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
        enqueue_trials=[prior_params] if prior_params else None 
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "Loss"],
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
    params["mode"]    = "manifold"
    params["out_dim"] = 5

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
            project_path("data", "datasets", "viirs_pooled.mat"),
            project_path("data", "datasets", "saipe_pooled.mat")
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

def test_viirs_extract_with_logits(
    *,
    data_path: str = VIIRS_ROOT, 
    model_key: str = VIIRS_KEY,
    out_path: str = VIIRS_OUT_WITH_LOGITS,
    canvas_hw: tuple[int, int] = (512, 512),
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    n_components: int = 11,
    folds: int = 5,
    **_
): 
    ds, labels, fips, collate_fn, in_channels, sample_labels, sample_ids_full, sample_groups = \
        load_spatial_dataset(data_path, canvas_hw)

    params = load_model_params(config_path, model_key)
    params = normalize_spatial_params(params, random_state=random_state, collate_fn=collate_fn)

    is_packed = sample_groups.size > 0 and sample_labels.size > 0 
    if is_packed: 
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state
        )
        splits = (
            (np.sort(tr), np.sort(va)) 
            for tr, va in splitter.split(
                np.zeros_like(sample_labels),
                sample_labels,
                groups=sample_groups
            )
        )
        subset_fn = partial(subset_by_groups, groups=sample_groups) 
        fit_fn    = fit_without_labels 
        labels_in = sample_labels 
        fips_out  = sample_ids_full 
    else: 
        splitter  = cv_config(folds, random_state).get_splitter(OPT_TASK)
        splits    = splitter.split(np.arange(len(labels)), labels)
        subset_fn = None 
        fit_fn    = None 
        labels_in = labels 
        fips_out  = fips 

    model_factory = partial(
        make_spatial_classifier,
        in_channels,
        strategy,
        params,
    )

    projector, pca_stats = _pca_logits_fold_factory(n_components=n_components)

    embs = holdout_embeddings(
        ds, labels_in, splits, 
        model_factory=model_factory,
        extract_fn=extract_with_logits,
        postprocess=projector,
        random_state=random_state,
        subset_fn=subset_fn,
        fit_fn=fit_fn,
        devices=strategy.visible_devices()
    )

    logit_dim = embs.shape[1] - n_components 
    feature_names = np.array(
        [f"{model_key.split('/')[-1].lower()}_pca_{i}" for i in range(n_components)] +
        [f"{model_key.split('/')[-1].lower()}_logit_{i}" for i in range(logit_dim)],
        dtype="U32"
    )

    evr = float(np.mean(pca_stats)) if pca_stats else float("nan")

    savemat(out_path, {
        "features": embs,
        "labels": labels_in.reshape(-1, 1),
        "fips_codes": fips_out, 
        "feature_names": feature_names,
        "n_counties": np.array([len(labels_in)], dtype=np.int64)
    })

    return {
        "header": ["Name", "Logits Dim", "Emb Dim", "EVR"],
        "row": {
            "Name": model_key, 
            "Logits Dim": logit_dim, 
            "Emb Dim": n_components,
            "EVR": evr
        }
    }

def test_viirs_cnn_ablation(
    *,
    data_path: str = VIIRS_ROOT,
    model_key: str = VIIRS_KEY, 
    canvas_hw: tuple[int, int] = (512, 512),
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    pooling_features: list[str] | None = None, 
    ablation_groups: list[list[str]] | None = None, 
    **_
): 

    ds, labels, fips, collate_fn, in_channels, sample_labels, sample_ids_full, sample_groups = \
        load_spatial_dataset(data_path, canvas_hw)

    params = load_model_params(config_path, model_key)
    params = normalize_spatial_params(params, random_state=random_state, collate_fn=collate_fn)

    if pooling_features is None: 
        pooling_features = ["logsum", "gem", "max", "entropy", "var"]

    devices = strategy.visible_devices()

    ab = SpatialAblation(
        classifier_factory=SpatialClassifier,
        classifier_kwargs={
            "in_channels": in_channels,
            "compute_strategy": strategy,
            **params
        },
        pooling_features=pooling_features,
        ablation_groups=ablation_groups,
        device=strategy.device,
        random_state=random_state
    )

    labels_in = sample_labels if sample_labels.size else labels
    results = ab.run(ds, labels_in, devices=devices if devices else None)

    rows = []
    for r in results: 
        rows.append({
            "Ablation": r["name"],
            "Loss": format_metric(r["val_loss"]),
            "QWK": format_metric(r["qwk"]),
            "Dim": r["dim"]
        })

    return {
        "header": ["Ablation", "Loss", "QWK", "Dim"],
        "rows": rows
    }

def test_viirs_pooled_dependence(
    *,
    data_path: str = VIIRS_ROOT,
    model_key: str = VIIRS_KEY, 
    canvas_hw: tuple[int, int] = (512, 512),
    folds: int = 5, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH,
    pooling_features: list[str] | None = None, 
    cca_components: int = 3, 
    mi_components: int = 1, 
    **_
):
    ds, labels, fips, collate_fn, in_channels, sample_labels, sample_ids_full, sample_groups = \
        load_spatial_dataset(data_path, canvas_hw)

    params = load_model_params(config_path, model_key)
    params = normalize_spatial_params(params, random_state=random_state, collate_fn=collate_fn)

    if pooling_features is None: 
        pooling_features = ["logsum", "gem", "max", "entropy", "var"]

    base = getattr(ds, "dataset", ds) 
    is_packed = hasattr(base, "is_packed") and base.is_packed

    if is_packed: 
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True, 
            random_state=random_state
        )
        labels_in = sample_labels 
        splits    = splitter.split(
            np.zeros(len(sample_labels)),
            sample_labels,
            sample_groups 
        )
        subset_fn = partial(subset_by_groups, groups=sample_groups)
        fit_fn    = fit_without_labels
    else: 
        splitter  = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        labels_in = labels 
        splits    = splitter.split(np.zeros(len(labels)), labels, fips)
        subset_fn = None 
        fit_fn    = None 

    embs = holdout_embeddings(
        ds, labels_in, splits, 
        model_factory=partial(
            make_spatial_classifier,
            in_channels,
            strategy,
            params,
            pooling_features,
        ),
        extract_fn=extract_pooled,
        random_state=random_state,
        subset_fn=subset_fn,
        fit_fn=fit_fn,
        devices=strategy.visible_devices()
    )

    X = np.asarray(embs, dtype=np.float32)
    n_blocks = len(pooling_features)
    dim = X.shape[1] // n_blocks 
   
    blocks = {}
    for i, name in enumerate(pooling_features):
        blocks[name] = X[:, i * dim:(i + 1) * dim]

    rows = []
    names = list(pooling_features)
    for i in range(len(names)): 
        for j in range(i + 1, len(names)): 
            a, b = names[i], names[j]
            Xa   = blocks[a] 
            Xb   = blocks[b]

            Xa   = StandardScaler().fit_transform(Xa)
            Xb   = StandardScaler().fit_transform(Xb)

            cka  = linear_cka(Xa, Xb)
            cca  = cca_score(Xa, Xb, n_components=cca_components)
            mi   = block_mi(Xa, Xb, n_components=mi_components, random_state=random_state)
            dcor = distance_correlation(Xa, Xb)

            rows.append({
                "A": a,
                "B": b,
                "CKA": f"{cka:.4f}",
                "CCA": f"{cca:.4f}",
                "MI": f"{mi:.4f}",
                "dCor": f"{dcor:.4f}"
            })

    return {
        "header": ["A", "B", "CKA", "CCA", "MI", "dCor"],
        "rows": rows 
    }

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
    "reduce-all": test_reduce_all,
    "viirs-extract-with-logits": test_viirs_extract_with_logits,
    "viirs-ablation": test_viirs_cnn_ablation,
    "viirs-pooled-dependence": test_viirs_pooled_dependence
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
        canvas_hw=tuple(args.canvas_hw)
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
