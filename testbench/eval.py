#!/usr/bin/env python3 
# 
# eval.py  Andrew Belles  Jan 30th, 2026 
# 
# Train on full dataset A and evaluate on full dataset B. 
# 
# 

import argparse, io, torch, inspect  

import numpy as np

from numpy.typing import NDArray

from typing import Literal, Optional 

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from torch.utils.data import TensorDataset, DataLoader, Subset  

from analysis.cross_validation import (
    CVConfig,
    ScaledEstimator
)

from scipy.io                  import (
    savemat 
) 

from sklearn.preprocessing     import StandardScaler 

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from optimization.evaluators   import (
    TabularEvaluator,
    StandardEvaluator,
    HierarchicalFusionEvaluator
)

from optimization.spaces       import (
    define_tabular_space,
    define_fusion_joint_space
)

from testbench.utils.data      import (
    BASE, 
    load_spatial_dataset,
) 

from testbench.utils.etc       import (
    format_metric,
    get_factory, 
    get_param_space,
    run_tests_table
) 

from testbench.utils.metrics   import (
    OPT_TASK,
    metrics_from_probs,
    mape_wape_from_probs
)

from testbench.utils.config    import (
    load_model_params,
    make_residual_tabular,
    normalize_params,
    normalize_spatial_params,
    load_node_anchors
)

from models.ssfe import (
    TabularSSFE, 
    SpatialSSFE, 
    MultiviewManagerSSFE
)
    
from models.estimators import HierarchicalFusionModel

from preprocessing.labels import PopulationLabels, build_label_map

from testbench.utils.paths     import (
    CONFIG_PATH
)

from preprocessing.loaders     import (
    load_compact_dataset, 
    load_wide_deep_inputs, 
    load_ssfe_expert_inputs
)
    
from utils.resources           import ComputeStrategy

from utils.helpers             import project_path, save_model_config

strategy = ComputeStrategy.from_env() 

CENSUS_DIR     = project_path("data", "census")

VIIRS_KEY  = "Manifold/VIIRS" 
SAIPE_KEY  = "Manifold/SAIPE"
USPS_KEY   = "Manifold/USPS"
FUSION_KEY = "Fusion/2013"
MULTI_KEY  = "Manifold/Multiview"

VIIRS_ANCHORS = project_path("data", "anchors", "viirs.npy")
USPS_ANCHORS  = project_path("data", "anchors", "usps.npy")

EXPERT_PROBA = {
    "SAIPE_2023": project_path("data", "stacking", "saipe_optimized_probs.mat"), 
    "VIIRS_2023": project_path("data", "stacking", "viirs_optimized_probs.mat"), 
    "USPS_2023":  project_path("data", "stacking", "usps_optimized_probs.mat"), 
    "USPS_MANIFOLD": project_path("data", "stacking", "usps_pooled_probs.mat"), 
    "SAIPE_MANIFOLD": project_path("data", "stacking", "saipe_pooled_probs.mat"), 
    "VIIRS_MANIFOLD": project_path("data", "stacking", "viirs_pooled_probs.mat"), 
}

SCALAR_SWEEP_DATASETS = {
    "SAIPE": {
        "train": "SAIPE_2013", 
        "test": "SAIPE_2023", 
        "model_key": "Manifold/SAIPE"
    },
    "USPS": {
        "train": project_path("data", "datasets", "usps_scalar_2013.mat"), 
        "test":  project_path("data", "datasets", "usps_scalar_2023.mat"), 
        "model_key": "Manifold/USPS"
    }
}

def _resolve_dataset(spec: str): 
    if spec in BASE: 
        base = BASE[spec]
        return base["path"], base["loader"]

    def _loader(path): 
        return load_compact_dataset(path)

    return spec, _loader 

def _load_dataset(path, loader): 
    data   = loader(path)
    X      = np.asarray(data["features"], dtype=np.float64)
    y      = np.asarray(data["labels"]).reshape(-1)
    coords = data.get("coords")
    if coords is not None: 
        coords = np.asarray(coords, dtype=np.float64)
    return X, y, coords 

def _edges_for_year(year: int) -> NDArray:
    return np.asarray(
        PopulationLabels(year=year, census_dir=CENSUS_DIR).fit().edges_, dtype=np.float64
    )

def _canon_fips(fips: NDArray) -> NDArray:
    return np.asarray([str(x).strip().zfill(5) for x in np.asarray(fips).reshape(-1)], dtype="U5")

def _soft_rank_for_fips(fips: NDArray, year: int) -> NDArray:
    f   = _canon_fips(fips)
    pl  = PopulationLabels(year=year, census_dir=CENSUS_DIR).fit(feature_fips=f)
    srm = pl.to_soft_rank_map(f)
    return np.asarray([srm.get(fid, np.nan) for fid in f], dtype=np.float32)

def _load_external_wide_cond(
    *,
    source_fips: NDArray,
    year: int, 
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

def _save_embedding_mat(
    out_path: str, 
    *,
    features: NDArray,
    labels: NDArray,
    fips: NDArray,
    soft_rank: NDArray,
    feature_prefix: str,
    year: int
): 
    X  = np.asarray(features, dtype=np.float32)
    y  = np.asarray(labels).reshape(-1, 1)
    f  = _canon_fips(fips)
    sr = np.asarray(soft_rank, dtype=np.float32).reshape(-1, 1)
    names = np.asarray([f"{feature_prefix}_emb_{i}" for i in range(X.shape[1])], dtype="U64")

    if not (X.shape[0] == y.shape[0] == f.shape[0] == sr.shape[0]): 
        raise ValueError("features/labels/fips/soft_rank size mismatch")

    savemat(out_path, {
        "features": X, 
        "labels": y,
        "soft_rank": sr, 
        "fips_codes": f,
        "feature_names": names, 
        "year": np.asarray([int(year)], dtype=np.int64),
    })

class _ExpertZipLoader:
    def __init__(self, loaders: dict[str, DataLoader]):
        if not loaders:
            raise ValueError("expected at least one expert loader")
        self.expert_ids = list(loaders.keys())
        self.loaders = loaders

    def __iter__(self):
        iters = [iter(self.loaders[eid]) for eid in self.expert_ids]
        for batches in zip(*iters):
            yield {eid: b for eid, b in zip(self.expert_ids, batches)}

    def __len__(self):
        return min(len(self.loaders[eid]) for eid in self.expert_ids)

def _filter_ctor_kwargs(params: dict, cls, *, deny: tuple[str, ...] = ()) -> dict:
    allowed = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    deny_set = set(deny)
    return {k: v for k, v in dict(params).items() if k in allowed and k not in deny_set}

def _make_expert_loaders(
    admin_ds,
    viirs_ds,
    *,
    collate_fn,
    batch_size: int,
    shuffle: bool,
):
    pin = str(strategy.device).startswith("cuda")
    common = dict(
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, 
        pin_memory=pin, 
        drop_last=False
    )
    return {
        "admin": DataLoader(admin_ds, **common), 
        "viirs": DataLoader(viirs_ds, collate_fn=collate_fn, **common)
    }

def _optimize_on_train(
    train_path, 
    train_loader, 
    model_name,
    trials,
    folds,
    random_state
): 
    param_space = get_param_space(model_name)
    cv = CVConfig(
        n_splits=folds,
        n_repeats=1,
        stratify=True,
        random_state=random_state
    )

    evaluator = StandardEvaluator(
        filepath=train_path,
        loader_func=train_loader,
        base_factory_func=get_factory(model_name, strategy=strategy),
        param_space=param_space,
        task=OPT_TASK,
        config=cv,
        metric="mae"
    )

    config = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state
    )

    best_params, best_value, _ = run_optimization(
        name=f"{model_name}_trainopt", 
        evaluator=evaluator,
        config=config 
    )

    return dict(best_params), float(best_value) 

def _fit_eval(train_path, train_loader, test_path, test_loader, model_name, params): 
    X_tr, y_tr, coords_tr = _load_dataset(train_path, train_loader)
    X_te, y_te, coords_te = _load_dataset(test_path, test_loader)

    params = normalize_params(model_name, dict(params))
    base   = get_factory(model_name, strategy=strategy)(**params)
    model  = ScaledEstimator(base, scale_X=True, scale_y=False)
    model.fit(X_tr, y_tr, coords_tr)

    probs        = model.predict_proba(X_te, coords_te)
    class_labels = np.sort(np.unique(y_tr))
    metrics      = metrics_from_probs(y_te, probs, class_labels)
    return metrics

def _ssfe_fit_extract(
    *,
    source: str, 
    out_path: str, 
    model_key: str, 
    modality: Literal["tabular", "spatial"], 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    tile_shape: tuple[int, int, int] = (3, 256, 256), 
    year: int = 2013, 
    wide_cond_path: Optional[str] = None, 
    **_
): 
    if modality == "tabular": 
        src_path, src_loader = _resolve_dataset(source)
        src = src_loader(src_path)

        X = np.asarray(src["features"], dtype=np.float32)
        y = np.asarray(src["labels"]).reshape(-1)
        fips = _canon_fips(np.asarray(src["sample_ids"]))

        if wide_cond_path is None: 
            wide_cond_path = project_path("data", "datasets", f"wide_scalar_{year}.mat")

        wide_raw, keep = _load_external_wide_cond(
            source_fips=fips, year=year, wide_cond_path=wide_cond_path
        )
        
        X = X[keep]
        y = y[keep]
        fips = fips[keep]

        scaler_x = StandardScaler() 
        scaler_w = StandardScaler() 
        Xs = scaler_x.fit_transform(X).astype(np.float32, copy=False)
        wide = scaler_w.fit_transform(wide_raw).astype(np.float32, copy=False)

        node_ids = np.arange(Xs.shape[0], dtype=np.int64)
        ds = TensorDataset(
            torch.from_numpy(Xs),
            torch.from_numpy(node_ids),
            torch.from_numpy(wide)
        )

        params = load_model_params(config_path, model_key)
        params.setdefault("random_state", random_state)

        model = TabularSSFE(
            in_dim=Xs.shape[1],
            **params
        )

        model.fit(ds)
        emb = model.extract(ds)

    else: 
        ds, y, fips, collate_fn, in_ch = load_spatial_dataset(
            source, tile_shape=tile_shape, random_state=random_state
        )

        fips = _canon_fips(np.asarray(fips))
        y    = np.asarray(y).reshape(-1)

        params = load_model_params(config_path, model_key)
        params = normalize_spatial_params(
            params, random_state=random_state, collate_fn=collate_fn
        )

        model = SpatialSSFE(**params)

        model.fit(ds)
        emb = model.extract(ds)

    score = model.best_val_score_

    soft_rank = _soft_rank_for_fips(fips, year)

    prefix = model_key.split("/")[-1].lower() 
    _save_embedding_mat(
        out_path, 
        features=emb,
        labels=y,
        fips=fips,
        soft_rank=soft_rank,
        feature_prefix=prefix,
        year=year
    )

    return {
        "header": ["Name", "Year", "SSFE Loss", "Dim", "N Samples"], 
        "row": {
            "Name": model_key, 
            "Year": year, 
            "SSFE Loss": f"{score:.4f}",
            "Dim": emb.shape[1],
            "N Samples": emb.shape[0]
        }
    }

# ---------------------------------------------------------
# Tests 
# ---------------------------------------------------------

def test_fit_eval(
    _buf,
    *,
    train: str, 
    test: str, 
    models: list[str] | None = None, 
    trials: int = 100, 
    folds: int = 3, 
    random_state: int = 0,
    **_
): 

    train_path, train_loader = _resolve_dataset(train)
    test_path, test_loader   = _resolve_dataset(test)

    rows = []
    for model_name in (models or ["Logistic"]):
        best_params, _ = _optimize_on_train(
            train_path, train_loader, model_name, trials, folds, random_state 
        )
        metrics = _fit_eval(
            train_path, train_loader, test_path, test_loader, model_name, best_params 
        )
        rows.append(_row(f"{train}->{test}/{model_name}", metrics))

    return {
        "header": ["Name", "Acc", "F1", "ROC", "ECE", "QWK", "MAE"],
        "rows": rows,
    }

def test_saipe_manifold(
    _buf,
    *,
    test: str, 
    out: str | None = None, 
    model_key: str = SAIPE_KEY, 
    random_state: int = 0,
    config_path: str = CONFIG_PATH,
    year: int = 2013, 
    **_
): 
    if out is None: 
        out = project_path("data", "datasets", f"saipe_embeddings_{year}.mat")

        return _ssfe_fit_extract(
            source=test,
            year=year, 
            out_path=out,
            model_key=model_key,
            modality="tabular",
            random_state=random_state,
            config_path=config_path
        )

def test_usps_manifold(
    _buf,
    *,
    test: str, 
    out: str | None = None, 
    model_key: str = "Manifold/USPS", 
    random_state: int = 0,
    config_path: str = CONFIG_PATH,
    year: int = 2013, 
    **_
):
    if out is None: 
        out = project_path("data", "datasets", f"admin_embeddings_{year}.mat")

    return _ssfe_fit_extract(
        source=test,
        year=year, 
        out_path=out,
        model_key=model_key,
        modality="tabular",
        random_state=random_state,
        config_path=config_path
    )

def test_viirs_manifold(
    _buf,
    *,
    test: str, 
    out: str | None,
    model_key: str = VIIRS_KEY, 
    tile_shape: tuple[int, int, int] = (3, 256, 256), 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    year: int = 2013, 
    **_
): 
    if out is None: 
        out = project_path("data", "datasets", f"viirs_embeddings_{year}.mat")

    return _ssfe_fit_extract(
        source=test,
        year=year, 
        tile_shape=tile_shape, 
        out_path=out,
        model_key=model_key,
        modality="spatial",
        random_state=random_state,
        config_path=config_path
    )

def test_fusion_opt(
    _buf,
    *,
    model_key: str = FUSION_KEY, 
    trials: int = 50, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    year: int = 2013, 
    **_ 
):
    viirs_embeddings = project_path(
        "data", "datasets", f"viirs_embeddings_{year}.mat"
    )
    admin_embeddings = project_path(
        "data", "datasets", f"admin_embeddings_{year}.mat"
    )
    wide_dataset = project_path(
        "data", "datasets", f"wide_scalar_{year}.mat"
    )

    X = load_wide_deep_inputs(
        expert_paths={
            "viirs": viirs_embeddings,
            "admin": admin_embeddings,
        },
        wide_path=wide_dataset
    )

    cut_edges = _edges_for_year(year)
    fixed = {
        "expert_dims": X["expert_dims"], 
        "wide_in_dim": X["wide_in_dim"], 
        "cut_edges": cut_edges
    }

    def model_factory(**params): 
        merged = dict(fixed)
        merged.update(params)
        return HierarchicalFusionModel(
            **merged,
            random_state=random_state,
            compute_strategy=strategy
        )

    evaluator = HierarchicalFusionEvaluator(
        X=X,
        model_factory=model_factory,
        param_space=define_fusion_joint_space,
        cv_folds=3,
        random_state=random_state,
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
        enqueue_trials=[prior_params] if prior_params else None
    )

    best_params, best_value, _ = run_optimization(
        name=model_key,
        evaluator=evaluator,
        config=config
    )

    save_model_config(config_path, model_key, best_params)

    return {
        "header": ["Name", "Year", "Ordinal + Regression Loss"], 
        "row": {
            "Name": model_key, 
            "Year": year, 
            "Ordinal + Regression Loss": f"{best_value:.4f}"
        }
    }

def test_multiview_manifold(
    _buf, 
    *, 
    admin_path: str = project_path("data", "datasets", "usps_scalar_2013.mat"), 
    viirs_root: str = project_path("data", "tensors", "viirs_2013"), 
    out_admin: Optional[str] = None, 
    out_viirs: Optional[str] = None, 
    out_global: Optional[str] = None,
    model_key: str = MULTI_KEY,
    admin_key: str = USPS_KEY, 
    viirs_key: str = VIIRS_KEY, 
    random_state: int = 0, 
    config_path: str = CONFIG_PATH, 
    tile_shape: tuple[int, int, int] = (3, 256, 256), 
    year: int = 2013, 
    eval_fraction: float = 0.2, 
    wide_cond_path: Optional[str] = None, 
    semantic_depth: int = 2,
    mv_global_dim: int = 128,
    mv_gate_floor: float = 0.05,
    mv_lr: float = 1.5e-4,
    mv_weight_decay: float = 1e-6,
    **_ 
): 
    if out_admin is None: 
        out_admin = project_path("data", "datasets", f"admin_embeddings_mv_{year}.mat")
    if out_viirs is None: 
        out_viirs = project_path("data", "datasets", f"viirs_embeddings_mv_{year}.mat")
    if out_global is None: 
        out_global = project_path("data", "datasets", f"shared_embedding_mv_{year}.mat")
    if wide_cond_path is None: 
        wide_cond_path = project_path("data", "datasets", f"wide_scalar_{year}.mat")

    aligned = load_ssfe_expert_inputs(
        admin_path=admin_path,
        viirs_root=viirs_root, 
        tile_shape=tile_shape,
        random_state=random_state,
        wide_mat_path=wide_cond_path,
        year=year 
    )

    fips = _canon_fips(np.asarray(aligned["sample_ids"]))
    y    = np.asarray(aligned["labels"]).reshape(-1)
    soft_rank = _soft_rank_for_fips(fips, year)

    admin_x  = np.asarray(aligned["admin"]["features"], dtype=np.float32)
    admin_w  = np.asarray(aligned["admin"]["wide_cond"], dtype=np.float32)
    node_ids = np.asarray(aligned["admin"]["node_ids"], dtype=np.int64)
    if not (admin_x.shape[0] == admin_w.shape[0] == node_ids.shape[0] == y.shape[0]): 
        raise ValueError("aligned admin tensors have mismatch rows.")

    viirs_ds   = aligned["viirs"]["dataset"]
    collate_fn = aligned["viirs"].get("collate_fn")
    if len(viirs_ds) != y.shape[0]: 
        raise ValueError("aligned viirs dataset rows mismatch labels.")

    admin_ds = TensorDataset(
        torch.from_numpy(admin_x),
        torch.from_numpy(node_ids), 
        torch.from_numpy(admin_w) 
    )
    n = int(y.shape[0])
    if n < 4:
        raise ValueError("need at least 4 aligned rows for train/val split")

    idx = np.arange(n, dtype=np.int64)
    y_bucket = np.floor(np.clip(y, 0.0, None)).astype(np.int64)
    _, counts = np.unique(y_bucket, return_counts=True)
    strat = y_bucket if (counts.size > 1 and counts.min() >= 2) else None
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(eval_fraction),
        random_state=int(random_state),
        shuffle=True,
        stratify=strat
    )

    admin_train = Subset(admin_ds, tr_idx.tolist())
    admin_val   = Subset(admin_ds, va_idx.tolist())
    viirs_train = Subset(viirs_ds, tr_idx.tolist())
    viirs_val   = Subset(viirs_ds, va_idx.tolist())

    admin_params = load_model_params(config_path, admin_key)
    viirs_params = normalize_spatial_params(
        load_model_params(config_path, viirs_key),
        random_state=random_state,
        collate_fn=collate_fn
    )

    mv_params    = load_model_params(config_path, model_key)
    admin_config = _filter_ctor_kwargs(
        admin_params, TabularSSFE, deny=("in_dim", "random_state", "device")
    )
    viirs_config = _filter_ctor_kwargs(
        viirs_params, SpatialSSFE, deny=("in_channels", "tile_size", "random_state", "device")
    )
    mv_config    = _filter_ctor_kwargs(
        mv_params, MultiviewManagerSSFE, deny=("experts", "random_state", "device")
    )

    # Force manifold-generation init to the requested baseline trial config.
    admin_config["semantic_depth"] = int(semantic_depth)
    viirs_config["semantic_depth"] = int(semantic_depth)
    mv_config["global_dim"] = int(mv_global_dim)
    mv_config["gate_floor"] = float(mv_gate_floor)
    mv_config["lr"] = float(mv_lr)
    mv_config["weight_decay"] = float(mv_weight_decay)

    batch_size = mv_config.pop("batch_size", 64)

    experts = {
        "admin": TabularSSFE(in_dim=admin_x.shape[1], **admin_config),
        "viirs": SpatialSSFE(in_channels=aligned["viirs"]["in_channels"], 
                             tile_size=tile_shape[1], **viirs_config) 
    }

    manager = MultiviewManagerSSFE(
        experts=experts,
        random_state=random_state,
        device=str(strategy.device),
        **mv_config 
    )

    train_loaders = _make_expert_loaders(
        admin_train, viirs_train, 
        collate_fn=collate_fn, batch_size=batch_size, shuffle=True
    )

    val_loaders   = _make_expert_loaders(
        admin_val, viirs_val, 
        collate_fn=collate_fn, batch_size=batch_size, shuffle=False 
    )

    full_loaders = _make_expert_loaders(
        admin_ds, viirs_ds, 
        collate_fn=collate_fn, batch_size=batch_size, shuffle=False 
    )

    for eid, ex in manager.experts.items():
        ex.init_fit(
            train_loaders[eid],
            state_loader=full_loaders[eid]
        )

    manager.fit(_ExpertZipLoader(train_loaders), val_loader=_ExpertZipLoader(val_loaders))
    emb = manager.extract(full_loaders)

    _save_embedding_mat(out_admin, features=emb["admin"], labels=y, fips=fips, 
                        soft_rank=soft_rank, feature_prefix=f"admin_mv", year=year)
    _save_embedding_mat(out_viirs, features=emb["viirs"], labels=y, fips=fips, 
                        soft_rank=soft_rank, feature_prefix=f"viirs_mv", year=year)
    _save_embedding_mat(out_global, features=emb["shared_global"], labels=y, fips=fips, 
                        soft_rank=soft_rank, feature_prefix=f"shared_mv", year=year)
    
    return {
        "header": ["Name", "Year", "Dim", "N Samples"],
        "rows": [
            {
                "Name": f"{model_key}/admin", 
                "Year": year, 
                "Dim": emb["admin"].shape[1], 
                "N Samples": emb["admin"].shape[0]
            },
            {
                "Name": f"{model_key}/viirs", 
                "Year": year, 
                "Dim": emb["viirs"].shape[1], 
                "N Samples": emb["viirs"].shape[0]
            },
            {
                "Name": f"{model_key}/shared", 
                "Year": year, 
                "Dim": emb["shared_global"].shape[1], 
                "N Samples": emb["shared_global"].shape[0]
            },
        ]
    }


TESTS = {
    "fit-eval": test_fit_eval,
    "saipe-manifold": test_saipe_manifold,
    "viirs-manifold": test_viirs_manifold,
    "usps-manifold": test_usps_manifold,
    "multiview-manifolds": test_multiview_manifold,
    "fusion-opt": test_fusion_opt
}


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())

    run_tests_table(
        buf, 
        TESTS,
        targets=targets,
        train=args.train,
        test=args.test,
        out=args.out,
        models=args.models,
        trials=args.trials,
        folds=args.folds,
        random_state=args.random_state,
    )

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
