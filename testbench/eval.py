#!/usr/bin/env python3 
# 
# eval.py  Andrew Belles  Jan 30th, 2026 
# 
# Train on full dataset A and evaluate on full dataset B. 
# 
# 

import argparse, io, torch 

import numpy as np

from numpy.typing import NDArray

from typing import Literal 

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from torch.utils.data import TensorDataset

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

from models.ssfe import TabularSSFE, SpatialSSFE 

from models.estimators import HierarchicalFusionModel

from preprocessing.labels import PopulationLabels, build_label_map

from testbench.utils.paths     import (
    CONFIG_PATH
)

from preprocessing.loaders     import load_compact_dataset, load_wide_deep_inputs 

from utils.resources           import ComputeStrategy

from utils.helpers             import project_path, save_model_config

strategy = ComputeStrategy.from_env() 

CENSUS_DIR     = project_path("data", "census")

VIIRS_KEY  = "Manifold/VIIRS" 
SAIPE_KEY  = "Manifold/SAIPE"
USPS_KEY   = "Manifold/USPS"
FUSION_KEY = "Fusion/2013"

VIIRS_ARTIFACT = project_path("data", "datasets", "viirs_2013_artifact.mat")
SAIPE_ARTIFACT = project_path("data", "datasets", "saipe_2013_artifact.mat")
USPS_ARTIFACT = project_path("data", "datasets", "usps_2013_artifact.mat")

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
    **_
): 
    if modality == "tabular": 
        src_path, src_loader = _resolve_dataset(source)
        src = src_loader(src_path)

        X = np.asarray(src["features"], dtype=np.float32)
        y = np.asarray(src["labels"]).reshape(-1)
        fips = _canon_fips(np.asarray(src["sample_ids"]))

        scaler = StandardScaler() 
        Xs = scaler.fit_transform(X).astype(np.float32, copy=False)

        node_ids = np.arange(Xs.shape[0], dtype=np.int64)
        ds = TensorDataset(
            torch.from_numpy(Xs),
            torch.from_numpy(node_ids)
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
        out = project_path("data", "datasets", f"usps_embeddings_{year}.mat")

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
    usps_embeddings = project_path(
        "data", "datasets", f"usps_embeddings_{year}.mat"
    )
    saipe_embeddings = project_path(
        "data", "datasets", f"saipe_embeddings_{year}.mat"
    )
    wide_dataset = project_path(
        "data", "datasets", f"wide_scalar_{year}.mat"
    )

    X = load_wide_deep_inputs(
        expert_paths={
            "viirs": viirs_embeddings,
            "usps": usps_embeddings,
            "saipe": saipe_embeddings 
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

TESTS = {
    "fit-eval": test_fit_eval,
    "saipe-manifold": test_saipe_manifold,
    "viirs-manifold": test_viirs_manifold,
    "usps-manifold": test_usps_manifold,
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
