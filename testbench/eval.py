#!/usr/bin/env python3 
# 
# eval.py  Andrew Belles  Jan 30th, 2026 
# 
# Train on full dataset A and evaluate on full dataset B. 
# 
# 

import argparse, io 

import numpy as np

from analysis.cross_validation import (
    CVConfig,
    ScaledEstimator
)

from models.estimators         import EmbeddingProjector

from scipy.io                  import savemat 

from sklearn.preprocessing     import StandardScaler 

from optimization.engine       import (
    run_optimization,
    EngineConfig
)

from optimization.evaluators   import ProjectorEvaluator, StandardEvaluator

from optimization.spaces import define_manifold_projector_space
from testbench.utils.data      import BASE 

from testbench.utils.etc       import (
    format_metric,
    get_factory, 
    get_param_space,
    run_tests_table
) 

from testbench.utils.metrics   import (
    OPT_TASK,
    metrics_from_probs
)

from testbench.utils.config    import normalize_params

from preprocessing.loaders     import load_compact_dataset 

from utils.resources           import ComputeStrategy

from utils.helpers             import project_path

strategy = ComputeStrategy.from_env() 


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
        metric="rps"
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

def _row(name, metrics): 
    return {
        "Name": name, 
        "Acc":  format_metric(metrics.get("accuracy")),
        "F1":   format_metric(metrics.get("f1_macro")),
        "ROC":  format_metric(metrics.get("roc_auc")),
        "ECE":  format_metric(metrics.get("ece")),
        "QWK":  format_metric(metrics.get("qwk")),
        "RPS":  format_metric(metrics.get("rps"))
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
        "header": ["Name", "Acc", "F1", "ROC", "ECE", "QWK", "RPS"],
        "rows": rows,
    }

def test_manifold_generate(
    _buf,
    *,
    train: str, 
    test: str, 
    out_path: str | None = None, 
    trials: int = 50, 
    random_state: int = 0, 
    **_
): 
    train_path, train_loader = _resolve_dataset(train)
    test_path, test_loader   = _resolve_dataset(test)

    X_tr, y_tr, _ = _load_dataset(train_path, train_loader)
    X_te, y_te, _ = _load_dataset(test_path, test_loader)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    evaluator = ProjectorEvaluator(
        X_tr_s, y_tr,
        define_manifold_projector_space,
        random_state=random_state,
        compute_strategy=strategy 
    )

    config = EngineConfig(
        n_trials=trials,
        direction="minimize",
        random_state=random_state
    )
    
    best_params, best_value, _ = run_optimization(
        name=f"{train}_manifold_opt",
        evaluator=evaluator,
        config=config
    )

    params = dict(best_params)
    params["hidden_dims"] = (int(params.pop("base_width")),)
    params["mode"]        = "manifold"

    proj = EmbeddingProjector(
        in_dim=X_tr_s.shape[1],
        random_state=random_state,
        device=strategy.device,
        **params
    )

    proj.fit(X_tr_s, y_tr)
    embs = proj.transform(X_te_s)

    if out_path is None: 
        out_path = project_path("data", "datasets", f"{test.lower()}_pooled.mat")
    feature_names = np.array([f"{test.lower()}_manifold_{i}" for i in range(embs.shape[1])],
                             dtype="U64")

    savemat(out_path, {
        "features": embs, 
        "labels": y_te.reshape(-1, 1),
        "fips_codes": np.asarray(load_compact_dataset(test_path)["sample_ids"]),
        "feature_names": feature_names,
        "n_counties": np.array([len(y_te)], dtype=np.int64)
    })

    return {
        "header": ["Name", "Dim", "RPS"],
        "row": {
            "Name": f"{train}->{test}/manifold",
            "Dim": embs.shape[1],
            "RPS": format_metric(best_value)
        }
    }


TESTS = {
    "fit_eval": test_fit_eval,
    "manifold": test_manifold_generate
}


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())

    run_tests_table(
        buf, 
        TESTS,
        targets=targets,
        train=args.train,
        test=args.test,
        models=args.models,
        trials=args.trials,
        folds=args.folds,
        random_state=args.random_state
    )

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
