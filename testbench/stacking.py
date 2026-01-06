#!/usr/bin/env python3 
# 
# stacking.py  Andrew Belles  Dec 27th, 2025 
# 
# Comprehensive Test to determine optimal model over Bayesian Optimization of 
# hyperparameters, saves oof, and compares best in class for stacked classifier 
# 
# Saves out of fold predictions for final, best overall classifier 
# 

import os, argparse 
from pathlib import Path
import numpy as np
import optuna
import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm

from analysis.cross_validation import (
    CrossValidator,
    CVConfig,
    CLASSIFICATION
)

from analysis.hyperparameter import (
    StandardEvaluator,
    CorrectAndSmoothEvaluator,
    run_nested_cv,
    _load_yaml_config,
    _save_model_config,
    run_optimization, 
)

from preprocessing.loaders import (
    load_viirs_nchs,
    load_stacking, 
    load_oof_predictions,
    _align_on_fips 
)

from preprocessing.disagreement import (
    load_pass_through_stacking,
)

from preprocessing.loaders import load_concat_datasets 
from models.graph.learning import EdgeLearner 
from testbench.test_utils import _select_specs_psv

from utils.helpers import make_cfg_gap_factory, project_path

from models.graph.processing import (
    CorrectAndSmooth, 
) 

from models.graph.construction import (
    make_queen_adjacency_factory,
    make_mobility_adjacency_factory,
    compute_probability_lag_matrix
)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from testbench.test_utils import (
    _metrics_from_summary,
    _score_from_summary, 
    _get_cached_params,
    _normalize_params,
    _best_score,
    _majority_vote,
    _metrics_from_preds
)

def _build_specs(oof_files): 
    return  [
    {
        "name": "viirs", 
        "raw_path": project_path("data", "datasets", "viirs_nchs_2023.mat"),
        "raw_loader": load_viirs_nchs,
        "oof_path": oof_files[0], 
        "oof_loader": load_oof_predictions 
    },
    {
        "name": "tiger", 
        "raw_path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
        "raw_loader": load_viirs_nchs,
        "oof_path": oof_files[1],
        "oof_loader": load_oof_predictions
    },
    {
        "name": "nlcd",
        "raw_path": project_path("data", "datasets", "nlcd_nchs_2023.mat"),
        "raw_loader": load_viirs_nchs,
        "oof_path": oof_files[2],
        "oof_loader": load_oof_predictions 
    }
]


def plot_class_maps(
    *,
    y_true,
    y_pred,
    fips,
    shapefile,
    out_dir,
    prefix,
    vmin=-5,
    vmax=5
): 
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fips   = np.asarray(fips).astype("U5")

    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != fips.shape[0]:
        raise ValueError("y_true, y_pred, fips must have same length")

    # -5 means was most urban guessed most rural  
    # likewise 5 means most rural and guessed most urban
    class_distance = y_true - y_pred 
    dist_abs       = np.abs(class_distance)

    df = pd.DataFrame({
        "FIPS": fips, 
        "True_Class": y_true, 
        "Predicted_Class": y_pred, 
        "Class_Distance": class_distance
    })

    counties = gpd.read_file(shapefile)
    counties["FIPS"] = counties["GEOID"].astype(str).str.zfill(5)

    exclude  = {"02", "15", "60", "66", "69", "72", "78"}
    counties = counties[~counties["STATEFP"].isin(exclude)]

    gdf = counties.merge(df, on="FIPS", how="left")

    def _plot(column, title, out_path, cmap, vmin=None, vmax=None): 
        vals = gdf[column].dropna() 
        if vals.empty: 
            return 
        
        lo = int(vals.min()) if vmin is None else vmin 
        hi = int(vals.max()) if vmax is None else vmax 
        bounds = np.arange(lo, hi + 2) - 0.5 
        ticks  = np.arange(lo, hi + 1)

        fig, ax = plt.subplots(1, 1, figsize=(9,6))
        norm = BoundaryNorm(bounds, cmap.N)
        gdf.plot(
            column=column, 
            ax=ax,
            cmap=cmap,
            norm=norm,
            linewidth=0,
            edgecolor="none",
            missing_kwds={"color": "lightgrey"}
        )
        ax.set_title(title)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        setattr(sm, "_A", [])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.3, pad=0.02, ticks=ticks)
        cbar.ax.set_yticklabels([str(t) for t in ticks])

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    os.makedirs(out_dir, exist_ok=True)

    _plot(
        "True_Class",
        f"True Class", 
        os.path.join(out_dir, "true_class.png"),
        cmap=plt.cm.viridis
    )
    _plot(
        "Predicted_Class",
        f"{prefix}: Predicted Class",
        os.path.join(out_dir, f"{prefix}_pred_class.png"),
        cmap=plt.cm.viridis
    )
    _plot(
        "Class_Distance",
        f"{prefix}: Class Distance (MAE={dist_abs.mean():.3f})",
        os.path.join(out_dir, f"{prefix}_class_distance.png"),
        cmap=plt.cm.magma,
        vmin=vmin,
        vmax=vmax
    )

    return {
        "class_distance_mae": float(dist_abs.mean()),
        "class_distance_mean": float(class_distance.mean()),
        "within_1": float((dist_abs <= 1).mean())
    }


def evaluate_model(
    filepath, 
    loader_func, 
    model_name, 
    params, 
    *, 
    config, 
    oof_path=None
):
    model = get_factory(model_name)(**params)
    cv = CrossValidator(
        filepath=filepath, 
        loader=loader_func, 
        task=CLASSIFICATION, 
        scale_y=False
    )
    results = cv.run(
        models={"model": model},
        config=config,
        oof=oof_path is not None,
        collect=True
    )
    summary = cv.summarize(results)

    pred_rows = cv.predictions_[cv.predictions_["model"] == "model"]
    n = cv.n_samples 
    preds  = np.full(n, np.nan, dtype=float)
    labels = np.full(n, np.nan, dtype=float)

    idx = pred_rows["idx"].to_numpy(dtype=int)
    preds[idx]  = pred_rows["y_pred"].to_numpy(dtype=float)
    labels[idx] = pred_rows["y_true"].to_numpy(dtype=float)

    if np.isnan(preds).any() or np.isnan(labels).any(): 
        raise ValueError("missing predictions for some samples")

    preds  = preds.astype(int)
    labels = labels.astype(int)
    fips   = cv.sample_ids 

    if oof_path: 
        cv.save_oof(oof_path)
    return _metrics_from_summary(summary), _score_from_summary(summary), preds, labels, fips


def evaluate_correct_and_smooth(
    oof_path,
    shapefile,
    mobility_path, 
    params, 
    random_state=0
): 
    params = dict(params)
    adjacency = params.pop("adjacency")

    oof = load_oof_predictions(oof_path)
    y_train = np.asarray(oof["labels"]).reshape(-1)
    class_labels = np.asarray(oof["class_labels"]).reshape(-1)
    oof_fips = np.asarray(oof["fips_codes"]).astype("U5")

    train_mask = make_train_mask(
        y_train,
        train_size=0.3,
        random_state=random_state, 
        stratify=True
    )
    test_mask  = ~train_mask 

    if adjacency == "qg":
        cfg = _load_yaml_config(Path(project_path("testbench", "model_config.yaml")))
        qg_key = "StackingPassthrough/QG" if "passthrough" in oof_path else "Stacking/QG"
        qg_params = cfg.get("models", {}).get(qg_key)
        if qg_params is None:
            raise ValueError(f"missing QG config: {qg_key}")

        qg_params = dict(qg_params)
        qg_params.pop("dataset", None)

        data = load_concat_datasets(
            specs=_select_specs_psv("VIIRS+TIGER+NLCD+COORDS+PASSTHROUGH"),
            labels_path=project_path("data", "datasets", "viirs_nchs_2023.mat"),
            labels_loader=load_viirs_nchs
        )
        X = data["features"]
        fips = np.asarray(data["sample_ids"], dtype="U5")

        if len(fips) != len(oof_fips) or not np.array_equal(fips, oof_fips):
            idx = _align_on_fips(oof_fips, fips)
            X = X[idx]

        base_adj = make_queen_adjacency_factory(shapefile)(list(oof_fips))
        qg = QueenGateLearner(**qg_params)
        qg.fit(X, y_train, adj=base_adj, train_mask=train_mask)
        W = qg.get_graph(X)

        P = np.asarray(oof["probs"], dtype=np.float64)
        if P.ndim == 3:
            P = P.mean(axis=1)
    else:
        queen_factory    = make_queen_adjacency_factory(shapefile)
        mobility_factory = make_mobility_adjacency_factory(mobility_path, oof_path)

        P, _, W_queen, _ = compute_probability_lag_matrix(oof_path, queen_factory)
        _, _, W_mob, _   = compute_probability_lag_matrix(oof_path, mobility_factory)
        W = W_queen if adjacency == "queen" else W_mob

    cs = CorrectAndSmooth(class_labels=class_labels, **params)

    P_cs = cs.fit(P, y_train, W, train_mask)
    P_cs_norm = normalized_proba(P_cs, test_mask)

    pred_idx    = np.argmax(P_cs_norm, axis=1)
    pred_labels = class_labels[pred_idx]
    y_true      = y_train[test_mask]

    return {
        "accuracy": accuracy_score(y_true, pred_labels), 
        "f1_macro": f1_score(y_true, pred_labels, average="macro"), 
        "roc_auc": roc_auc_score(y_true, P_cs_norm, multi_class="ovr", average="macro")
    }, oof["fips_codes"], y_train, cs.predict(),

# ---------------------------------------------------------
# "Enum" into parameter space and model factory 
# ---------------------------------------------------------

def get_param_space(model_type: str): 
    
    from analysis.hyperparameter import (
        define_xgb_space,
        define_rf_space,
        define_svm_space,
        define_logistic_space
    )

    mapping = {
        "XGBoost": define_xgb_space, 
        "RandomForest": define_rf_space, 
        "SVM": define_svm_space, 
        "Logistic": define_logistic_space
    }
    return mapping[model_type]

def get_factory(model_type: str):
    
    from models.estimators import (
        make_xgb_classifier,
        make_rf_classifier,
        make_svm_classifier,
        make_logistic
    )

    mapping = {
        "XGBoost": make_xgb_classifier, 
        "RandomForest": make_rf_classifier, 
        "SVM": make_svm_classifier, 
        "Logistic": make_logistic
    }
    return mapping[model_type]

# ---------------------------------------------------------
# Test Helper Functions 
# ---------------------------------------------------------

def optimize_correct_and_smooth(
    *,
    oof_path: str, 
    shapefile: str,
    mobility_path: str, 
    n_trials: int = 100,
    early_stopping_rounds=40,
    early_stopping_delta=1e-4,
    random_state: int = 0,
    config_key: str = "CorrectAndSmooth", 
    config_path: str | None = None,
    qg_key: str | None = None, 
    force_qg: bool = False 
): 
    oof          = load_oof_predictions(oof_path)
    y_train      = np.asarray(oof["labels"]).reshape(-1)
    class_labels = np.asarray(oof["class_labels"]).reshape(-1)
    train_mask   = make_train_mask(
        y_train, 
        train_size=0.3, 
        random_state=random_state,
        stratify=True
    )
    test_mask    = ~train_mask 

    queen_factory    = make_queen_adjacency_factory(shapefile)
    mobility_factory = make_mobility_adjacency_factory(mobility_path, oof_path)

    P, _, W_queen, _ = compute_probability_lag_matrix(
        proba_path=oof_path, 
        adj_factory=queen_factory
    )

    _, _, W_mob, _   = compute_probability_lag_matrix(
        proba_path=oof_path, 
        adj_factory=mobility_factory
    )

    W_by_name = {
        "queen": W_queen,
        "mobility_adaptive": W_mob 
    }

    if qg_key is not None: 
        if config_path is None: 
            raise ValueError("qg_key requires config_path")

        cfg = _load_yaml_config(Path(config_path))
        qg_params = cfg.get("models", {}).get(qg_key)
        if qg_params is None: 
            raise ValueError(f"missing QG config: {qg_key}")

        qg_params = dict(qg_params)
        qg_params.pop("dataset", None)

        h = qg_params.get("hidden_layer_size")
        if isinstance(h, str): 
            qg_params["hidden_layer_size"] = tuple(int(x) for x in h.split("-") if x)

        data = load_concat_datasets(
            specs=_select_specs_psv("VIIRS+TIGER+NLCD+COORDS+PASSTHROUGH"),
            labels_path=project_path("data", "datasets", "viirs_nchs_2023.mat"),
            labels_loader=load_viirs_nchs 
        )

        X    = data["features"]
        fips = np.asarray(data["sample_ids"], dtype="U5")
        oof_fips = np.asarray(oof["fips_codes"]).astype("U5")

        if len(fips) != len(oof_fips) or not np.array_equal(fips, oof_fips): 
            idx = _align_on_fips(oof_path, fips)
            X = X[idx]

        base_adj = make_queen_adjacency_factory(shapefile)(list(oof_fips))
        qg = QueenGateLearner(**qg_params)
        qg.fit(X, y_train, adj=base_adj, train_mask=train_mask)
        W_qg = qg.get_graph(X)

        if force_qg: 
            W_by_name = {"qg": W_qg}
        else: 
            W_by_name["qg"] = W_qg

    evaluator = CorrectAndSmoothEvaluator(
        P=P,
        W_by_name=W_by_name,
        y_train=y_train,
        train_mask=train_mask,
        test_mask=test_mask,
        class_labels=class_labels
    )

    best_params, best_score = run_optimization(
        name="CorrectAndSmooth", 
        evaluator=evaluator,
        n_trials=n_trials,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_delta=early_stopping_delta
    )

    if config_path is not None: 
        _save_model_config(config_path, config_key, best_params)

    return best_params, best_score 

def optimize_dataset(
    dataset_name, 
    filepath, 
    loader_func, 
    n_trials=100, 
    *,
    outer_config=None, 
    inner_config=None,
    resume=False, 
    cache=None,
    config_path=None,
    random_state: int = 0,
    transforms=None
): 

    models_list = ["Logistic", "RandomForest", "XGBoost"]
    leaderboard = []

    if outer_config is None: 
        outer_config = CVConfig(
            n_splits=5,
            n_repeats=1,
            stratify=True,
            random_state=random_state
        )
    if inner_config is None: 
        inner_config = CVConfig(
            n_splits=3,
            n_repeats=1,
            stratify=True,
            random_state=random_state
        )

    cache = cache or {}

    for m in models_list:

        sampler_type = "multivariate-tpe"

        config_key = f"{dataset_name}/{m}"
        cached_params = _get_cached_params(cache, config_key) if resume else None 

        if cached_params is not None: 
            evaluator = StandardEvaluator(
                filepath=filepath, 
                loader_func=loader_func, 
                base_factory_func=get_factory(m),
                param_space=get_param_space(m),
                task=CLASSIFICATION,
                config=CVConfig(n_splits=5, n_repeats=1)
            )

            cached_params = _normalize_params(m, cached_params)
            best_val = evaluator.evaluate(cached_params)
            print(f"[{dataset_name}] resume hit {m} (score: {best_val:.4f})")
            leaderboard.append({
                "model": m, 
                "params": cached_params, 
                "score": best_val
            })
            continue 

        evaluator = StandardEvaluator(
            filepath=filepath,
            loader_func=loader_func,
            base_factory_func=get_factory(m), 
            param_space=get_param_space(m),
            task=CLASSIFICATION,
            config=CVConfig(n_splits=5, n_repeats=1), 
            feature_transform_factory=transforms 
        )

        mean_score, best_params, _, _ = run_nested_cv(
            name=f"{dataset_name}_{m}", 
            filepath=filepath,
            loader_func=loader_func, 
            model_factory=get_factory(m),
            param_space=get_param_space(m), 
            task=CLASSIFICATION,
            outer_config=outer_config,
            inner_config=inner_config,
            n_trials=n_trials,
            random_state=random_state,
            early_stopping_rounds=25,
            early_stopping_delta=1e-4,
            sampler_type=sampler_type
        )

        best_params = _normalize_params(m, best_params)
        if config_path is not None: 
            _save_model_config(config_path, config_key, best_params)

        leaderboard.append({
            "model": m,
            "params": best_params,
            "score": mean_score
        })

    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)
    winner = leaderboard[0]

    print(f"\n [{dataset_name}] winner: {winner['model']} "
          f"(score: {winner['score']:.4f})")
    return winner 


def generate_oof(dataset_name, filepath, loader_func, winner, output_path):

    print(f"[{dataset_name}] generating oof using {winner['model']}...")

    factory        = get_factory(winner['model'])
    model_instance = factory(**winner['params'])

    config = CVConfig(
        n_splits=5,
        n_repeats=5,
        stratify=True,
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath,
        loader=loader_func, 
        task=CLASSIFICATION,
        scale_y=False
    )

    cv.run(
        models={"best_model": model_instance}, 
        config=config,
        oof=True 
    )

    cv.save_oof(output_path)
    print(f"[{dataset_name}] oof saved to {output_path}")

# ---------------------------------------------------------
# Main Test Pipeline  
# ---------------------------------------------------------

def stacking(config_path, args):

    cache = _load_yaml_config(Path(config_path)) if args.resume else {}

    datasets = {
        "VIIRS": {
            "path": project_path("data", "datasets", "viirs_nchs_2023.mat"), 
            "loader": load_viirs_nchs
        },
        "TIGER": {
            "path": project_path("data", "datasets", "tiger_nchs_2023.mat"),
            "loader": load_viirs_nchs 
        },
        "NLCD": {
            "path": project_path("data", "datasets", "nlcd_nchs_2023.mat"), 
            "loader": load_viirs_nchs
        }
    }

    oof_files = []
    
    for name, config in datasets.items(): 
        winner = optimize_dataset(
            name, 
            config["path"],
            config["loader"],
            n_trials=250, 
            cache=cache, 
            resume=args.resume, 
            config_path=config_path
        )

        oof_path = project_path("data", "stacking", f"{name.lower()}_optimized_oof.mat")
        generate_oof(name, config["path"], config["loader"], winner, oof_path)
        oof_files.append(oof_path)

    print("[Stacking] starting meta-learner optimization")

    if args.passthrough: 
        cross_modal_path = project_path("data", "datasets", "cross_modal_2023.mat")

        passthrough_specs = [
            {
                "name": "cross",
                "raw_path": cross_modal_path,
                "raw_loader": load_viirs_nchs
            }
        ]

        passthrough = [
            "cross__cross_viirs_log_mean", 
            "cross__cross_tiger_integ",
            "cross__cross_radiance_entropy",
            "cross__cross_dev_intensity_gradient",
            "cross__cross_vanui_proxy",
            "cross__cross_effective_mesh_proxy",
            "cross__cross_road_effect_intensity",
        ]

        stack_loader = lambda fp: load_pass_through_stacking(
            _build_specs(oof_files), 
            label_path=project_path("data", "datasets", "viirs_nchs_2023.mat"), 
            label_loader=load_viirs_nchs,
            passthrough_features=passthrough, 
            passthrough_specs=passthrough_specs
        )
    else: 
        stack_loader = lambda _: load_stacking(oof_files)

    dummy  = oof_files[0]
    prefix = "StackingPassthrough" if args.passthrough else "Stacking"
    cfg_gap_factory = None 
    if args.passthrough: 
        stack_data = stack_loader(dummy)
        cfg_gap_factory = make_cfg_gap_factory(stack_data["feature_names"])

    stack_winner = optimize_dataset(
        prefix,
        dummy,
        stack_loader, 
        n_trials=250,
        config_path=config_path,
        transforms=cfg_gap_factory if args.passthrough else None 
    )

    if args.passthrough:
        final_output = project_path("data", "results", "final_stacked_passthrough.mat")
    else: 
        final_output = project_path("data", "results", "final_stacked_predictions.mat")
    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    generate_oof("Stacking", dummy, stack_loader, stack_winner, final_output)
    print(f"> stacking predictions saved to {final_output}")


def correct_and_smooth(config_path, args): 

    if args.passthrough: 
        stacking_oof_path = project_path("data", "results", "final_stacked_passthrough.mat")
    else: 
        stacking_oof_path = project_path("data", "results", "final_stacked_predictions.mat")
    shapefile = project_path("data", "geography", "county_shapefile", 
                             "tl_2020_us_county.shp")
    mobility_mat = project_path("data", "datasets", "travel_proxy.mat")

    cs_params, cs_score = optimize_correct_and_smooth(
        oof_path=stacking_oof_path,
        shapefile=shapefile,
        mobility_path=mobility_mat,
        n_trials=150,
        random_state=0,
        config_path=config_path,
        qg_key="Stacking/QG",
        force_qg=True,
        config_key="CorrectAndSmooth"
    )

    print(cs_params, cs_score)


def report(
    config_path,
    args
): 

    shapefile  = project_path("data", "geography", "county_shapefile",
                             "tl_2020_us_county.shp")
    out_dir    = project_path("testbench", "images") 
    cfg         = _load_yaml_config(Path(config_path))
    models_cfg  = cfg.get("models", {})

    datasets = {
        "VIIRS": project_path("data", "datasets", "viirs_nchs_2023.mat"), 
        "TIGER": project_path("data", "datasets", "tiger_nchs_2023.mat"), 
        "NLCD": project_path("data", "datasets", "nlcd_nchs_2023.mat")
    }

    stacking_oof_path = project_path("data", "results", "stacking_oof_path.mat")

    rows = []
    oof_files = []

    config = CVConfig(
        n_splits=5,
        n_repeats=3,
        stratify=True, 
        random_state=0,
        verbose=False
    )

    stage_preds  = []
    stage_labels = None
    stage_fips   = [] 
    stage_label_fips = None

    for name, path in datasets.items(): 
        best = None
        best_preds = None 
        best_fips = None 
        best_labels = None

        for model_name in ("Logistic", "RandomForest", "XGBoost"): 
            key = f"{name}/{model_name}"
            params = models_cfg.get(key)
            if params is None: 
                continue 

            params = _normalize_params(model_name, params)
            metrics, score, preds, labels, fips = evaluate_model(
                path, 
                load_viirs_nchs, 
                model_name, 
                params,
                config=config
            )

            score = _best_score(metrics)
            if best is None or score > best[0]: 
                best = (score, model_name, params, metrics)
                best_preds = preds 
                best_labels = labels 
                best_fips = fips 

        if best is None: 
            continue 

        _, model_name, params, metrics = best 

        stage_preds.append(best_preds.reshape(-1)) 
        stage_fips.append(best_fips)
        if stage_labels is None: 
            stage_labels = best_labels 
            stage_label_fips = best_fips 

        _, model_name, params, metrics = best 
        oof_path = project_path("data", "stacking", f"{name.lower()}_optimized_oof.mat")
        evaluate_model(
            path, 
            load_viirs_nchs, 
            model_name, 
            params, 
            config=config, 
            oof_path=oof_path
        )
        oof_files.append(oof_path)

        rows.append({
            "Stage": name, 
            "Model": model_name, 
            **metrics 
        })

    common = stage_fips[0]
    for f in stage_fips[1:]:
        fset = set(f)
        common = [x for x in common if x in fset]

    aligned_preds = [
        preds[_align_on_fips(common, fips)] 
        for preds, fips in zip(stage_preds, stage_fips)
    ]

    majority_pred   = _majority_vote(np.column_stack(aligned_preds))
    majority_labels = stage_labels[_align_on_fips(common, stage_label_fips)]
    majority_metrics = _metrics_from_preds(majority_labels, majority_pred)

    rows.append({
        "Stage": "Majority", 
        "Model": "Vote", 
        **majority_metrics
    })

    plot_class_maps(
        y_true=majority_labels,
        y_pred=majority_pred,
        fips=np.array(common, dtype="U5"),
        shapefile=shapefile,
        out_dir=out_dir,
        prefix="majority"
    )

    if args.passthrough:
        cross_modal_path = project_path("data", "datasets", "cross_modal_2023.mat")

        passthrough_specs = [
            {
                "name": "cross",
                "raw_path": cross_modal_path,
                "raw_loader": load_viirs_nchs
            }
        ]

        passthrough = [
            "cross__cross_viirs_log_mean",
            "cross__cross_tiger_integ",
            "cross__cross_radiance_entropy",
            "cross__cross_dev_intensity_gradient",
            "cross__cross_vanui_proxy",
            "cross__cross_effective_mesh_proxy",
            "cross__cross_road_effect_intensity",
        ]

        stack_loader = lambda fp: load_pass_through_stacking(
            _build_specs(oof_files),
            label_path=project_path("data", "datasets", "viirs_nchs_2023.mat"),
            label_loader=load_viirs_nchs,
            passthrough_features=passthrough,
            passthrough_specs=passthrough_specs
        )
    else:
        stack_loader = lambda _: load_stacking(oof_files)
    
    best = {
        "stack": None,
        "preds": None, 
        "labels": None, 
        "fips": None
    }

    for model_name in ("Logistic", "RandomForest", "XGBoost"):
        if args.passthrough: 
            key = f"StackingPassthrough/{model_name}"
        else: 
            key = f"Stacking/{model_name}"
        params = models_cfg.get(key)
        if params is None: 
            continue 

        params = _normalize_params(model_name, params)
        metrics, score, preds, labels, fips = evaluate_model(
            "virtual", 
            stack_loader, 
            model_name, 
            params, 
            config=config
        )
        if best["stack"] is None or score > best["stack"][0]: 
            best["stack"]  = (score, model_name, params, metrics)
            best["preds"]  = preds 
            best["labels"] = labels 
            best["fips"]   = fips 

    if best["stack"] is None: 
        raise ValueError

    _, model_name, params, metrics = best["stack"]
    evaluate_model(
        oof_files[0], 
        stack_loader,
        model_name, 
        params, 
        config=config,
        oof_path=stacking_oof_path
    )
    rows.append({
        "Stage": "Stacking",
        "Model": model_name,
        **metrics
    })

    plot_class_maps(
        y_true=best["labels"],
        y_pred=best["preds"].reshape(-1),
        fips=best["fips"],
        shapefile=shapefile,
        out_dir=out_dir,
        prefix="stacking"
    )

    cs_params  = models_cfg.get("CorrectAndSmooth")
    mobility   = project_path("data", "datasets", "travel_proxy.mat")
    cs_metrics, fips, labels, preds = evaluate_correct_and_smooth(
        stacking_oof_path, 
        shapefile,
        mobility,
        cs_params,
        random_state=0
    ) 

    rows.append({
        "Stage": "CorrectAndSmooth",
        "Model": cs_params.get("adjacency"),
        **cs_metrics
    })

    plot_class_maps(
        y_true=labels,
        y_pred=preds,
        fips=fips,
        shapefile=shapefile,
        out_dir=out_dir,
        prefix="correct_and_smooth"
    )

    headers = ["Stage", "Model", "accuracy", "f1_macro", "roc_auc"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows: 
        lines.append("| " + " | ".join([
            str(r.get("Stage", "")),
            str(r.get("Model", "")),
            f"{r.get('accuracy', float('nan')):.4f}",
            f"{r.get('f1_macro', float('nan')):.4f}",
            f"{r.get('roc_auc', float('nan')):.4f}",
        ]) + " |")
    print("\n".join(lines))


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stacking", action="store_true")
    parser.add_argument("--cs", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--suppress", action="store_true")
    parser.add_argument("--passthrough", action="store_true")
    args = parser.parse_args()

    config_path = project_path("testbench", "model_config.yaml")

    if args.suppress: 
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.stacking: 
        stacking(config_path, args)

    if args.cs: 
        correct_and_smooth(config_path, args)

    if args.report: 
        report(config_path, args)


if __name__ == "__main__": 
    main()
