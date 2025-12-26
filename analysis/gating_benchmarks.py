#!/usr/bin/env python3 
# 
# gating_benchmakrs.py  Andrew Belles  Dec 23rd, 2025 
# 
# Analysis of Classification Problem for Identifying NCHS from 
# various datasets identified as key for gating layer of HMoE model 
# 

import argparse, os 
import itertools 

from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt

from analysis.cross_validation import (
    CLASSIFICATION,
    CrossValidator, 
    CVConfig, 
)

from analysis.optimizer import optimize_hyperparameters

from models.estimators import (
    make_rf_classifier,
    make_xgb_classifier,
    make_logistic, 
    make_svm_classifier
)

from preprocessing.loaders import (
    DatasetDict,
    load_oof_errors,
    load_oof_predictions,
    load_viirs_nchs, # also loads tiger/nlcd datasets for now 
    load_stacking 
)

from preprocessing.disagreement import (
    DisagreementSpec,
    build_disagreement_dataset,
    load_pass_through_stacking
)

from models.graph_utils import (
    compute_probability_lag_matrix
)

from models.post_processing import (
    CorrectAndSmooth,
    make_train_mask,
    normalized_proba
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

REPEATS = 10

from support.helpers import project_path

def run_viirs_nchs_full(): 
    
    print("CLASSIFICATION: VIIRS Nighttime Lights classification via NCHS Scheme (2023)")

    filepath = project_path("data", "datasets", "viirs_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        )
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_viirs_nchs_smushed(): 
    
    print("CLASSIFICATION: VIIRS Nighttime Lights classification via NCHS Scheme (2023)")
    print("3-Class Scheme")

    filepath = project_path("data", "datasets", "viirs_nchs_2023_smushed.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        )
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_tiger_nchs_full(): 

    print("CLASSIFICATION: (TIGER 2023) Road Topology, Density, and Texture against NCHS")

    filepath = project_path("data", "datasets", "tiger_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config,
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_tiger_nchs_smushed(): 

    print("CLASSIFICATION: (TIGER 2023) Road Topology, Density, and Texture against 3-Class NCHS")

    filepath = project_path("data", "datasets", "tiger_nchs_2023_smushed.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(C=1.0), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_nlcd_nchs_full():
    print("CLASSIFICATION: (NLCD 2023) Land Usage Prediction of Urban-Rural Classifications (2023)")

    filepath = project_path("data", "datasets", "nlcd_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config,
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_nlcd_nchs_smushed():
    print("CLASSIFICATION: (NLCD 2023) Land Usage Prediction of 3-Class Urban-Rural"
          "Classifications (2023)")

    filepath = project_path("data", "datasets", "nlcd_nchs_2023_smushed.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    models = {
        "Logistic": make_logistic(), 
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier(), 
        "SVM": make_svm_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config,
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def create_oof_datasets(): 
    print("DATASET: Creates OOF datasets for each best performer per dataset.")

    viirs = project_path("data", "datasets", "viirs_nchs_2023.mat")
    tiger = project_path("data", "datasets", "tiger_nchs_2023.mat")
    nlcd  = project_path("data", "datasets", "nlcd_nchs_2023.mat")
    
    # Pre-load data to get common index 
    viirs_data = load_viirs_nchs(viirs)
    tiger_data = load_viirs_nchs(tiger)
    nlcd_data  = load_viirs_nchs(nlcd)

    common_fips = sorted(
        set(viirs_data["sample_ids"]) & 
        set(tiger_data["sample_ids"]) & 
        set(nlcd_data["sample_ids"])
    )

    def _align_loader(base_loader, fips):
        fips_set = set(fips)
        def _loader(fp) -> DatasetDict: 
            data = base_loader(fp)
            idx_map = {f: i for i, f in enumerate(data["sample_ids"])}
            idx = np.array([idx_map[f] for f in fips_set if f in idx_map], dtype=int)
            return {
                "features": data["features"][idx], 
                "labels": data["labels"][idx], 
                "coords": data["coords"][idx], 
                "feature_names": data["feature_names"], 
                "sample_ids": data["sample_ids"][idx]
            }
        return _loader 

    loader = _align_loader(load_viirs_nchs, common_fips)

    viirs_model = {
        "SVM": make_svm_classifier(
            C=140.0, 
            kernel="rbf", 
            gamma=0.06
        ),
    }

    tiger_model = {
        "Logistic": make_logistic(),
    }

    nlcd_model = {
        "RandomForest": make_rf_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    viirs_cv = CrossValidator(
        filepath=viirs, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    tiger_cv = CrossValidator(
        filepath=tiger, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False 
    )

    nlcd_cv = CrossValidator(
        filepath=nlcd, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False 
    )

    viirs_cv.run(
        models=viirs_model, 
        config=config,
        oof=True 
    )
    splits = viirs_cv.splits_ 

    tiger_cv.run(
        models=tiger_model, 
        config=config, 
        oof=True, 
        splits=splits
    )

    nlcd_cv.run(
        models=nlcd_model,
        config=config,
        oof=True,
        splits=splits
    )

    viirs_cv.save_oof(project_path("data", "stacking", "viirs_2023_oof.mat"))
    tiger_cv.save_oof(project_path("data", "stacking", "tiger_2023_oof.mat"))
    nlcd_cv.save_oof(project_path("data", "stacking", "nlcd_2023_oof.mat"))


def run_stacking(): 

    print("CLASSIFICATION: Stacking Classifier from Layer A Classifier's Probabilities")
    
    viirsstack = project_path("data", "stacking", "viirs_2023_oof.mat") 
    tigerstack = project_path("data", "stacking", "tiger_2023_oof.mat")
    nlcdstack  = project_path("data", "stacking", "nlcd_2023_oof.mat")

    files = [
        viirsstack, 
        tigerstack, 
        nlcdstack
    ]

    def stacking_loader(fp: str):
        _ = fp 
        return load_stacking(files)

    models = {
        "Logistic": make_logistic(), 
        "XGBoost": make_xgb_classifier(), 
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=viirsstack, 
        loader=stacking_loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    return summary, results 


def run_disagreement():
    print("DISAGREEMENT: Classification of rows as conflicting vs. agreeing predictions")

    viirs = project_path("data", "datasets", "viirs_nchs_2023.mat")
    tiger = project_path("data", "datasets", "tiger_nchs_2023.mat")
    nlcd  = project_path("data", "datasets", "nlcd_nchs_2023.mat")

    viirsstack = project_path("data", "stacking", "viirs_2023_oof.mat") 
    tigerstack = project_path("data", "stacking", "tiger_2023_oof.mat")
    nlcdstack  = project_path("data", "stacking", "nlcd_2023_oof.mat")

    specs: list[DisagreementSpec] = [
        {
            "name": "viirs", 
            "raw_path": viirs, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": viirsstack, 
            "oof_loader": load_oof_predictions 
        }, 
        {
            "name": "tiger",
            "raw_path": tiger, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": tigerstack,
            "oof_loader": load_oof_predictions 
        },
        {
            "name": "nlcd",
            "raw_path": nlcd, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": nlcdstack,
            "oof_loader": load_oof_predictions 
        }
    ]

    loader = lambda _: build_disagreement_dataset(specs)
    feature_names  = build_disagreement_dataset(specs)["feature_names"]
    
    models = {
        "Logistic": make_logistic(), 
        "RandomForest": make_rf_classifier(), 
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=viirs, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config, 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    # Load feature importances 
    rf = cv.get_model("RandomForest")
    importances = pd.Series(rf.estimator.feature_importances_, index=feature_names) 
    top = importances.sort_values(ascending=False).head(10)

    print(top)

    return summary, results 


def run_passthrough_stacking(): 
    print("CLASSIFICATION: Meta-Learner classification of Urban-Rural w/ Passthroug Features")
    print("[viirs_variance, nlcd_urban_core, tiger_density_deadend, nlcd_edge_dens]")
    viirs = project_path("data", "datasets", "viirs_nchs_2023.mat")
    tiger = project_path("data", "datasets", "tiger_nchs_2023.mat")
    nlcd  = project_path("data", "datasets", "nlcd_nchs_2023.mat")

    viirsstack = project_path("data", "stacking", "viirs_2023_oof.mat") 
    tigerstack = project_path("data", "stacking", "tiger_2023_oof.mat")
    nlcdstack  = project_path("data", "stacking", "nlcd_2023_oof.mat")

    specs: list[DisagreementSpec] = [
        {
            "name": "viirs", 
            "raw_path": viirs, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": viirsstack, 
            "oof_loader": load_oof_predictions 
        },
        {
            "name": "tiger",
            "raw_path": tiger, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": tigerstack,
            "oof_loader": load_oof_predictions 
        },
        {
            "name": "nlcd",
            "raw_path": nlcd, 
            "raw_loader": load_viirs_nchs, 
            "oof_path": nlcdstack,
            "oof_loader": load_oof_predictions 
        }
    ]

    nchs_loader = lambda fp: load_viirs_nchs(fp)

    loader = lambda label_fp: load_pass_through_stacking(
        specs,
        label_path=label_fp, 
        label_loader=nchs_loader,
        passthrough_features=[
            "viirs_variance", 
            "nlcd_urban_core", 
            "tiger_density_deadend", 
            "nlcd_edge_dens"
        ]
    )
    
    models = {
        "RandomForest": make_rf_classifier(), 
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=viirs, 
        loader=loader, 
        task=CLASSIFICATION, 
        scale_y=False
    )

    results = cv.run(
        models=models, 
        config=config,
        oof=True 
    )

    summary = cv.summarize(results)
    cv.format_summary(summary)

    stacking_oof_path = project_path("data", "stacking", "stacking_passthrough_oof.mat")
    cv.save_oof(stacking_oof_path)

    return summary, results 


def run_cs_postprocess(): 

    print("CLASSIFICATION: Urban-Rural classification using Correct and Smooth "
          "from Stacking Classifier's Output")

    stacking_oof = project_path("data", "stacking", "stacking_passthrough_oof.mat")
    shapefile    = project_path("data", "geography", "county_shapefile", "tl_2020_us_county.shp")

    P, _, W, _ = compute_probability_lag_matrix(stacking_oof, shapefile)

    oof          = load_oof_predictions(stacking_oof)
    y_train      = np.asarray(oof["labels"]).reshape(-1)
    class_labels = np.asarray(oof["class_labels"]).reshape(-1) 

    train_mask   = make_train_mask(y_train)

    def _run(correction_alpha, correction_iter, smoothing_alpha, smooth_iter):
        cs = CorrectAndSmooth(
            class_labels=class_labels,
            correction_alpha=correction_alpha,
            correction_max_iter=correction_iter,
            smoothing_alpha=smoothing_alpha,
            smoothing_max_iter=smooth_iter
        )

        P_cs = cs.fit(
            P,
            y_train,
            W,
            train_mask
        )

        test_mask = ~train_mask 
        y_true    = y_train[test_mask]

        P_cs_norm = normalized_proba(P_cs, test_mask)
        cs_idx    = np.argmax(P_cs_norm, axis=1)
        cs_pred   = class_labels[cs_idx] 

        return {
            "acc": accuracy_score(y_true, cs_pred),  
            "f1": f1_score(y_true, cs_pred, average="macro"),
            "roc": roc_auc_score(y_true, P_cs_norm, multi_class="ovr", average="macro"),
            "params": [
                correction_alpha,
                correction_iter,
                smoothing_alpha,
                smooth_iter
            ]
        } 

    res = _run(0.0, 1, 0.05, 5)

    print("Results:")
    print(f"> acc: {res['acc']}")
    print(f">  f1: {res['f1']}")
    print(f"> roc: {res['roc']}")


def optimize_svm_viirs(): 
    print("OPTIMIZATION: SVM For VIIRS multi-classification for NCHS Urban-Rural Labels (2023)")

    filepath = project_path("data", "datasets", "viirs_nchs_2023.mat") 
    
    loader = lambda fp: load_viirs_nchs(filepath=fp)

    svm_grid = {
        "C": np.linspace(50.0, 150.0, 10), 
        "kernel": ["rbf"], 
        "gamma": np.linspace(0.01, 0.1, 8) 
    }

    _, best_model_name = optimize_hyperparameters(
        base_name="SVM", 
        filepath=filepath,
        loader_func=loader,
        base_factory_func=make_svm_classifier, 
        param_grid=svm_grid,
        task=CLASSIFICATION, 
    )

    return best_model_name


def run_gating_visualization(): 
    print("VISUALIZATION: True, Predicted, and Class Distance plots for CONUS (2023)")

    oof_path   = project_path("data", "stacking", "stacking_passthrough_oof.mat")
    label_path = project_path("data", "datasets", "viirs_nchs_2023.mat")
    shapefile  = project_path("data", "geography", "county_shapefile", "tl_2020_us_county.shp")
    out_dir    = project_path("analysis", "figures")

    os.makedirs(out_dir, exist_ok=True)

    df = load_oof_errors(
        oof_path=oof_path,
        label_path=label_path,
        coords_path=None, 
    )
     
    avg_dist = float(np.abs(df["Class_Distance"]).mean())

    counties = gpd.read_file(shapefile)
    counties["FIPS"] = counties["GEOID"].astype(str).str.zfill(5)
    
    exclude  = {"02", "15", "60", "66", "69", "72", "78"}
    counties = counties[~counties["STATEFP"].isin(exclude)] 

    gdf = counties.merge(df, on="FIPS", how="left")

    def _plot_discrete(column, title, out_path, cmap, vmin=None, vmax=None): 
        vals = gdf[column].dropna() 
        if vals.empty: 
            return 

        lo = int(vals.min()) if vmin is None else vmin 
        hi = int(vals.max()) if vmax is None else vmax 
        bounds = np.arange(lo, hi + 2) - 0.5 
        ticks  = np.arange(lo, hi + 1) 

        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        norm = BoundaryNorm(bounds, cmap.N)
        gdf.plot(
            column=column, ax=ax, 
            cmap=cmap, norm=norm, 
            linewidth=0, edgecolor="none", 
            missing_kwds={"color": "lightgrey"}
        )
        ax.set_title(title)
        ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if hasattr(sm, "_A"):
            setattr(sm, "_A", []) 
        else: 
            raise AttributeError("expected _A on ScalarMappable")
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=ticks)
        cbar.ax.set_yticklabels([str(t) for t in ticks])

        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    _plot_discrete(
        "True_Class",
        "True Class",
        os.path.join(out_dir, "true_class.png"),
        cmap=plt.cm.viridis
    )

    _plot_discrete(
        "Predicted_Class",
        "Predicted Classes",
        os.path.join(out_dir, "predicted_class.png"),
        cmap=plt.cm.viridis
    )

    _plot_discrete(
        "Class_Distance",
        f"Class Distance (avg={avg_dist:.3f})", 
        os.path.join(out_dir, "class_distance.png"),
        vmin=-5,
        vmax=5,
        cmap=plt.cm.magma
    )


def run_all(): 

    run_viirs_nchs_full() 
    run_viirs_nchs_smushed()
    run_tiger_nchs_full()
    run_tiger_nchs_smushed()
    run_nlcd_nchs_full()
    run_nlcd_nchs_smushed()
    create_oof_datasets()
    run_stacking() 
    run_gating_visualization()
    optimize_svm_viirs()


def main(): 

    TASKS = [
        "viirs", 
        "viirs_smushed", 
        "tiger", 
        "tiger_smushed", 
        "nlcd",
        "nlcd_smushed", 
        "oof", 
        "stacking", 
        "disagree", 
        "stacking_passthrough", 
        "optimize_svm", 
        "visualize_gating",
        "full",
        "all"
    ] 

    task_dict = {
        "viirs": run_viirs_nchs_full, 
        "viirs_smushed": run_viirs_nchs_smushed, 
        "tiger": run_tiger_nchs_full,
        "tiger_smushed": run_tiger_nchs_smushed, 
        "nlcd": run_nlcd_nchs_full,
        "nlcd_smushed": run_nlcd_nchs_smushed, 
        "oof": create_oof_datasets,
        "stacking": run_stacking, 
        "disagree": run_disagreement,
        "stacking_passthrough": run_passthrough_stacking,
        "optimize_svm": optimize_svm_viirs,  
        "visualize_gating": run_gating_visualization, 
        "full": run_cs_postprocess,
        "all": run_all
    }

    parser = argparse.ArgumentParser() 
    parser.add_argument("--task", choices=TASKS, nargs="+", default=["all"])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()    

    if args.list: 
        print(f"Available tasks: {task_dict.keys()}")
        return 

    tasks_to_run = args.task if isinstance(args.task, list) else [args.task]

    for task in tasks_to_run: 
        if task == "all": 
            run_all() 
            break 
        else: 
            fn = task_dict.get(task)
            if fn is None: 
                raise KeyError(f"invalid task: {task}")
            fn() 


if __name__ == "__main__": 
    main()
