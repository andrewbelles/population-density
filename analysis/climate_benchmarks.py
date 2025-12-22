#!/usr/bin/env python
# 
# climate_geospatial_benchmark.py  Andrew Belles  Dec 16th, 2025 
# 
# Flexible Testbench for Datasets Derived from Climate Based Datasets
# 
# 

REPEATS=5

import argparse 

from analysis.cross_validation import (
    CLASSIFICATION,
    CrossValidator, 
    CVConfig, 
    TaskSpec,
    REGRESSION,
)

from models.estimators import (
    make_gcn_regressor,
    make_linear,
    make_rf_regressor,
    make_rf_classifier,
    make_xgb_regressor,
    make_xgb_classifier,
    make_logistic
)

import support.helpers as h

from preprocessing.loaders import (
    DatasetLoader,
    load_climate_geospatial,
    load_geospatial_climate, 
    load_climate_population, 
    load_compact_dataset,
    load_neighbors_by_density 
)

from analysis.ablation import AblationSpec, FeatureAblation


def run_geospatial_from_climate_regression():

    print("\nREGRESSION: Climate -> Coordinates")

    filepath = h.project_path("data", "datasets", "climate_geospatial.mat")

    loader = lambda fp: load_climate_geospatial(
        fp, target="all", groups=["degree_days", "palmer_indices"]
    )

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor(),
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_climate_from_geospatial_regression(): 

    print("\nREGRESSION: Coordinates -> Climate")

    filepath = h.project_path("data", "datasets", "climate_geospatial.mat")

    loader = lambda fp: load_geospatial_climate(
        fp, target="all", groups=["lat", "lon"]
    )

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_contrastive_raw_climate_representation(): 

    print("\nCLASSIFICATION: Contrastive Pairs on Raw Climate Representation")

    filepath = h.project_path("data", "datasets", "climate_norepr_contrastive.mat")

    loader = load_compact_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_contrastive_pca_compact_climate_representation(): 

    print("\nCLASSIFICATION: Contrastive Pairs on Compact PCA Representation (95% threshold)")

    filepath = h.project_path("data", "datasets", "climate_pca_contrastive.mat")

    loader = load_compact_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_contrastive_kernel_pca_compact_climate_representation(): 

    print("\nCLASSIFICATION: Contrastive Pairs on Compact KernelPCA Representation (95% threshold)")

    filepath = h.project_path("data", "datasets", "climate_kpca_contrastive.mat")

    loader = load_compact_dataset 

    task = TaskSpec("classification", metrics=("accuracy", "f1", "roc_auc"))

    models = {
        "Logistic": make_logistic(C=1.0),
        "RandomForest": make_rf_classifier(),
        "XGBoost": make_xgb_classifier()
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    cv = CrossValidator(
        filepath=filepath, loader=loader, task=task, scale_y=False 
    )

    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_climate_to_population(): 

    print("\nREGRESSION: Climate (Raw) -> Population (decade: 2020)")

    filepath = h.project_path("data", "datasets", "climate_population.mat")

    loader = lambda fp: load_climate_population(
        filepath=fp, 
        decade=2020, 
        groups=["climate"]
    )

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor(), 
        "GraphNN": make_gcn_regressor(hidden_dims=(64,64))
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_pca_climate_to_population(): 

    print("\nREGRESSION: Climate (PCA) -> Population (decade: 2020)")

    filepath = h.project_path("data", "datasets", "climate_population_pca_supervised.mat")

    loader = lambda fp: load_compact_dataset(filepath=fp)

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor(),
        "GraphNN": make_gcn_regressor(hidden_dims=(64,64))
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_kpca_climate_to_population(): 

    print("\nREGRESSION: Climate (KernelPCA) -> Population (decade: 2020)")

    filepath = h.project_path("data", "datasets", "climate_population_kpca_supervised.mat")

    loader = lambda fp: load_compact_dataset(filepath=fp)

    models = {
        "Linear": make_linear(),
        "RandomForest": make_rf_regressor(max_depth=6), 
        "XGBoost": make_xgb_regressor(),
        "GraphNN": make_gcn_regressor(hidden_dims=(64,64))
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        random_state=0 
    )

    cv = CrossValidator(filepath=filepath, loader=loader, task=REGRESSION)
    results = cv.run(models=models, config=config)
    summary = cv.summarize(results)
    cv.format_summary(summary)

    return results, summary 


def run_raw_similarity_classification(): 
    pass 


def run_pca_similarity_classification(): 
    
    print("\nCLASSIFICATION: Train classifier to separate neighbors from non-neighbors\n"
          " in terms of similary population density (PCA repr, 2020).")

    compact_filepath = h.project_path("data", "datasets", "climate_population_pca_supervised.mat")
    census_filepath  = h.project_path("data", "datasets", "climate_population.mat")

    def proxy_loader_factory(tags: list[str]) -> DatasetLoader:
        return lambda fp: load_neighbors_by_density(
            compact_filepath=compact_filepath, 
            label_filepath=fp, 
            groups=tags, 
            decade=2020, 
            pos_threshold=0.20, 
            neg_threshold=0.60,
            neg_ratio=2.0,
            local_radius_km=1000.0 
        )

    abl = FeatureAblation(
        filepath=census_filepath, 
        loader_factory=proxy_loader_factory
    )

    specs = [
        AblationSpec(name="Baseline (Geography Only)", tags=["coords"]), 
        AblationSpec(name="Embeddings", tags=["embeddings"]),
        AblationSpec(name="Hybrid (Geo + Embeddings)", tags=["coords", "embeddings"])
    ]

    models = {
        "Logistic": make_logistic(C=1.0),
        "XGBoost": make_xgb_classifier(n_estimators=600)
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0,
    )

    results = abl.run(specs=specs, models=models, config=config, task=CLASSIFICATION)
    summary = abl.interpret(results)

    return results, summary 

def run_null_test_pca_similarity(): 

    print("\nCLASSIFICATION: (From PCA repr, 2020) Determine if embeddings carry real signal")

    compact_filepath = h.project_path("data", "datasets", "climate_population_pca_supervised.mat")
    census_filepath  = h.project_path("data", "datasets", "climate_population.mat")

    def proxy_loader_factory(tags: list[str]) -> DatasetLoader:
        return lambda fp: load_neighbors_by_density(
            compact_filepath=compact_filepath, 
            label_filepath=fp, 
            groups=tags, 
            decade=2020, 
            pos_threshold=0.20, 
            neg_threshold=0.60,
            neg_ratio=2.0,
            null_test=True,
            local_radius_km=1000.0
        )

    abl = FeatureAblation(
        filepath=census_filepath, 
        loader_factory=proxy_loader_factory
    )

    specs = [
        AblationSpec(name="Baseline (Geography Only)", tags=["coords"]), 
        AblationSpec(name="Embeddings", tags=["embeddings"]),
        AblationSpec(name="Hybrid (Geo + Embeddings)", tags=["coords", "embeddings"])
    ]

    models = {
        "Logistic": make_logistic(C=1.0),
        "XGBoost": make_xgb_classifier(n_estimators=600)
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0,
    )

    results = abl.run(specs=specs, models=models, config=config, task=CLASSIFICATION)
    summary = abl.interpret(results)

    return results, summary 


def run_kpca_similarity_classification(): 

    print("\nCLASSIFICATION: Train classifier to separate neighbors from non-neighbors\n"
          " in terms of similary population density (KernelPCA repr, 2020).")

    compact_filepath = h.project_path("data", "datasets", "climate_population_kpca_supervised.mat")
    census_filepath  = h.project_path("data", "datasets", "climate_population.mat")

    def proxy_loader_factory(tags: list[str]) -> DatasetLoader:
        return lambda fp: load_neighbors_by_density(
            compact_filepath=compact_filepath, 
            label_filepath=fp, 
            groups=tags, 
            decade=2020, 
            pos_threshold=0.20, 
            neg_threshold=0.60,
            neg_ratio=2.0,
            local_radius_km=1000.0
        )

    abl = FeatureAblation(
        filepath=census_filepath, 
        loader_factory=proxy_loader_factory
    )

    specs = [
        AblationSpec(name="Baseline (Geography Only)", tags=["coords"]), 
        AblationSpec(name="Embeddings", tags=["embeddings"]),
        AblationSpec(name="Hybrid (Geo + Embeddings)", tags=["embeddings", "coords"])
    ]

    models = {
        "Logistic": make_logistic(C=1.0),
        "XGBoost": make_xgb_classifier(n_estimators=600)
    }

    config = CVConfig(
        n_splits=5, 
        n_repeats=REPEATS, 
        stratify=True, 
        random_state=0 
    )

    results = abl.run(specs=specs, models=models, config=config, task=CLASSIFICATION)
    summary = abl.interpret(results)

    return results, summary 


def run_all(): 

    run_climate_from_geospatial_regression() 
    run_geospatial_from_climate_regression()
    run_contrastive_raw_climate_representation() 
    run_contrastive_pca_compact_climate_representation() 
    run_contrastive_kernel_pca_compact_climate_representation()
    run_climate_to_population()
    run_pca_climate_to_population()
    run_kpca_climate_to_population()


def main(): 

    TASKS = [
        "coords_to_climate", 
        "climate_to_coords", 
        "norepr_contrast", 
        "pca_contrast", 
        "kpca_contrast", 
        "climate_to_pop", 
        "pca_to_pop", 
        "kpca_to_pop", 
        "pca_similarity",
        "null_test_pca_similarity",
        "kpca_similarity", 
        "all"
    ] 

    task_dict = {
        "coords_to_climate": run_climate_from_geospatial_regression, 
        "climate_to_coords": run_geospatial_from_climate_regression, 
        "norepr_contrast": run_contrastive_raw_climate_representation, 
        "pca_contrast": run_contrastive_pca_compact_climate_representation, 
        "kpca_contrast": run_contrastive_kernel_pca_compact_climate_representation, 
        "climate_to_pop": run_climate_to_population, 
        "pca_to_pop": run_pca_climate_to_population, 
        "kpca_to_pop": run_kpca_climate_to_population,
        "pca_similarity": run_pca_similarity_classification, 
        "null_test_pca_similarity": run_null_test_pca_similarity, 
        "kpca_similarity": run_kpca_similarity_classification, 
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
