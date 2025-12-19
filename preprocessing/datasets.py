#!/usr/bin/env python3 
# 
# datasets.py  Andrew Belles  Dec 18th, 2025 
# 
# Testbench/Environment for generating specific datasets 
# 
# 

import argparse 
import numpy as np 

from sklearn.decomposition import PCA, KernelPCA 
from sklearn.metrics import pairwise_distances 

from support.helpers import project_path

from preprocessing.loaders import (
    load_climate_population, 
    UnsupervisedDatasetDict, 
)

from preprocessing.encodings import Encoder 


def test_climate_population_pca_supervised(): 

    print("DATASET: Climate -> Population PCA supervised dataset (2020)")

    filepath = project_path("data", "climate_population.mat")
    dataset  = load_climate_population(filepath, decade=2020, groups=["climate"])

    n_features    = dataset["features"].shape[1]
    feature_names = np.array([f"climate_feature_{i}" for i in range(n_features)], dtype="U64")

    unsupervised_dataset = UnsupervisedDatasetDict({
        "X": dataset["features"],
        "feature_names": feature_names,
        "coords": np.zeros((dataset["features"].shape[0],2), dtype=np.float64), 
        "coord_names": np.empty((0,), dtype="U1"), 
        "sample_ids": np.empty((0,), dtype="U1"),
        "groups": {}
    })

    encoder = Encoder(dataset=unsupervised_dataset, standardize=True)
    encoder.fit_pca(pca_class=PCA)

    pca_compact, k = encoder.get_reduced_pca_scores(threshold=0.95)

    print(f"PCA: {encoder.n_features} -> {k} components")

    out_path = project_path("data", "climate_population_pca_supervised.mat")
    Encoder.save_as_compact_supervised(out_path, pca_compact, dataset["labels"])

    print(f"Saved: {out_path}")

    return encoder, pca_compact


def test_climate_population_kpca_supervised(): 

    print("DATASET: Climate -> Population KernelPCA supervised dataset (2020)")

    filepath = project_path("data", "climate_population.mat")
    dataset  = load_climate_population(filepath, decade=2020, groups=["climate"])

    n_features    = dataset["features"].shape[1]
    feature_names = np.array([f"climate_feature_{i}" for i in range(n_features)], dtype="U64")

    unsupervised_dataset = UnsupervisedDatasetDict({
        "X": dataset["features"],
        "feature_names": feature_names,
        "coords": np.zeros((dataset["features"].shape[0],2), dtype=np.float64), 
        "coord_names": np.empty((0,), dtype="U1"), 
        "sample_ids": np.empty((0,), dtype="U1"),
        "groups": {}
    })

    encoder = Encoder(dataset=unsupervised_dataset, standardize=True)

    Xp  = encoder._X_for_pca()
    d   = pairwise_distances(Xp, metric="euclidean")
    med = np.median(d[d > 0])
    gamma = 1.0 / (2.0 * med * med) 

    encoder.fit_pca(
        pca_class=KernelPCA, 
        kernel="rbf", 
        gamma=gamma, 
        eigen_solver="auto", 
        remove_zero_eig=True 
    )

    kpca_compact, k = encoder.get_reduced_pca_scores(threshold=0.95)

    print(f"KernelPCA: {encoder.n_features} -> {k} components (gamma={gamma:.6f})")

    out_path = project_path("data", "climate_population_kpca_supervised.mat")
    Encoder.save_as_compact_supervised(out_path, kpca_compact, dataset["labels"])

    print(f"Saved: {out_path}")

    return encoder, kpca_compact 


def run_all(): 

    test_climate_population_pca_supervised()
    test_climate_population_kpca_supervised()


def main(): 

    TASKS = [
        "climate_pop_pca",
        "climate_pop_kpca", 
        "all"
    ]

    parser = argparse.ArgumentParser() 
    parser.add_argument("--task", choices=TASKS, default="all")
    args = parser.parse_args()

    task_dict = {
        "climate_pop_pca": test_climate_population_pca_supervised, 
        "climate_pop_kpca": test_climate_population_kpca_supervised, 
        "all": run_all 
    }

    fn = task_dict.get(args.task)
    if fn is None: 
        raise KeyError(f"invalid task: {args.task}")

    fn() 


if __name__ == "__main__": 
    main() 
