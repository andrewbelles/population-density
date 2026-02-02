#!/usr/bin/env python3 
# 
# collinearity.py  Andrew Belles  Jan 10th, 2026 
# 
# Testbench driver for variance inflaction analysis between feature groups. 
# 
# 

import argparse, io 

import numpy as np 

from sklearn.linear_model      import LinearRegression 

from sklearn.decomposition     import PCA 

from sklearn.neighbors         import NearestNeighbors

from analysis.vif              import (
    PairwiseVIF,
    FullVIF,
    align_and_merge_features,
    coerce_feature_mat 
)

from preprocessing.loaders import load_coords_from_mobility
from testbench.utils.config    import (
    eval_config,
    load_model_params,
    normalize_params 
)

from testbench.utils.data      import (
    load_raw,
    load_dataset,
    DATASETS,
    BASE
)

from testbench.utils.paths     import (
    PAIRWISE_CSV,
    FULL_CSV,
    CONFIG_PATH,
    MOBILITY_PATH
)

from testbench.utils.metrics   import moran_i

from models.graph.construction import (
    build_knn_graph_from_coords
)

from testbench.utils.etc       import (
    full_long, 
    pairwise_long, 
    render_table,
    run_tests_table, 
    write_csv,
    get_factory
)

from testbench.utils.graph     import (
    coords_for_fips
)

from utils.helpers import align_on_fips
from utils.resources           import ComputeStrategy 

strategy = ComputeStrategy.from_env()

RANDOM_STATE = 0 

# --------------------------------------------------------- 
# Test Helper Functions 
# ---------------------------------------------------------

def _log(buf: io.StringIO, msg: str): 
    buf.write(msg.rstrip() + "\n")

def _resolve_datasets(datasets: list[str] | None = None) -> list[str]: 
    if not datasets: 
        return list(BASE.keys())
    missing = [d for d in datasets if d not in BASE] 
    if missing: 
        raise ValueError(f"unknown datasets not in BASE: {missing}")
    return list(datasets)

def build_matrix(datasets: list[str]): 
    mats = [load_raw(k) for k in datasets]
    return align_and_merge_features(mats, feature_groups=datasets)

def _model_feature_mats(
    datasets: list[str], 
    model: str, 
    config_path: str = CONFIG_PATH 
): 
    mats = []
    for key in datasets: 
        data = load_dataset(key)
        X    = np.asarray(data["features"], dtype=np.float64)
        y    = np.asarray(data["labels"]).reshape(-1)
        fips = np.asarray(data.get("sample_ids"))

        params = load_model_params(config_path, f"{key}/{model}")
        params = normalize_params(model, params)

        clf = get_factory(model, strategy=strategy)(**params)
        clf.fit(X, y)

        P = clf.predict_proba(X)
        if P.ndim == 1: 
            P = P[:, None]

        feature_names = np.array(
            [f"{key}_p{i}" for i in range(P.shape[1])], dtype="U64"
        )

        mats.append({
            "features": P, 
            "coords": np.zeros((P.shape[0], 2), dtype=np.float64),
            "feature_names": feature_names, 
            "sample_ids": fips 
        })

    return align_and_merge_features(mats, feature_groups=datasets)

def _binary_weights(W):
    if W.nnz == 0: 
        return W 
    W = W.copy() 
    W.data = np.ones_like(W.data, dtype=np.float64)
    return W 

def _align_coords(fips, X, y=None): 
    coord_data = load_coords_from_mobility(MOBILITY_PATH)
    coord_fips = np.asarray(coord_data["sample_ids"]).astype("U5")

    coord_set = set(coord_fips)
    keep_idx  = [i for i, f in enumerate(fips) if f in coord_set]
    if not keep_idx: 
        raise ValueError("no overlap between data fips and coordinate fips")

    if len(keep_idx) != len(fips):
        dropped = len(fips) - len(keep_idx)
        print(f"[manifold] dropped {dropped} missing sampled.")

    fips_keep = np.asarray(fips)[keep_idx]
    X = np.asarray(X)[keep_idx]
    y = np.asarray(y)[keep_idx]

    coord_idx = align_on_fips(fips_keep, coord_fips)
    coords    = np.asarray(coord_data["coords"])[coord_idx]
    return coords, fips_keep, X, y 

# --------------------------------------------------------- 
# Unittests  
# ---------------------------------------------------------

def test_pairwise_vif(
    _buf,
    *,
    datasets: list[str] | None = None, 
    method: str = "corr", 
    return_r2: bool = True,
    top_k: int = 15, 
    out_csv: str = PAIRWISE_CSV,
    **_
): 
    datasets       = _resolve_datasets(datasets)
    matrix, groups = build_matrix(datasets)
    config         = eval_config(RANDOM_STATE)
    model_factory  = lambda: LinearRegression()
    vif            = PairwiseVIF(model_factory, config, verbose=False)
    loader         = lambda _: {
        "features": matrix.X, 
        "coords": matrix.coords, 
        "feature_names": matrix.feature_names,
        "sample_ids": matrix.sample_ids 
    }

    result = vif.compute(
        "__in_memory__",
        loader, 
        method=method,
        return_r2=return_r2, 
        feature_groups=groups
    )

    df   = pairwise_long(result, groups).sort_values("vif", ascending=False)
    top  = df.head(top_k)

    rows = [
        {
            "#": i + 1, 
            "VIF": f"{r.vif:.3f}", 
            "r2": f"{r.r2:.3f}",
            "Group A": r.group_a,
            "Feature A": r.feature_a,
            "Group B": r.group_b,
            "Feature B": r.feature_b
        }
        for i, r in top.reset_index(drop=True).iterrows()
    ]

    csv_path = write_csv(df, out_csv)

    return {
        "header": ["#", "VIF", "r2", "Group A", "Feature A", "Group B", "Feature B"], 
        "rows": rows, 
        "csv": csv_path 
    }

def test_full_vif(
    buf: io.StringIO, 
    *, 
    datasets: list[str] | None = None, 
    return_r2: bool = True,
    top_k: int = 20, 
    out_csv: str = FULL_CSV,
    **_
): 
    datasets       = _resolve_datasets(datasets)
    matrix, groups = build_matrix(datasets)
    config         = eval_config(RANDOM_STATE)
    model_factory  = lambda: LinearRegression() 
    vif            = FullVIF(model_factory, config, verbose=False)
    loader         = lambda _: {
        "features": matrix.X, 
        "coords": matrix.coords, 
        "feature_names": matrix.feature_names, 
        "sample_ids": matrix.sample_ids
    }

    result = vif.compute(
        "__in_memory__",
        loader,
        return_r2=return_r2,
        feature_groups=groups 
    )

    df   = full_long(result, groups).sort_values("vif", ascending=False)
    top  = df.head(top_k)

    rows = [
        {
            "#": i + 1, 
            "VIF": f"{r.vif:.3f}", 
            "r2": f"{r.r2:.3f}",
            "Group": r.group,
            "Feature": r.feature,
        }
        for i, r in top.reset_index(drop=True).iterrows()
    ]

    csv_path = write_csv(df, out_csv)

    return {
        "header": ["#", "VIF", "r2", "Group", "Feature"], 
        "rows": rows, 
        "csv": csv_path 
    }

def test_manifold_stats(
    _buf,
    *,
    datasets: list[str] | None = None, 
    k_neighbors: int = 15,
    **_
): 
    datasets = _resolve_datasets(datasets)

    rows = []
    for key in datasets: 
        data   = load_dataset(key)
        X      = np.asarray(data["features"], dtype=np.float64)
        y      = np.asarray(data["labels"]).reshape(-1)
        fips   = np.asarray(data.get("sample_ids"))
        coords, fips, X, y = _align_coords(fips, X, y) 

        w = build_knn_graph_from_coords(
            coords, 
            k=k_neighbors,
            directed=False,
            include_self=False,
            compute_strategy=strategy
        )
        W = _binary_weights(w)

        var = X.var(axis=0, ddof=0)
        sum_var  = float(var.sum())
        eff_dim  = float((sum_var**2) / (np.sum(var**2) + 1e-9)) if sum_var > 0 else 0.0 

        pc1       = PCA(n_components=1, random_state=RANDOM_STATE).fit_transform(X).reshape(-1)
        moran_pc1 = moran_i(pc1, W)

        moran_y   = moran_i(y, W) 

        nn = NearestNeighbors(n_neighbors=2, n_jobs=strategy.n_jobs)
        nn.fit(X)
        dists = nn.kneighbors(X, return_distance=True)[0][:, 1]
        nn_dist = float(dists.mean())

        rows.append({
            "Name": key,
            "N": X.shape[0],
            "Dim": X.shape[1],
            "EffDim": f"{eff_dim:.2f}",
            "NN_Dist": f"{nn_dist:.4f}",
            "Moran_PC1": f"{moran_pc1:.4f}",
            "Moran_Label": f"{moran_y:.4f}"
        })

    return {
        "header": ["Name", "N", "Dim", "EffDim", 
                   "NN_Dist", "Moran_PC1", "Moran_Label"],
        "rows": rows 
    }


TESTS = {
    "pairwise-vif": test_pairwise_vif,
    "full-vif": test_full_vif,
    "manifold-stats": test_manifold_stats
}

# ---------------------------------------------------------
# Test Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--model", default="Logistic")
    parser.add_argument("--method", choices=["cv", "corr"], default="corr")
    parser.add_argument("--no-r2", action="store_true")
    parser.add_argument("--knn", type=int, default=15)
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    run_tests_table(
        buf,
        TESTS,
        targets=targets,
        datasets=args.datasets,
        model=args.model,
        method=args.method,
        return_r2=(not args.no_r2),
        k_neighbors=args.knn
    )

    print(buf.getvalue().strip())

if __name__ == "__main__": 
    main() 
