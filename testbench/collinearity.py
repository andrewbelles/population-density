#!/usr/bin/env python3 
# 
# collinearity.py  Andrew Belles  Jan 10th, 2026 
# 
# Testbench driver for variance inflaction analysis between feature groups. 
# 
# 

import argparse, io 

from sklearn.linear_model import LinearRegression 

from analysis.vif import (
    PairwiseVIF,
    FullVIF,
    align_and_merge_features,
    coerce_feature_mat 
)

from testbench.utils.config  import eval_config 

from testbench.utils.data    import (
    load_raw, 
    DATASETS,
)

from testbench.utils.paths   import (
    PAIRWISE_CSV,
    FULL_CSV
)

from testbench.utils.etc     import full_long, pairwise_long, render_table, write_csv

RANDOM_STATE = 0 

# --------------------------------------------------------- 
# Test Helper Functions 
# ---------------------------------------------------------

def _log(buf: io.StringIO, msg: str): 
    buf.write(msg.rstrip() + "\n")


def build_matrix(): 
    datasets = [load_raw(k) for k in DATASETS]
    return align_and_merge_features(datasets, feature_groups=["VIIRS", "TIGER", "NLCD"])

# --------------------------------------------------------- 
# Unittests  
# ---------------------------------------------------------

def test_pairwise(
    buf: io.StringIO,
    *,
    method: str = "corr", 
    return_r2: bool = True,
    top_k: int = 15, 
    out_csv: str = PAIRWISE_CSV,
    **_
): 
    matrix, groups = build_matrix()
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

    df   = pairwise_long(result, groups)
    dfs  = df.sort_values("vif", ascending=False)
    top  = dfs.head(top_k)
    rows = [
        (i + 1, f"{r.vif:.3f}", f"{r.r2:.3f}", r.group_a, r.feature_a, r.group_b, r.feature_b)
        for i, r in top.reset_index(drop=True).iterrows() 
    ]
    header = ("#", "VIF", "r2", "Group A", "Feature A", "Group B", "Feature B")
    _log(buf, render_table(header, rows))

    csv_path = write_csv(dfs, out_csv)
    _log(buf, f"[pairwise] csv={csv_path}]")

    return {
        "result": result.to_dict(), 
        "csv": csv_path
    }

def test_full(
    buf: io.StringIO, 
    *, 
    return_r2: bool = True,
    top_k: int = 20, 
    out_csv: str = FULL_CSV,
    **_
): 
    matrix, groups = build_matrix() 
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

    df   = full_long(result, groups)
    dfs  = df.sort_values("vif", ascending=False)
    top  = dfs.head(top_k)
    rows = [
        (i + 1, f"{r.vif:.3f}", f"{r.r2:.3f}", r.group, r.feature)
        for i, r in top.reset_index(drop=True).iterrows() 
    ]
    header = ("#", "VIF", "r2", "Group", "Feature")
    _log(buf, render_table(header, rows))

    csv_path = write_csv(dfs, out_csv)
    _log(buf, f"[full] csv={csv_path}]")

    return {
        "result": result.to_dict(),
        "csv": csv_path
    }

TESTS = {
    "pairwise": test_pairwise,
    "full": test_full
}

# ---------------------------------------------------------
# Test Entry 
# ---------------------------------------------------------

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--method", choices=["cv", "corr"], default="corr")
    parser.add_argument("--no-r2", action="store_true")
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    for name in targets: 
        fn = TESTS.get(name)
        if fn is None: 
            raise ValueError(f"unknown test: {name}")
        fn(buf, method=args.method, return_r2=(not args.no_r2))

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
