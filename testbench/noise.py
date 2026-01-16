#!/usr/bin/env python3 
# 
# nosie.py  Andrew Belles  Jan 14th, 2026 
# 
# Rigorous Analysis of features to determine whether they are noise or contribute meaningful 
# information. 
# 

import argparse, io 

import numpy as np 

from preprocessing.loaders import ConcatSpec, load_concat_datasets

from pathlib import Path 

from analysis.boruta import (
    BorutaProbe,
    BorutaConfig
)

from testbench.utils.data  import (
    BASE
)

from testbench.utils.paths import (
    BORUTA_CSV,
    LABELS_PATH,
)

from testbench.utils.metrics import summarize_boruta

from testbench.utils.etc   import (
    render_table,
    infer_groups,
    clean_feature,
    split_boruta,
    write_boruta_splits,
    write_csv,
)

def _log(buf: io.StringIO, msg: str): 
    buf.write(msg.rstrip() + "\n")

def build_candidate_loader(): 
    specs: list[ConcatSpec] = [
        {"name": "viirs", "path": BASE["VIIRS"]["path"], "loader": BASE["VIIRS"]["loader"]},
        {"name": "nlcd", "path": BASE["NLCD"]["path"], "loader": BASE["NLCD"]["loader"]},
        {"name": "saipe", "path": BASE["SAIPE"]["path"], "loader": BASE["SAIPE"]["loader"]},
        {"name": "cross", "path": BASE["PASSTHROUGH"]["path"], 
         "loader": BASE["PASSTHROUGH"]["loader"]}
    ]

    def _loader(_): 
        return load_concat_datasets(
            specs=specs,
            labels_path=LABELS_PATH,
            labels_loader=BASE["VIIRS"]["loader"]
        )
    return _loader 

def build_boruta_config(
    *,
    n_iter: int, 
    n_estimators: int, 
    max_depth: int, 
    max_features,
    min_samples_leaf: int,
    alpha: float 
) -> BorutaConfig: 
    return BorutaConfig(
        n_iter=n_iter,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        alpha=alpha 
    )

def boruta_table(result, top_k: int): 
    df            = result.summary.copy() 
    names         = np.asarray(result.feature_names, dtype="U128")
    groups        = infer_groups(names)
    df["group"]   = groups 
    df["feature"] = [clean_feature(n) for n in names]

    status_rank   = {"confirmed": 2, "tentative": 1, "rejected": 0}
    df["status_rank"] = df["status"].map(status_rank)
    df = df.sort_values(
        ["status_rank", "hit_rate", "importance_mean"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    top  = df.head(top_k) if top_k > 0 else df 
    rows = [
        (
            i + 1, 
            r.status,
            f"{r.hit_rate:.3f}",
            f"{r.importance_mean:.4f}",
            f"{r.importance_std:.4f}",
            f"{r.p_greater:.4f}",
            f"{r.p_less:.4f}",
            r.group, 
            r.feature
        ) for i, r in top.iterrows() 
    ]
    header = (
        "#", 
        "Status", 
        "HitRate", 
        "ImpMean", 
        "ImpStd", 
        "p>", 
        "p<", 
        "Group", 
        "Feature"
    )
    return render_table(header, rows), df 

def run_boruta_probe(loader, config: BorutaConfig): 
    probe = BorutaProbe(config, verbose=False)
    return probe.compute("__in_memory__", loader)

def test_boruta(
    buf: io.StringIO,
    *,
    n_iter: int = 50, 
    n_estimators: int = 500, 
    max_depth: int = 7, 
    max_features="sqrt",
    min_samples_leaf: int = 1, 
    alpha: float = 0.05, 
    top_k: int = 30, 
    out_csv: str | None = BORUTA_CSV,
    **_
): 
    loader     = build_candidate_loader()
    config     = build_boruta_config(
        n_iter=n_iter,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        alpha=alpha
    )
    result     = run_boruta_probe(loader, config) 
    counts     = result.summary["status"].value_counts().to_dict() 
    _log(buf, "== Boruta Summary ==")
    _log(buf, f"confirmed={counts.get('confirmed', 0)} "
              f"tentative={counts.get('tentative', 0)} "
              f"rejected={counts.get('rejected', 0)}")
    table, df  = boruta_table(result, top_k)
    _log(buf, table)

    csv_path   = None 
    if out_csv: 
        csv_path = write_csv(df, out_csv)
        _log(buf, f"[boruta] csv={csv_path}")

    return {
        "result": result.to_dict(),
        "csv": csv_path
    }

def export_boruta(
    buf: io.StringIO, 
    *,
    filter_csv: str = BORUTA_CSV,
    out_dir: str | None = None, 
    **_
): 
    if out_dir is None: 
        out_dir = str(Path(filter_csv).with_name("boruta_splits"))

    df        = summarize_boruta(filter_csv)
    df        = split_boruta(df)
    split_csv = write_boruta_splits(df, out_dir)

    counts    = df["status"].value_counts().to_dict()
    _log(buf, "== Boruta Export ==")
    _log(buf, f"confirmed={counts.get('confirmed', 0)} "
              f"tentative={counts.get('tentative', 0)} "
              f"rejected={counts.get('rejected', 0)}")
    _log(buf, f"[boruta] split_csv={split_csv}")
    _log(buf, f"[boruta] split_dir={out_dir}")


TESTS = {
    "boruta": test_boruta, 
    "boruta_export": export_boruta
}

def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--tests", nargs="*", default=None)
    parser.add_argument("--out-csv", default=BORUTA_CSV)
    parser.add_argument("--top-k", default=30)
    parser.add_argument("--filter-csv", default=BORUTA_CSV)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    buf = io.StringIO() 

    targets = args.tests or list(TESTS.keys())
    for name in targets: 
        fn = TESTS.get(name)
        if fn is None: 
            raise ValueError(f"unknown test: {name}")
        fn(
            buf,
            top_k=args.top_k,
            out_csv=args.out_csv,
            out_dir=args.out_dir,
            filter_csv=args.filter_csv
        )

    print(buf.getvalue().strip())


if __name__ == "__main__": 
    main() 
