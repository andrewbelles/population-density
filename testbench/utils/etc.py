#!/usr/bin/env python3 
# 
# etc.py  Andrew Belles  Jan 7th, 2026 
# 
# Miscellanous Helper Functions necessary for Testbench Execution 
# 
# 

import io, contextlib, re  

import numpy as np

import pandas as pd 

from pathlib import Path 

from utils.helpers import bind, load_yaml_config

from testbench.utils.paths import CONFIG_PATH

from analysis.graph_metrics import MetricAnalyzer

from utils.resources import ComputeStrategy

from optimization.spaces import (
    define_xgb_space,
    define_rf_space,
    define_svm_space,
    define_logistic_space,
)

from models.estimators import (
    make_xgb_classifier,
    make_rf_classifier,
    make_svm_classifier,
    make_logistic,
)

def get_param_space(model_type: str): 
    mapping = {
        "XGBoost": define_xgb_space,
        "RandomForest": define_rf_space,
        "SVM": define_svm_space, 
        "Logistic": define_logistic_space,
    }
    return mapping[model_type]

def get_factory(
    model_type: str, 
    strategy: ComputeStrategy = ComputeStrategy.create(greedy=False)
): 
    mapping = {
        "XGBoost": make_xgb_classifier,
        "RandomForest": make_rf_classifier,
        "SVM": make_svm_classifier,
        "Logistic": make_logistic,
    }
    factory_fn = mapping[model_type] 
    return bind(_factory_with_strategy, factory_fn=factory_fn, strategy=strategy)

def _factory_with_strategy(*, factory_fn, strategy, **p):
    return factory_fn(compute_strategy=strategy, **p)

def load_metric_params(model_key: str) -> dict: 
    cfg    = load_yaml_config(Path(CONFIG_PATH))
    params = cfg.get("models", {}).get(model_key)
    if params is None: 
        raise ValueError(f"missing model config for key: {model_key}")
    return dict(params)

def write_model_metrics(buf: io.StringIO, title: str, metrics: dict): 
    buf.write(f"\n== {title} ==\n")
    for key in ("accuracy", "f1_macro", "roc_auc"): 
        if key in metrics: 
            buf.write(f"{key}: {metrics[key]:.4f}\n")

def write_graph_metrics(buf: io.StringIO, title: str, adj, P, y, coords): 
    '''
    Pretty prints metric output to a buffer so all staged metric results can be 
    printed simultaneously 
    '''
    metrics = MetricAnalyzer.compute_metrics(
        adj,
        probs=P, 
        y_true=y,
        train_mask=None,
        coords=coords,
        verbose=False 
    )
    buf.write(f"\n== {title} ==\n")
    with contextlib.redirect_stdout(buf): 
        MetricAnalyzer._print_report(metrics)

    return metrics 

def write_model_summary(buf: io.StringIO, title: str, best_value: float): 
    buf.write(f"\n== {title} ==\n")
    buf.write(f"Best value: {best_value:.6f}\n")

def format_metric(value): 
    if value is None or np.isnan(value): 
        return "nan"
    return f"{value:.3f}"

def format_cell(item): 
    m    = item["metrics"]
    name = f"{item['dataset']}/{item['model']}"
    acc  = format_metric(m.get("accuracy"))
    f1   = format_metric(m.get("f1_macro"))
    roc  = format_metric(m.get("roc_auc"))
    return f"{name} Acc={acc} f1={f1} roc_auc={roc}"

def render_table(header, rows): 
    cols   = list(zip(*([header] + rows)))
    widths = [max(len(str(cell)) for cell in col) for col in cols]

    def _row(values): 
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))
    
    line = "-+-".join("-" * w for w in widths)

    out = [_row(header), line]
    for row in rows: 
        out.append(_row(row))
    return "\n".join(out)

def infer_groups(names): 
    groups = []
    for n in names: 
        s = str(n)
        if "::" in s: 
            groups.append(s.split("::", 1)[0])
        elif "__" in s: 
            groups.append(s.split("__", 1)[0])
        else: 
            groups.append("features")
    return np.asarray(groups, dtype="U64")

def clean_feature(name: str) -> str: 
    s = str(name).strip() 
    s = s.replace("::", "/").replace("__", "/")
    s = re.sub(r"\s", " ", s)
    return s 

def pairwise_long(result, groups) -> pd.DataFrame: 
    names   = np.asarray(result.vif.index, dtype="U128")
    n       = names.shape[0]
    rows    = []
    r2_df   = result.r2 
    r2_vals = r2_df.to_numpy() if r2_df is not None else None 

    for i in range(n): 
        for j in range(i + 1, n): 
            rows.append({
                "feature_a": clean_feature(names[i]),
                "group_a": str(groups[i]),
                "feature_b": clean_feature(names[j]),
                "group_b": str(groups[j]),
                "vif": float(result.vif.iloc[i, j]),
                "r2": float(r2_vals[i, j]) if r2_vals is not None else np.nan 
            })

    df = pd.DataFrame(rows)
    return df 

def full_long(result, groups) -> pd.DataFrame: 
    names = np.asarray(result.vif.index, dtype="U128")
    v     = result.vif["vif"].to_numpy(dtype=np.float64)
    r2    = result.r2["r2"].to_numpy(dtype=np.float64) if result.r2 is not None else None 
    rows  = []

    for i, name in enumerate(names): 
        rows.append({
            "feature": clean_feature(name),
            "group": str(groups[i]),
            "vif": float(v[i]),
            "r2": float(r2[i]) if r2 is not None else np.nan 
        })

    return pd.DataFrame(rows)

def write_csv(df: pd.DataFrame, path: str) -> str: 
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)

def to_prefixed_name(group: str, feature: str) -> str: 
    g      = str(group).strip().lower()
    f      = str(feature).strip() 
    prefix = f"{g}/"
    if f.startswith(prefix): 
        f = f[len(prefix):]
    f = f.replace("/", "__")
    return f"{g}__{f}"

def split_boruta(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    df["group"] = df["group"].astype(str).str.lower() 
    df["feature_name"] = [
        to_prefixed_name(g, f) for g, f in zip(df["group"], df["feature"])
    ]
    return df 

def write_boruta_splits(df: pd.DataFrame, out_dir: str) -> str: 
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_csv = out / "boruta_split.csv"
    df.to_csv(split_csv, index=False)

    keep_mask = df["status"].isin(["confirmed", "tentative"])
    keep_df   = df[keep_mask]
    rej_df    = df[~keep_mask]

    keep_df.to_csv(out / "boruta_keep.csv", index=False)
    rej_df.to_csv(out / "boruta_reject.csv", index=False)

    for group, gdf in df.groupby("group"): 
        g_keep    = gdf[gdf["status"].isin(["confirmed", "tentative"])]
        g_rej     = gdf[~gdf["status"].isin(["confirmed", "tentative"])]

        keep_path = out / f"boruta_keep_{group}.txt"
        rej_path  = out / f"boruta_reject_{group}.txt"

        g_keep["feature_name"].to_csv(keep_path, index=False, header=False)
        g_rej["feature_name"].to_csv(rej_path, index=False, header=False)

    return str(split_csv)

def flatten_imaging(spatial, mask, gaf, mode): 
    n = spatial.shape[0]
    if mode == "spatial": 
        return np.hstack([spatial.reshape(n, -1), mask.reshape(n, -1)])
    if mode == "gaf": 
        return gaf.reshape(n, -1)
    if mode == "dual": 
        return np.hstack([spatial.reshape(n, -1), mask.reshape(n, -1), gaf.reshape(n, -1)])
    raise ValueError("mode must be spatial/gaf/dual")

# ---------------------------------------------------------
# Test Harness + Pretty Print 
# ---------------------------------------------------------

def collect_rows(result): 
    if not isinstance(result, dict): 
        return None, []
    header = result.get("header")
    rows   = []
    if result.get("rows"): 
        rows.extend(result["rows"])
    elif result.get("row"): 
        rows.append(result["row"])
    return header, rows 

def merge_results(*results): 
    header = None 
    rows   = []
    for result in results: 
        h, r = collect_rows(result)
        if not r: 
            continue 
        if header is None: 
            header = h 
        if h == header: 
            rows.extend(r)
    return {"header": header, "rows": rows} if header and rows else {}

def run_tests_table(
    buf, 
    tests: dict, 
    targets: list[str] | None = None, 
    *,
    caller=None,
    **kwargs
):

    header_rows  = {} 
    header_order = []
    targets = targets or list(tests.keys())

    # Lambda to handle pushing row into list for each test header 
    def _coerce_row(row, header): 
        if isinstance(row, dict): 
            return [row.get(h, "") for h in header]
        return list(row)

    def _handle(row, header): 
        if not header: 
            if isinstance(row, dict): 
                header = list(row.keys())
            else: 
                return  

        key = tuple(header)
        if key not in header_rows: 
            header_rows[key] = []
            header_order.append(key)

        header_rows[key].append(_coerce_row(row, header))

    for name in targets: 
        fn = tests.get(name)
        if fn is None: 
            raise ValueError(f"unknown test: {name}")

        result = caller(fn, name, **kwargs) if caller else fn(buf, **kwargs)

        if not isinstance(result, dict): 
            continue 
    
        rows   = result.get("rows")
        header = result.get("header")

        if rows: 
            for row in rows: 
                _handle(row, header)
        else: 
            row = result.get("row")
            if not row: 
                continue 

            _handle(row, header)

    for key in header_order:
        rows = header_rows.get(key, [])
        if not rows: 
            continue 
        buf.write(render_table(list(key), rows) + "\n")

    return {
        "tables": [
            {"header": list(key), "rows": header_rows[key]}
            for key in header_order if header_rows[key]
        ]
    }
