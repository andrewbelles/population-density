#!/usr/bin/env python3 
# 
# etc.py  Andrew Belles  Jan 7th, 2026 
# 
# Miscellanous Helper Functions necessary for Testbench Execution 
# 
# 

import io, contextlib 

import numpy as np

from pathlib import Path 

from utils.helpers import load_yaml_config

from testbench.utils.paths import CONFIG_PATH

from analysis.graph_metrics import MetricAnalyzer

from analysis.hyperparameter import (
    define_xgb_space,
    define_rf_space,
    define_svm_space,
    define_logistic_space
)

from models.estimators import (
    make_xgb_classifier,
    make_rf_classifier,
    make_svm_classifier,
    make_logistic
)

def get_param_space(model_type: str): 
    mapping = {
        "XGBoost": define_xgb_space,
        "RandomForest": define_rf_space,
        "SVM": define_svm_space, 
        "Logistic": define_logistic_space
    }
    return mapping[model_type]

def get_factory(model_type: str): 
    mapping = {
        "XGBoost": make_xgb_classifier,
        "RandomForest": make_rf_classifier,
        "SVM": make_svm_classifier,
        "Logistic": make_logistic
    }
    return mapping[model_type]

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

def _format_metric(value): 
    if value is None or np.isnan(value): 
        return "nan"
    return f"{value:.3f}"

def format_cell(item): 
    m    = item["metrics"]
    name = f"{item['dataset']}/{item['model']}"
    acc  = _format_metric(m.get("accuracy"))
    f1   = _format_metric(m.get("f1_macro"))
    roc  = _format_metric(m.get("roc_auc"))
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
