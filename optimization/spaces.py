#!/usr/bin/env python3 
# 
# spaces.py  Andrew Belles  Jan 24th, 2026 
# 
# Defines all parameter spaces for used evaluators 
# 
# 

import itertools, optuna 

# ---------------------------------------------------------
# Standard model spaces 
# ---------------------------------------------------------

def define_xgb_space(trial: optuna.Trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),

        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),

        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1e2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e2, log=True),

        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),

        "gamma": trial.suggest_float("gamma", 1e-1, 1e1, log=True),

        "tree_method": "hist",
    }


def define_rf_space(trial: optuna.Trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 15),

        "min_samples_split": trial.suggest_int("min_samples_split", 10, 60),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),

        "class_weight": trial.suggest_categorical(
            "class_weight", [None, "balanced", "balanced_subsample"]
        ),

        "criterion": trial.suggest_categorical(
            "criterion", ["gini", "entropy", "log_loss"]
        ),
    }


def define_svm_space(trial: optuna.Trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
        "probability": True
    }

    if kernel == "rbf":
        gamma_type = trial.suggest_categorical("gamma_mode", ["auto_scale", "custom"])
        if gamma_type == "custom":
            params["gamma"] = trial.suggest_float("gamma_custom", 1e-4, 1e1, log=True)
        else:
            params["gamma"] = trial.suggest_categorical("gamma_rbf", ["scale", "auto"])

    params["shrinking"] = trial.suggest_categorical("shrinking", [True, False])
    return params


def define_logistic_space(trial: optuna.Trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

# ---------------------------------------------------------
# EdgeLearner 
# ---------------------------------------------------------

def make_layer_choices(
    sizes=(32, 64, 128, 256),
    min_layers=1,
    max_layers=3
): 
    choices = {}
    for L in range(min_layers, max_layers+1): 
        for combo in itertools.product(sizes, repeat=L): 
            if any(combo[i] < combo[i + 1] for i in range(len(combo) - 1)): 
                continue 
            key = "-".join(str(x) for x in combo)
            choices[key] = combo 
    return choices 

def define_gate_space(trial):
    layer_choices = make_layer_choices()
    key = trial.suggest_categorical("hidden_dims", list(layer_choices.keys()))
    return {
        "hidden_dims": layer_choices[key], 
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True), 
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True), 
        "epochs": trial.suggest_int("epochs", 1000, 6000), 
        "batch_size": trial.suggest_categorical("batch_size", [4096, 8192, 16384]), 
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "residual": trial.suggest_categorical("residual", [True, False]), 
        "batch_norm": trial.suggest_categorical("batch_norm", [True, False])
    }

# ---------------------------------------------------------
# Spatial Classifier (CNN architecture) 
# ---------------------------------------------------------

def define_spatial_space(trial): 
    conv_choices = {
        # shallow 
        "32-64": (32, 64),
        "64-128": (64, 128),

        # standard 
        "32-64-64": (32, 64, 64),
        "32-64-128": (32, 64, 128),
        "64-128-256": (64, 128, 256), 

        # deep 
        "32-32-64-64": (32, 32, 64, 64),
        "32-64-64-128": (32, 64, 64, 128),
        "64-128-256-256": (64, 128, 256, 256)
    }
    key = trial.suggest_categorical("conv_channels", list(conv_choices.keys())) 
    return {
        # varying  
        "conv_channels": conv_choices[key],
        "roi_output_size": trial.suggest_categorical("roi_output_size", [32, 64, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [5, 7, 9, 11]),

        "fc_dim": trial.suggest_categorical("fc_dim", [64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "lambda_supcon": trial.suggest_float("lambda_supcon", 0.1, 1.0), 

        "use_bn": True, 
        "epochs": 125, 
        "early_stopping_rounds": 10, 
        "eval_fraction": 0.15,
        "min_delta": 1e-3,
        "sampling_ratio": 0
    }

# ---------------------------------------------------------
# MLP Projector  
# ---------------------------------------------------------

def define_projector_space(trial): 
    return {
        "mode": "single",     
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "eval_fraction": trial.suggest_categorical("eval_fraction", [0.15]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "epochs": trial.suggest_categorical("epochs", [300]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [15]),
        "out_dim": trial.suggest_categorical("out_dim", [5]),
    }

def define_manifold_projector_space(trial):
    n_layers   = trial.suggest_int("n_layers", 2, 4)
    base_width = trial.suggest_categorical("base_width", [32, 64, 128, 256]) 
    shaping    = trial.suggest_categorical("shaping", ["constant", "funnel", "diamond"])

    hidden_dims = []

    if shaping == "constant": 
        hidden_dims = [base_width] * n_layers 
    elif shaping == "funnel": 
        current = base_width 
        for _ in range(n_layers): 
            hidden_dims.append(current)
            current = max(16, current // 2)
    elif shaping == "diamond": 
        hidden_dims.append(base_width)
        for _ in range(n_layers - 2): 
            hidden_dims.append(base_width * 2)
        hidden_dims.append(base_width)

    return {
        "mode": "manifold",     
        "hidden_dims": tuple(hidden_dims),
        
        "dropout": trial.suggest_float("dropout", 0.15, 0.5),
        "lr": trial.suggest_float("lr", 5e-6, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "epochs": trial.suggest_categorical("epochs", [400]),

        "lambda_supcon": trial.suggest_float("lambda_supcon", 0.1, 1.0),
        "temperature": trial.suggest_float("temperature", 0.07, 0.2),

        "use_residual": trial.suggest_categorical("use_residual", [True, False]),
        "eval_fraction": trial.suggest_categorical("eval_fraction", [0.15]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [10]),
        "out_dim": trial.suggest_categorical("out_dim", [5]),
    }

# ---------------------------------------------------------
# XGBoost Ordinal Space 
# (same params as xgb but tuned for regularization) 
# ---------------------------------------------------------

def define_xgb_ordinal_space(trial): 
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e1, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True)
    }
