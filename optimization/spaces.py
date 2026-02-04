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
    return {
        "kernel": "rbf",
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "shrinking": True, 
        "probability": True, 
        "ordinal": True
    }

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
    return {
        "attn_dim": trial.suggest_categorical("attn_dim", [128, 256, 384]),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3), 

        "fc_dim": trial.suggest_categorical("fc_dim", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        "alpha_rps": trial.suggest_float("alpha_rps", 0.5, 10.0, log=True),
        "beta_supcon": trial.suggest_float("beta_supcon", 0.1, 2.0), 
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.05, 0.3),
        "supcon_dim": trial.suggest_categorical("supcon_dim", [64, 128, 256]), 

        "batch_size": trial.suggest_int("batch_size", 256, 512, step=128), 

        "ens": 0.999, # effective number of samples hyperparam   
        "epochs": 350, 
        "early_stopping_rounds": 10, 
        "eval_fraction": 0.2,
        "min_delta": 1e-4,
        "target_global_batch": 2048
    }

# ---------------------------------------------------------
# HyperGAT Spatial Classifier 
# ---------------------------------------------------------

def define_hgnn_space(trial: optuna.Trial): 
    return {
        "patch_size": trial.suggest_categorical("patch_size", [16, 32, 64]), 
        "embed_dim": trial.suggest_categorical("embed_dim", [32, 64, 128]), 
        "gnn_dim": trial.suggest_categorical("gnn_dim", [64, 128, 256]),
        #"gnn_layers": trial.suggest_int("gnn_layers", 1, 3), 
        #"gnn_heads": trial.suggest_categorical("gnn_heads", [2, 4, 8]), 
        "fc_dim": trial.suggest_categorical("fc_dim", [64, 128, 256]),
        
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.4), 

        "max_bag_frac": trial.suggest_float("max_bag_frac", 0.5, 1.0), 

        "alpha_rps": trial.suggest_float("alpha_rps", 0.5, 10.0, log=True),
        "beta_supcon": trial.suggest_float("beta_supcon", 0.1, 2.0), 
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.05, 0.3),
        "ens": trial.suggest_float("ens", 0.9, 0.999),
        "supcon_dim": trial.suggest_categorical("supcon_dim", [64, 128, 256]), 

        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        "threshold_scale": trial.suggest_float("threshold_scale", 0.8, 1.2), 

        "batch_size": trial.suggest_int("batch_size", 256, 512, step=128), 
        "gnn_layers": 3, 
        "gnn_heads": 8, 
        "epochs": 350, 
        "early_stopping_rounds": 15, 
        "eval_fraction": 0.2,
        "min_delta": 1e-4,
        "target_global_batch": 2048
    }

# ---------------------------------------------------------
# MLP Projector  
# ---------------------------------------------------------

def define_tabular_space(trial: optuna.Trial): 
    return {
        "hidden_dim": trial.suggest_int("hidden_dim", 768, 1024, step=64), 
        "depth": trial.suggest_int("depth", 12, 16), 
        "dropout": trial.suggest_float("dropout", 0.0, 0.3), 

        "mix_alpha": trial.suggest_float("mix_alpha", 0.1, 0.6),
        "mix_mult": trial.suggest_categorical("mix_mult", [4, 8]), 

        "beta_supcon": trial.suggest_float("beta_supcon", 0.1, 1.5), 
        "alpha_rps": trial.suggest_float("alpha_rps", 1.0, 10.0, log=True),
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.05, 0.25), 
        "supcon_dim": trial.suggest_categorical("supcon_dim", [64, 128]),

        "transformer_dim": trial.suggest_categorical("transformer_dim", [64, 128]),
        "transformer_tokens": trial.suggest_categorical("transformer_tokens", [4, 8]),
        "transformer_heads": trial.suggest_categorical("transformer_heads", [4, 8]),
        "transformer_layers": trial.suggest_int("transformer_layers", 3, 4),
        "transformer_dropout": trial.suggest_float("transformer_dropout", 0.0, 0.2),
        "transformer_attn_dropout": trial.suggest_float("transformer_attn_dropout", 0.0, 0.2),

        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        "eval_fraction": 0.2,
        "min_delta": 1e-4, 
        "early_stopping_rounds": 30, 
        "batch_size": 1024,
        "epochs": 1200, 
        "max_mix": None, 
        "anchor_power": 1.0 

    }

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

    width = trial.suggest_int("base_width", 128, 1024, step=64)

    return {
        "mode": "manifold",     
        "use_residual": True, 

        "hidden_dims": (width,),
        
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": trial.suggest_categorical("epochs", [500]),

        "beta_supcon": trial.suggest_float("beta_supcon", 0.1, 1.5),
        "alpha_rps": trial.suggest_float("alpha_rps", 1.0, 10.0, log=True),
        "temperature": trial.suggest_float("temperature", 0.05, 0.2),

        "eval_fraction": trial.suggest_categorical("eval_fraction", [0.2]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [20]),
        "out_dim": trial.suggest_int("out_dim", 8, 32, step=4),
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
