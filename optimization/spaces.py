#!/usr/bin/env python3 
# 
# spaces.py  Andrew Belles  Jan 24th, 2026 
# 
# Defines all parameter spaces for used evaluators 
# 
# 

import itertools, optuna 

from models.graph.construction import (
    LOGCAPACITY_GATE_HIGH,
    LOGCAPACITY_GATE_LOW
)

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

        "alpha_mae": trial.suggest_float("alpha_mae", 0.5, 10.0, log=True),
        "beta_supcon": trial.suggest_float("beta_supcon", 0.1, 1.0), 
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.7, 6.0, log=True),
        "supcon_dim": trial.suggest_categorical("supcon_dim", [64, 128, 256]), 

        "batch_size": trial.suggest_int("batch_size", 256, 512, step=128), 

        "ens": 0.995, # effective number of samples hyperparam   
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
        "patch_size": trial.suggest_categorical("patch_size", [32]), 
        "embed_dim": trial.suggest_categorical("embed_dim", [128, 256]), 
        "gnn_dim": trial.suggest_categorical("gnn_dim", [128, 256]),
        "gnn_layers": trial.suggest_int("gnn_layers", 1, 3), 
        "gnn_heads": trial.suggest_categorical("gnn_heads", [4, 8]), 
        "fc_dim": trial.suggest_categorical("fc_dim", [128, 256]),
        
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.2), 
        "max_bag_frac": trial.suggest_float("max_bag_frac", 0.8, 1.0), 

        "alpha_mae": trial.suggest_float("alpha_mae", 5.0, 10.0),
        "beta_supcon": trial.suggest_float("beta_supcon", 0.0, 2.5), 
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.2, 2.0, log=True),
        "supcon_dim": trial.suggest_categorical("supcon_dim", [128, 256]), 

        "lr": trial.suggest_float("lr", 3e-6, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),

        #"reduce_depth": trial.suggest_int("reduce_depth", 1, 4), 
        #"reduce_dropout": trial.suggest_float("reduce_dropout", 0.0, 0.15), 

        "reduce_dim": None, 
        "batch_size": 256, 
        "soft_epochs": 400, 
        "hard_epochs": 200, 
        "patch_stat": "viirs", 
        "early_stopping_rounds": 10, 
        "eval_fraction": 0.2,
        "min_delta": 1e-4,
    }

def define_usps_space(trial: optuna.Trial): 
    params = define_hgnn_space(trial)
    params.pop("threshold_scale", None)

    params["patch_stat"]     = "usps"
    params["patch_quantile"] = 1.0
    return params 

# ---------------------------------------------------------
# SSFE Models   
# ---------------------------------------------------------

def define_spatial_ssfe_space(trial: optuna.Trial): 
    return {
        "embed_dim": trial.suggest_categorical("embed_dim", [64, 128]),

        # semantic branch
        "semantic_hidden_dim": trial.suggest_categorical("semantic_hidden_dim", [128, 256, 512]),
        "semantic_out_dim": trial.suggest_categorical("semantic_out_dim", [64, 128, 256]),
        "semantic_dropout": trial.suggest_float("semantic_dropout", 0.0, 0.2),

         # structural branch
        "gnn_dim": trial.suggest_categorical("gnn_dim", [128, 256]),
        "gnn_layers": trial.suggest_int("gnn_layers", 1, 3), 
        "gnn_heads": trial.suggest_categorical("gnn_heads", [4, 8]), 
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.2),
        "global_active_eps": trial.suggest_float("global_active_eps", 1e-6, 1e-3, log=True),

        # ssfe loss + optimization
        "w_contrast": 1.5,#trial.suggest_float("w_contrast", 1.0, 2.0),
        "w_cluster": 0.9,#trial.suggest_float("w_cluster", 0.75, 2.0),
        "w_recon": 0.9,#trial.suggest_float("w_recon", 0.5, 2.0),
        "contrast_temperature": trial.suggest_float("contrast_temperature", 0.05, 0.5, log=True),
        "cluster_temperature": trial.suggest_float("cluster_temperature", 0.05, 0.5, log=True),
        "n_prototypes": 16,#trial.suggest_categorical("n_prototypes", [16, 32, 64]),
        "proj_dim": trial.suggest_categorical("proj_dim", [64, 128]),

        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),

        "batch_size": 64, 
        "epochs": 400, 
        "patch_size": 32, 
        "early_stopping_rounds": 25, 
        "eval_fraction": 0.20,
        "min_delta": 1e-4 
    }

def define_tabular_ssfe_space(trial: optuna.Trial):
    return {
        "embed_dim": trial.suggest_categorical("embed_dim", [64, 128]),

        # semantic branch
        "semantic_out_dim": trial.suggest_categorical("semantic_out_dim", [64, 128, 256]),
        "transformer_dim": trial.suggest_categorical("transformer_dim", [64, 128]),
        "transformer_heads": trial.suggest_categorical("transformer_heads", [4, 8]),
        "transformer_layers": trial.suggest_int("transformer_layers", 1, 3),
        "transformer_attn_dropout": trial.suggest_float("transformer_attn_dropout", 0.0, 0.2),
        "semantic_proj_dim": trial.suggest_categorical("semantic_proj_dim", [64, 128, 256]),
        "semantic_hidden_dim": trial.suggest_categorical("semantic_hidden_dim", [128, 256, 512]),
        "semantic_depth": trial.suggest_int("semantic_depth", 1, 4),
        "semantic_dropout": trial.suggest_float("semantic_dropout", 0.0, 0.2),

        # structural branch
        "tabular_knn": trial.suggest_categorical("tabular_knn", [8, 16, 24]),
        "gnn_dim": trial.suggest_categorical("gnn_dim", [128, 256]),
        "gnn_layers": trial.suggest_int("gnn_layers", 1, 3),
        "gnn_heads": trial.suggest_categorical("gnn_heads", [4, 8]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.2),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.2),
        "global_active_eps": trial.suggest_float("global_active_eps", 1e-6, 1e-3, log=True),

        # loss + optimization
        "w_contrast": 1.5,
        "w_cluster": 0.9,
        "w_recon": 0.9,
        "contrast_temperature": trial.suggest_float("contrast_temperature", 0.05, 0.5, log=True),
        "cluster_temperature": trial.suggest_float("cluster_temperature", 0.05, 0.5, log=True),
        "n_prototypes": 16,
        "proj_dim": trial.suggest_categorical("proj_dim", [64, 128]),

        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),

        "batch_size": 64,
        "epochs": 400,
        "early_stopping_rounds": 25,
        "eval_fraction": 0.20,
        "min_delta": 1e-4,
    }

# ---------------------------------------------------------
# Fusion Spaces
# ---------------------------------------------------------

def define_deep_moe_space(trial: optuna.Trial): 

    return {
        "d_model": trial.suggest_categorical("d_model", [64, 128, 256]), 
        "transformer_heads": trial.suggest_categorical("transformer_heads", [2, 4, 8]), 
        "transformer_layers": trial.suggest_int("transformer_layers", 1, 3),
        "transformer_ff_mult": trial.suggest_categorical("transformer_ff_mult", [2, 4]), 
        "transformer_attn_dropout": trial.suggest_float("transformer_attn_dropout", 0.0, 0.2),
        "transformer_dropout": trial.suggest_float("transformer_dropout", 0.0, 0.2), 
        "gate_floor": trial.suggest_float("gate_floor", 0.0, 0.15), 
        "gate_num_tokens": trial.suggest_categorical("gate_num_tokens", [2, 4, 6]),
        "gateway_hidden_dim": trial.suggest_categorical("gateway_hidden_dim", [128, 256, 512]),

        "trunk_hidden_dim": trial.suggest_categorical("trunk_hidden_dim", [128, 256, 512]), 
        "trunk_depth": trial.suggest_int("trunk_depth", 1, 6), 
        "trunk_dropout": trial.suggest_float("trunk_dropout", 0.0, 0.25), 
        "trunk_out_dim": trial.suggest_categorical("trunk_out_dim", [None, 64, 128, 256]),

        "head_hidden_dim": trial.suggest_categorical("head_hidden_dim", [None, 64, 128, 256]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.25), 

        "log_var_min": -9.0, 
        "log_var_max": 9.0 
    }

def define_wide_rreg_space(trial: optuna.Trial): 
    return {
        "wide_l2_alpha": trial.suggest_float("wide_l2_alpha", 1e-7, 1e-2, log=True), 
        "wide_init_log_var": trial.suggest_float("wide_init_log_var", -2.0, 2.0), 
        "ridge_scale": trial.suggest_float("ridge_scale", 1e-5, 1e-1, log=True)
    }

def define_fusion_joint_space(trial: optuna.Trial): 
    p = {}
    p.update(define_deep_moe_space(trial))
    p.update(define_wide_rreg_space(trial))

    # Force high smoothing w/o fucking search up 
    grad_ema = 1.0 - trial.suggest_float("grad_ema_one_minus", 1e-3, 2e-1, log=True)

    p.update({
        "mix_alpha": trial.suggest_float("mix_alpha", 0.1, 0.8), 
        "mix_mult": trial.suggest_categorical("mix_mult", [2, 4]), 
        "mix_min_lambda": trial.suggest_float("mix_min_lambda", 0.1, 0.4),  

        "lr_deep": trial.suggest_float("lr_deep", 1e-4, 5e-3, log=True),
        "lr_wide": trial.suggest_float("lr_wide", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),

        "grad_beta": trial.suggest_float("grad_beta", 0.2, 0.9),
        "grad_ema": grad_ema,
        "grad_min_scale": trial.suggest_float("grad_min_scale", 0.35, 0.9),
        "grad_max_scale": trial.suggest_float("grad_max_scale", 1.5, 3.5),
        "grad_warmup_epochs": trial.suggest_int("grad_warmup_epochs", 0, 40),

        "aux_deep_weight": 0.6,#trial.suggest_float("aux_deep_weight", 3e-2, 5e-1, log=True),
        "aux_wide_weight": trial.suggest_float("aux_wide_weight", 3e-2, 5e-1, log=True),
        "aux_hard_scale": trial.suggest_float("aux_hard_scale", 0.1, 0.7),
        "aux_decay_power": trial.suggest_float("aux_decay_power", 0.5, 3.0),

        "hsic_weight": trial.suggest_float("hsic_weight", 1e-2, 5e-1, log=True),
        "hsic_sigma": 0.0, # median heuristic  
        
        "batch_size": trial.suggest_categorical("batch_size", [128]),

        "w_ordinal": 1.0, 
        "w_uncertainty": 1.0, 
        "var_floor": 1e-6, 
        "prob_eps": 1e-9, 
        "soft_epochs": 400, 
        "hard_epochs": 400, 
        "eval_fraction": 0.2, 
        "early_stopping_rounds": 20, 
        "min_delta": 1e-4, 
        "mix_with_replacement": True
    })
    return p 

# ---------------------------------------------------------
# MLP Projector  
# ---------------------------------------------------------

def define_tabular_space(trial: optuna.Trial): 
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 384, 512]), 
        "depth": trial.suggest_int("depth", 4, 16), 
        "dropout": trial.suggest_float("dropout", 0.05, 0.3), 

        "mix_alpha": trial.suggest_float("mix_alpha", 0.2, 1.0, log=True),
        "mix_mult": trial.suggest_categorical("mix_mult", [2, 4]), 

        "alpha_mae": trial.suggest_float("alpha_mae", 0.75, 3.0),
        "beta_supcon": trial.suggest_float("beta_supcon", 0.0, 2.0), 
        "supcon_temperature": trial.suggest_float("supcon_temperature", 0.2, 2.0, log=True), 
        "supcon_dim": trial.suggest_categorical("supcon_dim", [128, 256]),

        "transformer_dim": trial.suggest_categorical("transformer_dim", [64, 128]),
        "transformer_heads": trial.suggest_categorical("transformer_heads", [4, 8]),
        "transformer_layers": trial.suggest_int("transformer_layers", 1, 3),
        "transformer_dropout": trial.suggest_float("transformer_dropout", 0.0, 0.15),
        "transformer_attn_dropout": trial.suggest_float("transformer_attn_dropout", 0.0, 0.15),

        "reduce_depth": trial.suggest_int("reduce_depth", 1, 4), 
        "reduce_dropout": trial.suggest_float("reduce_dropout", 0.0, 0.15), 

        "lr": trial.suggest_float("lr", 3e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True),

        "reduce_dim": 256, 
        "eval_fraction": 0.2,
        "min_delta": 1e-4, 
        "early_stopping_rounds": 50, 
        "batch_size": 256,
        "soft_epochs": 500,
        "hard_epochs": 250, 
        "max_mix": None, 
        "anchor_power": 1.0 
    }
