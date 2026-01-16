#!/usr/bin/env python3 
# 
# imaging.py  Andrew Belles  Jan 15th, 2026 
# 
# Testbench for models that require CNN, rely on imaging 
# 
# 

import argparse 

import numpy as np 

from scipy.io import savemat 

from testbench.utils.paths     import (
    CONFIG_PATH,
    TENSOR_DATA,
    TENSOR_POOLED_OUT
)

from testbench.utils.data      import (
    make_tensor_loader,
    make_tensor_adapter
)

from testbench.utils.metrics   import OPT_TASK

from analysis.cross_validation import CVConfig 

from analysis.hyperparameter   import run_optimization, define_cnn_space, CNNEvaluator 

from models.estimators         import make_image_cnn 

from utils.helpers import (
    save_model_config,
)

DEFAULT_MODEL_KEY = "ImageCNN/VIIRS_TENSOR"


def test_viirs(
    *,
    data_path: str = TENSOR_DATA,
    out_path: str = TENSOR_POOLED_OUT,
    model_key: str = DEFAULT_MODEL_KEY,
    mode: str = "dual",
    canvas_h: int = 128, 
    canvas_w: int = 128, 
    gaf_size: int = 64, 
    trials: int = 50, 
    folds: int = 3, 
    random_state: int = 0, 
    config_path: str | None = None 
): 
    loader  = make_tensor_loader(mode, canvas_h, canvas_w, gaf_size) 
    adapter = make_tensor_adapter(mode, canvas_h, canvas_w, gaf_size) 
    factory = make_image_cnn(
        input_adapter=adapter, 
        normalize_main=True, 
        normalize_aux=False,
        pool_mode="avgmax"
    )

    config  = CVConfig(
        n_splits=folds, 
        n_repeats=1, 
        stratify=True, 
        random_state=random_state
    )
    config.verbose = False 

    evaluator = CNNEvaluator(
        filepath=data_path,
        loader_func=loader,
        model_factory=factory,
        param_space=define_cnn_space,
        task=OPT_TASK,
        config=config
    )

    best_params, best_value = run_optimization(
        name=model_key,
        evaluator=evaluator,
        n_trials=trials,
        direction="maximize",
        random_state=random_state,
        sampler_type="multivariate-tpe"
    )

    if config_path: 
        save_model_config(config_path, model_key, best_params)

    data  = loader(data_path)
    X     = np.asarray(data["features"], dtype=np.float32)
    y     = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    fips  = np.asarray(data["sample_ids"], dtype="U5")
    model = factory(**best_params) 
    model.fit(X, y)

    feats = model.extract_features(X)
    feat_names = np.array([f"cnn_f{i}" for i in range(feats.shape[1])], dtype="U")

    savemat(out_path, {
        "features": feats.astype(np.float32, copy=False),
        "labels": y.reshape(-1, 1),
        "fips_codes": fips,
        "feature_names": feat_names, 
        "mode": np.array([mode], dtype="U")
    })

    print(f"[cnn] best value: {best_value:.5f}")
    print(f"[cnn] saved pooled features: {out_path}")


def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--mode", choices=["spatial", "gaf", "dual"], default="spatial")
    parser.add_argument("--trials", default=80)
    parser.add_argument("--folds", default=2)
    parser.add_argument("--random-state", default=0)
    args = parser.parse_args()

    test_viirs(
        mode=args.mode,
        trials=args.trials,
        folds=args.folds,
        random_state=args.random_state,
        config_path=CONFIG_PATH
    )


if __name__ == "__main__": 
    main() 
