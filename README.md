# Population Regression via Wide & Deep MoE 

Wide and Deep architecture that aims to unify extensive counts highly correlated with population and intrinsic rates/texture of satellite imaging and more esoteric data sources. The Deep portion of the architecture implements a variety of self-supervised feature extractors which learn embeddings that are then fused via a transformer based residual MLP network. 

## Current Focus 

- Developing self-supervised feature extractors to extract high quality representations for use in early fusion.
- Looking to add Infrared satellite tuned to two different frequencies (Vegation and Concrete) as expert modalities. 

## Data Processing Pipeline

Dataset compilers live in `preprocessing/` and emit `.mat` files under `data/`:
- VIIRS nighttime lights processed via novel strategy for self-supervised learning on satellite based tensor datasets
- SAIPE and IRS intensive/rate based socioeconomic information. 
- USPS derived rates about changes in business and residential addresses. 

See `preprocessing/README.md` for the per‑dataset build steps.

## Evaluation and Optimization

Core tools:
- `analysis/cross_validation.py` for repeated CV with fold‑safe scaling.
- `optimization/` for Optuna optimization and nested CV.

This repository uses a nix shell for dependencies and build consistency. Tests are located in `testbench/` with the tests that align most with the repositories current directions being location in:
- `testbench/eval.py` 
- `testbench/embedding.py`

## Repository Layout

- `analysis/`: CV harness and diagnostic tooling.
- `preprocessing/`: dataset compilation and loaders.
- `models/`: estimators, metric learning, graph utilities, post‑processing.
- `testbench/`: end-to-end testing and visualization. 
- `scripts/`: data fetch and parsing scripts.
- `utils/`: shared helpers (paths  metrics, interfaces).
- `optimization/`: Configuration and optimization contracts for Optuna based evaluation. 
- `data/`: datasets and outputs. 
