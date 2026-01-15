# Testbench

End‑to‑end evaluation scripts for expert models, stacking, CNN imaging, graph diagnostics, and plots.

## Requirements 

Run from inside `nix-shell` (sets `PROJECT_ROOT`, installs deps, creates `data/`)

Paths and defaults are centralized in `testbench/utils/paths.py`.

All required datasets must be instantiated within `preprocessing/` prior to any tests being callable. 

## Quick Start 

Stacking pipeline: 
```bash
python testbench/stacking.py --tests pipeline --cross both 
```

Imaging (VIIRS tensor CNN): 

```bash
python testbench/imaging.py --mode dual --trials 50
```

## Scripts and CLI 

### `testbench/stacking.py`

Runs expert model optimization, out-of-fold probability generation, stacking, and correct-and-smooth. 

Tests (`--tests`):

- `expert_opt`
- `expert_oof`
- `stacking_opt`
- `stacking`
- `cs_opt`
- `pipeline`

Args: 

- `--tests [list]` subset of the above; default runs all 
- `--cross {off,on,both}` apply cross-modal features to cross-tests only 
- `--datasets [list]` subset of datasets (default from `testbench/utils/paths.py`)
- `--no-opt` skip tests ending in `_opt`
- `--filter` use Boruta filter config + splits 

Notes: 

- Cross-tests are `stacking`, `stacking_opt`, `cs_opt`, and `pipeline`. 
- `--filter` uses `testbench/filter_config.yaml` and `data/results/boruta_splits`

Example: 

```bash 
python testbench/stacking.py --tests pipeline --cross both 
```

### `testbench/imaging.py`

CNN optimization and pooled feature export for VIIRS tensor dataset. 

Args: 

- `--mode {spatial,gaf,dual}` input channel mode 
- `--trials <int>` Optuna trials (default 150)
- `--folds <int>` CV folds (default 3)
- `--random-state <int>` RNG seed (default 0)

Outputs pooled features to `data/results/viirs_cnn_features.mat` by default. 

Example: 

```bash
python testbench/imaging.py --mode gaf --trials 100 --folds 5 
```

### `testbench/adjacency.py`

Graph construction and adjacency diagnostics for queen/mobility/kNN and learned metrics. 

Args: 

- `--k <int>` neighbor count for kNN and mobility (default 12)
- `--metric-keys [list]` evaluate learned metric keys (optional)

Example: 

```bash
python testbench/adjacency.py --k 8 --metric-keys metric_v1 metric_v2
```

### `testbench/downstream.py`

Optimizes learned metric space and evaluates downstream Correct-and-Smooth. 

Tests (`--tests`): 

- `metric_base` 
- `metric_passthrough`

Args: 

- `--tests [list]`
- `--cross {off,on,both}`

Example: 

```bash
python testbench/downstream.py --tests metric_base --cross both
```

### `testbench/round_robin.py`

Round-robin evaluation across datasets and stacking variants. 

Tests (`--tests`): 
- `round_robin_opt`
- `stacking`
- `stacking_opt`
- `cs_opt`

Args: 

- `--tests [list]`
- `--no-opt` skip tests ending in `_opt`
- `--trials <int>` Optuna trials (default from script)
- `--quiet` suppress verbose logs 


Example: 

```bash
python testbench/round_robin.py --tests stacking_opt --trials 200
```

### `testbench/noise.py`

Boruta feature relevance + export. 

Tests (`--tests`): 
- `boruta`
- `boruta_export`

Args: 
- `--tests [list]`
- `--out-csv <path>` summary CSV 
- `--top-k <int>` number of features to retain in export 
- `--filter-csv <path>` input CSV for export 
- `--out-dir <path>` output directory for split CSVs 

Example: 

```bash
python testbench/noise.py --tests boruta_export --top-k 40 --out-dir data/results/boruta_splits 
```

### `testbench/collinearity.py`

Collinearity/VIF diagnostics. 

Tests (`--tests`): 

- `pairwise`
- `full`

Args: 

- `--tests [list]`
- `--method {cv,corr}` method for redundancy checks 
- `--no-r2` disable $R^2$ computation 

Example: 

```bash
python testbench/collinearity.py --tests full --method corr 
```

### `testbench/plots.py`

Plot rendering for stackign, adjacency, and downstream.

Groups (`--group`): 

- `stacking`
- `adjacency`
- `downstream`
- `all` (default)

Args: 

- `--group {stacking,adjacency,downstream,all}`
- `--plots [list]` subset of plots within group 
- `--out <dir>` output directory (default `testbench/images`)
- `--metric-keys [list]` learned metrics to include 
- `--log-hist` log-scale histograms 
- `--quiet` suppress verbose logs 
- `--round-robin` use round-robin stack outputs 

Examples: 

```bash
python testbench/plots.py --group stacking 
python testbench/plots.py --group adjacency --plots metric_avg_degree degree_dist 
```

## Config Files 

- `testbench/model_config.yaml`: Optuna best params cache. 
- `testbench/round_robin_config.yaml`: round-robin settings. 
- `testbench/filter_config.yaml`: Boruta filtering configuration 
