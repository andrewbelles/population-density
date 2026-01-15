# Preprocessing

Dataset compilation scripts. These read raw inputs under `data/` and write `.mat` datasets.

## Requirements

Run from the repo root inside `nix-shell` (sets `PROJECT_ROOT`, installs deps, creates `data/`).

## Datasets

### `viirs_nchs_dataset.py` -> `data/datasets/viirs_nchs_2023.mat`

Nighttime lights with NCHS labels. Feature set includes:
- Radiance summary stats: min, max, mean, variance, skew, kurtosis.
- VREI (radiance entropy over binned intensities).
- Coefficient of variation (std / mean).
- GLCM texture: contrast and homogeneity.
- Gradient magnitude (local radiance boundaries).

### `viirs_nchs_tensor_dataset.py` -> `data/datasets/viirs_nchs_tensor_2023.mat`

Tensor VIIRS dataset for CNNS: 
- Spatial canvas (fixed size).
- Binary geometry mask. 
- Gramian angular field imaging (GAF) from resampled intensity sequence. 

Optional flags: 
- `--all-touched` 
- `--log-scale`
- `--debug-png-dir <path>` for per-county PNGs. 

### `saipe_nchs_dataset.py` -> `data/datasets/saipe_nchs_2023.mat`

SAIPE socioeconomic features: 
- primary scalar expert model 

### `tiger_nchs_dataset.py` -> `data/datasets/tiger_nchs_2023.mat`

Road network topology with NCHS labels. Feature set includes:
- Road density (highway/local) and average segment length (highway/local).
- Intersection structure: 4‑way ratio, 3‑way ratio, dead‑end density.
- Circuity (local length / straight‑line length).
- Meshedness coefficient (graph connectivity).
- Approximate betweenness (mean, max), straightness centrality (mean).
- Orientation entropy (bearing diversity).
- Integration at radius 3 (space syntax).

### `nlcd_nchs_dataset.py` -> `data/datasets/nlcd_nchs_2023.mat`

Land cover configuration with NCHS labels. Feature set includes:
- Class proportions: developed (open/low/med/high), agriculture (pasture/crops), nature.
- Lawn index (open / developed) and urban core (med+high / developed).
- Edge density and Shannon diversity.
- Aggregation index for developed classes (open/low/med/high).
- Contagion (landscape clumping).
- Edge density for developed open and nature.
- Largest patch index for developed high.

### Mobility / Travel Time
`travel_times_dataset.py` -> `data/travel_times.mat`
- Distance / mobility adjacency sources.

## Loaders

`preprocessing/loaders.py` defines dataset loaders used by CV and testbench.
`preprocessing/disagreement.py` includes passthrough loaders for stacking.
