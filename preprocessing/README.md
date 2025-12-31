# Preprocessing

Dataset compilation scripts. These read raw inputs under `data/` and write `.mat` datasets.

## Requirements

Run from the repo root inside `nix-shell` (sets `PROJECT_ROOT`, installs deps, creates `data/`).

## Datasets

### VIIRS Nighttime Lights
`viirs_nchs_dataset.py` to `data/viirs_nchs.mat`
- GLCM texture features, radiance entropy (VREI), gradients.

### TIGER Road Network
`tiger_nchs_dataset.py` to `data/tiger_nchs.mat`
- Road topology metrics, orientation entropy, centrality summaries.

### NLCD Land Cover
`nlcd_nchs_dataset.py` to `data/nlcd_nchs.mat`
- Landscape configuration metrics (AI, contagion, edge density, LPI).

### Mobility / Travel Time
`travel_times_dataset.py` to `data/travel_times.mat`
- Distance / mobility adjacency sources.

### SAIPE / Climate Baselines
`saippe_population_dataset.py`, `climate_*_dataset.py` provide baseline datasets.

## Loaders

`preprocessing/loaders.py` defines dataset loaders used by CV and testbench.
`preprocessing/disagreement.py` includes passthrough loaders for stacking.
