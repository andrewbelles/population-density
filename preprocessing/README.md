# Preprocessing

This directory contains dataset compilation scripts. These scripts read raw inputs under `data/` and write `.mat` datasets (also under `data/`).

## Requirements

- Run in the project root, inside `nix-shell` (sets `PROJECT_ROOT`, creates `.venv`, installs Python deps, and creates `data/` subdirectories).
- Raw data should be present under `data/` (see `scripts/README.md`).

## Datasets

### `climate_population_dataset.py` → `data/climate_population.mat`

Builds a per-decade dataset of county-level climate features against population density labels.

- Climate features: NOAA NClimDiv county products (monthly), aggregated by county and aligned on a canonical FIPS ordering shared across decades.
- Labels: population density for the target decades.
- Also exports: `data/climate_counties.tsv` (county metadata used by the geospatial backend).

Run:
```bash
python preprocessing/climate_population_dataset.py
```

### `climate_geospatial_dataset.py` → `data/climate_geospatial.mat`

Builds a long-run dataset of county-level climate features against `(lat, lon)` labels.

This is used for experiments like coordinate→climate regression, residual workflows, and climate-aware geospatial encodings.

Run:
```bash
python preprocessing/climate_geospatial_dataset.py
```

