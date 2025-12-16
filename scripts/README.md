# Scripts

This directory contains the data fetch/validation/parsing scripts used to build the datasets in `data/`.

## Requirements

- `bash`, `curl`, `awk`, `sed`, `grep`
- Recommended: run inside `nix-shell` from the project root (`shell.nix` installs the Python environment; these scripts still rely on system CLI tools).

## Usage

### Fetch NClimDiv county products (raw)

Downloads NOAA NClimDiv county-level files for:
- Palmer indices: `pdsi`, `pmdi`, `phdi`, `zndx`
- Degree days: `hddc`, `cddc`

```bash
./scripts/fetch_nclimdiv_county.sh
./scripts/validate_nclimdiv.sh
```

Outputs to `data/climate/nclimdiv_county/raw/`.

### Parse NClimDiv county products (raw → CSV)

Converts fixed-width NOAA files to CSV with monthly columns:
`fips,year,jan,...,dec`

```bash
./scripts/parse_nclimdiv_county.sh 1990 2020
```

Outputs to `data/climate/nclimdiv_county/parsed/`.

### Legacy (gridded daily) fetch

```bash
./scripts/fetch_climdiv.sh
./scripts/fetch_census.sh
./scripts/fetch_geography.sh
```

`./scripts/fetch_all.sh` calls the legacy fetch scripts.

