# Scripts

Data fetch and parsing helpers.

## Requirements

`bash`, `curl`, `awk`, `sed`, `grep`

## Example Usage: (Climate, NClimDiv)

```bash
./scripts/fetch_nclimdiv_county.sh
./scripts/validate_nclimdiv.sh
./scripts/parse_nclimdiv_county.sh 1990 2020
```

Outputs to `data/climate/nclimdiv_county/`.

## Additional Sources

- `fetch_saipe.sh`: SAIPE population data.
- `fetch_bea_cainc.sh`: BEA income data.
- `fetch_usda_edu.sh`: USDA education data.
- `parse_ur_classification.sh`: urban/rural labels.
