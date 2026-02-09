#!/usr/bin/bash 
# 
# experts.sh  Andrew Belles  Feb 9th, 2026 
#
# Creates VIIRS, SAIPE, and USPS datasets for 2013 and 2020 
#
#

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

census="data/census"
counties="data/geography/county_shapefile/tl_2020_us_county.shp"

declare -A SAIPE_CSV=(
  [2013]="data/saipe/saipe_2013.csv" 
  [2020]="data/saipe/saipe_2020.csv" 
)

declare -A USPS_GPKG=(
  [2013]="data/usps/usps_master_tracts_2013.gpkg"
  [2020]="data/usps/usps_master_tracts_2020.gpkg" 
)

declare -A VIIRS_TIF=(
  [2013]="data/viirs/viirs_2013_median_masked.tif"
  [2020]="data/viirs/viirs_2020_median_masked.tif" 
)

check_file() {
  [[ -f "$1" ]] || { 
    echo "[missing] $1" >&2; 
    exit 1; 
  }
}

mkdir -p "data/datasets" "data/tensors" 

check_file "$counties" 
for y in 2013 2020; do 
  check_file "${SAIPE_CSV[$y]}"
  check_file "${USPS_GPKG[$y]}"
  check_file "${VIIRS_TIF[$y]}"
done

for y in 2013 2020; do
  python preprocessing/saipe_scalar_dataset.py \
    --csv "${SAIPE_CSV[$y]}" \
    --year "$y" \
    --census-dir "$census" \
    --out "data/datasets/saipe_scalar_${y}.mat" \
    --out-csv "data/datasets/saipe_scalar_${y}.csv"

  python preprocessing/usps_scalar_dataset.py \
    --usps-gpkg "${USPS_GPKG[$y]}" \
    --year "$y" \
    --census-dir "$census" \
    --counties-path "$counties" \
    --out-path "data/datasets/usps_scalar_${y}.mat" \
    --csv-path "data/datasets/usps_scalar_${y}.csv"

  python preprocessing/tensors.py \
    --viirs \
    --year "$y" \
    --census-dir "$census" \
    --counties-path "$counties" \
    --viirs-path "${VIIRS_TIF[$y]}" \
    --viirs-out "data/tensors/viirs_${y}"
done

echo "Done."
