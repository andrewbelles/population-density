#!/usr/bin/bash 
# 
# fetch_nclimdiv_county.sh  Andrew Belles  Dec 13th, 2025 
#
# Fetches NOAA's county-level products. Contains information 
# on drought indices and degree days already aggregated on county 
# FIPS 
#

set -euo pipefail 

BASE="https://www.ncei.noaa.gov/pub/data/cirs/climdiv" 
OUT="../data/climate/nclimdiv_county/raw"
mkdir -p "$OUT"

ELEMS=(
  tmpc pcpn tmin tmax 
  pdsi pmdi phdi zndx 
  hddc cddc 
)

curl -fSsSL "$BASE/" \
  | grep -oE 'climdiv-[a-z0-9]+cy-v[0-9.]+-[0-9]{8}[^"< ]*' \
  | sed -E 's/^climdiv-([a-z0-9]+)cy-.*/\1/' \
  | sort -u 

INDEX="$(curl -fsSL "$BASE/")"

for e in "${ELEMS[@]}"; do 
  pat="climdiv-${e}cy-v[0-9.]+-[0-9]{8}[A-Za-z0-9._-]*"
  f="$(printf '%s' "$INDEX" | grep -oE "$pat" | sort -V | tail -n 1 || true)"

  if [[ -z "$f" ]]; then 
    echo "WARN: no match for ${e}" >&2 
    continue 
  fi 

  curl -fSLo "$OUT/$f" "$BASE/$f" 
  echo "OK: $e -> $OUT/$f"
done

ls -lh "$OUT"
