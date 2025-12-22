#!/usr/bin/env bash
#
# fetch_usda_education.sh  Andrew Belles  Dec 21st, 2025
#
# Fetches USDA ERS county-level education attainment data.
# No API key required.
#
# Coverage: Decennial (1970, 1980, 1990, 2000) + ACS 5-year estimates
#
# Variables: 
# % adults with HS diploma, 
# % with Bachelors
#

set -euo pipefail

OUT="../data/socioeconomic/usda_education/raw"
mkdir -p "$OUT"

echo "> Fetching USDA ERS Education data"
echo "> Output: $OUT"

# Main education file (single file with all years)
EXCEL_URL="https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/48747/Education2023.xlsx"
CSV_URL="https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/48747/Education2023.csv"

echo "> Downloading Education.csv..."
if curl -fsSL "$CSV_URL" -o "$OUT/Education.csv"; then
  echo "  OK: Education.csv"
else
  echo "  WARN: CSV failed, trying Excel..." >&2
  if curl -fsSL "$EXCEL_URL" -o "$OUT/Education.xlsx"; then
    echo "  OK: Education.xlsx"
  else
    echo "  ERROR: Both downloads failed" >&2
    exit 1
  fi
fi
