#!/usr/bin/env bash
#
# fetch_bea_cainc.sh  Andrew Belles  Dec 21st, 2025
#
# Fetches county annual income data via API.
# Covers per capita income, total personal income, population.
#
# REQUIRES: API key from https://apps.bea.gov/API/signup/
# Set BEA_API_KEY environment variable or pass as first argument.
#
# Coverage: 1969-present
#

set -euo pipefail

API_KEY="${BEA_API_KEY:-${1:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: BEA API key required." >&2
  echo "  Set BEA_API_KEY env var or pass as first argument." >&2
  echo "  Get free key at: https://apps.bea.gov/API/signup/" >&2
  exit 1
fi
echo "$API_KEY"

OUT="../data/socioeconomic/bea_cainc/raw"
mkdir -p "$OUT"

BASE="https://apps.bea.gov/api/data"

# CAINC1: Per capita personal income, population
# CAINC4: Personal income components (wages, transfers, etc.)
# CAINC30: Economic profile summary
TABLES=("CAINC1" "CAINC4" "CAINC30")

declare -A LINECODES
LINECODES["CAINC1"]="1,2,3"        # Population, Personal Income, Per Capita PI
LINECODES["CAINC4"]="10,20,30,40"  # Wages, Proprietors, Dividends, Transfers  
LINECODES["CAINC30"]="10,20,30"    # Key summary lines

START_YEAR=1990
END_YEAR=2020

echo "> Fetching BEA CAINC data for $START_YEAR-$END_YEAR"
echo "> Output: $OUT"

for table in "${TABLES[@]}"; do
  echo "> Fetching $table..."
  
  # Split line codes into array
  IFS=',' read -ra codes <<< "${LINECODES[$table]}"
  
  for linecode in "${codes[@]}"; do
    for ((y=START_YEAR; y<=END_YEAR; y+=5)); do
      end=$((y + 4))
      [[ $end -gt $END_YEAR ]] && end=$END_YEAR
      
      years=""
      for ((yr=y; yr<=end; yr++)); do
        years+="$yr,"
      done
      years="${years%,}"
      
      outfile="$OUT/${table}_line${linecode}_${y}_${end}.json"
      
      url="${BASE}?UserID=${API_KEY}&method=GetData&datasetname=Regional"
      url+="&TableName=${table}&LineCode=${linecode}&GeoFIPS=COUNTY&Year=${years}"
      url+="&ResultFormat=JSON"
      
      if curl -fsSL "$url" -o "$outfile"; then
        if grep -q '"Error"' "$outfile" 2>/dev/null; then
          echo "  WARN: API error for $table line $linecode $y-$end" >&2
          grep -o '"APIErrorDescription":"[^"]*"' "$outfile" >&2
        else
          echo "  OK: $table line $linecode $y-$end"
        fi
      else
        echo "  ERROR: Failed $table line $linecode $y-$end" >&2
      fi
      
      sleep 2
    done
  done
done

echo "> Done."
ls -lh "$OUT"
