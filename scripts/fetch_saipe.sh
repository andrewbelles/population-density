#!/usr/bin/env bash
#
# fetch_saipe.sh  Andrew Belles  Dec 21st, 2025
#
# Fetches Census SAIPE (Small Area Income and Poverty Estimates) from API.
# County-level poverty rates and median household income.
#
# REQUIRES: API key from https://api.census.gov/data/key_signup.html
# Set CENSUS_API_KEY environment variable or pass as first argument.
#
# Coverage: 1989, 1993, 1995-2023 
#
# Variables: 
# SAEPOVRTALL_PT = All ages poverty rate
# SAEPOVALL_PT = All ages in poverty (count)
# SAEMHI_PT = Median household income
# SAEPOV0_17_PT = Children 0-17 in poverty

set -euo pipefail

API_KEY="${CENSUS_API_KEY:-${1:-}}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: Census API key required." >&2
  echo "  Set CENSUS_API_KEY env var or pass as first argument." >&2
  echo "  Get free key at: https://api.census.gov/data/key_signup.html" >&2
  exit 1
fi

OUT="../data/socioeconomic/saipe/raw"
mkdir -p "$OUT"

BASE="https://api.census.gov/data/timeseries/poverty/saipe"
VARS="NAME,SAEPOVRTALL_PT,SAEPOVALL_PT,SAEMHI_PT,SAEPOV0_17_PT,SAEPOVRTALL_MOE,SAEMHI_MOE"

# Years with SAIPE data (note gaps)
YEARS=(1989 1993 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020)

echo "> Fetching SAIPE poverty data"
echo "> Output: $OUT"
echo "> NOTE: No data exists for 1990, 1991, 1992, 1994"

for year in "${YEARS[@]}"; do
  [[ $year -gt 2020 ]] && continue
  
  outfile="$OUT/saipe_${year}.json"
  
  # Fetch all counties (state=* county=*)
  url="${BASE}?get=${VARS}&for=county:*&time=${year}&key=${API_KEY}"
  
  if curl -fsSL "$url" -o "$outfile"; then
    # Check for error response
    if grep -qE '^\["error"' "$outfile" 2>/dev/null; then
      echo "  WARN: API error for $year" >&2
      rm -f "$outfile"
    else
      lines=$(wc -l < "$outfile")
      echo "  OK: $year ($lines lines)"
    fi
  else
    echo "  ERROR: Failed to fetch $year" >&2
  fi
  
  sleep 0.3
done

echo "> Done."
ls -lh "$OUT"
