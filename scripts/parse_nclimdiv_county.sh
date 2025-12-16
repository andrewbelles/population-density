#!/usr/bin/env bash  
# 
# parse_nclimdiv_county.sh  Andrew Belles  Dec 15th, 2025 
#
# Parses raw nclimdiv county files filtering to 1990 and outputs 
# to a clean CSV with FIPS, year, and monthly values 
#

set -euo pipefail 

RAW="../data/climate/nclimdiv_county/raw"
OUT="../data/climate/nclimdiv_county/parsed"

mkdir -p "$OUT"

if [ "$#" -ne 2 ]; then 
  echo "usage: ./parse_nclimdiv_county.sh [start_year] [end_year]"
  exit 1  
fi 

START_YEAR="$1" 
END_YEAR="$2"

for f in "$RAW"/climdiv-*; do 
  echo "$f"
  [[ -f "$f" ]] || continue 

  elem="$(basename "$f" | sed -E 's/climdiv-([a-z]+)cy-.*/\1/')"
  outfile="$OUT/${elem}_${START_YEAR}_${END_YEAR}.csv" 

  echo "$outfile"
  echo "fips,year,jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec" > "$outfile"

  # format: 
  # cols 1-5: state-county fips 
  # col 6: element code 
  # cols 7-10 year 
  # cols 11+: 12 monthly values, each 7 chars wide 

  awk -v start="$START_YEAR" -v end="$END_YEAR" '
  {
    fips = substr($0, 1, 5)
    year = substr($0, 8, 4) + 0
    
    if (year < start || year > end) next
    
    printf "%s,%d", fips, year
    for (i = 0; i < 12; i++) {
      val = substr($0, 12 + i*7, 7)
      gsub(/^ +| +$/, "", val)
      printf ",%s", val
    }
    printf "\n"
  }' "$f" >> "$outfile"

  echo "OK: $elem -> $outfile ($(wc -l < "$outfile") rows)"
done 
