#!/usr/bin/bash 
# 
# parse_ur_classification.sh  Andrew Belles  Dec 22nd, 2025 
#
# Parses NCHS Urban-Rural Classification Dataset for 2023 at 
# target directory provided by command line argument 
#

set -euo pipefail 

if [ $# -ne 2 ]; then 
  echo "Usage: parse_ur_classification.sh <year> <target_dir>" >&2 
  exit 2 
fi 

year="$1"
dir="$2"
in="${dir}/urban_rural_classification_${year}.csv"
out="${dir}/nchs_classification_${year}.csv"

if [ ! -f "$in" ]; then 
  echo "error missing file: $in" >&2 
  exit 1 
fi 

gawk -v FPAT='([^,]*)|("([^"]|"")*")' -v OFS=',' '
NR==1 { print "FIPS,class_code"; next }
{
  # Column 2 is FIPS code, column 5 is 2023 Code
  fips = $2
  gsub(/"/, "", fips)
  gsub(/^[ \t]+|[ \t]+$/, "", fips)
  if (fips ~ /^[0-9]+$/) {
    fips = sprintf("%05d", fips)
  }

  code = $5
  gsub(/"/, "", code)
  if (match(code, /[0-9]+/)) {
    code = substr(code, RSTART, RLENGTH)
    print fips, code
  }
}' "$in" > "$out"

echo "Wrote $out"
