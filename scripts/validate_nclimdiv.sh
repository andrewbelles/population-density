#!/usr/bin/bash 
# 
# validate_nclimdiv.sh  Andrew Belles  Dec 13th, 2025 
#
#

NCLIMDIV="../data/climate/nclimdiv_county/raw" 

shopt -s nullglob
for f in "$NCLIMDIV"/climdiv-*cy-v*; do
  ls -lh "$f"
  test -s "$f" || { echo "ERROR: empty file: $f" >&2; continue; }

  num_fields="$(awk 'NR==1{
    rest=substr($0,12); gsub(/^ +/,"",rest);
    if (rest=="") { print 0; exit }
    n=split(rest,a,/ +/);
    print n; exit
  }' "$f")"

  num_counties="$(awk '{seen[substr($0,1,5)]=1} END{print length(seen)}' "$f")"

  read -r miny maxy < <(awk '{
    y=substr($0,8,4);
    if (min=="" || y<min) min=y;
    if (max=="" || y>max) max=y;
  } END{print min, max}' "$f")

  num_years=$((10#$maxy - 10#$miny + 1))

  echo "file: $f"
  echo "numeric_fields_per_record: $num_fields"
  echo "unique_counties          : $num_counties"
  echo "years                    : $miny..$maxy"
  echo "num_years                : $num_years"
done
