#!/usr/bin/bash
#
# fetch_climdiv.sh  Andrew Belles  Dec 8th, 2025
#
# Fetches Climate Division Data for the following blocks
# - 1960: Acts as an early set of values while still being good quality
# - 1988: Midpoint between early and modern US acting as a transition
# - 2023: Most up to date full year
#
# All 4 variables (tmax, tmin, tavg, prcp) are in each monthly file.
#
#
URL="https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/grids"
YEARS=(1960 1990 2020)

CLIMGRID="../data/climate"
mkdir -p $CLIMGRID

for year in "${YEARS[@]}"; do
    mkdir -p "$CLIMGRID/${year}"
    for month in $(seq -w 1 12); do
        file="ncdd-${year}${month}-grd-scaled.nc"
        url="${URL}/${year}/${file}"
        echo "Downloading: ${file}"
        curl -sS -f -o "$CLIMGRID/${year}/${file}" "${url}" || \
            echo "  Failed: ${url}"
    done
done

echo -e "\nClimdiv Summary:"
ls -lh $CLIMGRID*
