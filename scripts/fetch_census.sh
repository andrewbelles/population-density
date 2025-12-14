#!/usr/bin/bash 
# 
# fetch_census.sh  Andrew Belles  Dec 10th, 2025 
#
# Downloads Census data from government Census API 
# - Data is decennial, fetching 1960, '90, and 2020
#

CENSUS="../data/census"
mkdir -p $CENSUS 

echo "Downloading NBER county population 1900-1990..."
curl -sS -o "$CENSUS/county_population_1900_1990.csv" \
  "https://data.nber.org/census/population/cencounts/cencounts.csv" 

echo "Downloading 2000 county population..."
curl -sS -o "$CENSUS/county_population_2000.json" \
  "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=county:*"

echo "Downloading 2010 county population..."
curl -sS -o "$CENSUS/county_population_2010.json" \
  "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=county:*"

echo "Downloading 2020 county population..."
curl -sS -o "$CENSUS/county_population_2020.json" \
  "https://api.census.gov/data/2020/dec/pl?get=NAME,P1_001N&for=county:*"

echo -e "\nCensus Summary"
ls -lh ../data/census
