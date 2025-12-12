#!/usr/bin/bash 
# 
# fetch_geography.sh  Andrew Belles  Dec 10th, 2025 
#
# Script to download public data on county geometries as 
# well as total land area from government's census API 
#
 
GEOGRAPHY="../data/geography"
mkdir -p $GEOGRAPHY

# Main shapefile for geometries 
echo "Downloading 2020 county shapefile..."
curl -sS -f -o "$GEOGRAPHY/tl_2020_us_county.zip" \
  "https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip"

# County Gazetteer which has helpful auxilliary information like FIPS, Land Area, names 
echo "Downloading 2020 county gazetteer..."
curl -sS -f -o "$GEOGRAPHY/2020_Gaz_counties_national.zip" \
  "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"

echo "Extracting geography files..."
unzip -q -o "$GEOGRAPHY/tl_2020_us_county.zip" -d "$GEOGRAPHY/county_shapefile"
unzip -q -o "$GEOGRAPHY/2020_Gaz_counties_national.zip" -d "$GEOGRAPHY/"

echo -e "\nGeography Summary:" 
ls -lh $GEOGRAPHY
