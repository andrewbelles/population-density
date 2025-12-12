#!/usr/bin/bash 
# 
# fetch_all.sh  Andrew Belles  Dec 10th, 2025
#
# Calls all data fetching bash scripts 
#
#

# File paths 
CLIMDIV="../data/climate"
CENSUS="../data/census"
GEOGRAPHY="../data/geography"

./fetch_climdiv.sh 

./fetch_census.sh 

./fetch_geography.sh

echo "Summary:"
du -sh $CLIMDIV $CENSUS $GEOGRAPHY  
