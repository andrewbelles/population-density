#!/usr/bin/bash 

./climate_analysis.py --folds 5 --repeats 50 --target "lat" 
./climate_analysis.py --folds 5 --repeats 50 --target "lon"
