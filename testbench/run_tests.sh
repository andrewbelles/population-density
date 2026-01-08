#!/usr/bin/bash 
# 
# run_tests.sh  Andrew Belles  Jan 7th, 2025 
#
# Simple shell script to run all tests then valid plots for evaluation tests
# 

echo -e "[TEST START] Adjacency Tests\n"
./adjacency.py 

echo -e "[TEST START] Downstream Tests\n"
./downstream.py --cross both  

echo -e "[TEST START] Stacking Tests\n"
./stacking.py --cross both 

echo -e "[TEST START] Plotting\n"
./plots.py --group adjacency --log-hist 
./plots.py --group stacking 
./plot.py --group downstream 
