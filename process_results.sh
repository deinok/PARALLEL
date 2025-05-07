#!/bin/bash

# Remove non CSV lines from all CSV files in the current directory
sed -i '' '/;/!d' *.csv

# Join all CSV files into one
cat *.csv > all_results.csv

# Prepend header to the combined CSV file
sed -i '' '1i\
thread;max threads;time;N;M;steps
' all_results.csv