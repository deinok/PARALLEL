#!/bin/bash
# Force the shell to be the C-shell
#$ -S /bin/csh
# Request 2 GBytes of virtual memory
#$ -l h_vmem=2G
# Specify myresults as the output file
#$ -o myresults

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <program> <size> <steps>"
  exit 1
fi

program=$1
size=$2
steps=$3

output_file="${program}_${size}_${steps}.bmp"

"./$program" "$size" "$steps" "$output_file"
