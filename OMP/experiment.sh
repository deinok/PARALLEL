#!/bin/bash

# Check if the program name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <program>"
  exit 1
fi

program="$1"

sizes=(100 1000 2000)
steps_list=(100 1000 10000 100000)

# Loop over each combination of size and steps and submit a job using qsub
for size in "${sizes[@]}"; do
  for steps in "${steps_list[@]}"; do
    process_name="${program}_${size}_${steps}"
    echo "Submitting job: $process_name"
    /opt/gridengine/bin/lx-amd64/qsub -N "$process_name" ./hpc.sh "$program" "$size" "$steps"
  done
done
