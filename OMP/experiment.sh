#!/bin/bash

program="$1"  

sizes=(100 1000 2000)
steps_list=(100 1000 10000 100000)

# Loop over each combination of size and steps and call hpc.sh
for size in "${sizes[@]}"; do
  for steps in "${steps_list[@]}"; do
    echo "Running $program with size=$size and steps=$steps"
    ./hpc.sh "$program" "$size" "$steps"
  done
done