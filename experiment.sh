#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <hpc_file> <program>"
  exit 1
fi

hpc_file="$1"
program="$2"

machines=(2 4 10)
sizes=(100 1000 2000)
steps_list=(100 1000 10000 100000)

mkdir -p logs

for m in "${machines[@]}"; do
  for size in "${sizes[@]}"; do
    for steps in "${steps_list[@]}"; do
      jobname="th${m}_sz${size}_st${steps}"
      echo "Submitting $jobname (nodes=$m, size=$size, steps=$steps)"
      qsub \
        -N "$jobname" \
        -pe mpich "$m" \
        -o "./${jobname}.csv" \
        "$hpc_file" "$program" "$size" "$steps" 
    done
  done
done
