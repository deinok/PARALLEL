#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <hpc_file> <program>"
  exit 1
fi

hpc_file="$1"
program="$2"

machines=(2 4)              # number of nodes to request
sizes=(100 1000 2000)
steps_list=(100 1000 10000 100000)

mkdir -p logs

for m in "${machines[@]}"; do
  for size in "${sizes[@]}"; do
    for steps in "${steps_list[@]}"; do
      jobname="${program}_mach${m}_sz${size}_st${steps}"
      echo "Submitting $jobname (nodes=$m, size=$size, steps=$steps)"
      qsub \
        -N "$jobname" \
        -pe mpich "$m" \            # request m MPI slots → m nodes if allocation_rule=1
        -o "logs/${jobname}.out" \
        -e "logs/${jobname}.err" \
        -cwd \
        ./"$hpc_file" "$program" "$size" "$steps" "$m"
    done
  done
done