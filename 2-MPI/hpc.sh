#!/bin/bash
## Specifies the interpreting shell for the job.
#$ -S /bin/bash

## Specifies that all environment variables active within the qsub utility be exported to the context of the job.
#$ -V

## Execute the job from the current working directory.
#$ -cwd 

## The  name  of  the  job.
#$ -N OMP


if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <program> <size> <steps>"
  exit 1
fi

program=$1
size=$2
steps=$3

output_file="${program}_${size}_${steps}.bmp"

"./$program" "$size" "$steps" "$output_file"
