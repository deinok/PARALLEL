#!/bin/bash
## Specifies the interpreting shell for the job.
#$ -S /bin/bash

## Specifies that all environment variables active within the qsub utility be exported to the context of the job.
#$ -V

## Execute the job from the current working directory.
#$ -cwd

## Parallel programming environment (mpich) to instantiate and number of computing slots.
#$ -pe mpich 4

#$ -j y


if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <program> <size> <steps>"
  exit 1
fi
program=$1
size=$2
steps=$3

MPICH_MACHINES=$TMPDIR/mpich_machines
cat $PE_HOSTFILE | awk '{print $1":"$2}' > $MPICH_MACHINES


output_file="${program}_th${NSLOTS}_sz${size}_st${steps}.bmp"


## In this line you have to write the command that will execute your application.
mpiexec -f $MPICH_MACHINES -n $NSLOTS ./"$program" "$size" "$steps" "$output_file"

rm -rf $MPICH_MACHINES


