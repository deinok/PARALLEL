#!/bin/bash

# Compile the program
gcc -fopenmp -o heat_parallel heat_parallel.c -lm
gcc -fopenmp -o heat_serial heat_serial.c -lm