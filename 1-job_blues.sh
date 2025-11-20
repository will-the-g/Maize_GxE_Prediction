#!/bin/bash

## configs
#module purge
#module load gcc/9.3.1 mkl/19.0.5 intel/19.0.5 R/4.2.2

set -e

#mkdir -p output logs

## run
Rscript src/blues.R #| tee logs/blues.txt
