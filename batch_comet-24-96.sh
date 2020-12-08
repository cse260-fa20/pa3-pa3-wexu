#!/bin/bash

#SBATCH -A csd562
#SBATCH --job-name="pa3"
#SBATCH --output="pa3.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 00:05:00

#This job runs with 2  nodes, 24 cores per node for a total of 48 cores.

sbatch --output="comet-24.out" N1_comet-24.slurm
sbatch --output="comet-24-nocomm.out" N1_comet-24-nocomm.slurm

sbatch --output="comet-48.out" --nodes=2 N1_comet-48.slurm
sbatch --output="comet-48-nocomm.out" --nodes=2 N1_comet-48-nocomm.slurm

sbatch --output="comet-96.out" --nodes=4 N1_comet-96.slurm
sbatch --output="comet-96-nocomm.out" --nodes=4 N1_comet-96-nocomm.slurm
