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

sbatch --output="comet-192.out" --nodes=8 N2_comet-192.slurm
sbatch --output="comet-192-nocomm.out" --nodes=8 N2_comet-192-nocomm.slurm

sbatch --output="comet-240.out" --nodes=10 N2_comet-240.slurm
sbatch --output="comet-240-nocomm.out" --nodes=10 N2_comet-240-nocomm.slurm

sbatch --output="comet-384.out" --nodes=16 N2_comet-384.slurm
sbatch --output="comet-384-nocomm.out" --nodes=16 N2_comet-384-nocomm.slurm

sbatch --output="comet-480.out" --nodes=20 N2_comet-480.slurm
sbatch --output="comet-480-nocomm.out" --nodes=20 N2_comet-480-nocomm.slurm
