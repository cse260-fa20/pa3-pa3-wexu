#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=apf-strong-scale-48
#SBATCH --output="apf-strong-scale-48.out"
#SBATCH --partition="compute"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here

ibrun -np 240 ./apf -n 8000 -i 2000 -x 2 -y 120
ibrun -np 240 ./apf -n 8000 -i 2000 -x 4 -y 60
ibrun -np 240 ./apf -n 8000 -i 2000 -x 6 -y 40
ibrun -np 240 ./apf -n 8000 -i 2000 -x 8 -y 30
ibrun -np 240 ./apf -n 8000 -i 2000 -x 16 -y 15

