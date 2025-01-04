#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=nvprof_output.txt
#SBATCH --constraint=k20

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1

make > /dev/null 2>&1

# run app
nvprof ./fluid_sim_cuda
