#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=fluid_sim_output.txt
#SBATCH --constraint=k20

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1

make > /dev/null 2>&1

# run app
srun --partition=cpar --exclusive perf stat -r 1 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim_cuda
