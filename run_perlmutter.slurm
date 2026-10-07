#!/bin/bash
#SBATCH --job-name=slm_thermal
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu

# Load the GNU compiler and Cray MPI environment
module load PrgEnv-gnu
module load cray-mpich

# Tell OpenMP how many cores to use per MPI task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting SLM Simulation"

# Enter the build directory and run the MPI application
cd build
srun ./simulator

echo "Job completed."
