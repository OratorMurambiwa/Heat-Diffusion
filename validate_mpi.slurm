#!/bin/bash
#SBATCH --job-name=validate_heat
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --constraint=cpu

set -euo pipefail

module load PrgEnv-gnu
module load cray-mpich

export OMP_NUM_THREADS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close

cd "$SLURM_SUBMIT_DIR"

simulator="$SLURM_SUBMIT_DIR/build/simulator"
results_dir="$SLURM_SUBMIT_DIR/mpi_validation_$SLURM_JOB_ID"

mkdir -p "$results_dir/serial" "$results_dir/parallel"

echo "Running one MPI process"
cd "$results_dir/serial"
srun --nodes=1 --ntasks=1 --cpus-per-task=2 "$simulator"

echo "Running four MPI processes"
cd "$results_dir/parallel"
srun --nodes=1 --ntasks=4 --cpus-per-task=2 "$simulator"

echo "Both runs completed."
echo "Results: $results_dir"
