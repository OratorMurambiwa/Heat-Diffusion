#!/bin/bash
#SBATCH --job-name=heat_transfer_init
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=heat-transfer-init-%j.out

set -euo pipefail

# Submit with --account=<your GPU account>. Use the same module versions as build.
module load PrgEnv-gnu
module load cudatoolkit
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SLURM_CPU_BIND=cores

cd "${SLURM_SUBMIT_DIR:?}"
build_dir="${1:-$SLURM_SUBMIT_DIR/build-cuda}"
revision="$(git rev-parse HEAD)"
results_dir="${SCRATCH:?}/heat-diffusion-transfer-init/${SLURM_JOB_ID:?}_${revision:0:12}"
mkdir -p "$results_dir"
test -x "$build_dir/cpu_cuda_benchmark"

# Keep provenance with the results; do not dump the full environment.
module -t list > "$results_dir/modules.txt" 2>&1
printf '%s\n' "$revision" > "$results_dir/revision.txt"
git status --short > "$results_dir/worktree-status.txt"
cp "$build_dir/CMakeCache.txt" "$results_dir/CMakeCache.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    nvidia-smi > "$results_dir/nvidia-smi.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    lscpu > "$results_dir/lscpu.txt"

# A validation failure prevents benchmark execution.
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    ctest --test-dir "$build_dir" --output-on-failure -V
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    "$build_dir/cpu_cuda_benchmark" --threads "$OMP_NUM_THREADS" \
    --iterations 100 --warmup 10 --repeats 9 --gpu-init both \
    --csv "$results_dir/initialization-comparison.csv"
echo "Results: $results_dir"
