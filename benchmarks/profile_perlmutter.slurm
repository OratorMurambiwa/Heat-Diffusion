#!/bin/bash
#SBATCH --job-name=heat_nsys
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=heat-nsys-%j.out

set -euo pipefail

# Submit with --account=<your GPU account>; match the build's module versions.
module load PrgEnv-gnu
module load cudatoolkit
export OMP_NUM_THREADS=8
export OMP_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export SLURM_CPU_BIND=cores

cd "${SLURM_SUBMIT_DIR:?}"
build_dir="$(realpath "${1:-$SLURM_SUBMIT_DIR/build-cuda}")"
test -x "$build_dir/cpu_cuda_benchmark"
test -f "$build_dir/CMakeCache.txt"
command -v nsys
revision="$(git rev-parse HEAD)"
revision_short="${revision:0:12}"
run_id="${SLURM_JOB_ID:?}_${revision_short}"
results_dir="${SCRATCH:?}/heat-diffusion-profiles/$run_id"
report_prefix="$results_dir/heat_256_fp64_$run_id"
mkdir -p "$results_dir"

printf '%s\n' "$revision" > "$results_dir/revision.txt"
git status --short > "$results_dir/worktree-status.txt"
module -t list > "$results_dir/modules.txt" 2>&1
nsys --version > "$results_dir/nsys-version.txt" 2>&1
cp "$build_dir/CMakeCache.txt" "$results_dir/CMakeCache.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    nvidia-smi > "$results_dir/nvidia-smi.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    lscpu > "$results_dir/lscpu.txt"

# Tests are outside the trace; failure stops the job before profiling.
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    ctest --test-dir "$build_dir" --output-on-failure -V \
    | tee "$results_dir/validation.log"

# Keep Nsight-generated side files on scratch as well as the primary report.
cd "$results_dir"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
    --force-overwrite=false --output="$report_prefix" \
    "$build_dir/cpu_cuda_benchmark" --size 256 --threads "$OMP_NUM_THREADS" \
    --iterations 20 --warmup 5 --repeats 3 --csv "$results_dir/profile-run.csv" \
    | tee "$results_dir/profile.log"

srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    "$report_prefix.nsys-rep" > "$results_dir/nsys-stats.txt"
echo "Profile: $report_prefix.nsys-rep"
echo "Results: $results_dir"
