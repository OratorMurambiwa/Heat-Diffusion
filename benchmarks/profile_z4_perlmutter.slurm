#!/bin/bash
#SBATCH --job-name=heat_ncu_z4
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --signal=TERM@60
#SBATCH --output=heat-ncu-z4-%j.out

set -euo pipefail

# Submit with --account=<your GPU account>; match build module versions.
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
command -v ncu
revision="$(git rev-parse HEAD)"
run_id="${SLURM_JOB_ID:?}_${revision:0:12}"
results_dir="${SCRATCH:?}/heat-diffusion-ncu-z4/$run_id"
report_prefix="$results_dir/heat_256_fp64_z4_$run_id"
mkdir -p "$results_dir"
printf '%s\n' "$revision" > "$results_dir/revision.txt"
git status --short > "$results_dir/worktree-status.txt"
module -t list > "$results_dir/modules.txt" 2>&1
ncu --version > "$results_dir/ncu-version.txt" 2>&1
ncu --list-sections > "$results_dir/ncu-sections.txt" 2>&1
cp "$build_dir/CMakeCache.txt" "$results_dir/CMakeCache.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    nvidia-smi > "$results_dir/nvidia-smi.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    lscpu > "$results_dir/lscpu.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    ctest --test-dir "$build_dir" --output-on-failure -V \
    | tee "$results_dir/validation.log"

# Keep pause, profiling, and restoration on the same allocated compute node.
# The node-local EXIT trap runs on profiler errors and handled signals, too.
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    bash -s -- "$build_dir" "$results_dir" "$report_prefix" <<'PROFILE_NODE'
set -euo pipefail
build_dir="$1"
results_dir="$2"
report_prefix="$3"
command -v dcgmi
command -v ncu
resume_needed=0
cleanup() {
    local status=$?
    trap - EXIT
    trap '' HUP INT TERM
    if (( resume_needed )); then
        local restored=0
        for attempt in 1 2 3; do
            printf 'Resume attempt %s\n' "$attempt" >> "$results_dir/dcgm-resume.log"
            if dcgmi profile --resume >> "$results_dir/dcgm-resume.log" 2>&1; then
                restored=1
                break
            fi
        done
        if (( ! restored )); then
            echo "ERROR: DCGM resume failed; see $results_dir/dcgm-resume.log and restore it on this node." >&2
            if (( status == 0 )); then status=1; fi
        fi
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Arm cleanup before pause: even a partially successful pause must be restored.
resume_needed=1
dcgmi profile --pause > "$results_dir/dcgm-pause.log" 2>&1
cd "$results_dir"
ncu --kernel-name-base function --kernel-name regex:diffusion \
    --launch-skip 15 --launch-count 1 --kill no --check-exit-code yes \
    --replay-mode kernel --cache-control all --clock-control none \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section Occupancy --section ComputeWorkloadAnalysis \
    --section LaunchStats --section SchedulerStats \
    --export "$report_prefix" \
    "$build_dir/cpu_cuda_benchmark" --size 256 --threads "$OMP_NUM_THREADS" \
    --gpu-kernel z4 --iterations 20 --warmup 5 --repeats 3 --csv "$results_dir/profile-run.csv" \
    </dev/null | tee "$results_dir/profile.log"
test -s "$report_prefix.ncu-rep"
PROFILE_NODE

# DCGM has been restored before offline report export begins.
cd "$results_dir"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    ncu --import "$report_prefix.ncu-rep" --page details --print-details all \
    > "$results_dir/sections.txt"
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    ncu --import "$report_prefix.ncu-rep" --page raw --csv \
    > "$results_dir/metrics.csv"
echo "Profile: $report_prefix.ncu-rep"
echo "Results: $results_dir"
