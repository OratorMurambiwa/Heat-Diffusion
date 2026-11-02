Phase 4: FP64 stencil timeline with Nsight Systems
================================================

This phase profiles the existing naive CUDA stencil without changing numerical
code, validation, or benchmark measurement logic. The separate
`benchmarks/profile_perlmutter.slurm` job runs the existing comparison executable
at 256³, with 8 OpenMP threads, 5 warmups, 20 measured timesteps, and 3 repeats.
Both the benchmark and profiling scripts request 32 Slurm CPUs per GPU in their
allocation and every `srun`. This is the scheduler CPU request; physical-core
mapping depends on Slurm/SMT settings. The application still uses 8 OpenMP threads.

Build and submit
----------------

Use the existing [CUDA benchmark build instructions](cpu_cuda_benchmark.md).
No special profiling build flags or kernel changes are required. Use a matching,
compatible GNU/CUDA module pair for building and running, and preserve those
module versions with your results. `cudatoolkit` provides the Nsight tooling on
Perlmutter; check `command -v nsys` and `nsys --version` after loading it.

From the repository root on Perlmutter:

```sh
module load PrgEnv-gnu
module load cudatoolkit
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/profile_perlmutter.slurm
```

To use a different existing CUDA build directory:

```sh
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/profile_perlmutter.slurm /absolute/path/to/build-cuda
```

The script runs the existing CTest suite outside the profiler first. A failing
test prevents profiling. The profiled benchmark also validates every CPU and
GPU result, including both GPU timing modes and exact boundary preservation.
The short configuration reduces trace size; use the separate, unprofiled Phase 3
job for performance baselines and speedup claims.

Capture command
---------------

The script executes the following inside a one-GPU allocation, with absolute
`build_dir` and `report_prefix` paths on scratch:

```sh
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=8
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none \
  --force-overwrite=false --output="$report_prefix" \
  "$build_dir/cpu_cuda_benchmark" --size 256 --threads 8 \
  --iterations 20 --warmup 5 --repeats 3 --csv "$results_dir/profile-run.csv"
```

`--trace=cuda` captures device kernel execution, memory copies, and host CUDA API
calls. `osrt` adds supported OS-runtime calls. CPU statistical sampling and
context-switch tracing are disabled to reduce overhead and avoid requiring CPU
sampling privileges. This does not collect CPU hotspots, GPU hardware counters,
achieved DRAM bandwidth, or kernel occupancy. CUDA tracing does not require the
DCGM pause procedure used for conflicting hardware-counter collections.

The whole benchmark process is traced: CPU work, initialization, allocation,
warmups, checking, and both GPU measurement modes remain present in the timeline.
No capture-range controls or NVTX markers were added to the application. Do not
interpret the whole report as only the 20 measured kernel calls.

Saved outputs
-------------

The script creates:

```text
$SCRATCH/heat-diffusion-profiles/<job-id>_<12-character-git-revision>/
  heat_256_fp64_<job-id>_<revision>.nsys-rep
  nsys-stats.txt
  profile-run.csv
  validation.log
  profile.log
  revision.txt
  worktree-status.txt
  modules.txt
  nsys-version.txt
  CMakeCache.txt
  nvidia-smi.txt
  lscpu.txt
```

Nsight may also create SQLite/intermediate files in this directory. Reports
are not overwritten. The full Git revision is saved separately; for uncommitted
changes, retain the actual source snapshot too, since revision/status alone do
not reconstruct a dirty worktree. The normal Slurm output remains
`heat-nsys-<job-id>.out` in the submission directory. This script does not submit
additional jobs or modify Git state.

View and summarize
------------------

The script produces a text summary with:

```sh
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "$report_prefix.nsys-rep"
```

Run that command from the report directory to keep generated analysis files on
scratch. To inspect individual GPU operations:

```sh
nsys stats --report cuda_gpu_trace "$report_prefix.nsys-rep"
```

Report names can vary with older Nsight releases; use `nsys stats --help-reports`
to list those supported by the loaded version. A statistics-export error does
not remove the original `.nsys-rep`; re-run analysis with a compatible version.
Copy the `.nsys-rep` file to a workstation with a compatible Nsight Systems GUI
and open it with `nsys-ui /path/to/report.nsys-rep`.

Interpret the results
---------------------

1. **Kernel execution:** inspect `diffusion` on the GPU timeline and
   `cuda_gpu_kern_sum` for count, durations, and variation. The expected application
   kernel count is `3 * 2 * (5 + 20) = 150`: each repeat contains a warmup and timed
   batch for each of the two GPU modes. Warmup kernels are included in summaries.
   Within each repeat, the launch groups are 5, 20, 5, 20. CUDA-event API calls
   surround the first measured 20-step group; the transfer-inclusive group uses
   host timing without event instrumentation.
2. **CUDA API overhead and waiting:** `cuda_api_sum` and the CPU CUDA API track
   show launch, copy, event, allocation, and synchronization calls. Inspect
   `cudaLaunchKernel` duration and correlated GPU start times separately. Long
   synchronization calls can be waiting for GPU work, not pure runtime overhead.
   Do not add API waiting time to GPU execution time as independent work.
3. **Transfers:** inspect HtoD/DtoH operations and the memory time/size summaries.
   Each 256³ FP64 field is 128 MiB. Each GPU mode uploads both buffers for warmup
   and again before timing, then downloads one result. Across 3 repeats and both
   modes, expect 24 application HtoD copies and 6 DtoH copies. The kernel-only
   event interval excludes uploads/downloads; the transfer-inclusive interval
   includes two uploads and one download. Transfer throughput is not stencil
   DRAM bandwidth; pageable host staging can add API-side time.
4. **Gaps and overlap:** align host API calls, GPU kernels, and copies. Long idle
   regions can contain CPU reference work, initialization, or correctness checks.
   With CPU sampling disabled and no NVTX ranges, the trace does not directly
   identify those CPU source regions. Default-stream operations serialize here;
   report observed gaps without assuming a bandwidth or compute bottleneck.
5. **Scope and percentages:** summary percentages are relative to the operations
   in that report, not necessarily total application wall time. CUDA-event batch
   time can include launch gaps; it need not equal the sum of per-kernel durations.
   Profiling perturbs timings, so keep `profile-run.csv` separate from unprofiled
   Phase 3 results. Three short repeats support timeline inspection, not a new
   stable performance baseline.

Local validation can check script syntax and resource requests. Actual CUDA
capture, GPU validation, report generation, and timeline interpretation require
a Perlmutter GPU allocation; no local GPU or `nsys` is currently available.

References: [NERSC NVIDIA profiling tools](https://docs.nersc.gov/tools/performance/nvidiaproftools/),
[Nsight Systems CLI options](https://docs.nvidia.com/nsight-systems/UserGuide/),
[Nsight report definitions](https://docs.nvidia.com/nsight-systems/AnalysisGuide/).
