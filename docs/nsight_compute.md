Phase 5: one FP64 stencil launch with Nsight Compute
===================================================

This job prepares hardware-counter collection for the existing naive kernel;
it does not change the stencil, benchmark, or tests. Actual bottleneck findings
must come from the resulting Perlmutter measurements, not from the source alone.

Build and submit
----------------

Use the existing [Phase 3 CUDA build](cpu_cuda_benchmark.md) with compatible,
matching GNU/CUDA module versions and `CMAKE_CUDA_ARCHITECTURES=80`. Then submit
from the repository root:

```sh
module load PrgEnv-gnu
module load cudatoolkit
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/profile_compute_perlmutter.slurm
```

A different existing build directory can be passed as the script's first
argument. The job requests one node, one GPU, and 32 Slurm CPUs, including every
`srun`; the application uses 8 OpenMP threads with the Phase 3 affinity settings.
CTest runs before counter collection. Do not run GPU collection on login nodes.

The full report, `sections.txt`, raw `metrics.csv`, profiled benchmark CSV, logs,
full Git revision, worktree status, module/tool versions, CMake cache, and device
information are saved under:

```text
$SCRATCH/heat-diffusion-ncu/<job-id>_<12-character-git-revision>/
  heat_256_fp64_<job-id>_<revision>.ncu-rep
```

The report prefix is unique to the job/revision; existing reports are not forced
overwritten. Preserve the source snapshot for dirty worktrees: revision and
status alone do not reconstruct uncommitted changes. Slurm stdout is
`heat-ncu-<job-id>.out` in the submission directory.

DCGM restoration
----------------

NERSC requires pausing DCGM profiling before using hardware counters, followed
by resumption. A single `srun` launches a node-local Bash wrapper that:

1. Installs an EXIT cleanup trap and HUP/INT/TERM handlers.
2. Arms restoration **before** attempting `dcgmi profile --pause`.
3. Starts `ncu` only if pause succeeds.
4. Attempts `dcgmi profile --resume` on normal completion, pause/profiler failure,
   missing report, or handled termination; retries resume up to three times.
5. Preserves the failure exit code, or fails the job if resumption itself fails.

Pause/resume output is recorded in `dcgm-pause.log` and `dcgm-resume.log`. Report
export happens only after restoration. `--signal=TERM@60` requests advance
termination notification for job steps before the time limit, giving cleanup a
chance to run while the allocation still exists. No shell trap can guarantee
cleanup after SIGKILL, node loss, or an unresponsive DCGM service. If the resume
log reports failure, restore DCGM on the affected node using the site's supported
procedure, e.g. `srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 dcgmi profile
--resume` while the allocation is active; otherwise contact NERSC with the node,
job ID, and logs. Do not bypass a failed pause and collect counters anyway.

Selected launch and metrics
---------------------------

The executable runs `--size 256 --iterations 20 --warmup 5 --repeats 3`. Only
matching `diffusion` launches count toward the filter:

```sh
ncu --kernel-name-base function --kernel-name regex:diffusion \
  --launch-skip 15 --launch-count 1 --kill no --check-exit-code yes \
  --replay-mode kernel --cache-control all --clock-control none \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section ComputeWorkloadAnalysis \
  --section LaunchStats --section SchedulerStats \
  --export "$report_prefix" \
  "$build_dir/cpu_cuda_benchmark" --size 256 --threads 8 \
  --iterations 20 --warmup 5 --repeats 3 --csv "$results_dir/profile-run.csv"
```

This command is run **inside the protected pause/resume wrapper**, not directly
as a replacement for the Slurm job. Five matching launches are warmups; skipping
15 selects matching launch 16, the 11th step of the first timed GPU batch.
The timed batch is reset after warmup, so the selected kernel reads the field
after 10 timed steps. This is a mid-batch invocation rather than first-launch
startup. If the benchmark launch order changes, revisit this filter. Use the
Phase 4 timeline to cross-check the launch order and inspect the report to
confirm one logical launch was captured.

`--launch-count 1` limits the capture to one application launch, **not one physical
execution**: kernel replay may execute it several times to collect incompatible
counters. Only the six named sections are requested, rather than `--set full`.
The rest of the application runs unprofiled so CPU/GPU correctness checks still
execute; `--kill no` preserves that behavior and exit codes are checked.

`--cache-control all` flushes caches before replay passes for reproducible
counter collection. Cache state therefore differs from a naturally warm Phase 3
run, despite selecting a mid-batch launch. `--clock-control none` avoids requesting
clock changes; clocks can still vary naturally. Replay, serialization, cache
flushing, and instrumentation mean profiled durations are not baseline timings.

| Section | What to examine |
|---|---|
| SpeedOfLight | Kernel duration, SM/compute utilization, memory utilization relative to peak |
| MemoryWorkloadAnalysis | DRAM read/write traffic and throughput, L1/TEX and L2 hit behavior, cache/link or memory-pipeline pressure |
| Occupancy | Theoretical versus achieved active warps; register, block, and other residency limits |
| ComputeWorkloadAnalysis | Executed compute-pipeline utilization, including FP64 versus other instruction work |
| LaunchStats | Grid/block dimensions, registers per thread, shared memory, waves per SM |
| SchedulerStats | Active/eligible/issued warps and issue activity to help interpret latency hiding |

Section details and metric identifiers depend on the installed Nsight version
and GPU. Check the saved `ncu-sections.txt`, `ncu --list-sections`, and
`ncu --query-metrics` on the allocated node. Missing sections or unavailable
counters should be investigated, not silently discarded; cleanup still runs if
collection fails. Use reported units and denominators: active-cycle percentages
are not interchangeable with elapsed-cycle percentages.

Read the report and compare with Phase 3
--------------------------------------

The job exports human-readable sections and raw metrics after resuming DCGM:

```sh
ncu --import "$report_prefix.ncu-rep" --page details --print-details all
ncu --import "$report_prefix.ncu-rep" --page raw --csv > metrics.csv
```

The CSV includes metric names, units, and values, with base units by default.
Copy `.ncu-rep` to a workstation and open it with a compatible `ncu-ui`. Keep the
tool versions recorded with the report.

Interpret evidence together rather than declaring a bottleneck from one number:

- High DRAM throughput relative to sustained peak, together with memory pressure
  and lower compute utilization, supports a bandwidth-limit hypothesis. Check
  L1/L2 behavior and whether the busy resource is DRAM, a cache/link, or memory
  instruction issue. High aggregate "memory" utilization alone does not prove
  DRAM saturation. Cache hit rates alone do not establish good or bad efficiency.
- High compute-pipeline utilization can support a compute/issue-limit hypothesis.
  This naive index mapping also executes integer division/index arithmetic;
  overall SM utilization is not synonymous with FP64 arithmetic saturation.
- A gap between theoretical and achieved occupancy, or few eligible warps, can
  indicate limited latency hiding. Check residency limits and issue activity.
  Higher occupancy is not automatically faster and low occupancy alone does not
  identify the limiting resource.
- Low utilization of both memory and compute calls for checking scheduler data,
  launch configuration, and the Phase 4 timeline. If evidence is inconclusive,
  record it as inconclusive; this phase does not apply an optimization.

For the selected launch, updated cells are `254^3 = 16,387,064`. Divide by its
reported duration in seconds for a **profiled single-launch** updates/s estimate.
If read/write byte counters are available, divide their sum by duration for
measured kernel DRAM traffic rate, and by cell count for bytes/update. Prefer
the tool's reported DRAM throughput and matching time denominator. These are
hardware-counter results, unlike the earlier nominal 64-byte or idealized
16-byte traffic models; neither model predicts exact physical traffic.

Compare against an **unprofiled** Phase 3 run on Perlmutter with the same source,
build flags, GPU model, CPU thread count, and affinity. Filter `comparison.csv`
to `nx=ny=nz=256`:

- `gpu_kernel_batch` median / iterations gives an average per-step batch time,
  including launch gaps; it need not equal one replayed kernel's duration.
- `gpu_transfer_inclusive` includes transfers and host synchronization, which
  Nsight Compute's kernel metrics exclude. Use Phase 4 for those costs.
- Keep the Phase 3 same-machine CPU/GPU speedups as the performance baseline.
  Do not derive production speedups from the profiler's `profile-run.csv`, use
  WSL CPU timings as a denominator, or treat one launch as a variability study.

Suggested findings record: duration, DRAM bytes/s and peak %, L1/L2 hit rates,
SM/FP64 utilization, theoretical/achieved occupancy, register/residency limits,
eligible warps, and the supported limiting-resource hypothesis with caveats.
No hardware findings are claimed until the GPU collection has actually run.

References: [NERSC DCGM and Nsight Compute procedure](https://docs.nersc.gov/tools/performance/nvidiaproftools/),
[Nsight Compute CLI filters and export](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[Replay, caches, and section definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
