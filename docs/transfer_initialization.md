Phase 6A: one host upload plus a device copy
==========================================

This experiment changes only how the benchmark initializes its second GPU
buffer. The original naive FP64 kernel, `CudaSolver::diffuse`, numerical tests,
Phase 3 Slurm baseline, and Phase 4/5 profiling scripts are unchanged.

The comparison executable accepts `--gpu-init baseline|d2d|both`:

| CLI mode | CSV initialization_method | Each reset of the two GPU buffers |
|---|---|---|
| `baseline` (default) | `two_h2d` | Copy the initial host field into each device buffer: two H2D copies |
| `d2d` | `h2d_d2d` | Copy host field into the first device buffer once, then copy first to second using `cudaMemcpyDeviceToDevice` |
| `both` | One set of rows per method | Compare both methods in one invocation on the same GPU and device allocations |

Both copies cover the full grid, including all six fixed boundary planes. The
D2D copy reads the freshly uploaded first buffer **before** any stencil update.
Default-stream ordering preserves this dependency and both buffers exchange
roles normally. Every CUDA copy is error-checked. No memset, boundary-reset
kernel, reduced precision, pinned memory, or multi-GPU functionality is added.

"One host upload" means once **per buffer reset**, not once across an entire
multi-repeat benchmark: warmup and measurement resets still occur independently.

Equivalent measurement conditions
---------------------------------

The default sweep remains 64³, 128³, 256³ in FP64, with 100 measured timesteps,
10 warmups per measurement mode, and 9 repeats. In `both` mode:

- CPU reference and CPU timings are collected once per grid on the same node.
- GPU methods share one device, one pair of allocations, one output buffer,
  identical dimensions, coefficients, launch configuration, and thread count.
- Each method performs the same warmup and resets the entire initial field
  before each measured kernel batch and each transfer-inclusive batch.
- Baseline runs first on even-numbered repeats (starting at zero), experimental
  first on odd repeats. This reduces systematic ordering bias without claiming
  to remove clock, thermal, host-scheduling, or shared-node variation.
- CUDA-event kernel timing excludes initialization and all copies. A device
  synchronization before the start event ensures the D2D copy is excluded too.
- Transfer-inclusive `steady_clock` timing includes the selected initialization,
  timestep batch, synchronization, and result download. Allocation, warmup,
  checking, and output are outside the interval for both methods.

Every repeat's full GPU result is compared against the CPU reference separately
for both kernel and transfer-inclusive runs, using the existing tolerance
`1e-9 K + 1e-12 * abs(CPU temperature)`. Values must be finite and fixed boundaries
must match the initial field exactly. CPU results retain analytical sine-mode
validation. Errors stop the run before rows for that grid are saved.

The baseline still performs exactly two H2D copies per reset, and remains the
default for existing benchmark and profiling scripts. Its CSV columns retain
their meanings; Phase 6A appends new columns. No change to the stencil's compute
performance is assumed: kernel-only differences may reflect cache effects or
measurement variation rather than a faster kernel.

CSV results and comparisons
--------------------------

`both` produces five rows per grid: one CPU row, two GPU kernel rows, and two
GPU transfer-inclusive rows (15 rows across the default sweep). CPU-only mode
still writes only CPU rows, with `initialization_method=not_applicable`, regardless
of the selected GPU flag. Existing statistics and provenance are retained:
median/min/max/mean, population standard deviation, individual samples, update
counts/rates, CPU/GPU speedup, numerical errors, and hardware/build metadata.

New fields:

- `initialization_method`: `two_h2d`, `h2d_d2d`, or `not_applicable`.
- `variant_order`: records paired alternating order, single-variant, or CPU-only.
- `speedup_vs_two_h2d_median`: two-H2D median divided by this method's median for
  the **same timing type and grid**. Baseline GPU rows are 1. Experimental rows
  have this ratio only when a baseline was measured in the same invocation;
  CPU rows and D2D-only runs leave it blank. This is a ratio of medians, not a
  median of per-repeat speedups.
- `timed_h2d_bytes`, `timed_d2d_bytes`, `timed_d2h_bytes`: explicit copy payloads
  inside the measured interval. They are zero for kernel-only and CPU rows.
  They do not measure physical DRAM traffic or hidden pageable-memory staging.

For a field of B bytes, the baseline transfer-inclusive batch has H2D=2B,
D2D=0, D2H=B; the experiment has H2D=B, D2D=B, D2H=B. At 256³, B=128 MiB.
The experiment halves H2D payload and reduces total host/device payload from
3B to 2B, but still copies B within device memory. Whether it improves elapsed
time must be established by measurements; do not infer a speedup from byte
counts alone. The number of iterations amortizes initialization costs, so use
identical settings when comparing ratios.

Select `gpu_transfer_inclusive` rows to assess initialization cost reduction.
Compare `gpu_kernel_batch` rows to check that the unchanged computation remains
consistent. Keep separate `speedup_vs_cpu_median` and initialization-baseline
speedups; their denominators answer different questions. Inspect variability
and numerical errors before claiming an improvement. Do not mix WSL CPU
measurements with Perlmutter GPU times.

Perlmutter execution
--------------------

Rebuild using the [Phase 3 CUDA build instructions](cpu_cuda_benchmark.md), with
matching compiler/toolkit modules and `-DENABLE_CUDA=ON -DBUILD_TESTING=ON`.
Use the new, separate experiment job:

```sh
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/run_transfer_initialization_perlmutter.slurm
```

Optionally pass an absolute CUDA build directory as its first argument. The job
uses the existing 32-CPU/1-GPU allocation, 8 OpenMP threads and affinity settings,
runs CTest before benchmarking, and saves `initialization-comparison.csv` and
provenance under:

```text
$SCRATCH/heat-diffusion-transfer-init/<job-id>_<12-character-git-revision>/
```

The unmodified `benchmarks/run_perlmutter.slurm` continues to run the baseline
only. Within an existing allocation, the paired experiment command is:

```sh
export OMP_PLACES=cores OMP_PROC_BIND=close
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  build-cuda/cpu_cuda_benchmark --gpu-init both --threads 8 \
  --iterations 100 --warmup 10 --repeats 9 --csv "$SCRATCH/initialization-comparison.csv"
```

Omit `--size` to run all three grids; use `--size 17` and short counts for a GPU
smoke check. Run both odd and even iteration counts (for example 3 and 4) when
checking buffer swapping before the full sweep. Keep profiler-generated timings
out of this unprofiled baseline comparison.

Local checks without CUDA
-------------------------

```sh
cmake -S . -B /tmp/heat-phase6a-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/heat-phase6a-build --parallel 2
ctest --test-dir /tmp/heat-phase6a-build --output-on-failure
/tmp/heat-phase6a-build/cpu_cuda_benchmark --cpu-only --gpu-init both \
  --threads 2 --warmup 2 --iterations 3 --repeats 3 --csv /tmp/phase6a-cpu.csv
```

These check CPU behavior and CSV handling only. CUDA compilation, D2D execution,
GPU numerical checks, and actual initialization speedups require Perlmutter GPU
hardware (or another compatible CUDA environment). No GPU performance claim
should be made before that run completes.
