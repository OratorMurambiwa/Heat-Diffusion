CPU versus CUDA benchmark on Perlmutter
======================================

`cpu_cuda_benchmark` defaults to 64³, 128³, and 256³ full allocated grids,
including boundary planes, in one process on one node. It uses FP64 and the
unchanged CPU stencil plus the same naive CUDA kernel used by GPU validation.
The existing kernel body has only been moved into a shared internal header.
No MPI communication, laser calculations, I/O, tiling, graphs, reduced precision,
or multi-GPU execution occurs in timed computation.

Measurement protocol
--------------------

The common initial field is ambient temperature plus a deterministic product
of three sine modes in the interior, with fixed ambient outer planes in both
buffers. Every CPU, GPU kernel, and GPU transfer-inclusive repeat warms its
execution path, then resets to this identical initial field before measuring
exactly `iterations` timesteps. Warmup steps are never counted as timed updates.
Default settings are 10 warmups, 100 timed iterations, and 9 repeats per mode.

Three CSV rows are produced per grid:

| Timing type | Clock | Included | Excluded |
|---|---|---|---|
| `cpu_batch` | `std::chrono::steady_clock` | Stencil calls, OpenMP overhead, buffer swaps, timestep loop | Allocation, initialization, warmup, validation, output |
| `gpu_kernel_batch` | CUDA events on default stream | Entire batch of dependent kernel launches on device timeline | Allocation, initial uploads, downloads, warmup, validation, output |
| `gpu_transfer_inclusive` | `std::chrono::steady_clock` | Two full H2D uploads, the timestep batch, execution synchronization, one full D2H download, host launch/swap overhead | Allocation, initialization, warmup, validation, output |

The GPU transfer-inclusive result is end-to-end for **preallocated buffers**,
not a measurement of allocating and destroying the solver. Ordinary pageable
`std::vector` host buffers and blocking copies are used. Both buffers are uploaded
to establish identical boundary values. No pinned-memory or transfer optimization
is introduced. Synchronization before event recording ensures uploads are not
counted as kernel time. The stop event is synchronized before its elapsed time
is read. The transfer-inclusive measurement runs separately without event
instrumentation and stops only after the result has returned to host memory.

CUDA-event batch time can include gaps between launches if the GPU catches up
with the host. It is not a sum of per-kernel profiler durations. CPU batch time
includes swaps, unlike the older `cpu_stencil_benchmark` per-call measurement;
use the CPU baseline from this same invocation for the reported speedups.
All cases run sequentially: CPU repeats followed by alternating GPU kernel and
transfer-inclusive repeats. This avoids concurrent CPU/GPU benchmark workloads
but does not control thermal drift, clocks, or other tenants on a shared node.
The serial reset first-touches host pages; NUMA placement is not optimized.

After every CPU repeat, the full result is checked against analytical discrete
sine-mode decay, with absolute tolerance `1e-9 K`. Each GPU repeat in both modes
is compared with the CPU result using `abs(GPU-CPU) <= 1e-9 + 1e-12*abs(CPU)`.
All values must be finite; boundaries must remain exactly equal to the initial
values. Checking is outside all measured intervals. Any failed check or CUDA
error aborts the run with a nonzero exit; results from incomplete grids are not
written, though earlier validated grids may remain in a partial CSV.

Statistics and provenance
-------------------------

Each row records median/min/max/mean batch seconds, population standard
deviation, and all samples in execution order (semicolon-separated). Timed cell
updates are `(nx-2)*(ny-2)*(nz-2)*iterations`, excluding boundary and warmup cells.
Updates/s divides this count by median batch time. Each GPU speedup is the CPU
median divided by the corresponding GPU median, from the **same executable
invocation on the same node**, not a comparison against the earlier WSL CSV.
The CPU row's speedup is 1. No measured DRAM bandwidth is claimed.

CSV metadata includes FP64, dimensions, observed/requested-equal OpenMP team
size, settings, hostname, Slurm job ID, CPU model, GPU model/PCI bus/compute
capability/memory, CUDA runtime and driver versions, compiler identities and
paths, configured C++/CUDA flags, CUDA host compiler, target architectures,
build type, and OpenMP affinity settings. CUDA versions from runtime APIs are
encoded integers. Build metadata covers the documented single-configuration
Release build; it is not a complete dump of wrapper-injected flags. The Slurm
script also saves module versions, CMake cache, source revision/worktree status,
GPU details, and CPU topology. Preserve the source snapshot for uncommitted work;
a revision plus status listing alone cannot reconstruct a dirty worktree.

Build on Perlmutter
-------------------

Use a fresh build directory, CMake >= 3.18, and a compatible GNU/CUDA toolkit
module pair. Pin module versions for runs intended to be directly compared.
CUDA remains opt-in and the existing CPU and numerical-validation targets remain.

```sh
module load PrgEnv-gnu
module load cudatoolkit
cmake --version
nvcc --version
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build-cuda --parallel 4
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/run_perlmutter.slurm
```

Architecture 80 targets Perlmutter A100 GPUs. The script requests one GPU and
32 Slurm CPUs per GPU, uses 8 OpenMP threads with `OMP_PLACES=cores` and
`OMP_PROC_BIND=close`, runs the existing CTest suite before benchmarking, and
places CSV/provenance under `$SCRATCH/heat-diffusion-benchmarks/$SLURM_JOB_ID/`.
The comparison program rejects an OpenMP team smaller than requested. A different
build directory can be passed as the script's first positional argument.
Do not run GPU measurements on login nodes. The script has not been executed
on Perlmutter from this local environment.

Within an existing GPU allocation, one can instead run:

```sh
export OMP_PLACES=cores OMP_PROC_BIND=close
srun -n 1 -c 32 --gpus-per-task=1 \
  build-cuda/cpu_cuda_benchmark --threads 8 \
  --iterations 100 --warmup 10 --repeats 9 --csv "$SCRATCH/comparison.csv"
```

Use `--size N` to override the default sweep for a smoke test and `--help` for
all options. CSV paths are overwritten; use distinct paths for distinct jobs.

Local CPU-only checks
---------------------

No GPU results or speedups are fabricated when CUDA is unavailable:

```sh
cmake -S . -B /tmp/heat-phase3-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/heat-phase3-build --parallel 2
ctest --test-dir /tmp/heat-phase3-build --output-on-failure
/tmp/heat-phase3-build/cpu_cuda_benchmark --cpu-only --size 17 \
  --threads 2 --warmup 3 --iterations 7 --repeats 3 --csv /tmp/comparison-smoke.csv
```

CPU-only mode writes only CPU rows, with GPU metadata blank. Without `--cpu-only`,
a CPU-only build exits with an actionable diagnostic. Actual CUDA compilation,
event timing, transfer timing, CPU/GPU comparisons, and speedups require the
Perlmutter GPU run (or another compatible CUDA machine).

References: [NERSC CUDA modules](https://docs.nersc.gov/development/programming-models/cuda/),
[Perlmutter GPU jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/),
[CUDA event timing](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).

For a separate short Nsight Systems capture, see [Phase 4 profiling](gpu_profiling.md).
