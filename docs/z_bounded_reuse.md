Phase 6C: bounded Z-axis register reuse
======================================

The full-column Phase 6B experiment was reported slower than naive on A100.
This phase adds an experimental middle ground: 2, 4, or 8 consecutive Z cells
per thread. The naive kernel, full-column kernel, Phase 6A H2D/D2D choices,
existing validation tests, and previous Slurm scripts are preserved. No new
performance improvement is assumed before measurement.

Kernel mapping and boundaries
-----------------------------

`src/cuda_z_bounded_kernel.cuh` defines three compile-time specializations of
`diffusion_z_bounded<K>`, with K=2,4,8. Each thread owns one X-Y column within one
Z chunk. For `columns=(nx-2)*(ny-2)` and flattened thread index `item`:

```text
column  = item % columns
x       = 1 + column % (nx-2)
y       = 1 + column / (nx-2)
first_z = 1 + (item / columns)*K
```

The launch has `columns*ceil((nz-2)/K)` work items, rounded up to 256-thread
blocks. X/Y columns vary before the Z chunk, so adjacent lanes access consecutive
interior X coordinates within a row. Row and chunk boundaries can interrupt
contiguity; perfect coalescing or alignment is not assumed.

Each thread loads its lower neighbor and first center, then carries a rolling
`below, center, above` FP64 window through at most K cells. The X/Y neighbor loads
and stencil expression are unchanged. Values are read only from the current
time-level buffer and written only to the next buffer. Threads/chunks have
disjoint output cells; chunk boundaries are not physical boundaries and do not
require inter-thread synchronization. Each chunk reads its neighboring values
from the same current buffer, including across chunk boundaries.

The loop additionally requires `z < nz-1`, handling a final short chunk without
reading beyond the upper ghost/boundary plane. Outer X/Y/Z planes are never
written. Both initialization methods still initialize the complete buffers,
including nonuniform fixed boundary values. Kernel launches and buffer swaps
remain ordered in the default stream between timesteps.

This trades less reuse than a full column for more parallel work. For N interior
Z cells, source-level center/Z-neighbor loads are approximately
`N + 2*ceil(N/K)` per column versus `3N` for naive and `N+2` for full-column reuse.
That excludes four X/Y loads and one store per update. These are source-level
counts, not measured machine instructions or DRAM traffic. Startup work,
indexing, cache behavior, register allocation, and scheduling may offset reuse.
No shared-memory tiling, block tuning, fast math, reduced precision, or multi-GPU
execution is added.

Selections and preserved defaults
---------------------------------

| `--gpu-kernel` | Kernels selected |
|---|---|
| `naive` (default) | Original naive kernel |
| `z-register` | Original full-column experiment |
| `both` | Naive and full-column, preserving Phase 6B behavior |
| `z2`, `z4`, `z8` | One bounded specialization |
| `bounded` | Naive, Z2, Z4, Z8 |
| `all` | Naive, full-column, Z2, Z4, Z8 |

Combine any selection with the existing `--gpu-init baseline|d2d|both`. The new
Slurm job uses `--gpu-kernel all --gpu-init both`, collecting all ten combinations
on the same GPU with the same device allocations and CPU reference.

Benchmark protocol and CSV
--------------------------

The production sweep uses **64³, 128³, 256³**, **100 timed iterations**, **9
repeats**, and 10 warmups per timing mode. All variants use FP64, 256-thread
blocks, identical initial data, coefficients, physical boundaries, and CPU
reference. GPU method order remains initialization-major then kernel-major,
reversed on odd repeats. Every method warms up, then fully resets to the common
initial field before each timed batch. Reversal reduces systematic order bias
but does not eliminate drift or shared-node variation.

Measurement scopes remain unchanged:

- `gpu_kernel_batch`: CUDA events bracket the dependent timestep launches;
  initialization, transfers, allocation, warmup, and checking are excluded.
- `gpu_transfer_inclusive`: host `steady_clock` includes the selected full-buffer
  initialization, timestep batch, synchronization, and one result download;
  allocation and correctness checking are excluded.

Full-grid CPU/GPU comparison and exact fixed-boundary checks run after every
measurement repeat. The tolerance is still `1e-9 K + 1e-12*abs(CPU temperature)`;
nonfinite values or mismatches abort before saving that grid's rows.

The CSV schema is retained. New values of `kernel_variant` are `z_bounded_2`,
`z_bounded_4`, and `z_bounded_8`. For these kernels `z_cells_per_thread` records
the configured maximum (2,4,8), not the size of a final partial chunk. Naive
records 1; the full-column experiment records `nz-2`.

Each row retains individual samples, median/min/max/mean, population standard
deviation, updates/s, CPU/GPU speedup, and maximum numerical error. The update
count is still `(nx-2)*(ny-2)*(nz-2)*iterations`; it is not the number of threads.
`speedup_vs_naive_median` matches **grid, initialization, and timing type**.
`time_reduction_vs_naive_percent` is positive for reduced time and negative for
regression. `speedup_vs_two_h2d_median` matches the same kernel and timing type,
preserving Phase 6A comparisons. Missing references yield blank ratios.

With both initializers and all five kernels, expect 21 rows per grid (one CPU
and twenty GPU timing rows), or 63 rows for the sweep. Inspect GPU model/PCI ID,
source/build/module provenance, nine raw samples, variation, and numerical errors
before reporting improvements or regressions. Use only same-initializer naive
comparisons for kernel conclusions; do not mix WSL timings or profiler timing
with the unprofiled A100 comparison. CUDA-event batches can include host launch
gaps, so small differences need careful interpretation.

Correctness tests
-----------------

The new CUDA-only CTest `cuda_bounded_validation` supplements, without changing,
the existing CPU, naive CUDA, and full-column validation tests. It covers 14
shapes including 3³, 17³, noncubes, thin X/Y grids, and Z interior lengths 1
through 9. Those lengths exercise every remainder for K=2,4,8, exact multiples,
single-cell tails, and chunks larger than the interior. Cases include partial
thread blocks and a 64x33x18 shape.

Uniform, sine-shaped, and irregular fields with nonuniform fixed boundaries
are checked after 1,2,7,100 steps. Both initializers and naive/Z2/Z4/Z8 are tested
in both timing paths with two repeats, for 5,376 full-field comparisons across
168 shape/field/step cases. Production measurements validate all benchmark sizes
too. Existing `cuda_register_validation` continues to cover the full-column path.

Perlmutter build and run
-----------------------

Rebuild the existing [CUDA-enabled configuration](cpu_cuda_benchmark.md) with
`BUILD_TESTING=ON`, a compatible GNU/CUDA module pair, and architecture 80 for
Perlmutter A100. Then submit:

```sh
cmake --build build-cuda --parallel 4
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/run_z_bounded_perlmutter.slurm
```

Pass an absolute build directory as the script's first argument if necessary.
The job uses 32 Slurm CPUs, one GPU, 8 OpenMP threads, and the same affinity as
the earlier experiments. CTest runs first; any failure prevents measurement.
CSV and provenance are saved under:

```text
$SCRATCH/heat-diffusion-z-bounded/<job-id>_<12-character-git-revision>/
  bounded-kernel-comparison.csv
```

Within an existing allocation:

```sh
export OMP_PLACES=cores OMP_PROC_BIND=close
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  build-cuda/cpu_cuda_benchmark --gpu-kernel all --gpu-init both --threads 8 \
  --iterations 100 --warmup 10 --repeats 9 --csv "$SCRATCH/z-bounded.csv"
```

For independent GPU memory checking:

```sh
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  compute-sanitizer --tool memcheck --error-exitcode 1 build-cuda/cuda_bounded_validation
```

Previous benchmark/profiler scripts retain their default kernel choices and
launch filters. Do not reuse a naive-kernel filter to draw conclusions about
all five kernels. No A100 improvement is claimed until measured.

Local checks and limitations
---------------------------

CPU builds and CTest work without a toolkit. A CPU-only smoke command is:

```sh
build/cpu_cuda_benchmark --cpu-only --gpu-kernel all --gpu-init both \
  --threads 2 --warmup 2 --iterations 3 --repeats 3 --csv /tmp/phase6c-cpu.csv
```

Host execution of the actual kernel bodies with emulated CUDA indices can check
arithmetic and out-of-bounds cases under sanitizers. It is not GPU execution or
evidence about register allocation, coalescing, occupancy, or performance.
Actual NVCC compilation, GPU validation and measured timings require Perlmutter
or another compatible CUDA environment.
