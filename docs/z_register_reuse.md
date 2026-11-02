Phase 6B: experimental Z-axis register reuse
===========================================

The original naive FP64 kernel, public `CudaSolver::diffuse` API, CPU solver,
Phase 6A initialization methods, and existing validation tests remain intact.
The new `src/cuda_z_register_kernel.cuh` is a separate benchmark kernel named
`diffusion_z_register`. The default remains the naive kernel and two-H2D
initialization; existing benchmark/profiling jobs therefore retain their modes.

Implementation
--------------

One thread owns an interior X-Y column and visits all `nz-2` interior Z cells
in order. It initially loads the bottom neighbor and first center temperature.
Each iteration loads the next Z neighbor, evaluates the exact existing stencil,
then shifts `below = center; center = above`. These scalar values are intended
to remain in registers. Only current-buffer values are carried forward; newly
computed next-buffer values are never used within the same timestep.

The X/Y neighbors, FP64 coefficients, indexing, arithmetic expression, fixed
boundary planes, two-buffer swaps, error checking, and block size of 256 threads
are unchanged in meaning. Threads outside the interior column count return;
the Z loop never writes either outer plane and its final forward load is from
the valid upper boundary. Thin grids with only one interior Z cell are valid.
No shared-memory tiling, fast math, reduced precision, launch tuning, or multi-GPU
execution is introduced.

For a column with N interior cells, the rolling Z window needs N+2 source loads
for its centers/Z neighbors instead of roughly 3N stencil-level loads. This is
not a prediction of generated load instructions or DRAM traffic: caches already
reuse data, and the compiler may spill scalar state. Use hardware profiling to
inspect actual traffic, register count, and spills. There are fewer independent
threads/blocks than in the naive kernel, which can reduce available parallelism,
especially on small grids. An improvement is not assumed.

Benchmark modes
---------------

`cpu_cuda_benchmark` now accepts:

```text
--gpu-kernel naive       original kernel (default)
--gpu-kernel z-register  experimental full-column kernel
--gpu-kernel both        compare both kernels
--gpu-init baseline|d2d|both  existing Phase 6A initialization selection
```

For identical initialization, compare `--gpu-kernel both --gpu-init baseline`
or `--gpu-kernel both --gpu-init d2d`. The new Slurm job uses both flags set to
`both` to collect all four combinations on the same GPU with shared allocations.
It runs 64³, 128³, and 256³, with 10 warmups, **100 timed iterations and 9 repeats**
for every combination and timing type. It uses 8 CPU OpenMP threads and the
existing 32-CPU/1-GPU scheduler allocation.

Each repeat resets both buffers from the same deterministic sine-shaped field
after warmup. Combination order is initialization-major, then kernel-major,
and is reversed on odd repeats. For example, even repeats run naive/two-H2D,
Z-register/two-H2D, naive/H2D+D2D, Z-register/H2D+D2D; odd repeats reverse that
order. This controls systematic order bias without eliminating host load,
clock, thermal, cache-state, or shared-node effects.

CPU reference timing, CUDA-event kernel-only timing, transfer-inclusive chrono
timing, per-repeat validation, and payload accounting retain their prior scopes.
Every kernel-only initialization completes before the start event. Both methods
include their selected initialization and result download in transfer-inclusive
timing, with allocation excluded. Updated cells remain
`(nx-2)*(ny-2)*(nz-2)*iterations`, regardless of threads or launch count.

CSV additions
-------------

With four combinations there are nine rows per grid (one CPU, eight GPU timing
rows), or 27 rows for the full sweep. All raw timing samples, median/min/max,
mean/population standard deviation, updates/s, CPU/GPU speedups, numerical errors,
device identity, compiler flags, and Phase 6A fields remain available.

- `kernel_variant`: `naive`, `z_register`, or `not_applicable` for CPU.
- `z_cells_per_thread`: 1 for naive, `nz-2` for register reuse; blank for CPU.
- `speedup_vs_naive_median`: naive median divided by this row's median, matched
  on **grid, initialization method, and timing type**. Values above 1 indicate
  measured improvement; below 1 indicate regression. If naive was not measured
  in the invocation, the field is blank.
- `time_reduction_vs_naive_percent`: `100*(1 - experimental_time/naive_time)`;
  positive means reduced time, negative means regression. Blank without a
  matching reference. Naive self-comparisons are 1 and 0%, respectively.
- `speedup_vs_two_h2d_median` now explicitly matches the **same kernel** as well
  as grid and timing type. This preserves Phase 6A meaning and avoids crediting
  a kernel change to transfer initialization.

CPU reference rows have no GPU-relative ratios. Multiple-kernel runs record
their Cartesian ordering in `variant_order`; naive-only runs keep the Phase 6A
ordering label. Ratios are ratios of medians. Do not mix different initialization
methods, timing scopes, machines, or compiler configurations when comparing.
Kernel-only and transfer-inclusive speedups can differ because transfers can
mask or outweigh a compute-time change.

Correctness coverage
--------------------

The existing `solver_validation` and `cuda_solver_validation` are unchanged.
The new CUDA-only CTest `cuda_register_validation` checks both kernels and both
initializers for these shapes:

```text
3x3x3, 5x7x9, 17x17x17, 34x19x11, 64x33x18, 3x9x7, 9x3x4, 9x7x3
```

Each shape uses uniform, sine-shaped, and irregular deterministic data, including
nonuniform boundary temperatures. Step counts 1, 2, 7, and 100 exercise odd/even
buffer roles. Every kernel-only and transfer-inclusive repeat (two per path)
checks the full GPU field against the CPU reference, finiteness, and exactly
unchanged boundaries. Tolerance remains `1e-9 K + 1e-12*abs(CPU temperature)`.
The test performs 1,536 full-field comparisons across 96 shape/field/step cases.
The production benchmark also checks every repeat at all three benchmark sizes.

Perlmutter execution
--------------------

Use the [Phase 3 CUDA build configuration](cpu_cuda_benchmark.md), including
`ENABLE_CUDA=ON`, `BUILD_TESTING=ON`, matching GNU/CUDA modules, and architecture
80 for Perlmutter's A100. Rebuild and submit:

```sh
cmake --build build-cuda --parallel 4
sbatch --account=YOUR_GPU_ACCOUNT benchmarks/run_z_register_perlmutter.slurm
```

An absolute build directory may be passed as the first script argument. The
job runs all registered correctness tests before benchmarking, then writes
`kernel-comparison.csv` and hardware/build/source provenance under:

```text
$SCRATCH/heat-diffusion-z-register/<job-id>_<12-character-git-revision>/
```

Within an existing GPU allocation, the equivalent comparison command is:

```sh
export OMP_PLACES=cores OMP_PROC_BIND=close
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  build-cuda/cpu_cuda_benchmark --gpu-kernel both --gpu-init both --threads 8 \
  --iterations 100 --warmup 10 --repeats 9 --csv "$SCRATCH/z-register.csv"
```

For GPU memory checking, run the new test separately:

```sh
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
  compute-sanitizer --tool memcheck --error-exitcode 1 build-cuda/cuda_register_validation
```

Inspect the recorded GPU model to confirm the A100 comparison. Retain all nine
samples and report improvements **or regressions**, with variation and maximum
numerical error. Use unprofiled results for performance claims. Phase 4/5
scripts still select the naive default; do not apply their old launch filters
to a multi-kernel run without updating the intended kernel selection.

Local verification limits
-------------------------

CPU builds and CTest run with CUDA disabled. CPU-only smoke runs can parse both
experimental flags but produce only CPU rows:

```sh
cmake -S . -B /tmp/heat-phase6b-build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/heat-phase6b-build --parallel 2
ctest --test-dir /tmp/heat-phase6b-build --output-on-failure
/tmp/heat-phase6b-build/cpu_cuda_benchmark --cpu-only --gpu-kernel both --gpu-init both \
  --threads 2 --warmup 2 --iterations 3 --repeats 3 --csv /tmp/phase6b-cpu.csv
```

Local host execution of the kernel body with emulated thread indices, if used
for algorithm/bounds checks, is not CUDA compilation or GPU validation. Actual
NVCC compilation, register allocation, GPU correctness, and measured A100
improvements/regressions remain pending until the Perlmutter job completes.
