CPU stencil baseline
====================

Build `cpu_stencil_benchmark` with the project's CMake Release configuration:

```sh
cmake -S . -B /tmp/heat-cpu-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/heat-cpu-build --parallel 2
/tmp/heat-cpu-build/cpu_stencil_benchmark --nx 128 --ny 128 --nz 128 \
  --iterations 100 --warmup 10 --repeats 5 --threads 2 \
  --csv /tmp/heat-cpu-baseline.csv
```

Use `--help` for options. Dimensions include two fixed boundary planes per
axis. Both FP64 buffers start at ambient temperature plus a deterministic
interior sine mode; boundary values stay at ambient temperature. Each repeat
resets both buffers and performs its own warmup before timing.

The monotonic clock brackets each existing `Solver::compute_diffusion` call.
Times are summed over the requested iterations for each repeat. Allocation,
initialization, warmup, swaps, result checks, and CSV output are outside the
timed regions. No MPI or laser routines run. Timings include the stencil's
OpenMP team and synchronization overhead and small clock measurement overhead;
very small grids or very short calls can therefore give noisy results.

CSV output overwrites the requested path with one header and one summary row.
Times are batch seconds, and standard deviation is the population standard
deviation across repeated batches. Rates use exactly
`(nx-2)*(ny-2)*(nz-2)*iterations` cell updates and median batch time; warmup
updates and boundary cells are excluded. The checksum consumes the final field
outside timing. After every repeat, a full-field analytical check verifies the
discrete sine-mode decay over `warmup + iterations` steps (absolute tolerance
`1e-9 K`), finiteness, and exactly unchanged boundaries. Failure stops the run.
The CSV records the maximum validation error and individual batch samples in
execution order, separated by semicolons. It also records requested and observed
thread counts, since runtime limits can reduce
the team size. `--threads` overrides `OMP_NUM_THREADS`; dynamic teams are disabled.

Two traffic models multiply cell updates/s by bytes/update, in decimal GB/s:

- **Nominal memory-operation traffic:** seven FP64 reads plus one FP64 write,
  or 64 bytes/update. This counts stencil-level operations, not generated
  machine load/store instructions or external memory transactions.
- **Idealized minimum traffic:** one FP64 read plus one FP64 write, or
  16 bytes/update, assuming perfect reuse of input cells across neighbors.

Neither is measured DRAM bandwidth. Caching reuses neighbors, while write
allocation, boundary traffic, and other transfers are omitted. Small grids
can be cache-resident. Use hardware profiling for actual memory traffic.

CSV metadata includes CPU model from Linux `/proc/cpuinfo` (or an unknown
fallback), compiler version, build type, configured optimization/OpenMP/standard
flags, and `OMP_PLACES`/`OMP_PROC_BIND`. Build metadata is captured by CMake for
the single-configuration Release workflow shown above. The project currently
uses `-O3 -march=native`; results and binaries depend on the build host. Avoid
concurrent workloads and use identical affinity settings when comparing thread
counts. These metadata do not describe host scheduling or CPU clock changes.

Methodology review
------------------

The timed loop calls the unchanged stencil and swaps buffers outside timing, so
each iteration advances the previous field. The OpenMP worksharing construct
has an implicit completion barrier: work finishes before the ending timestamp.
Results from every repeat are checked against the analytical solution; the
last repeat's checksum is written to CSV. This makes the computation observable.
The baseline build uses separate translation units without LTO. Inspection of
the Release executable confirmed calls to the stencil between clock reads and
the stencil's OpenMP runtime call. No volatile accesses, artificial memory
barriers, or numerical changes were added to the solver.

Per-call timing preserves the exclusion of swaps but includes clock overhead
and interrupt/scheduling delays during calls. Warmups reduce startup effects
without ensuring a particular cache or thermal state. Nine repeats expose
variation but do not provide controlled hardware counters or a causal diagnosis
of performance limits. The serial reset first-touches the buffers; use care
when extending this setup to multi-socket/NUMA systems.
