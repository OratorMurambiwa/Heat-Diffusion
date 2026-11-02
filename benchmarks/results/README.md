WSL CPU baseline
================

CPU reported by WSL: Intel(R) Core(TM) Ultra 5 225U; 14 logical CPUs visible,
one NUMA node reported. These virtual topology details do not establish the
physical host core mapping or exclusive CPU access.

Compiler: GCC 13.3.0. CMake Release compile flags, verified against flags.make:
`-O3 -march=native -Wall -O3 -DNDEBUG -std=gnu++17 -fopenmp`.
All runs used FP64, 10 warmups per repeat, 100 timed iterations per batch,
9 repeats, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`, and disabled dynamic teams.
Cases ran sequentially in ascending grid size and thread count. Full dimensions
include fixed boundaries. Combined CSV contains all nine raw samples per case,
CPU/build metadata, checksums, update counts, and validation error.

| Grid | Threads | Median batch (ms) | Min–max (ms) | Stddev (ms) | Mupdates/s | Speedup vs 1 thread | Nominal 64 B (GB/s) | Minimum 16 B (GB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64³ | 1 | 73.939 | 64.529–117.031 | 14.558 | 322.33 | 1.00× | 20.63 | 5.16 |
| 64³ | 2 | 49.106 | 40.327–120.023 | 29.290 | 485.34 | 1.51× | 31.06 | 7.77 |
| 64³ | 4 | 44.380 | 37.765–85.421 | 13.649 | 537.02 | 1.67× | 34.37 | 8.59 |
| 128³ | 1 | 639.537 | 620.376–659.348 | 12.853 | 312.79 | 1.00× | 20.02 | 5.00 |
| 128³ | 2 | 489.550 | 470.591–533.315 | 17.726 | 408.62 | 1.31× | 26.15 | 6.54 |
| 128³ | 4 | 537.569 | 483.931–791.814 | 89.064 | 372.12 | 1.19× | 23.82 | 5.95 |
| 256³ | 1 | 5965.141 | 5752.020–6846.316 | 373.864 | 274.71 | 1.00× | 17.58 | 4.40 |
| 256³ | 2 | 4344.007 | 4261.367–5118.359 | 271.680 | 377.23 | 1.37× | 24.14 | 6.04 |
| 256³ | 4 | 4056.866 | 3817.535–11210.293 | 2253.620 | 403.93 | 1.47× | 25.85 | 6.46 |

Both traffic columns are models, **not measured DRAM bandwidth**. The 64-byte
model counts seven stencil-level FP64 reads and one write. The 16-byte model
assumes perfect input reuse and a single read/write per updated cell; neither
accounts for all actual memory-system traffic.

Observations and limits
-----------------------

- One-thread throughput falls from about 322 to 275 Mupdates/s as the two-buffer
  footprint grows from 4 to 32 to 256 MiB. More threads improve median throughput
  over one thread, but gains are sublinear.
- At 128³, four threads are about 10% slower than two. At 256³, four threads
  improve the median by only about 7% over two.
- Variation is substantial in several cases. For example, 256³/four threads
  ranges from 3.82 to 11.21 seconds per batch, with 2.25 seconds standard
  deviation. All samples are retained; no outlier filtering was applied.
- Cache effects, OpenMP overhead, frequency changes, host scheduling, and
  resource sharing are possible contributors, not established causes. No
  memory counters, clock measurements, or host isolation were collected.
  Treat this as a preliminary WSL baseline rather than a stable hardware limit.

Validation
----------

All 81 repeated fields passed analytical sine-decay, finiteness, and exact
fixed-boundary checks. Maximum analytical error was 8.53e-13 K. Checksums match
across thread counts at each grid size. The existing CTest suite passed 1/1.
CSV statistics, update counts, both traffic estimates, and actual thread counts
were independently checked from the saved raw samples.

To reproduce a case, use the build instructions in `../README.md`, then:

```sh
OMP_PLACES=cores OMP_PROC_BIND=close /tmp/heat-cpu-build/cpu_stencil_benchmark \
  --nx 256 --ny 256 --nz 256 --iterations 100 --warmup 10 --repeats 9 \
  --threads 4 --csv /tmp/cpu_256_4.csv
```

Repeat for grid dimensions 64, 128, 256 and thread counts 1, 2, 4 sequentially.
Each invocation overwrites its CSV path; use distinct paths for separate cases.
