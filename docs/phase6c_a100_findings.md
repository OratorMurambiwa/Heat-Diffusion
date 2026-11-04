# Phase 6C: Bounded Z-Register CUDA Optimization

## Objective
Evaluate whether processing multiple adjacent Z cells per CUDA
thread improves FP64 3D heat-diffusion stencil performance.

## Experimental Setup
- System: NERSC Perlmutter
- GPU: NVIDIA A100-SXM4-80GB
- Precision: FP64
- Grid sizes: 64^3, 128^3, 256^3
- Benchmark: 100 iterations, 10 warmups, 9 repeats
- Kernels: naive, full-column, Z2, Z4, Z8

## Performance Results
At 256^3, using H2D+D2D initialization:

| Metric | Naive | Z4 |
|---|---:|---:|
| Kernel batch time | 18.798 ms | 17.647 ms |
| Speedup vs 8-thread CPU | 61.63x | 65.65x |
| Transfer-inclusive time | 38.482 ms | 37.323 ms |

Z4 reduced unprofiled kernel-batch time by approximately 6.1%.

## Nsight Compute Findings
Controlled naive/Z4 profiling on the same GPU allocation:

| Metric | Naive | Z4 |
|---|---:|---:|
| Profiled duration | 173.79 us | 156.64 us |
| Executed instructions | 76.81M | 34.67M |
| Achieved occupancy | 88.86% | 95.22% |
| DRAM throughput | 1.50 TB/s | 1.66 TB/s |
| Registers per thread | 28 | 30 |

Z4 executed approximately 54.9% fewer GPU instructions.
Estimated total DRAM traffic was nearly unchanged.

## Correctness
All four CTest suites passed.
The bounded-kernel validation completed 5,376 full-field
comparisons with a maximum numerical difference of 0 K.

## Conclusion
Bounded Z-axis register reuse improved single-GPU stencil
performance while preserving FP64 numerical correctness.
The results suggest reduced instruction overhead and improved
execution efficiency rather than substantially lower DRAM traffic.

These measurements apply to the standalone CUDA stencil benchmark,
not the complete distributed MPI application.
