First CUDA stencil: correctness before performance
=================================================

`CudaSolver::diffuse(initial, steps)` returns a new FP64 `Grid3D` after advancing
the existing seven-point diffusion expression. It does not change the CPU
solver, MPI simulator, laser code, or CPU benchmark. It operates on the current
CUDA device and treats all six outer planes as fixed boundaries. In the MPI
solver the outer Z planes serve as ghosts; this standalone path does not
exchange halos or implement distributed GPU execution.

The implementation allocates two device buffers, copies the initial field into
both (including boundary planes), launches one thread per interior cell, and
swaps device pointers between steps. Blocks have 256 threads; excess threads
return before touching memory. X remains contiguous, with index
`z*(nx*ny) + y*nx + x`. All coefficients come from `config.hpp` and use the same
FP64 formula as `Solver::compute_diffusion`. There is no shared-memory tiling,
fast-math flag, kernel fusion, or multi-GPU code.

Default-stream ordering preserves timestep dependencies. Kernel launch errors
are checked after every launch, and a final device synchronization checks
asynchronous execution errors before copying the result to the host. Allocation,
copies, device queries, and normal-path deallocation are checked. Exception-path
cleanup is best effort and does not throw. Invalid dimensions, storage sizes,
step counts, and launch sizes are rejected. Zero steps returns a host copy
without requiring a GPU.

Build and validate on Perlmutter
-------------------------------

Use CMake 3.18 or newer and a CUDA toolkit supporting C++17. The CPU-only build
still uses the existing CMake minimum and does not detect or require CUDA.
CUDA is disabled by default; explicitly requesting it without a compiler gives
an actionable configuration error.

From the repository root on Perlmutter, load the GNU programming environment
and CUDA toolkit. Choose a toolkit/GCC pair supported by the site's modules:

```sh
module load PrgEnv-gnu
module load cudatoolkit
cmake --version
nvcc --version

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA=ON -DBUILD_TESTING=ON
cmake --build build-cuda --parallel 4
```

`CC` is the Cray C++ wrapper used for the project's MPI dependencies. `nvcc`
compiles the separate `.cu` implementation. Architecture 80 targets Perlmutter's
A100 GPUs. Use a fresh build directory when changing compiler environments.
The module environment must contain a CMake executable of the required version.

Request a GPU allocation using your GPU account (replace `YOUR_GPU_ACCOUNT`):

```sh
salloc -A YOUR_GPU_ACCOUNT -C gpu -q shared -N 1 -n 1 -c 2 \
  --gpus-per-task=1 -t 00:10:00
```

Once the allocation is granted, run from the repository root:

```sh
export OMP_NUM_THREADS=2
srun -n 1 -c 2 --gpus-per-task=1 nvidia-smi
srun -n 1 -c 2 --gpus-per-task=1 \
  ctest --test-dir build-cuda --output-on-failure -V
```

This runs the existing CPU validation and the new GPU validation. GPU-only:

```sh
srun -n 1 -c 2 --gpus-per-task=1 \
  ctest --test-dir build-cuda -R '^cuda_solver_validation$' --output-on-failure -V
```

If Compute Sanitizer is available, additionally check device memory accesses:

```sh
srun -n 1 -c 2 --gpus-per-task=1 \
  compute-sanitizer --tool memcheck --error-exitcode 1 build-cuda/cuda_solver_validation
```

The CUDA test requires a working GPU; it reports missing devices or runtime
errors as failures when CUDA is explicitly enabled. Do not run it on login
nodes without an allocated GPU. The code does not initialize MPI; only one
Slurm task is needed. These instructions have not been executed on Perlmutter.

References: [NERSC CUDA environment](https://docs.nersc.gov/development/programming-models/cuda/),
[CMake and Cray wrappers](https://docs.nersc.gov/development/build-tools/cmake/),
[GPU allocation examples](https://docs.nersc.gov/systems/perlmutter/running-jobs/).

Validation coverage
-------------------

The test compares the complete CPU/GPU fields for five shapes: 3x3x3, 5x7x9,
17x17x17, 34x19x11, and 64x33x18. These exercise minimal dimensions, cubes,
non-cubes, and partial thread blocks. Each shape uses uniform, sine-mode, and
deterministic irregular initial data, with 0, 1, 2, 7, and 100 steps (75 cases).
The irregular case includes nonuniform boundary values to check that boundaries
are preserved rather than reset. Odd and even step counts exercise both buffers.

Every value must be finite, with CPU/GPU difference bounded by
`1e-9 K + 1e-12 * abs(CPU temperature)`. Boundaries, uniform fields, and zero-step
results must match the initial values exactly. The test reports maximum absolute
differences. A tolerance is used because host/device floating-point contraction
can differ; CPU/GPU bitwise equality is not assumed.

Local CPU regression checks require no CUDA installation:

```sh
cmake -S . -B /tmp/heat-cpu-check -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/heat-cpu-check --parallel 2
ctest --test-dir /tmp/heat-cpu-check --output-on-failure
```
