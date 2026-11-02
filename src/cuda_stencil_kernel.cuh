#pragma once

#include <cuda_runtime.h>

// Shared unchanged naive kernel for correctness tests and benchmarks.
namespace {
__global__ void diffusion(const double* current, double* next,
                          int nx, int ny, size_t interior_count,
                          double cx, double cy, double cz) {
    const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= interior_count) return;

    // X is contiguous, matching Grid3D. Boundary cells are never written.
    const size_t interior_x = static_cast<size_t>(nx - 2);
    const size_t interior_y = static_cast<size_t>(ny - 2);
    const size_t x = 1 + cell % interior_x;
    const size_t y = 1 + (cell / interior_x) % interior_y;
    const size_t z = 1 + cell / (interior_x * interior_y);
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t idx = z * plane + y * nx + x;
    const double t_center = current[idx];
    const double t_x = current[idx + 1] - 2.0 * t_center + current[idx - 1];
    const double t_y = current[idx + nx] - 2.0 * t_center + current[idx - nx];
    const double t_z = current[idx + plane] - 2.0 * t_center + current[idx - plane];
    next[idx] = t_center + cx * t_x + cy * t_y + cz * t_z;
}
}
