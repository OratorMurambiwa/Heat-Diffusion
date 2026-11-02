#pragma once

#include <cuda_runtime.h>

// Experimental full-column Z sweep. The naive kernel remains separate.
namespace {
__global__ void diffusion_z_register(const double* current, double* next,
                                     int nx, int ny, int nz, size_t columns,
                                     double cx, double cy, double cz) {
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    const size_t x = 1 + column % static_cast<size_t>(nx - 2);
    const size_t y = 1 + column / static_cast<size_t>(nx - 2);
    const size_t plane = static_cast<size_t>(nx) * ny;
    size_t idx = plane + y * nx + x; // First interior Z plane.
    double below = current[idx - plane];
    double center = current[idx];
    for (int z = 1; z < nz - 1; ++z, idx += plane) {
        const double above = current[idx + plane];
        const double t_x = current[idx + 1] - 2.0 * center + current[idx - 1];
        const double t_y = current[idx + nx] - 2.0 * center + current[idx - nx];
        const double t_z = above - 2.0 * center + below;
        next[idx] = center + cx * t_x + cy * t_y + cz * t_z;
        below = center;
        center = above;
    }
}
}
