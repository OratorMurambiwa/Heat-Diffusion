#pragma once

#include <cuda_runtime.h>

// Bounded rolling Z window: X/Y columns vary before the Z-chunk index.
namespace {
template<int ZCells>
__global__ void diffusion_z_bounded(const double* current, double* next,
                                    int nx, int ny, int nz, size_t columns,
                                    size_t work_items, double cx, double cy, double cz) {
    static_assert(ZCells == 2 || ZCells == 4 || ZCells == 8, "Unsupported Z bound");
    const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (item >= work_items) return;
    const size_t column = item % columns;
    const size_t x = 1 + column % static_cast<size_t>(nx - 2);
    const size_t y = 1 + column / static_cast<size_t>(nx - 2);
    const int first_z = static_cast<int>(1 + (item / columns) * ZCells);
    const size_t plane = static_cast<size_t>(nx) * ny;
    size_t idx = static_cast<size_t>(first_z) * plane + y * nx + x;
    double below = current[idx - plane];
    double center = current[idx];
    for (int offset = 0, z = first_z; offset < ZCells && z < nz - 1;
         ++offset, ++z, idx += plane) {
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
