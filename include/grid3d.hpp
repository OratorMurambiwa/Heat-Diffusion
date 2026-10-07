#pragma once
#include <vector>
#include <cstddef>

class Grid3D {
public:
    int nx, ny, nz;
    std::vector<double> data;

    Grid3D(int nx_local, int ny_local, int nz_local, double initial_temp);

    inline size_t get_index(int x, int y, int z) const {
        return static_cast<size_t>(z) * (nx * ny) + static_cast<size_t>(y) * nx + static_cast<size_t>(x);
    }
};
