#include "solver.hpp"
#include "config.hpp"
#include <omp.h>

void Solver::compute_diffusion(const Grid3D& current, Grid3D& next) {
    // Pre-compute thermal multipliers for loop efficiency
    double cx = (Config::ALPHA * Config::DT) / (Config::DX * Config::DX);
    double cy = (Config::ALPHA * Config::DT) / (Config::DY * Config::DY);
    double cz = (Config::ALPHA * Config::DT) / (Config::DZ * Config::DZ);

    // 1 to n-1 to skip MPI ghost cells
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < current.nz - 1; ++z) {
        for (int y = 1; y < current.ny - 1; ++y) {
            for (int x = 1; x < current.nx - 1; ++x) {
                
                size_t idx = current.get_index(x, y, z);
                double t_center = current.data[idx];
                
                // 7-point 3D stencil
                double t_x = current.data[current.get_index(x+1, y, z)] - 2.0*t_center + current.data[current.get_index(x-1, y, z)];
                double t_y = current.data[current.get_index(x, y+1, z)] - 2.0*t_center + current.data[current.get_index(x, y-1, z)];
                double t_z = current.data[current.get_index(x, y, z+1)] - 2.0*t_center + current.data[current.get_index(x, y, z-1)];

                // Explicit update
                next.data[idx] = t_center + cx*t_x + cy*t_y + cz*t_z;
            }
        }
    }
}

void Solver::apply_laser(Grid3D& grid, int laser_x, int laser_y) {
    // Top metal layer (skip ghost padding)
    int z = grid.nz - 2;
    
    // Delta T = (Power * dt) / (Mass * Cp)
    double volume = Config::DX * Config::DY * Config::DZ;
    double mass = Config::DENSITY * volume;
    double temp_increase = (Config::LASER_POWER * Config::DT) / (mass * Config::SPECIFIC_HEAT);

    // Check if laser is in current MPI domain bounds
    if (laser_x >= 1 && laser_x < grid.nx - 1 && laser_y >= 1 && laser_y < grid.ny - 1) {
        size_t idx = grid.get_index(laser_x, laser_y, z);
        grid.data[idx] += temp_increase;
    }
}
