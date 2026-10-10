#pragma once
#include "grid3d.hpp"

class Solver {
public:
    // 1 timestep of 3D heat eq (finite difference)
    static void compute_diffusion(const Grid3D& current, Grid3D& next);

    // Injects laser heat into top powder bed layer
    static void apply_laser(Grid3D& grid, double laser_x, double laser_y);
};
