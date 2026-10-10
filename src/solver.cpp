#include "solver.hpp"
#include "config.hpp"
#include <omp.h>
#include <cmath>

void Solver::compute_diffusion(
    const Grid3D& current,
    Grid3D& next
) {
    const double cx =
        Config::ALPHA * Config::DT / (Config::DX * Config::DX);
    const double cy =
        Config::ALPHA * Config::DT / (Config::DY * Config::DY);
    const double cz =
        Config::ALPHA * Config::DT / (Config::DZ * Config::DZ);

    // Update interior cells, excluding boundaries and ghost layers.
    #pragma omp parallel for collapse(3)
    for (int z = 1; z < current.nz - 1; ++z) {
        for (int y = 1; y < current.ny - 1; ++y) {
            for (int x = 1; x < current.nx - 1; ++x) {
                const size_t idx = current.get_index(x, y, z);
                const double t_center = current.data[idx];

                // Seven-point stencil: center and six neighbors.
                const double t_x =
                    current.data[current.get_index(x + 1, y, z)]
                    - 2.0 * t_center
                    + current.data[current.get_index(x - 1, y, z)];

                const double t_y =
                    current.data[current.get_index(x, y + 1, z)]
                    - 2.0 * t_center
                    + current.data[current.get_index(x, y - 1, z)];

                const double t_z =
                    current.data[current.get_index(x, y, z + 1)]
                    - 2.0 * t_center
                    + current.data[current.get_index(x, y, z - 1)];

                next.data[idx] =
                    t_center + cx * t_x + cy * t_y + cz * t_z;
            }
        }
    }
}

void Solver::apply_laser(
    Grid3D& grid,
    double laser_x,
    double laser_y
) {
    // Top active layer, excluding the ghost layer.
    const int z = grid.nz - 2;

    const double radius_squared =
        Config::LASER_RADIUS * Config::LASER_RADIUS;

    // Gaussian intensity: strongest at the beam center.
    auto weight = [&](int x, int y) {
        const double dx = (x - laser_x) * Config::DX;
        const double dy = (y - laser_y) * Config::DY;

        return std::exp(
            -2.0 * (dx * dx + dy * dy) / radius_squared
        );
    };

    // Normalize over active surface cells to conserve absorbed energy.
    // This assumes the entire absorbed beam lands on the bed.
    double total_weight = 0.0;

    for (int y = 1; y < grid.ny - 1; ++y) {
        for (int x = 1; x < grid.nx - 1; ++x) {
            total_weight += weight(x, y);
        }
    }

    if (total_weight == 0.0) {
        return;
    }

    const double cell_heat_capacity =
        Config::DENSITY *
        Config::SPECIFIC_HEAT *
        Config::DX *
        Config::DY *
        Config::DZ;

    const double absorbed_energy =
        Config::ABSORPTIVITY *
        Config::LASER_POWER *
        Config::DT;

    const double temperature_scale =
        absorbed_energy / (cell_heat_capacity * total_weight);

    for (int y = 1; y < grid.ny - 1; ++y) {
        for (int x = 1; x < grid.nx - 1; ++x) {
            grid.data[grid.get_index(x, y, z)] +=
                temperature_scale * weight(x, y);
        }
    }
}