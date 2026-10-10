#include "config.hpp"
#include "grid3d.hpp"
#include "solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        constexpr int n = 17;

        Grid3D current(n, n, n, Config::AMBIENT_TEMP);
        Grid3D next(n, n, n, Config::AMBIENT_TEMP);

        // Test 1: A uniform temperature field should remain unchanged.
        Solver::compute_diffusion(current, next);

        for (double temperature : next.data) {
            require(
                std::abs(temperature - Config::AMBIENT_TEMP) < 1e-10,
                "Uniform temperature field changed"
            );
        }

        // Test 2: A sine-shaped field has a known discrete decay rate.
        const double pi = std::acos(-1.0);

        auto mode = [&](int x, int y, int z) {
            return
                std::sin(pi * x / (n - 1)) *
                std::sin(pi * y / (n - 1)) *
                std::sin(pi * z / (n - 1));
        };

        for (int z = 1; z < n - 1; ++z) {
            for (int y = 1; y < n - 1; ++y) {
                for (int x = 1; x < n - 1; ++x) {
                    current.data[current.get_index(x, y, z)] +=
                        10.0 * mode(x, y, z);
                }
            }
        }

        const double amplification =
            1.0 - 4.0 * Config::DIFFUSION_SUM *
            std::pow(std::sin(pi / (2.0 * (n - 1))), 2);

        constexpr int steps = 100;

        for (int step = 0; step < steps; ++step) {
            Solver::compute_diffusion(current, next);
            current.data.swap(next.data);
        }

        double max_error = 0.0;

        for (int z = 1; z < n - 1; ++z) {
            for (int y = 1; y < n - 1; ++y) {
                for (int x = 1; x < n - 1; ++x) {
                    const double expected =
                        Config::AMBIENT_TEMP +
                        10.0 * mode(x, y, z) *
                        std::pow(amplification, steps);

                    const double actual =
                        current.data[current.get_index(x, y, z)];

                    max_error = std::max(
                        max_error,
                        std::abs(actual - expected)
                    );
                }
            }
        }

        require(max_error < 1e-9, "Sine-mode diffusion mismatch");

        // Test 3: Laser energy must equal absorbed power times timestep.
        Grid3D laser(41, 41, 7, Config::AMBIENT_TEMP);
        Solver::apply_laser(laser, 20.0, 20.0);

        const double cell_heat_capacity =
            Config::DENSITY *
            Config::SPECIFIC_HEAT *
            Config::DX *
            Config::DY *
            Config::DZ;

        double deposited_energy = 0.0;

        for (int z = 0; z < laser.nz; ++z) {
            for (int y = 0; y < laser.ny; ++y) {
                for (int x = 0; x < laser.nx; ++x) {
                    const double delta =
                        laser.data[laser.get_index(x, y, z)] -
                        Config::AMBIENT_TEMP;

                    require(
                        std::isfinite(delta) && delta >= 0.0,
                        "Invalid laser temperature"
                    );

                    if (z != laser.nz - 2) {
                        require(
                            delta == 0.0,
                            "Laser heated a non-surface layer"
                        );
                    }

                    deposited_energy += delta * cell_heat_capacity;
                }
            }
        }

        const double expected_energy =
            Config::ABSORPTIVITY *
            Config::LASER_POWER *
            Config::DT;

        require(
            std::abs(deposited_energy - expected_energy) <
                1e-10 * expected_energy,
            "Laser energy is not conserved"
        );

        require(
            laser.data[laser.get_index(20, 20, 5)] >
                laser.data[laser.get_index(21, 20, 5)],
            "Laser intensity does not decrease away from the center"
        );

        std::cout
            << "PASS: uniform temperature\n"
            << "PASS: sine-mode diffusion; max error = "
            << max_error << " K\n"
            << "PASS: laser energy, surface placement, and Gaussian profile\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}