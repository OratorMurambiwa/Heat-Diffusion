#include "config.hpp"
#include "cuda_solver.hpp"
#include "solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        constexpr double absolute_tolerance = 1e-9; // Kelvin
        constexpr double relative_tolerance = 1e-12;
        const std::array<std::array<int, 3>, 5> shapes{{
            {{3, 3, 3}}, {{5, 7, 9}}, {{17, 17, 17}}, {{34, 19, 11}}, {{64, 33, 18}}
        }};
        const std::array<int, 5> step_counts{{0, 1, 2, 7, 100}};
        const double pi = std::acos(-1.0);
        double overall_max_error = 0.0;
        int cases = 0;
        for (const auto& shape : shapes) {
            for (int field = 0; field < 3; ++field) {
                Grid3D initial(shape[0], shape[1], shape[2], Config::AMBIENT_TEMP);
                for (int z = 0; z < initial.nz; ++z)
                    for (int y = 0; y < initial.ny; ++y)
                        for (int x = 0; x < initial.nx; ++x) {
                            const size_t idx = initial.get_index(x, y, z);
                            const bool boundary = x == 0 || x == initial.nx - 1 ||
                                y == 0 || y == initial.ny - 1 || z == 0 || z == initial.nz - 1;
                            if (field == 1 && !boundary) {
                                initial.data[idx] += 10.0 * std::sin(pi * x / (initial.nx - 1)) *
                                    std::sin(pi * y / (initial.ny - 1)) *
                                    std::sin(pi * z / (initial.nz - 1));
                            } else if (field == 2) {
                                // Irregular interior and nonuniform fixed boundaries.
                                initial.data[idx] += static_cast<double>((idx * 37 + 13) % 101) / 10.0;
                            }
                        }
                for (int steps : step_counts) {
                    Grid3D cpu = initial;
                    Grid3D next = initial;
                    for (int step = 0; step < steps; ++step) {
                        Solver::compute_diffusion(cpu, next);
                        cpu.data.swap(next.data);
                    }
                    const Grid3D gpu = CudaSolver::diffuse(initial, steps);
                    double max_error = 0.0;
                    for (int z = 0; z < initial.nz; ++z)
                        for (int y = 0; y < initial.ny; ++y)
                            for (int x = 0; x < initial.nx; ++x) {
                                const size_t idx = initial.get_index(x, y, z);
                                const bool boundary = x == 0 || x == initial.nx - 1 ||
                                    y == 0 || y == initial.ny - 1 || z == 0 || z == initial.nz - 1;
                                const double expected = cpu.data[idx];
                                const double actual = gpu.data[idx];
                                const double error = std::abs(actual - expected);
                                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                                    error > absolute_tolerance + relative_tolerance * std::abs(expected)) {
                                    throw std::runtime_error("CPU/GPU mismatch at index " + std::to_string(idx));
                                }
                                if ((boundary || steps == 0 || field == 0) && actual != initial.data[idx]) {
                                    throw std::runtime_error("Boundary, zero-step, or uniform-field value changed");
                                }
                                max_error = std::max(max_error, error);
                            }
                    overall_max_error = std::max(overall_max_error, max_error);
                    ++cases;
                    std::cout << "PASS: " << shape[0] << 'x' << shape[1] << 'x' << shape[2]
                              << ", field=" << field << ", steps=" << steps
                              << ", max error=" << max_error << " K\n";
                }
            }
        }
        std::cout << "PASS: " << cases << " CPU/GPU cases; maximum difference="
                  << overall_max_error << " K\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
