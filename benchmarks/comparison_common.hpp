#pragma once

#include "config.hpp"
#include "grid3d.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace Comparison {
inline void initialize(Grid3D& grid) {
    const double pi = std::acos(-1.0);
    std::fill(grid.data.begin(), grid.data.end(), Config::AMBIENT_TEMP);
    for (int z = 1; z < grid.nz - 1; ++z)
        for (int y = 1; y < grid.ny - 1; ++y)
            for (int x = 1; x < grid.nx - 1; ++x)
                grid.data[grid.get_index(x, y, z)] += 10.0 *
                    std::sin(pi * x / (grid.nx - 1)) *
                    std::sin(pi * y / (grid.ny - 1)) *
                    std::sin(pi * z / (grid.nz - 1));
}

inline bool boundary(const Grid3D& grid, int x, int y, int z) {
    return x == 0 || x == grid.nx - 1 || y == 0 || y == grid.ny - 1 ||
           z == 0 || z == grid.nz - 1;
}

inline double check_cpu(const Grid3D& grid, int steps) {
    const double pi = std::acos(-1.0);
    const double amplification = 1.0 - 4.0 * Config::ALPHA * Config::DT * (
        std::pow(std::sin(pi / (2.0 * (grid.nx - 1))) / Config::DX, 2) +
        std::pow(std::sin(pi / (2.0 * (grid.ny - 1))) / Config::DY, 2) +
        std::pow(std::sin(pi / (2.0 * (grid.nz - 1))) / Config::DZ, 2));
    const double amplitude = 10.0 * std::pow(amplification, steps);
    double maximum = 0.0;
    for (int z = 0; z < grid.nz; ++z)
        for (int y = 0; y < grid.ny; ++y)
            for (int x = 0; x < grid.nx; ++x) {
                const bool edge = boundary(grid, x, y, z);
                const double expected = Config::AMBIENT_TEMP + (edge ? 0.0 : amplitude *
                    std::sin(pi * x / (grid.nx - 1)) *
                    std::sin(pi * y / (grid.ny - 1)) *
                    std::sin(pi * z / (grid.nz - 1)));
                const double actual = grid.data[grid.get_index(x, y, z)];
                if (!std::isfinite(actual) || (edge && actual != Config::AMBIENT_TEMP))
                    throw std::runtime_error("CPU nonfinite value or changed boundary");
                maximum = std::max(maximum, std::abs(actual - expected));
            }
    if (maximum > 1e-9) throw std::runtime_error("CPU analytical validation failed");
    return maximum;
}

inline double check_gpu(const Grid3D& initial, const Grid3D& cpu, const Grid3D& gpu) {
    double maximum = 0.0;
    for (int z = 0; z < initial.nz; ++z)
        for (int y = 0; y < initial.ny; ++y)
            for (int x = 0; x < initial.nx; ++x) {
                const size_t idx = initial.get_index(x, y, z);
                const double actual = gpu.data[idx], expected = cpu.data[idx];
                const double error = std::abs(actual - expected);
                if (!std::isfinite(actual) || !std::isfinite(expected) ||
                    error > 1e-9 + 1e-12 * std::abs(expected) ||
                    (boundary(initial, x, y, z) && actual != initial.data[idx]))
                    throw std::runtime_error("CPU/GPU mismatch or changed boundary at index " + std::to_string(idx));
                maximum = std::max(maximum, error);
            }
    return maximum;
}

struct Statistics {
    double median, minimum, maximum, mean, stddev;
    explicit Statistics(std::vector<double> values) {
        if (values.size() < 2) throw std::runtime_error("Need at least two timing samples");
        for (double value : values)
            if (!(value > 0.0) || !std::isfinite(value)) throw std::runtime_error("Invalid timing sample");
        mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        double variance = 0.0;
        for (double value : values) variance += (value - mean) * (value - mean);
        stddev = std::sqrt(variance / values.size());
        std::sort(values.begin(), values.end());
        median = (values[values.size() / 2] + values[(values.size() - 1) / 2]) / 2.0;
        minimum = values.front(); maximum = values.back();
    }
};

inline std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (char c : value) result += c == '"' ? "\"\"" : std::string(1, c);
    return result + '"';
}

inline std::string samples_string(const std::vector<double>& samples) {
    std::ostringstream output;
    output << std::setprecision(17);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) output << ';';
        output << samples[i];
    }
    return output.str();
}
}
