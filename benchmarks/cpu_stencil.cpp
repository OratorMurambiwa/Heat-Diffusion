#include "config.hpp"
#include "grid3d.hpp"
#include "solver.hpp"
#include "cpu_benchmark_build.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>

namespace {
std::string csv_string(const std::string& value) {
    std::string result = "\"";
    for (char c : value) result += c == '"' ? "\"\"" : std::string(1, c);
    return result + '"';
}

std::string cpu_model() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) return line.substr(colon + 2);
        }
    }
    return "unknown (/proc/cpuinfo unavailable)";
}

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "unset";
}

#if defined(__clang__)
constexpr const char* compiler = "Clang " __clang_version__;
#elif defined(__GNUC__)
constexpr const char* compiler = "GCC " __VERSION__;
#else
constexpr const char* compiler = "Unknown C++ compiler";
#endif

int positive_integer(const char* value) {
    const std::string input(value);
    size_t consumed = 0;
    const int result = std::stoi(input, &consumed);
    if (consumed != input.size() || result <= 0) {
        throw std::runtime_error("Expected a positive integer: " + input);
    }
    return result;
}

void initialize(Grid3D& grid) {
    std::fill(grid.data.begin(), grid.data.end(), Config::AMBIENT_TEMP);
    const double pi = std::acos(-1.0);
    for (int z = 1; z < grid.nz - 1; ++z)
        for (int y = 1; y < grid.ny - 1; ++y)
            for (int x = 1; x < grid.nx - 1; ++x)
                grid.data[grid.get_index(x, y, z)] += 10.0 *
                    std::sin(pi * x / (grid.nx - 1)) *
                    std::sin(pi * y / (grid.ny - 1)) *
                    std::sin(pi * z / (grid.nz - 1));
}

// Independently check the exact discrete sine-mode decay and fixed boundaries.
// This consumes every repeat's full result, entirely outside the timed region.
double validate(const Grid3D& grid, double steps) {
    const double pi = std::acos(-1.0);
    const double amplification = 1.0 - 4.0 * Config::ALPHA * Config::DT * (
        std::pow(std::sin(pi / (2.0 * (grid.nx - 1))) / Config::DX, 2) +
        std::pow(std::sin(pi / (2.0 * (grid.ny - 1))) / Config::DY, 2) +
        std::pow(std::sin(pi / (2.0 * (grid.nz - 1))) / Config::DZ, 2));
    const double amplitude = 10.0 * std::pow(amplification, steps);
    double max_error = 0.0;
    for (int z = 0; z < grid.nz; ++z)
        for (int y = 0; y < grid.ny; ++y)
            for (int x = 0; x < grid.nx; ++x) {
                const bool boundary = x == 0 || x == grid.nx - 1 ||
                    y == 0 || y == grid.ny - 1 || z == 0 || z == grid.nz - 1;
                const double expected = Config::AMBIENT_TEMP + (boundary ? 0.0 : amplitude *
                    std::sin(pi * x / (grid.nx - 1)) *
                    std::sin(pi * y / (grid.ny - 1)) *
                    std::sin(pi * z / (grid.nz - 1)));
                const double actual = grid.data[grid.get_index(x, y, z)];
                if (!std::isfinite(actual) || (boundary && actual != Config::AMBIENT_TEMP))
                    throw std::runtime_error("Nonfinite result or changed fixed boundary");
                max_error = std::max(max_error, std::abs(actual - expected));
            }
    if (max_error > 1e-9) throw std::runtime_error("Analytical sine-mode validation failed");
    return max_error;
}
}

int main(int argc, char** argv) {
    try {
        int nx = 128, ny = 128, nz = 128;
        int iterations = 100, warmup = 10, repeats = 5, threads = 2;
        std::string csv_path = "cpu_benchmark.csv";
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--help") {
                std::cout << "Usage: cpu_stencil_benchmark [--nx N --ny N --nz N]"
                          << " [--iterations N --warmup N --repeats N --threads N]"
                          << " [--csv PATH]\n"
                          << "Dimensions include fixed boundary planes; all counts must be positive.\n"
                          << "CSV is overwritten. Times are seconds per timed batch.\n";
                return 0;
            }
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + option);
            const char* value = argv[++i];
            if (option == "--csv") csv_path = value;
            else if (option == "--nx") nx = positive_integer(value);
            else if (option == "--ny") ny = positive_integer(value);
            else if (option == "--nz") nz = positive_integer(value);
            else if (option == "--iterations") iterations = positive_integer(value);
            else if (option == "--warmup") warmup = positive_integer(value);
            else if (option == "--repeats") repeats = positive_integer(value);
            else if (option == "--threads") threads = positive_integer(value);
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (nx < 3 || ny < 3 || nz < 3 || repeats < 2)
            throw std::runtime_error("Dimensions must be >= 3 and repeats >= 2");
        // Grid3D currently multiplies nx * ny as int when calculating indices.
        if (nx > std::numeric_limits<int>::max() / ny)
            throw std::runtime_error("nx * ny exceeds Grid3D indexing capacity");
        const size_t plane = static_cast<size_t>(nx) * ny;
        if (plane > std::vector<double>().max_size() / static_cast<size_t>(nz))
            throw std::runtime_error("Grid exceeds allocation capacity");

        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        int actual_threads = 0;
        #pragma omp parallel
        {
            #pragma omp single
            actual_threads = omp_get_num_threads();
        }
        Grid3D current(nx, ny, nz, Config::AMBIENT_TEMP);
        Grid3D next(nx, ny, nz, Config::AMBIENT_TEMP);
        std::vector<double> samples;
        double checksum = 0.0;
        double max_error = 0.0;
        using Clock = std::chrono::steady_clock;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            initialize(current);
            initialize(next);
            for (int step = 0; step < warmup; ++step) {
                Solver::compute_diffusion(current, next);
                current.data.swap(next.data);
            }
            double seconds = 0.0;
            for (int step = 0; step < iterations; ++step) {
                const auto start = Clock::now();
                Solver::compute_diffusion(current, next);
                const auto end = Clock::now();
                seconds += std::chrono::duration<double>(end - start).count();
                current.data.swap(next.data);
            }
            if (!(seconds > 0.0)) throw std::runtime_error("Timer resolution insufficient");
            samples.push_back(seconds);
            // Consume and validate the result outside the timed region.
            checksum = std::accumulate(current.data.begin(), current.data.end(), 0.0);
            if (!std::isfinite(checksum)) throw std::runtime_error("Nonfinite result");
            max_error = std::max(max_error, validate(current, static_cast<double>(warmup) + iterations));
        }
        std::ostringstream raw_samples;
        raw_samples << std::setprecision(17);
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i) raw_samples << ';';
            raw_samples << samples[i];
        }
        std::sort(samples.begin(), samples.end());
        const double median = (samples[repeats / 2] + samples[(repeats - 1) / 2]) / 2.0;
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / repeats;
        double variance = 0.0;
        for (double sample : samples) variance += (sample - mean) * (sample - mean);
        const double stddev = std::sqrt(variance / repeats);
        const double updates = static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations;
        const double updates_per_second = updates / median;
        constexpr int nominal_bytes_per_update = 8 * sizeof(double);
        constexpr int minimum_bytes_per_update = 2 * sizeof(double);
        const double nominal_bandwidth = updates_per_second * nominal_bytes_per_update / 1e9;
        const double minimum_bandwidth = updates_per_second * minimum_bytes_per_update / 1e9;

        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Cannot open CSV: " + csv_path);
        csv << "nx,ny,nz,precision,requested_threads,actual_threads,iterations,warmup,repeats,compiler,"
               "median_seconds,min_seconds,max_seconds,mean_seconds,stddev_seconds,"
               "cell_updates_per_second,nominal_bytes_per_update,nominal_memory_operation_GB_s,"
               "idealized_minimum_bytes_per_update,idealized_minimum_GB_s,checksum,"
               "cpu_model,build_type,compiler_flags,omp_places,omp_proc_bind,"
               "timed_cell_updates,max_validation_error_K,batch_samples_seconds\n";
        csv << std::setprecision(17) << nx << ',' << ny << ',' << nz << ",FP64,"
            << threads << ',' << actual_threads << ',' << iterations << ',' << warmup << ',' << repeats
            << ',' << csv_string(compiler) << ',' << median << ',' << samples.front() << ',' << samples.back()
            << ',' << mean << ',' << stddev << ',' << updates_per_second << ',' << nominal_bytes_per_update
            << ',' << nominal_bandwidth << ',' << minimum_bytes_per_update << ',' << minimum_bandwidth
            << ',' << checksum << ',' << csv_string(cpu_model()) << ',' << csv_string(benchmark_build_type)
            << ',' << csv_string(benchmark_compiler_flags) << ',' << csv_string(environment("OMP_PLACES"))
            << ',' << csv_string(environment("OMP_PROC_BIND")) << ',' << updates << ',' << max_error
            << ',' << csv_string(raw_samples.str()) << '\n';
        csv.close();
        if (!csv) throw std::runtime_error("Failed to write CSV: " + csv_path);
        std::cout << std::setprecision(8)
                  << "Grid: " << nx << 'x' << ny << 'x' << nz << " (including boundaries), FP64\n"
                  << "OpenMP threads: " << actual_threads << " (requested " << threads << ")\n"
                  << "Iterations: " << iterations << ", warmup per repeat: " << warmup
                  << ", repeats: " << repeats << "\n"
                  << "Batch seconds: median=" << median << ", min=" << samples.front()
                  << ", max=" << samples.back() << ", population stddev=" << stddev << '\n'
                  << "Cell updates/s: " << updates_per_second << '\n'
                  << "Nominal memory-operation traffic: " << nominal_bandwidth
                  << " GB/s (64 bytes/update; 7 FP64 reads + 1 write)\n"
                  << "Idealized minimum traffic: " << minimum_bandwidth
                  << " GB/s (16 bytes/update; perfect reuse, 1 FP64 read + 1 write)\n"
                  << "Neither traffic estimate is measured DRAM bandwidth.\n"
                  << "Analytical validation max error: " << max_error << " K\n"
                  << "CSV: " << csv_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
