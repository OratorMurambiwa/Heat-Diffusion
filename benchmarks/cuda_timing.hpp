#pragma once
#include "grid3d.hpp"
#include <string>
#include <vector>
#include <stdexcept>

enum class GpuInitialization { TwoHostCopies, HostThenDeviceCopy };
enum class GpuKernel { Naive, ZRegister, Z2, Z4, Z8 };

inline const char* kernel_name(GpuKernel kernel) {
    switch (kernel) {
        case GpuKernel::Naive: return "naive";
        case GpuKernel::ZRegister: return "z_register";
        case GpuKernel::Z2: return "z_bounded_2";
        case GpuKernel::Z4: return "z_bounded_4";
        case GpuKernel::Z8: return "z_bounded_8";
    }
    throw std::invalid_argument("Unknown GPU kernel");
}

inline int z_cells_per_thread(GpuKernel kernel, int nz) {
    switch (kernel) {
        case GpuKernel::Naive: return 1;
        case GpuKernel::ZRegister: return nz - 2;
        case GpuKernel::Z2: return 2;
        case GpuKernel::Z4: return 4;
        case GpuKernel::Z8: return 8;
    }
    throw std::invalid_argument("Unknown GPU kernel");
}

inline const char* initialization_name(GpuInitialization mode) {
    return mode == GpuInitialization::TwoHostCopies ? "two_h2d" : "h2d_d2d";
}

struct GpuMeasurements {
    GpuInitialization initialization = GpuInitialization::TwoHostCopies;
    GpuKernel kernel = GpuKernel::Naive;
    std::vector<double> kernel_seconds, end_to_end_seconds;
    double kernel_error = 0.0, end_to_end_error = 0.0;
    std::string device, pci_bus_id;
    int compute_major = 0, compute_minor = 0, runtime_version = 0, driver_version = 0;
    size_t global_memory_bytes = 0;
};

std::vector<GpuMeasurements> measure_gpu(const Grid3D& initial, const Grid3D& cpu_reference,
                            int iterations, int warmup, int repeats,
                            const std::vector<GpuInitialization>& modes,
                            const std::vector<GpuKernel>& kernels);
