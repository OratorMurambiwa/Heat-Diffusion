#pragma once
#include "grid3d.hpp"
#include <string>
#include <vector>

struct GpuMeasurements {
    std::vector<double> kernel_seconds, end_to_end_seconds;
    double kernel_error = 0.0, end_to_end_error = 0.0;
    std::string device, pci_bus_id;
    int compute_major = 0, compute_minor = 0, runtime_version = 0, driver_version = 0;
    size_t global_memory_bytes = 0;
};

GpuMeasurements measure_gpu(const Grid3D& initial, const Grid3D& cpu_reference,
                            int iterations, int warmup, int repeats);
