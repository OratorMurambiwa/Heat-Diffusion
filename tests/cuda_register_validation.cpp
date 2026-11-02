#include "comparison_common.hpp"
#include "cuda_timing.hpp"
#include "solver.hpp"
#include <array>
#include <iostream>

int main() {
    try {
        const std::array<std::array<int,3>,8> shapes{{
            {{3,3,3}}, {{5,7,9}}, {{17,17,17}}, {{34,19,11}},
            {{64,33,18}}, {{3,9,7}}, {{9,3,4}}, {{9,7,3}}
        }};
        const std::array<int,4> steps{{1,2,7,100}};
        size_t comparisons = 0;
        double maximum_error = 0.0;
        for (const auto& shape : shapes) {
            for (int field = 0; field < 3; ++field) {
                Grid3D initial(shape[0],shape[1],shape[2],Config::AMBIENT_TEMP);
                if (field == 1) Comparison::initialize(initial);
                if (field == 2) {
                    // Nonuniform values on every boundary as well as in the interior.
                    for (size_t i = 0; i < initial.data.size(); ++i)
                        initial.data[i] += static_cast<double>((37*i+13)%101)/10.0;
                }
                for (int count : steps) {
                    Grid3D cpu = initial, next = initial;
                    for (int step = 0; step < count; ++step) {
                        Solver::compute_diffusion(cpu,next);
                        cpu.data.swap(next.data);
                    }
                    // Each path checks its full field against CPU and its boundaries
                    // exactly after every repeat, outside any measurement interval.
                    const auto measurements = measure_gpu(initial,cpu,count,1,2,
                        {GpuInitialization::TwoHostCopies,GpuInitialization::HostThenDeviceCopy},
                        {GpuKernel::Naive,GpuKernel::ZRegister});
                    if (measurements.size() != 4) throw std::runtime_error("Missing kernel/initialization variant");
                    for (const auto& measurement : measurements) {
                        if (measurement.kernel_seconds.size() != 2 || measurement.end_to_end_seconds.size() != 2)
                            throw std::runtime_error("Missing validation repeats");
                        maximum_error = std::max({maximum_error,measurement.kernel_error,measurement.end_to_end_error});
                        comparisons += measurement.kernel_seconds.size() + measurement.end_to_end_seconds.size();
                    }
                    std::cout << "PASS: " << shape[0] << 'x' << shape[1] << 'x' << shape[2]
                              << ", field=" << field << ", steps=" << count << '\n';
                }
            }
        }
        std::cout << "PASS: " << comparisons << " full-field comparisons across both kernels and initializers; "
                  << "max error=" << maximum_error << " K\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
