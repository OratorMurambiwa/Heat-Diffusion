#include "cuda_timing.hpp"
#include "comparison_common.hpp"
#include "cuda_stencil_kernel.cuh"
#include "cuda_z_register_kernel.cuh"
#include "cuda_z_bounded_kernel.cuh"
#include <chrono>
#include <utility>

namespace {
void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Buffer {
    double* data = nullptr;
    explicit Buffer(size_t bytes) { check(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc"); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void release() { auto pointer = data; data = nullptr; check(cudaFree(pointer), "cudaFree"); }
};
struct Event {
    cudaEvent_t event{};
    Event() { check(cudaEventCreate(&event), "cudaEventCreate"); }
    ~Event() { if (event) cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void release() { auto value = event; event = nullptr; check(cudaEventDestroy(value), "cudaEventDestroy"); }
};
}

std::vector<GpuMeasurements> measure_gpu(const Grid3D& initial, const Grid3D& cpu_reference,
                            int iterations, int warmup, int repeats,
                            const std::vector<GpuInitialization>& modes,
                            const std::vector<GpuKernel>& kernels) {
    if (modes.empty() || modes.size() > 2 ||
        (modes.size() == 2 && modes[0] == modes[1]))
        throw std::runtime_error("Expected one or two distinct GPU initialization modes");
    if (kernels.empty() || kernels.size() > 5)
        throw std::runtime_error("Expected one to five distinct GPU kernels");
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernel_name(kernels[i]); // Reject invalid enum values before allocating.
        if (std::find(kernels.begin(), kernels.begin() + i, kernels[i]) != kernels.begin() + i)
            throw std::runtime_error("Duplicate GPU kernel");
    }
    GpuMeasurements metadata;
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    metadata.device = properties.name;
    metadata.compute_major = properties.major; metadata.compute_minor = properties.minor;
    metadata.global_memory_bytes = properties.totalGlobalMem;
    char bus[32]{};
    check(cudaDeviceGetPCIBusId(bus, sizeof(bus), device), "cudaDeviceGetPCIBusId");
    metadata.pci_bus_id = bus;
    check(cudaRuntimeGetVersion(&metadata.runtime_version), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&metadata.driver_version), "cudaDriverGetVersion");
    std::vector<GpuMeasurements> results;
    for (auto mode : modes) {
        for (auto kernel : kernels) {
            results.push_back(metadata);
            results.back().initialization = mode;
            results.back().kernel = kernel;
        }
    }

    const size_t bytes = initial.data.size() * sizeof(double);
    const size_t columns = static_cast<size_t>(initial.nx - 2) * (initial.ny - 2);
    const size_t cells = columns * (initial.nz - 2);
    constexpr unsigned int threads = 256;
    auto work_items = [&](GpuKernel kernel) -> size_t {
        if (kernel == GpuKernel::Naive) return cells;
        if (kernel == GpuKernel::ZRegister) return columns;
        const size_t bound = static_cast<size_t>(z_cells_per_thread(kernel, initial.nz));
        const size_t chunks = 1 + (static_cast<size_t>(initial.nz - 2) - 1) / bound;
        return columns * chunks;
    };
    for (auto kernel : kernels) {
        const size_t count = (work_items(kernel) + threads - 1) / threads;
        if (count > static_cast<size_t>(properties.maxGridSize[0]))
            throw std::runtime_error("Grid exceeds CUDA launch limit");
    }
    Buffer first(bytes), second(bytes);
    Event start, stop;
    Grid3D output = initial;
    double* current = first.data;
    double* next = second.data;
    const double cx = Config::ALPHA * Config::DT / (Config::DX * Config::DX);
    const double cy = Config::ALPHA * Config::DT / (Config::DY * Config::DY);
    const double cz = Config::ALPHA * Config::DT / (Config::DZ * Config::DZ);
    auto upload = [&](GpuInitialization mode) {
        current = first.data; next = second.data;
        check(cudaMemcpy(current, initial.data.data(), bytes, cudaMemcpyHostToDevice), "first H2D");
        if (mode == GpuInitialization::TwoHostCopies) {
            check(cudaMemcpy(next, initial.data.data(), bytes, cudaMemcpyHostToDevice), "second H2D");
        } else {
            // Copy the full initialized field, including all six boundary planes.
            check(cudaMemcpy(next, current, bytes, cudaMemcpyDeviceToDevice), "second buffer D2D");
        }
    };
    auto advance = [&](int steps, GpuKernel kernel) {
        const size_t items = work_items(kernel);
        const auto blocks = static_cast<unsigned int>((items + threads - 1) / threads);
        for (int step = 0; step < steps; ++step) {
            switch (kernel) {
                case GpuKernel::Naive:
                    diffusion<<<blocks, threads>>>(current, next, initial.nx, initial.ny, cells, cx, cy, cz);
                    break;
                case GpuKernel::ZRegister:
                    diffusion_z_register<<<blocks, threads>>>(
                        current, next, initial.nx, initial.ny, initial.nz, columns, cx, cy, cz);
                    break;
                case GpuKernel::Z2:
                    diffusion_z_bounded<2><<<blocks, threads>>>(
                        current, next, initial.nx, initial.ny, initial.nz, columns, items, cx, cy, cz);
                    break;
                case GpuKernel::Z4:
                    diffusion_z_bounded<4><<<blocks, threads>>>(
                        current, next, initial.nx, initial.ny, initial.nz, columns, items, cx, cy, cz);
                    break;
                case GpuKernel::Z8:
                    diffusion_z_bounded<8><<<blocks, threads>>>(
                        current, next, initial.nx, initial.ny, initial.nz, columns, items, cx, cy, cz);
                    break;
            }
            check(cudaGetLastError(), "diffusion launch");
            std::swap(current, next);
        }
    };
    auto warm = [&](GpuInitialization mode, GpuKernel kernel) {
        upload(mode); advance(warmup, kernel);
        check(cudaDeviceSynchronize(), "warmup synchronization");
    };
    auto download = [&] {
        check(cudaMemcpy(output.data.data(), current, bytes, cudaMemcpyDeviceToHost), "D2H");
    };
    using Clock = std::chrono::steady_clock;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        // Reverse paired order on odd repeats to reduce consistent ordering bias.
        // Single-mode runs retain the original baseline measurement sequence.
        for (size_t slot = 0; slot < results.size(); ++slot) {
            const size_t index = repeat % 2 == 0 ? slot : results.size() - 1 - slot;
            auto& result = results[index];
            const auto mode = result.initialization;
            const auto kernel = result.kernel;
            warm(mode, kernel);
            upload(mode);
            // Finish uploads before starting device-only timing (pageable host data).
            check(cudaDeviceSynchronize(), "pre-timing synchronization");
            check(cudaEventRecord(start.event), "start event record");
            advance(iterations, kernel);
            check(cudaEventRecord(stop.event), "stop event record");
            check(cudaEventSynchronize(stop.event), "stop event synchronization");
            float milliseconds = 0.0f;
            check(cudaEventElapsedTime(&milliseconds, start.event, stop.event), "event elapsed time");
            result.kernel_seconds.push_back(milliseconds / 1000.0);
            download();
            result.kernel_error = std::max(result.kernel_error, Comparison::check_gpu(initial, cpu_reference, output));

            warm(mode, kernel);
            const auto begin = Clock::now();
            upload(mode);
            advance(iterations, kernel);
            check(cudaDeviceSynchronize(), "end-to-end synchronization");
            download();
            const auto end = Clock::now();
            result.end_to_end_seconds.push_back(std::chrono::duration<double>(end - begin).count());
            result.end_to_end_error = std::max(result.end_to_end_error, Comparison::check_gpu(initial, cpu_reference, output));
        }
    }
    first.release(); second.release(); start.release(); stop.release();
    return results;
}
