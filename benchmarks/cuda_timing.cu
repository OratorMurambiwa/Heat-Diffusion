#include "cuda_timing.hpp"
#include "comparison_common.hpp"
#include "cuda_stencil_kernel.cuh"
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
                            const std::vector<GpuInitialization>& modes) {
    if (modes.empty() || modes.size() > 2 ||
        (modes.size() == 2 && modes[0] == modes[1]))
        throw std::runtime_error("Expected one or two distinct GPU initialization modes");
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
        results.push_back(metadata);
        results.back().initialization = mode;
    }

    const size_t bytes = initial.data.size() * sizeof(double);
    const size_t cells = static_cast<size_t>(initial.nx - 2) * (initial.ny - 2) * (initial.nz - 2);
    constexpr unsigned int threads = 256;
    const size_t blocks = (cells + threads - 1) / threads;
    if (blocks > static_cast<size_t>(properties.maxGridSize[0]))
        throw std::runtime_error("Grid exceeds CUDA launch limit");
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
    auto advance = [&](int steps) {
        for (int step = 0; step < steps; ++step) {
            diffusion<<<static_cast<unsigned int>(blocks), threads>>>(
                current, next, initial.nx, initial.ny, cells, cx, cy, cz);
            check(cudaGetLastError(), "diffusion launch");
            std::swap(current, next);
        }
    };
    auto warm = [&](GpuInitialization mode) {
        upload(mode); advance(warmup);
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
            warm(mode);
            upload(mode);
            // Finish uploads before starting device-only timing (pageable host data).
            check(cudaDeviceSynchronize(), "pre-timing synchronization");
            check(cudaEventRecord(start.event), "start event record");
            advance(iterations);
            check(cudaEventRecord(stop.event), "stop event record");
            check(cudaEventSynchronize(stop.event), "stop event synchronization");
            float milliseconds = 0.0f;
            check(cudaEventElapsedTime(&milliseconds, start.event, stop.event), "event elapsed time");
            result.kernel_seconds.push_back(milliseconds / 1000.0);
            download();
            result.kernel_error = std::max(result.kernel_error, Comparison::check_gpu(initial, cpu_reference, output));

            warm(mode);
            const auto begin = Clock::now();
            upload(mode);
            advance(iterations);
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
