#include "cuda_solver.hpp"
#include "config.hpp"
#include "cuda_stencil_kernel.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) {
        check(cudaMalloc(reinterpret_cast<void**>(&data_), bytes), "cudaMalloc");
    }
    // Cleanup during exception unwinding is best effort and must not throw.
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    double* data() const { return data_; }
    void release() {
        double* pointer = data_;
        data_ = nullptr;
        check(cudaFree(pointer), "cudaFree");
    }
private:
    double* data_ = nullptr;
};
}

Grid3D CudaSolver::diffuse(const Grid3D& initial, int steps) {
    if (steps < 0 || initial.nx < 3 || initial.ny < 3 || initial.nz < 3) {
        throw std::invalid_argument("CUDA diffusion requires steps >= 0 and dimensions >= 3");
    }
    // Keep inputs within the existing CPU Grid3D indexing range as well.
    if (initial.nx > std::numeric_limits<int>::max() / initial.ny) {
        throw std::invalid_argument("nx * ny exceeds Grid3D indexing capacity");
    }
    const size_t plane = static_cast<size_t>(initial.nx) * initial.ny;
    if (plane > std::numeric_limits<size_t>::max() / sizeof(double) / initial.nz) {
        throw std::invalid_argument("CUDA grid byte size overflows size_t");
    }
    const size_t count = plane * initial.nz;
    if (initial.data.size() != count) {
        throw std::invalid_argument("Grid dimensions do not match data size");
    }
    Grid3D result = initial;
    if (steps == 0) return result;

    const size_t interior_count = static_cast<size_t>(initial.nx - 2) *
        (initial.ny - 2) * (initial.nz - 2);
    constexpr unsigned int threads = 256;
    const size_t blocks = (interior_count + threads - 1) / threads;
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (blocks > static_cast<size_t>(properties.maxGridSize[0])) {
        throw std::invalid_argument("Grid exceeds the naive one-dimensional CUDA launch limit");
    }

    const size_t bytes = count * sizeof(double);
    DeviceBuffer first(bytes);
    DeviceBuffer second(bytes);
    double* current = first.data();
    double* next = second.data();
    // Both buffers need identical boundaries because they exchange roles.
    check(cudaMemcpy(current, initial.data.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy first H2D");
    check(cudaMemcpy(next, initial.data.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy second H2D");
    const double cx = Config::ALPHA * Config::DT / (Config::DX * Config::DX);
    const double cy = Config::ALPHA * Config::DT / (Config::DY * Config::DY);
    const double cz = Config::ALPHA * Config::DT / (Config::DZ * Config::DZ);
    for (int step = 0; step < steps; ++step) {
        diffusion<<<static_cast<unsigned int>(blocks), threads>>>(
            current, next, initial.nx, initial.ny, interior_count, cx, cy, cz);
        check(cudaGetLastError(), "diffusion kernel launch");
        std::swap(current, next);
    }
    // Default-stream ordering keeps dependent steps sequential. Synchronize
    // before returning so asynchronous execution errors are reported here.
    check(cudaDeviceSynchronize(), "diffusion kernel execution");
    check(cudaMemcpy(result.data.data(), current, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    first.release();
    second.release();
    return result;
}
