#include "cuda_solver.hpp"
#include "config.hpp"

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

__global__ void diffusion(const double* current, double* next,
                          int nx, int ny, size_t interior_count,
                          double cx, double cy, double cz) {
    const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= interior_count) return;

    // X is contiguous, matching Grid3D. Boundary cells are never written.
    const size_t interior_x = static_cast<size_t>(nx - 2);
    const size_t interior_y = static_cast<size_t>(ny - 2);
    const size_t x = 1 + cell % interior_x;
    const size_t y = 1 + (cell / interior_x) % interior_y;
    const size_t z = 1 + cell / (interior_x * interior_y);
    const size_t plane = static_cast<size_t>(nx) * ny;
    const size_t idx = z * plane + y * nx + x;
    const double t_center = current[idx];
    const double t_x = current[idx + 1] - 2.0 * t_center + current[idx - 1];
    const double t_y = current[idx + nx] - 2.0 * t_center + current[idx - nx];
    const double t_z = current[idx + plane] - 2.0 * t_center + current[idx - plane];
    next[idx] = t_center + cx * t_x + cy * t_y + cz * t_z;
}
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
