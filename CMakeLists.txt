cmake_minimum_required(VERSION 3.10)
project(Heat-Diffusion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimization and warnings.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(simulator
    src/main.cpp
    src/grid3d.cpp
    src/mpi_halo.cpp
    src/solver.cpp
    src/vtk_writer.cpp
)

target_link_libraries(simulator PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

# Enable validation through CTest.
include(CTest)

if(BUILD_TESTING)
    add_executable(solver_validation
        tests/solver_validation.cpp
        src/grid3d.cpp
        src/solver.cpp
    )

    target_link_libraries(solver_validation PRIVATE
        OpenMP::OpenMP_CXX
    )

    add_test(
        NAME solver_validation
        COMMAND solver_validation
    )

    set_tests_properties(solver_validation PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2"
    )
endif()

# Standalone CPU stencil benchmark (no MPI communication or output).
add_executable(cpu_stencil_benchmark
    benchmarks/cpu_stencil.cpp
    src/grid3d.cpp
    src/solver.cpp
)
target_link_libraries(cpu_stencil_benchmark PRIVATE OpenMP::OpenMP_CXX)

# Capture the benchmark compile configuration for CSV reproducibility.
string(TOUPPER "${CMAKE_BUILD_TYPE}" CPU_BENCHMARK_CONFIG)
set(CPU_BENCHMARK_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CPU_BENCHMARK_CONFIG}} -std=gnu++${CMAKE_CXX_STANDARD} ${OpenMP_CXX_FLAGS}")
configure_file(benchmarks/cpu_benchmark_build.hpp.in
    generated/cpu_benchmark_build.hpp @ONLY)
target_include_directories(cpu_stencil_benchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Same-process CPU/GPU comparison; --cpu-only also works without a toolkit.
add_executable(cpu_cuda_benchmark
    benchmarks/cpu_cuda_benchmark.cpp src/grid3d.cpp src/solver.cpp)
target_link_libraries(cpu_cuda_benchmark PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(cpu_cuda_benchmark PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Keep CUDA entirely opt-in so CPU-only configurations need no CUDA toolkit.
option(ENABLE_CUDA "Build the standalone FP64 CUDA stencil and validation" OFF)
if(ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "ENABLE_CUDA requires CMake 3.18 or newer")
    endif()
    cmake_policy(SET CMP0104 NEW)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "ENABLE_CUDA=ON requires a CUDA compiler; load the CUDA toolkit or use ENABLE_CUDA=OFF")
    endif()
    enable_language(CUDA)

    target_sources(cpu_cuda_benchmark PRIVATE benchmarks/cuda_timing.cu)
    target_include_directories(cpu_cuda_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(cpu_cuda_benchmark PRIVATE HEAT_BENCH_HAS_CUDA=1)
    set_target_properties(cpu_cuda_benchmark PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

    add_library(cuda_stencil STATIC src/cuda_solver.cu src/grid3d.cpp)
    target_include_directories(cuda_stencil PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set_target_properties(cuda_stencil PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    if(BUILD_TESTING)
        add_executable(cuda_solver_validation tests/cuda_solver_validation.cpp src/solver.cpp)
        target_link_libraries(cuda_solver_validation PRIVATE cuda_stencil OpenMP::OpenMP_CXX)
        add_test(NAME cuda_solver_validation COMMAND cuda_solver_validation)
        set_tests_properties(cuda_solver_validation PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=2"
            LABELS "cuda"
        )

        add_executable(cuda_register_validation
            tests/cuda_register_validation.cpp benchmarks/cuda_timing.cu
            src/grid3d.cpp src/solver.cpp)
        target_include_directories(cuda_register_validation PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks" "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(cuda_register_validation PRIVATE OpenMP::OpenMP_CXX)
        set_target_properties(cuda_register_validation PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME cuda_register_validation COMMAND cuda_register_validation)
        set_tests_properties(cuda_register_validation PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2" LABELS "cuda")
    endif()
endif()

set(COMPARISON_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${CPU_BENCHMARK_CONFIG}} -std=c++17")
configure_file(benchmarks/comparison_build.hpp.in generated/comparison_build.hpp @ONLY)
