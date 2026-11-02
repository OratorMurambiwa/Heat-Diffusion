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
