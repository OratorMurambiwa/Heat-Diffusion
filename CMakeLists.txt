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