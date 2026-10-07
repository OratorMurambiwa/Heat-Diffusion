cmake_minimum_required(VERSION 3.10)
project(Heat-Diffusion CXX)

# Enforce modern C++ standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Apply aggressive HPC performance optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# Find required parallel libraries
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Include the header files directory
include_directories(include)

# Define the source files
set(SOURCES
    src/main.cpp
    src/grid3d.cpp
    src/mpi_halo.cpp
    src/solver.cpp
    src/vtk_writer.cpp
)

# Create the executable binary
add_executable(simulator ${SOURCES})

# Link MPI and OpenMP to the executable
target_link_libraries(simulator PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
