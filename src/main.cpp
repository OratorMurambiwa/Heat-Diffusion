#include <iostream>
#include <string>
#include <mpi.h>
#include "config.hpp"
#include "grid3d.hpp"
#include "solver.hpp"
#include "vtk_writer.hpp"

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "Starting SLM Thermal Simulation..." << std::endl;
    }

    // Initialize double buffer grids for time-stepping
    Grid3D grid_current(Config::NX_GLOBAL, Config::NY_GLOBAL, Config::NZ_GLOBAL, Config::AMBIENT_TEMP);
    Grid3D grid_next(Config::NX_GLOBAL, Config::NY_GLOBAL, Config::NZ_GLOBAL, Config::AMBIENT_TEMP);

    // Main time loop
    for (int step = 0; step < Config::TOTAL_STEPS; ++step) {
        
        // Move laser across the X axis
        int laser_x = (step % (Config::NX_GLOBAL - 2)) + 1;
        int laser_y = Config::NY_GLOBAL / 2;

        // Compute heat diffusion
        Solver::compute_diffusion(grid_current, grid_next);

        // Apply laser heat
        Solver::apply_laser(grid_next, laser_x, laser_y);

        // Swap buffers (O(1) pointer swap)
        grid_current.data.swap(grid_next.data);

        // Export data for ParaView every 100 steps
        if (step % 100 == 0) {
            std::string filename = "output_step_" + std::to_string(step) + ".vtk";
            VTKWriter::write(grid_current, filename);
        }

        if (rank == 0 && step % 100 == 0) {
            std::cout << "Step " << step << " / " << Config::TOTAL_STEPS << " complete." << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "Simulation finished successfully!" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
