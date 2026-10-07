#include <iostream>
#include <string>
#include <mpi.h>
#include "config.hpp"
#include "grid3d.hpp"
#include "solver.hpp"
#include "vtk_writer.hpp"
#include "mpi_halo.hpp"

int main(int argc, char** argv) {
    // Initialize MPI network
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // My ID (e.g., 0, 1, 2, 3)
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Total processors

    if (rank == 0) {
        std::cout << "Starting Distributed SLM Thermal Simulation on " << size << " nodes" << std::endl;
    }

    // 1. Slice the global grid along the Z-axis
    int nz_local = Config::NZ_GLOBAL / size;
    int remainder = Config::NZ_GLOBAL % size;
    if (rank < remainder) {
        nz_local += 1; // Distribute any leftover layers evenly
    }

    // Add 2 extra layers for our Ghost Cells (Top and Bottom)
    int nz_with_ghosts = nz_local + 2;

    // Initialize this processor's specific chunk of the grid
    Grid3D grid_current(Config::NX_GLOBAL, Config::NY_GLOBAL, nz_with_ghosts, Config::AMBIENT_TEMP);
    Grid3D grid_next(Config::NX_GLOBAL, Config::NY_GLOBAL, nz_with_ghosts, Config::AMBIENT_TEMP);

    // Main time loop
    for (int step = 0; step < Config::TOTAL_STEPS; ++step) {
        
        // 2. Exchange Ghost Cells BEFORE doing the math so boundaries are accurate
        MPIHalo::exchange(grid_current, rank, size);

        int laser_x = (step % (Config::NX_GLOBAL - 2)) + 1;
        int laser_y = Config::NY_GLOBAL / 2;

        // Compute heat diffusion for this local chunk
        Solver::compute_diffusion(grid_current, grid_next);

        // 3. Only the processor owning the top of the powder bed fires the laser
        if (rank == size - 1) {
            Solver::apply_laser(grid_next, laser_x, laser_y);
        }

        // Swap buffers
        grid_current.data.swap(grid_next.data);

        // Export data every 100 steps
        if (step % 100 == 0) {
            // Include the rank ID in the filename so they don't overwrite each other
            std::string filename = "output_step_" + std::to_string(step) + "_rank_" + std::to_string(rank) + ".vtk";
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
