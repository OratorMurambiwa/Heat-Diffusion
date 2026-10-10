#include <iostream>
#include <string>
#include <cmath>
#include <mpi.h>

#include "config.hpp"
#include "grid3d.hpp"
#include "solver.hpp"
#include "vtk_writer.hpp"
#include "mpi_halo.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > Config::NZ_GLOBAL) {
        if (rank == 0) {
            std::cerr << "Too many MPI processes for the Z grid.\n";
        }

        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout
            << "Starting Distributed SLM Thermal Simulation on "
            << size << " MPI processes\n";
    }

    // Divide the global grid into Z slabs.
    int nz_local = Config::NZ_GLOBAL / size;
    const int remainder = Config::NZ_GLOBAL % size;

    if (rank < remainder) {
        ++nz_local;
    }

    // Global starting Z index of this rank's real layers.
    const int z_offset =
        rank * (Config::NZ_GLOBAL / size)
        + (rank < remainder ? rank : remainder);

    // One ghost layer on each side of the local slab.
    const int nz_with_ghosts = nz_local + 2;

    Grid3D grid_current(
        Config::NX_GLOBAL,
        Config::NY_GLOBAL,
        nz_with_ghosts,
        Config::AMBIENT_TEMP
    );

    Grid3D grid_next(
        Config::NX_GLOBAL,
        Config::NY_GLOBAL,
        nz_with_ghosts,
        Config::AMBIENT_TEMP
    );

    for (int step = 0; step < Config::TOTAL_STEPS; ++step) {
        MPIHalo::exchange(grid_current, rank, size);

        // Move the beam using physical speed and elapsed time.
        // Start near the center and wrap along X.
        const double scan_length =
            (Config::NX_GLOBAL - 2) * Config::DX;

        const double laser_x = 1.0 + std::fmod(
            0.5 * scan_length
                + Config::SCAN_SPEED * step * Config::DT,
            scan_length
        ) / Config::DX;

        const double laser_y = Config::NY_GLOBAL / 2.0;

        Solver::compute_diffusion(grid_current, grid_next);

        // Only the final rank owns the top surface.
        if (rank == size - 1) {
            Solver::apply_laser(grid_next, laser_x, laser_y);
        }

        grid_current.data.swap(grid_next.data);

        if (step % 100 == 0) {
            const std::string filename =
                "output_step_" + std::to_string(step)
                + "_rank_" + std::to_string(rank) + ".vtk";

            VTKWriter::write(grid_current, filename, z_offset);
        }

        if (rank == 0 && step % 100 == 0) {
            std::cout
                << "Step " << step << " / "
                << Config::TOTAL_STEPS << " complete.\n";
        }
    }

    if (rank == 0) {
        std::cout << "Simulation finished successfully!\n";
    }

    MPI_Finalize();
    return 0;
}