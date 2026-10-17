
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>
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

    // Select MPI halo communication precision.
    // Default to FP64 if HALO_PRECISION is not set.
    HaloPrecision halo_precision = HaloPrecision::FP64;

    const char* env = std::getenv("HALO_PRECISION");

    if (env != nullptr) {
        const std::string mode(env);

        if (mode == "fp32") {
            halo_precision = HaloPrecision::FP32;
        } else if (mode == "fp64") {
            halo_precision = HaloPrecision::FP64;
        } else {
            if (rank == 0) {
                std::cerr
                    << "Invalid HALO_PRECISION value: "
                    << mode << "\n"
                    << "Supported values: fp32, fp64\n";
            }

            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    if (size > Config::NZ_GLOBAL) {
        if (rank == 0) {
            std::cerr
                << "Too many MPI processes for the Z grid.\n";
        }

        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::cout
            << "Starting Distributed SLM Thermal Simulation on "
            << size << " MPI processes\n";

        std::cout
            << "MPI Halo Communication Precision: "
            << (halo_precision == HaloPrecision::FP32
                    ? "FP32" : "FP64")
            << "\n";

        std::cout
            << "Internal Solver Precision: FP64\n";
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

    // Main heat-diffusion simulation loop.
    for (int step = 0; step < Config::TOTAL_STEPS; ++step) {

        // Exchange halo data using the selected precision.
        // The solver itself continues using FP64.
        MPIHalo::exchange(
            grid_current,
            rank,
            size,
            halo_precision
        );

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

        // Compute diffusion using FP64.
        Solver::compute_diffusion(
            grid_current,
            grid_next
        );

        // Only the final rank owns the top surface.
        if (rank == size - 1) {
            Solver::apply_laser(
                grid_next,
                laser_x,
                laser_y
            );
        }

        // Swap current and next temperature grids.
        grid_current.data.swap(grid_next.data);

        // Write VTK temperature output every 100 steps.
        if (step % 100 == 0) {
            const std::string filename =
                "output_step_" + std::to_string(step)
                + "_rank_" + std::to_string(rank)
                + ".vtk";

            VTKWriter::write(
                grid_current,
                filename,
                z_offset
            );
        }

        if (rank == 0 && step % 100 == 0) {
            std::cout
                << "Step " << step << " / "
                << Config::TOTAL_STEPS
                << " complete.\n";
        }
    }

    if (rank == 0) {
        std::cout
            << "Simulation finished successfully!\n";
    }

    MPI_Finalize();
    return 0;
}

