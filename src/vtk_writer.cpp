#include "vtk_writer.hpp"
#include "config.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

void VTKWriter::write(
    const Grid3D& grid,
    const std::string& filename,
    int z_offset
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr
            << "Error: Could not open "
            << filename << " for writing.\n";
        return;
    }

    // Exclude the two MPI ghost layers.
    const int active_nz = grid.nz - 2;
    const size_t point_count =
        static_cast<size_t>(grid.nx) * grid.ny * active_nz;

    file << std::setprecision(17);

    file << "# vtk DataFile Version 3.0\n";
    file << "SLM Thermal Simulation Data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    file << "DIMENSIONS "
         << grid.nx << " "
         << grid.ny << " "
         << active_nz << "\n";

    // Position this slab in the global grid, in metres.
    file << "ORIGIN 0 0 "
         << z_offset * Config::DZ << "\n";

    file << "SPACING "
         << Config::DX << " "
         << Config::DY << " "
         << Config::DZ << "\n";

    file << "POINT_DATA " << point_count << "\n";
    file << "SCALARS Temperature double 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write only real layers; skip bottom and top ghosts.
    for (int z = 1; z < grid.nz - 1; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                file << grid.data[grid.get_index(x, y, z)]
                     << "\n";
            }
        }
    }

    file.close();

    if (!file) {
        std::cerr
            << "Error: Failed to finish writing "
            << filename << "\n";
    }
}