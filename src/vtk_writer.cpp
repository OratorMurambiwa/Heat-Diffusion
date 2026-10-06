#include "vtk_writer.hpp"
#include <fstream>
#include <iostream>

void VTKWriter::write(const Grid3D& grid, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    // VTK legacy format header
    file << "# vtk DataFile Version 3.0\n";
    file << "SLM Thermal Simulation Data\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << grid.nx << " " << grid.ny << " " << grid.nz << "\n";
    file << "ORIGIN 0 0 0\n";
    
    // Using 1 unit spacing for visualization (actual physical scale is in config.hpp)
    file << "SPACING 1 1 1\n"; 
    
    file << "POINT_DATA " << grid.nx * grid.ny * grid.nz << "\n";
    file << "SCALARS Temperature double 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write grid data
    for (int z = 0; z < grid.nz; ++z) {
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                size_t idx = grid.get_index(x, y, z);
                file << grid.data[idx] << "\n";
            }
        }
    }

    file.close();
}
