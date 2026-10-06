#pragma once
#include "grid3d.hpp"
#include <string>

class VTKWriter {
public:
    // Exports grid data to a VTK file for ParaView
    static void write(const Grid3D& grid, const std::string& filename);
};
