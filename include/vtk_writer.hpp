#pragma once

#include "grid3d.hpp"
#include <string>

class VTKWriter {
public:
    // Export active layers using their global Z position.
    static void write(
        const Grid3D& grid,
        const std::string& filename,
        int z_offset
    );
};