#pragma once
#include "grid3d.hpp"

class MPIHalo {
public:
    // Swaps ghost cell layers with neighboring MPI ranks
    static void exchange(Grid3D& grid, int rank, int size);
};
