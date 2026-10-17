
#pragma once
#include "grid3d.hpp"

enum class HaloPrecision {
    FP64,
    FP32
};

class MPIHalo {
public:
    static void exchange(
        Grid3D& grid,
        int rank,
        int size,
        HaloPrecision precision = HaloPrecision::FP64
    );
};

