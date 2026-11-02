#pragma once

#include "grid3d.hpp"

namespace CudaSolver {
// Advance a standalone FP64 field, preserving every outer plane exactly.
// Uses the current CUDA device. Does not perform MPI exchange or laser heating.
// Throws on invalid arguments or CUDA errors; zero steps returns a host copy.
Grid3D diffuse(const Grid3D& initial, int steps);
}
