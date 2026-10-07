#include "grid3d.hpp"

Grid3D::Grid3D(int nx_local, int ny_local, int nz_local, double initial_temp) 
    : nx(nx_local), ny(ny_local), nz(nz_local) {
    
    size_t total_size = static_cast<size_t>(nx) * ny * nz;
    data.assign(total_size, initial_temp);
}
