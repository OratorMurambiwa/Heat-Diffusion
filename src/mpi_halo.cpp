#include "mpi_halo.hpp"
#include <mpi.h>

void MPIHalo::exchange(Grid3D& grid, int rank, int size) {
    // Slicing the Z-axis. One "slice" is a whole X-Y plane.
    int slice_size = grid.nx * grid.ny;

    // Pointers to the actual outermost layers we computed
    double* send_bottom = &grid.data[grid.get_index(0, 0, 1)]; 
    double* send_top = &grid.data[grid.get_index(0, 0, grid.nz - 2)];

    // Pointers to the empty ghost padding where incoming data goes
    double* recv_bottom = &grid.data[grid.get_index(0, 0, 0)]; 
    double* recv_top = &grid.data[grid.get_index(0, 0, grid.nz - 1)]; 

    MPI_Request reqs[4];
    int req_count = 0;

    // If we aren't the top-most processor, swap data with the guy above us
    if (rank < size - 1) {
        MPI_Isend(send_top, slice_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Irecv(recv_top, slice_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    // If we aren't the bottom-most processor, swap data with the guy below us
    if (rank > 0) {
        MPI_Isend(send_bottom, slice_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Irecv(recv_bottom, slice_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    // Wait for all the async network traffic to finish before letting the physics loop continue
    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }
}
