
#include "mpi_halo.hpp"
#include <mpi.h>
#include <algorithm>
#include <vector>

void MPIHalo::exchange(
    Grid3D& grid, int rank, int size, HaloPrecision precision
) {
    const int n = grid.nx * grid.ny;

    double* send_bottom = &grid.data[grid.get_index(0, 0, 1)];
    double* send_top = &grid.data[grid.get_index(0, 0, grid.nz - 2)];
    double* recv_bottom = &grid.data[grid.get_index(0, 0, 0)];
    double* recv_top = &grid.data[grid.get_index(0, 0, grid.nz - 1)];

    MPI_Request requests[4];
    int count = 0;

    if (precision == HaloPrecision::FP64) {
        if (rank < size - 1) {
            MPI_Isend(send_top, n, MPI_DOUBLE, rank+1, 0,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Irecv(recv_top, n, MPI_DOUBLE, rank+1, 1,
                      MPI_COMM_WORLD, &requests[count++]);
        }
        if (rank > 0) {
            MPI_Isend(send_bottom, n, MPI_DOUBLE, rank-1, 1,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Irecv(recv_bottom, n, MPI_DOUBLE, rank-1, 0,
                      MPI_COMM_WORLD, &requests[count++]);
        }
    } else {
        // Convert outgoing FP64 temperatures into FP32.
        std::vector<float> send_top_f, send_bottom_f;
        std::vector<float> recv_top_f, recv_bottom_f;

        if (rank < size - 1) {
            send_top_f.assign(send_top, send_top + n);
            recv_top_f.resize(n);

            MPI_Isend(send_top_f.data(), n, MPI_FLOAT, rank+1, 0,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Irecv(recv_top_f.data(), n, MPI_FLOAT, rank+1, 1,
                      MPI_COMM_WORLD, &requests[count++]);
        }

        if (rank > 0) {
            send_bottom_f.assign(send_bottom, send_bottom + n);
            recv_bottom_f.resize(n);

            MPI_Isend(send_bottom_f.data(), n, MPI_FLOAT, rank-1, 1,
                      MPI_COMM_WORLD, &requests[count++]);
            MPI_Irecv(recv_bottom_f.data(), n, MPI_FLOAT, rank-1, 0,
                      MPI_COMM_WORLD, &requests[count++]);
        }

        if (count > 0) {
            MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
        }

        // Convert received data back into FP64.
        if (rank < size - 1)
            std::copy(recv_top_f.begin(), recv_top_f.end(), recv_top);

        if (rank > 0)
            std::copy(recv_bottom_f.begin(), recv_bottom_f.end(),
                      recv_bottom);

        return;
    }

    if (count > 0) {
        MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    }
}

