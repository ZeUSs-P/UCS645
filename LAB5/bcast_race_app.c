#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "broadcast_utils.h"

// 10 million doubles * 8 bytes = 80MB
#define NUM_ELEMENTS 10000000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *buffer = (double*)malloc(NUM_ELEMENTS * sizeof(double));
    if (!buffer) {
        if (my_rank == 0) printf("Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (my_rank == 0) {
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            buffer[i] = 1.0;
        }
    }

    // Warmup
    MPI_Bcast(buffer, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Time MyBcast (Linear)
    double start_my = MPI_Wtime();
    my_bcast(buffer, NUM_ELEMENTS, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_my = MPI_Wtime();

    // Time MPI_Bcast (Tree)
    double start_mpi = MPI_Wtime();
    MPI_Bcast(buffer, NUM_ELEMENTS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    double end_mpi = MPI_Wtime();

    if (my_rank == 0) {
        double my_time = end_my - start_my;
        double mpi_time = end_mpi - start_mpi;
        double gain = my_time / mpi_time;

        printf("Processes: %d\n", num_procs);
        printf("MyBcast Time: %f s\n", my_time);
        printf("MPI_Bcast Time: %f s\n", mpi_time);
        printf("Performance Gain: %.2fx\n\n", gain);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
