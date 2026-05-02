#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "vec_dot_math.h"

// 500 Million total elements across all nodes
#define TOTAL_ELEMENTS 500000000LL

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    long long local_n = TOTAL_ELEMENTS / num_procs;
    double multiplier = 0.0;
    
    // Time tracking
    double start_time, end_time, total_time;
    double max_total_time;

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Step 1: Broadcast the multiplier
    if (my_rank == 0) {
        multiplier = 3.14159; // Arbitrary scalar
    }
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 2 & 3: Local Generation & Compute
    double local_dot = compute_local_dot(local_n, multiplier);
    if (local_dot < 0.0) {
        printf("Rank %d failed to allocate memory for %lld elements\n", my_rank, local_n);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Step 4: Reduce to get the final sum
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    total_time = end_time - start_time;

    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("Processes: %d\n", num_procs);
        printf("Global Dot Product: %f\n", global_dot);
        printf("Max Total Time: %f s\n\n", max_total_time);
    }

    MPI_Finalize();
    return 0;
}
