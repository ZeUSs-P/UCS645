#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 65536 // 2^16

int main(int argc, char** argv) {
    int my_rank, num_procs;
    const double scalar_A = 2.5;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (VECTOR_SIZE % num_procs != 0) {
        if (my_rank == 0) {
            printf("[ERROR] Vector size (%d) must be perfectly divisible by the number of processes (%d).\n", VECTOR_SIZE, num_procs);
        }
        MPI_Finalize();
        return 1;
    }

    int chunk_size = VECTOR_SIZE / num_procs;
    double *global_vec_x = NULL, *global_vec_y = NULL, *seq_vec_x = NULL;
    double *chunk_x = (double*)malloc(chunk_size * sizeof(double));
    double *chunk_y = (double*)malloc(chunk_size * sizeof(double));

    double seq_exec_time = 0.0;

    if (my_rank == 0) {
        global_vec_x = (double*)malloc(VECTOR_SIZE * sizeof(double));
        global_vec_y = (double*)malloc(VECTOR_SIZE * sizeof(double));
        seq_vec_x = (double*)malloc(VECTOR_SIZE * sizeof(double));

        int idx = 0;
        while (idx < VECTOR_SIZE) {
            global_vec_x[idx] = 1.0;
            global_vec_y[idx] = 2.0;
            seq_vec_x[idx] = 1.0;
            idx++;
        }

        // Sequential calculation for baseline comparison
        double start_seq = MPI_Wtime();
        for (int i = 0; i < VECTOR_SIZE; i++) {
            seq_vec_x[i] = (scalar_A * seq_vec_x[i]) + global_vec_y[i];
        }
        seq_exec_time = MPI_Wtime() - start_seq;
    }

    MPI_Barrier(MPI_COMM_WORLD); 
    
    double t_start, t_end, comm_tick, local_comm_acc = 0.0;

    // Parallel Execution
    t_start = MPI_Wtime();

    comm_tick = MPI_Wtime();
    MPI_Scatter(global_vec_x, chunk_size, MPI_DOUBLE, chunk_x, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(global_vec_y, chunk_size, MPI_DOUBLE, chunk_y, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm_acc += (MPI_Wtime() - comm_tick);

    for (int k = 0; k < chunk_size; k++) {
        chunk_x[k] = (scalar_A * chunk_x[k]) + chunk_y[k];
    }

    comm_tick = MPI_Wtime();
    MPI_Gather(chunk_x, chunk_size, MPI_DOUBLE, global_vec_x, chunk_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    local_comm_acc += (MPI_Wtime() - comm_tick);

    t_end = MPI_Wtime();
    double local_exec_time = t_end - t_start;

    double max_exec_time, max_comm_time;
    MPI_Reduce(&local_exec_time, &max_exec_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comm_acc, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        double calc_speedup = seq_exec_time / max_exec_time;
        double calc_efficiency = (calc_speedup / num_procs) * 100.0;
        double comm_overhead = (max_comm_time / max_exec_time) * 100.0;

        printf("\n*** MPI DAXPY PERFORMANCE REPORT ***\n");
        printf("  Vector Size : %d\n", VECTOR_SIZE);
        printf("  Processors  : %d\n", num_procs);
        printf("**************************************\n");
        printf("  Seq Time    : %f seconds\n", seq_exec_time);
        printf("  Par Time    : %f seconds\n", max_exec_time);
        printf("  Comm Time   : %f seconds\n", max_comm_time);
        printf("**************************************\n");
        printf("  Speedup     : %.2fx\n", calc_speedup);
        printf("  Efficiency  : %.2f%%\n", calc_efficiency);
        printf("  Comm Load   : %.2f%%\n", comm_overhead);
        printf("**************************************\n\n");

        free(global_vec_x);
        free(global_vec_y);
        free(seq_vec_x);
    }

    free(chunk_x);
    free(chunk_y);
    MPI_Finalize();
    return 0;
}
