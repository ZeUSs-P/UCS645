#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "prime_num_validator.h"

#define LIMIT 100000
#define MSG_TASK 100
#define MSG_HALT 200

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int my_id, total_nodes;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_nodes);

    if (total_nodes < 2) {
        if (my_id == 0) {
            fprintf(stderr, "Need at least 2 nodes (1 Master, 1+ Slaves).\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (my_id == 0) {
        int next_value = 2;
        int running_slaves = total_nodes - 1;
        int slave_response;
        MPI_Status comm_status;
        
        double clock_start = MPI_Wtime();
        int prime_count = 0;

        while (running_slaves > 0) {
            MPI_Recv(&slave_response, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &comm_status);
            int src_rank = comm_status.MPI_SOURCE;

            if (slave_response > 0) {
                prime_count++;
            }
            
            if (next_value <= LIMIT) {
                MPI_Send(&next_value, 1, MPI_INT, src_rank, MSG_TASK, MPI_COMM_WORLD);
                next_value++;
            } else {
                int kill_signal = 0;
                MPI_Send(&kill_signal, 1, MPI_INT, src_rank, MSG_HALT, MPI_COMM_WORLD);
                running_slaves--;
            }
        }
        
        double clock_end = MPI_Wtime();
        double total_dur = clock_end - clock_start;
        
        printf("Processes: %d\n", total_nodes);
        printf("Primes Found: %d\n", prime_count);
        printf("Time (s): %f\n\n", total_dur);
        
    } else {
        int task_val;
        int return_val = 0; 
        MPI_Status comm_status;

        for (;;) {
            MPI_Send(&return_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&task_val, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &comm_status);
            
            if (comm_status.MPI_TAG == MSG_HALT) {
                break;
            }

            return_val = test_prime(task_val);
        }
    }

    MPI_Finalize();
    return 0;
}
