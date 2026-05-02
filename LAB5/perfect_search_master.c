#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "perf_num_validator.h"

#define LIMIT 10000
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
        // --- MASTER NODE ---
        int next_value = 2;
        int running_slaves = total_nodes - 1;
        int slave_response;
        MPI_Status comm_status;
        
        double clock_start = MPI_Wtime();
        double comm_dur_total = 0.0;
        int eval_count = 0;

        printf("[MASTER] Initiating Perfect Number Search up to %d\n", LIMIT);

        while (running_slaves > 0) {
            double comm_tick = MPI_Wtime();
            MPI_Recv(&slave_response, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &comm_status);
            comm_dur_total += (MPI_Wtime() - comm_tick);
            
            int src_rank = comm_status.MPI_SOURCE;

            if (slave_response > 0) {
                printf(">>> Perfect Number Discovered: %d\n", slave_response);
            }
            if (slave_response != 0) {
                eval_count++;
            }
            
            comm_tick = MPI_Wtime();
            if (next_value <= LIMIT) {
                MPI_Send(&next_value, 1, MPI_INT, src_rank, MSG_TASK, MPI_COMM_WORLD);
                next_value++;
            } else {
                int kill_signal = 0;
                MPI_Send(&kill_signal, 1, MPI_INT, src_rank, MSG_HALT, MPI_COMM_WORLD);
                running_slaves--;
            }
            comm_dur_total += (MPI_Wtime() - comm_tick);
        }
        
        double clock_end = MPI_Wtime();
        double total_dur = clock_end - clock_start;
        double comp_dur = total_dur - comm_dur_total;
        
        printf("\n--- EXECUTION METRICS ---\n");
        printf(" Total Runtime    : %.4f s\n", total_dur);
        printf(" Total Comm Time  : %.4f s\n", comm_dur_total);
        printf(" Total Comp Time  : %.4f s\n", comp_dur);
        printf(" Comm Percentage  : %.2f %%\n", (comm_dur_total / total_dur) * 100.0);
        printf(" Items Evaluated  : %d\n", eval_count);
        printf(" Avg Time/Item    : %.6f s\n", total_dur / eval_count);
        
    } else {
        // --- SLAVE NODE ---
        int task_val;
        int return_val = 0; 
        MPI_Status comm_status;
        
        double clock_start = MPI_Wtime();
        double comm_dur_total = 0.0;
        double comp_dur_total = 0.0;
        int items_checked = 0;

        for (;;) {
            double comm_tick = MPI_Wtime();
            MPI_Send(&return_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            comm_dur_total += (MPI_Wtime() - comm_tick);
            
            comm_tick = MPI_Wtime();
            MPI_Recv(&task_val, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &comm_status);
            comm_dur_total += (MPI_Wtime() - comm_tick);
            
            if (comm_status.MPI_TAG == MSG_HALT) {
                break;
            }

            double comp_tick = MPI_Wtime();
            return_val = test_perfect(task_val);
            comp_dur_total += (MPI_Wtime() - comp_tick);
            items_checked++;
        }
        
        double clock_end = MPI_Wtime();
        double total_dur = clock_end - clock_start;
        
        printf("Worker %d -> Total: %.4fs | Comm: %.4fs (%.1f%%) | Comp: %.4fs (%.1f%%) | Checked: %d\n", 
               my_id, total_dur, comm_dur_total, 
               (comm_dur_total / total_dur) * 100.0, comp_dur_total, 
               (comp_dur_total / total_dur) * 100.0, items_checked);
    }

    MPI_Finalize();
    return 0;
}
