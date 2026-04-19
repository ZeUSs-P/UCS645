#include "broadcast_utils.h"

void my_bcast(double* buffer, int count, int root_id, MPI_Comm communicator) {
    int current_proc, num_procs;
    MPI_Comm_rank(communicator, &current_proc);
    MPI_Comm_size(communicator, &num_procs);

    if (current_proc == root_id) {
        // The root process loops over all processes and sends data to them
        int target = 0;
        while (target < num_procs) {
            if (target != root_id) {
                MPI_Send(buffer, count, MPI_DOUBLE, target, 0, communicator);
            }
            target++;
        }
    } else {
        // Non-root processes wait to receive the broadcasted data
        MPI_Recv(buffer, count, MPI_DOUBLE, root_id, 0, communicator, MPI_STATUS_IGNORE);
    }
}
