#ifndef BCAST_FUNCTIONS_H
#define BCAST_FUNCTIONS_H

#include <mpi.h>

/*
 * Distributes an array of doubles from a root process to all other processes
 * using basic MPI send and receive operations.
 */
void my_bcast(double* buffer, int count, int root_id, MPI_Comm communicator);

#endif
