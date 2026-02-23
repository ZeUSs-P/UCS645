#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get total number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get rank of current process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get processor name
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print hello message
    std::cout << "Hello from processor " 
              << processor_name 
              << ", rank " << world_rank 
              << " out of " << world_size 
              << " processors" << std::endl;

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
