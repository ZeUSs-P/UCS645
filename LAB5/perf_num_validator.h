#ifndef PERFECT_WORKER_H
#define PERFECT_WORKER_H

#include <mpi.h>

/*
 * Determines if a given integer is a perfect number.
 * Returns the positive number if true, otherwise returns the negative of the number.
 */
int test_perfect(int candidate_num);

#endif
