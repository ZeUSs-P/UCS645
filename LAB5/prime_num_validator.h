#ifndef PRIME_WORKER_H
#define PRIME_WORKER_H

#include <mpi.h>

/*
 * Validates if the given integer is a prime number.
 * Returns the number itself if it is prime, otherwise returns its negative form.
 */
int test_prime(int value);

#endif
