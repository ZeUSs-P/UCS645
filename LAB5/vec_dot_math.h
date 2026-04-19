#ifndef DOT_PRODUCT_H
#define DOT_PRODUCT_H

#include <mpi.h>

/*
 * Evaluates the dot product for a local subset of two arrays.
 * The arrays are generated internally to conserve memory footprint.
 */
double compute_local_dot(long long elements_count, double scale_factor);

#endif
