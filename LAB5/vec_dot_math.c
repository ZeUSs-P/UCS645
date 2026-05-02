#include "vec_dot_math.h"
#include <stdlib.h>

double compute_local_dot(long long elements_count, double scale_factor) {
    double result_sum = 0.0;
    
    // Allocate memory for local vector portions
    double *array_x = (double*)malloc(elements_count * sizeof(double));
    double *array_y = (double*)malloc(elements_count * sizeof(double));
    
    if (!array_x || !array_y) {
        // Memory allocation failed
        if (array_x) free(array_x);
        if (array_y) free(array_y);
        return -1.0;
    }

    // Populate the vectors locally
    long long idx = 0;
    while (idx < elements_count) {
        array_x[idx] = 1.0;
        array_y[idx] = scale_factor * 2.0;
        idx++;
    }

    // Calculate the partial dot product
    for (long long j = 0; j < elements_count; ++j) {
        result_sum += (array_x[j] * array_y[j]);
    }

    // Cleanup
    free(array_x);
    free(array_y);
    
    return result_sum;
}
