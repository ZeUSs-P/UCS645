#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define N (1 << 16)  // 2^16 = 65536 elements

int main() {
    // Allocate memory for vectors
    double *X = (double*)malloc(N * sizeof(double));
    double *Y = (double*)malloc(N * sizeof(double));
    double scalar = 2.5;
    
    // Vector initialization
    for(int idx = 0; idx < N; idx++) {
        X[idx] = idx * 0.5;
        Y[idx] = (N - idx) * 0.3;
    }
    
    printf("========================================\n");
    printf("DAXPY Operation: X = %.1f*X + Y\n", scalar);
    printf("Vector Size: %d elements\n", N);
    printf("========================================\n");
    printf("Thread Count | Execution Time (ms) | Speedup Factor\n");
    printf("--------------------------------------------------------\n");
    
    double serial_exec_time = 0.0;
    double best_speedup = 0.0;
    int optimal_threads = 1;
    
    // Test various thread configurations
    for(int thread_num = 1; thread_num <= 16; thread_num++) {
        omp_set_num_threads(thread_num);
        
        // Perform warm-up execution
        #pragma omp parallel for
        for(int idx = 0; idx < N; idx++) {
            X[idx] = scalar * X[idx] + Y[idx];
        }
        
        // Reset vector X
        for(int idx = 0; idx < N; idx++) {
            X[idx] = idx * 0.5;
        }
        
        // Multiple iterations for accurate timing
        double cumulative_time = 0.0;
        int iterations = 100;
        
        for(int iter = 0; iter < iterations; iter++) {
            double time_start = omp_get_wtime();
            
            // DAXPY kernel
            #pragma omp parallel for schedule(static)
            for(int idx = 0; idx < N; idx++) {
                X[idx] = scalar * X[idx] + Y[idx];
            }
            
            double time_end = omp_get_wtime();
            cumulative_time += (time_end - time_start);
            
            // Restore X for next iteration
            for(int idx = 0; idx < N; idx++) {
                X[idx] = idx * 0.5;
            }
        }
        
        double mean_time = (cumulative_time / iterations) * 1000.0; // Convert to ms
        
        // Store baseline time for single thread
        if(thread_num == 1) {
            serial_exec_time = mean_time;
        }
        
        double speedup_ratio = serial_exec_time / mean_time;
        
        printf("     %2d      |      %.6f       |     %.4f\n", 
               thread_num, mean_time, speedup_ratio);
        
        // Track best configuration
        if(speedup_ratio > best_speedup) {
            best_speedup = speedup_ratio;
            optimal_threads = thread_num;
        }
    }
    
    printf("--------------------------------------------------------\n");
    printf("\n** Analysis Summary **\n");
    printf("Peak speedup: %.4fx with %d threads\n", best_speedup, optimal_threads);
    printf("========================================\n");
    
    // Cleanup
    free(X);
    free(Y);
    
    return 0;
}
