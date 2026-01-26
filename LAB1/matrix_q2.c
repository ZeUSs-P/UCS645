#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define MATRIX_DIM 1000
#define NUM_ITERATIONS 5
#define THREAD_LIMIT 32

// Matrices stored as 1D arrays
double *matrix_A, *matrix_B, *result_matrix, *baseline_result;

// Macro for 2D indexing in 1D array
#define INDEX(mat, row, col) (mat[(row) * MATRIX_DIM + (col)])

// Memory allocation helper
double* create_matrix() {
    double *mat = (double*)malloc(MATRIX_DIM * MATRIX_DIM * sizeof(double));
    if (!mat) {
        fprintf(stderr, "ERROR: Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

// Initialize input matrices with values
void setup_matrices() {
    for (int row = 0; row < MATRIX_DIM; row++) {
        for (int col = 0; col < MATRIX_DIM; col++) {
            INDEX(matrix_A, row, col) = (double)(row + col);
            INDEX(matrix_B, row, col) = (double)(row - col);
            INDEX(result_matrix, row, col) = 0.0;
        }
    }
}

// Clear result matrix
void clear_result() {
    memset(result_matrix, 0, MATRIX_DIM * MATRIX_DIM * sizeof(double));
}

// Sequential matrix multiplication (baseline)
double sequential_multiply() {
    double time_begin = omp_get_wtime();
    
    for (int row = 0; row < MATRIX_DIM; row++) {
        for (int col = 0; col < MATRIX_DIM; col++) {
            double accumulator = 0.0;
            for (int k = 0; k < MATRIX_DIM; k++) {
                accumulator += INDEX(matrix_A, row, k) * INDEX(matrix_B, k, col);
            }
            INDEX(result_matrix, row, col) = accumulator;
        }
    }
    
    return omp_get_wtime() - time_begin;
}

// Version 1: Single-dimension parallelization (rows)
double parallel_version1(int num_threads) {
    omp_set_num_threads(num_threads);
    double time_begin = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < MATRIX_DIM; row++) {
        for (int col = 0; col < MATRIX_DIM; col++) {
            double accumulator = 0.0;
            for (int k = 0; k < MATRIX_DIM; k++) {
                accumulator += INDEX(matrix_A, row, k) * INDEX(matrix_B, k, col);
            }
            INDEX(result_matrix, row, col) = accumulator;
        }
    }
    
    return omp_get_wtime() - time_begin;
}

// Version 2: Two-dimension parallelization (rows and columns)
double parallel_version2(int num_threads) {
    omp_set_num_threads(num_threads);
    double time_begin = omp_get_wtime();
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < MATRIX_DIM; row++) {
        for (int col = 0; col < MATRIX_DIM; col++) {
            double accumulator = 0.0;
            for (int k = 0; k < MATRIX_DIM; k++) {
                accumulator += INDEX(matrix_A, row, k) * INDEX(matrix_B, k, col);
            }
            INDEX(result_matrix, row, col) = accumulator;
        }
    }
    
    return omp_get_wtime() - time_begin;
}

// Check if results match baseline
int validate_output(double *reference, double *computed) {
    int sample_rate = MATRIX_DIM / 10;
    for (int row = 0; row < MATRIX_DIM; row += sample_rate) {
        for (int col = 0; col < MATRIX_DIM; col += sample_rate) {
            double delta = INDEX(reference, row, col) - INDEX(computed, row, col);
            if (delta < 0) delta = -delta;
            if (delta > 1e-6) {
                printf("VALIDATION ERROR at [%d][%d]: expected=%.2f, got=%.2f\n", 
                       row, col, INDEX(reference, row, col), INDEX(computed, row, col));
                return 0;
            }
        }
    }
    return 1;
}

void display_line() {
    printf("================================================================\n");
}

int main() {
    printf("\n>>> Allocating memory for matrices...\n");
    
    matrix_A = create_matrix();
    matrix_B = create_matrix();
    result_matrix = create_matrix();
    baseline_result = create_matrix();
    
    double mem_usage = (4.0 * MATRIX_DIM * MATRIX_DIM * sizeof(double)) / (1024.0 * 1024.0);
    printf(">>> Memory allocated: %.2f MB\n\n", mem_usage);
    
    setup_matrices();
    
    display_line();
    printf("       MATRIX MULTIPLICATION PERFORMANCE STUDY\n");
    display_line();
    printf("Configuration:\n");
    printf("  Matrix dimensions: %d x %d\n", MATRIX_DIM, MATRIX_DIM);
    printf("  Total operations: %ld FLOPs\n", (long)2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM);
    printf("  Iterations per test: %d\n", NUM_ITERATIONS);
    printf("  Maximum threads: %d\n", THREAD_LIMIT);
    printf("  System threads: %d\n", omp_get_max_threads());
    display_line();
    
    // Baseline sequential execution
    printf("\n>>> Computing baseline (sequential)...\n");
    double baseline_time = 0.0;
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        printf("  Iteration %d/%d...\r", iter + 1, NUM_ITERATIONS);
        fflush(stdout);
        clear_result();
        baseline_time += sequential_multiply();
    }
    baseline_time /= NUM_ITERATIONS;
    
    memcpy(baseline_result, result_matrix, MATRIX_DIM * MATRIX_DIM * sizeof(double));
    
    double baseline_gflops = (2.0 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM) / (baseline_time * 1e9);
    printf("\nBaseline time: %.4f seconds (%.2f GFLOPS)\n\n", baseline_time, baseline_gflops);
    
    display_line();
    printf("APPROACH 1: ROW-WISE PARALLELIZATION\n");
    display_line();
    printf("Description: Outer loop parallelized, each thread handles complete rows\n");
    printf("Work allocation: Thread T processes rows where (row %% num_threads == T)\n\n");
    printf("Threads | Time(s) | Speedup | Efficiency | GFLOPS\n");
    printf("--------|---------|---------|------------|--------\n");
    
    double best_speedup_v1 = 0.0;
    int best_threads_v1 = 1;
    
    for (int threads = 1; threads <= THREAD_LIMIT; threads++) {
        double cumulative_time = 0.0;
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            clear_result();
            cumulative_time += parallel_version1(threads);
        }
        
        double mean_time = cumulative_time / NUM_ITERATIONS;
        double speedup_factor = baseline_time / mean_time;
        double efficiency_pct = (speedup_factor / threads) * 100.0;
        double gflops = (2.0 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM) / (mean_time * 1e9);
        
        if (speedup_factor > best_speedup_v1) {
            best_speedup_v1 = speedup_factor;
            best_threads_v1 = threads;
        }
        
        // Verify at key points
        if (threads == 2 || threads == THREAD_LIMIT) {
            if (!validate_output(baseline_result, result_matrix)) {
                printf("WARNING: Validation failed at %d threads!\n", threads);
            }
        }
        
        printf("  %2d    | %.4f  | %.3fx   | %.2f%%     | %.2f\n", 
               threads, mean_time, speedup_factor, efficiency_pct, gflops);
    }
    
    printf("\n");
    display_line();
    printf("APPROACH 2: 2D GRID PARALLELIZATION\n");
    display_line();
    printf("Description: Both row and column loops parallelized using collapse(2)\n");
    printf("Work allocation: 2D iteration space divided into blocks\n");
    printf("Benefits: Finer granularity, improved load distribution\n\n");
    printf("Threads | Time(s) | Speedup | Efficiency | GFLOPS\n");
    printf("--------|---------|---------|------------|--------\n");
    
    double best_speedup_v2 = 0.0;
    int best_threads_v2 = 1;
    
    for (int threads = 1; threads <= THREAD_LIMIT; threads++) {
        double cumulative_time = 0.0;
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            clear_result();
            cumulative_time += parallel_version2(threads);
        }
        
        double mean_time = cumulative_time / NUM_ITERATIONS;
        double speedup_factor = baseline_time / mean_time;
        double efficiency_pct = (speedup_factor / threads) * 100.0;
        double gflops = (2.0 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM) / (mean_time * 1e9);
        
        if (speedup_factor > best_speedup_v2) {
            best_speedup_v2 = speedup_factor;
            best_threads_v2 = threads;
        }
        
        // Verify at key points
        if (threads == 2 || threads == THREAD_LIMIT) {
            if (!validate_output(baseline_result, result_matrix)) {
                printf("WARNING: Validation failed at %d threads!\n", threads);
            }
        }
        
        printf("  %2d    | %.4f  | %.3fx   | %.2f%%     | %.2f\n", 
               threads, mean_time, speedup_factor, efficiency_pct, gflops);
    }
    
    printf("\n");
    display_line();
    printf("COMPARATIVE ANALYSIS\n");
    display_line();
    printf("\nApproach 1 (Row-wise):\n");
    printf("  Best configuration: %d threads\n", best_threads_v1);
    printf("  Peak speedup: %.3fx\n", best_speedup_v1);
    
    printf("\nApproach 2 (2D Grid):\n");
    printf("  Best configuration: %d threads\n", best_threads_v2);
    printf("  Peak speedup: %.3fx\n", best_speedup_v2);
    
    double performance_ratio = best_speedup_v2 / best_speedup_v1;
    if (performance_ratio > 1.05) {
        printf("\nResult: 2D approach outperforms by %.2f%%\n", (performance_ratio - 1.0) * 100);
    } else if (performance_ratio < 0.95) {
        printf("\nResult: Row-wise approach is more efficient (lower overhead)\n");
    } else {
        printf("\nResult: Both approaches show comparable performance\n");
    }
    
    display_line();
    printf("\n>>> Cleaning up memory...\n");
    
    free(matrix_A);
    free(matrix_B);
    free(result_matrix);
    free(baseline_result);
    
    printf(">>> Execution completed successfully.\n\n");
    
    return 0;
}
