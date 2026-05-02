#include <stdio.h>
#include <omp.h>
#include <math.h>

#define STEPS 100000000L
#define MAX_T 32
#define ITER 3

double delta;

double compute_pi_sequential(long n, double *result) {
    double total = 0.0, pos;
    delta = 1.0 / (double)n;
    double t0 = omp_get_wtime();
    
    for (long idx = 0; idx < n; idx++) {
        pos = (idx + 0.5) * delta;
        total += 4.0 / (1.0 + pos * pos);
    }
    
    *result = delta * total;
    return omp_get_wtime() - t0;
}

double compute_pi_threaded(long n, int threads, double *result) {
    double total = 0.0;
    delta = 1.0 / (double)n;
    omp_set_num_threads(threads);
    double t0 = omp_get_wtime();
    
    #pragma omp parallel
    {
        double pos, local_total = 0.0;
        #pragma omp for
        for (long idx = 0; idx < n; idx++) {
            pos = (idx + 0.5) * delta;
            local_total += 4.0 / (1.0 + pos * pos);
        }
        #pragma omp critical
        total += local_total;
    }
    
    *result = delta * total;
    return omp_get_wtime() - t0;
}

int main() {
    printf("\n============================================================\n");
    printf("PI COMPUTATION USING NUMERICAL INTEGRATION\n");
    printf("============================================================\n");
    printf("Integration steps: %ld | Actual π: %.15f\n", STEPS, M_PI);
    printf("System threads: %d\n", omp_get_max_threads());
    printf("============================================================\n");
    
    double baseline = 0.0, pi_val;
    for (int r = 0; r < ITER; r++) 
        baseline += compute_pi_sequential(STEPS, &pi_val);
    baseline /= ITER;
    
    printf("\nBaseline (serial): %.4f sec | π = %.15f | Error: %.2e\n\n", 
           baseline, pi_val, fabs(pi_val - M_PI));
    
    printf("Threads | Time(s) | Speedup | Efficiency | Computed π\n");
    printf("--------|---------|---------|------------|-----------------\n");
    
    double peak_speedup = 0.0;
    int best_t = 1;
    
    for (int t = 1; t <= MAX_T; t++) {
        double elapsed = 0.0, pi_out = 0.0;
        for (int r = 0; r < ITER; r++) 
            elapsed += compute_pi_threaded(STEPS, t, &pi_out);
        elapsed /= ITER;
        
        double speedup = baseline / elapsed;
        double eff = (speedup / t) * 100.0;
        
        if (speedup > peak_speedup) {
            peak_speedup = speedup;
            best_t = t;
        }
        
        printf("  %2d    | %.4f  | %.3fx   | %.2f%%     | %.15f\n", 
               t, elapsed, speedup, eff, pi_out);
    }
    
    printf("--------|---------|---------|------------|-----------------\n");
    printf("\nOptimal: %d threads with %.3fx speedup\n", best_t, peak_speedup);
    printf("============================================================\n\n");
    
    return 0;
}
