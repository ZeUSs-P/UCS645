/*
 * ============================================================
 * CUDA DIY Exercise 1: CUDA Basics — Vector & Element-wise Ops
 * ============================================================
 * TOPIC        : GPU Architecture, Kernel Launch, Memory Management
 * CUDA VERSION : 12.x
 *
 * Learning Objectives:
 *   1. Understand Thread -> Block -> Grid hierarchy
 *   2. Write and launch basic CUDA kernels
 *   3. Manage GPU memory (cudaMalloc / cudaMemcpy / cudaFree)
 *   4. Measure kernel timing with CUDA Events
 *
 * Compile:
 *   nvcc -O2 -arch=sm_86 ex01_cuda_basics.cu -o ex01_cuda_basics
 *   (replace sm_86 with your GPU's compute capability, e.g. sm_75, sm_89)
 *
 * Run:
 *   ./ex01_cuda_basics
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ── Error-checking macro (always use this around CUDA calls) ── */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define N_DEFAULT (1 << 20)   /* 1M elements */
#define THREADS   256


/* ================================================================
 * SECTION A — PROVIDED REFERENCE KERNELS
 *   Study these carefully before filling in the DIY sections.
 * ================================================================ */

/* ----- A1. Vector Addition (fully provided) ------------------- */
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    /* Each thread computes exactly ONE output element */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

/* ----- A2. CPU Baseline --------------------------------------- */
void cpu_vectorAdd(const float* A, const float* B, float* C, int N)
{
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}

/* Timing helper (wall-clock ms) */
static double wall_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* Verify two float arrays match within tolerance */
int allclose(const float* a, const float* b, int N, float atol)
{
    for (int i = 0; i < N; i++)
        if (fabsf(a[i] - b[i]) > atol) return 0;
    return 1;
}

/* Run and time the provided vectorAdd */
void run_vector_add(int N)
{
    size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    /* Initialise with random data */
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    /* CPU reference */
    double t0 = wall_ms();
    cpu_vectorAdd(h_A, h_B, h_ref, N);
    double cpu_ms = wall_ms() - t0;

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    /* Copy Host -> Device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch configuration: ceiling division */
    int threads = THREADS;
    int blocks  = (N + threads - 1) / threads;

    /* CUDA Events for precise GPU timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    /* Copy result Device -> Host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose(h_C, h_ref, N, 1e-4f);
    printf("  [A1-VectorAdd] N=%d  CPU=%.1f ms  GPU=%.2f ms  Speedup=%.1fx  %s\n",
           N, cpu_ms, gpu_ms, cpu_ms / gpu_ms, ok ? "[PASS]" : "[FAIL]");

    /* Cleanup */
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C); free(h_ref);
}


/* ================================================================
 * SECTION B — DIY KERNELS
 *   Fill in every TODO block.  Hints are in the comments.
 * ================================================================ */

/* ----- B1. DIY: Vector Scaling --------------------------------
 * Goal  : Multiply every element of A by scalar k → C[i] = k * A[i]
 * Why   : This is the core of learning-rate scaling in gradient descent.
 * -------------------------------------------------------------- */
__global__ void vectorScale(const float* A, float* C, float k, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * TODO (B1): Write the scaling operation.
         *   HINT: C[i] = k * A[i];
         * ══════════════════════════════════════════════════════ */
        C[i] = k * A[i];
    }
}

void diy_vector_scale(int N)
{
    size_t bytes = N * sizeof(float);
    float k = 3.14f;

    float *h_A   = (float*)malloc(bytes);
    float *h_C   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { h_A[i] = (float)rand() / RAND_MAX; h_ref[i] = k * h_A[i]; }

    float *d_A, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    vectorScale<<<blocks, threads>>>(d_A, d_C, k, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose(h_C, h_ref, N, 1e-4f);
    printf("  [B1-VectorScale] k=%.2f  %s\n", k, ok ? "[PASS]" : "[FAIL] -- check your kernel");

    cudaFree(d_A); cudaFree(d_C);
    free(h_A); free(h_C); free(h_ref);
}


/* ----- B2. DIY: Element-wise Squared Difference ---------------
 * Goal  : C[i] = (A[i] - B[i])^2
 * Why   : This is the per-element MSE loss computation.
 * -------------------------------------------------------------- */
__global__ void squaredDiff(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * TODO (B2): Compute squared difference.
         *   HINT:
         *     float diff = A[i] - B[i];
         *     C[i] = diff * diff;
         *   Alternatively: C[i] = powf(A[i] - B[i], 2.0f);
         * ══════════════════════════════════════════════════════ */
        float diff = A[i] - B[i];
        C[i] = diff * diff;
    }
}

void diy_squared_diff(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_A   = (float*)malloc(bytes);
    float *h_B   = (float*)malloc(bytes);
    float *h_C   = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
        float d = h_A[i] - h_B[i];
        h_ref[i] = d * d;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    squaredDiff<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose(h_C, h_ref, N, 1e-4f);
    printf("  [B2-SquaredDiff] %s\n", ok ? "[PASS]" : "[FAIL] -- check your kernel");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
}


/* ----- B3. DIY: Launch Config Calculator ----------------------
 * Goal  : Print the correct (blocks, threads) for a set of N values
 *         and confirm total_threads >= N in every case.
 * -------------------------------------------------------------- */
void diy_launch_config(void)
{
    int N_values[] = {1, 100, 256, 257, 1024, 10000, 1 << 20};
    int n_cases = sizeof(N_values) / sizeof(N_values[0]);
    int threads = THREADS;

    printf("\n  [B3-LaunchConfig] threads_per_block=%d\n", threads);
    printf("  %10s  %8s  %15s  %12s\n", "N", "blocks", "total_threads", "covers_all?");
    printf("  %s\n", "---------------------------------------------------");

    for (int c = 0; c < n_cases; c++) {
        int N = N_values[c];

        /* ══════════════════════════════════════════════════════
         * TODO (B3): Compute 'blocks' using ceiling division.
         *   HINT: blocks = (N + threads - 1) / threads;
         * ══════════════════════════════════════════════════════ */
        int blocks = (N + threads - 1) / threads;

        int total = blocks * threads;
        int covers = (total >= N);
        printf("  %10d  %8d  %15d  %12s\n", N, blocks, total, covers ? "[OK]" : "[FAIL]");
    }
}


/* ----- B4. DIY: Memory Bandwidth Benchmark --------------------
 * Goal  : Measure H2D and D2H transfer bandwidth (GB/s) for
 *         sizes: 1, 8, 64, 256, 512 MB.
 * Real-world: PCIe bandwidth is often the training bottleneck!
 * -------------------------------------------------------------- */
void diy_memory_bandwidth(void)
{
    int sizes_mb[] = {1, 8, 64, 256, 512};
    int n_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);

    printf("\n  [B4-MemoryBandwidth]\n");
    printf("  %10s  %12s  %12s\n", "Size (MB)", "H2D (GB/s)", "D2H (GB/s)");
    printf("  %s\n", "----------------------------------------");

    for (int s = 0; s < n_sizes; s++) {
        int mb = sizes_mb[s];
        size_t bytes = (size_t)mb * 1024 * 1024;

        float *h_data, *d_data;
        CUDA_CHECK(cudaMallocHost(&h_data, bytes));   /* pinned host memory */
        CUDA_CHECK(cudaMalloc(&d_data, bytes));

        /* Initialise */
        for (size_t i = 0; i < bytes / sizeof(float); i++)
            h_data[i] = (float)i;

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        float elapsed_ms = 0.0f;

        /* ══════════════════════════════════════════════════════
         * TODO (B4-H2D): Measure Host->Device transfer time.
         *   HINT:
         *     cudaEventRecord(t0);
         *     cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
         *     cudaEventRecord(t1);
         *     cudaEventSynchronize(t1);
         *     cudaEventElapsedTime(&elapsed_ms, t0, t1);
         *     float h2d_GBps = (mb / 1024.0f) / (elapsed_ms / 1000.0f);
         * ══════════════════════════════════════════════════════ */
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
        float h2d_GBps = (mb / 1024.0f) / (elapsed_ms / 1000.0f);

        /* ══════════════════════════════════════════════════════
         * TODO (B4-D2H): Measure Device->Host transfer time.
         *   HINT: Same pattern but cudaMemcpyDeviceToHost direction.
         * ══════════════════════════════════════════════════════ */
        CUDA_CHECK(cudaEventRecord(t0));
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, t0, t1));
        float d2h_GBps = (mb / 1024.0f) / (elapsed_ms / 1000.0f);

        printf("  %10d  %12.1f  %12.1f\n", mb, h2d_GBps, d2h_GBps);

        cudaFree(d_data);
        cudaFreeHost(h_data);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }
}


/* ================================================================
 * SECTION C — STRETCH GOALS (Advanced)
 * ================================================================ */

/* ----- C1. Stretch: ReLU from scratch -------------------------
 * Implement: out[i] = max(0, x[i])
 * -------------------------------------------------------------- */
__global__ void reluKernel(const float* x, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * STRETCH (C1): Implement ReLU.
         *   HINT: out[i] = fmaxf(0.0f, x[i]);
         * ══════════════════════════════════════════════════════ */
        out[i] = fmaxf(0.0f, x[i]);
    }
}

void stretch_relu(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_x   = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_x[i]  = ((float)rand() / RAND_MAX - 0.5f) * 8.0f;
        h_ref[i] = h_x[i] > 0.0f ? h_x[i] : 0.0f;
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    reluKernel<<<blocks, threads>>>(d_x, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose(h_out, h_ref, N, 1e-5f);
    printf("  [C1-ReLU-Stretch] %s\n", ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_x); cudaFree(d_out);
    free(h_x); free(h_out); free(h_ref);
}


/* ----- C2. Stretch: Warp Divergence Experiment ----------------
 * Design a kernel that artificially diverges based on threadIdx.
 * Measure time vs a branch-free equivalent.
 * -------------------------------------------------------------- */
__global__ void divergentKernel(float* data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * STRETCH (C2): Introduce warp divergence.
         *   HINT: if (threadIdx.x % 2 == 0)
         *             data[i] = data[i] * 2.0f;
         *         else
         *             data[i] = data[i] + 1.0f;
         *   Then write a branchFreeKernel that computes the same
         *   result without branching using predicated execution.
         * ══════════════════════════════════════════════════════ */
        if (threadIdx.x % 2 == 0)
            data[i] = data[i] * 2.0f;
        else
            data[i] = data[i] + 1.0f;
    }
}

__global__ void branchFreeKernel(float* data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * STRETCH (C2): Branch-free equivalent of divergentKernel.
         *   HINT:
         *     int even = (threadIdx.x % 2 == 0);
         *     data[i] = even * (data[i] * 2.0f) + (1-even) * (data[i] + 1.0f);
         * ══════════════════════════════════════════════════════ */
        int even = (threadIdx.x % 2 == 0);
        data[i] = even * (data[i] * 2.0f) + (1 - even) * (data[i] + 1.0f);
    }
}

void stretch_warp_divergence(int N)
{
    size_t bytes = N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = (float)rand() / RAND_MAX;

    float *d_div, *d_bf;
    CUDA_CHECK(cudaMalloc(&d_div, bytes));
    CUDA_CHECK(cudaMalloc(&d_bf,  bytes));
    CUDA_CHECK(cudaMemcpy(d_div, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bf,  h_data, bytes, cudaMemcpyHostToDevice));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    int REPS = 1000;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float div_ms, bf_ms;

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < REPS; r++)
        divergentKernel<<<blocks, threads>>>(d_div, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&div_ms, t0, t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int r = 0; r < REPS; r++)
        branchFreeKernel<<<blocks, threads>>>(d_bf, N);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&bf_ms, t0, t1));

    printf("  [C2-WarpDivergence] Divergent=%.2fms  BranchFree=%.2fms  "
           "Overhead=%.1fx\n", div_ms, bf_ms, div_ms / (bf_ms + 1e-6f));

    cudaFree(d_div); cudaFree(d_bf); free(h_data);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}


/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 1: Basics & Memory\n");
    printf("========================================================\n");

    /* Print device info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  (SM %d.%d)  VRAM: %.0f MB\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e6);

    printf("[Section A] Reference:\n");
    run_vector_add(N_DEFAULT);

    printf("\n[Section B] DIY Exercises:\n");
    diy_vector_scale(N_DEFAULT);
    diy_squared_diff(N_DEFAULT);
    diy_launch_config();
    diy_memory_bandwidth();

    printf("\n[Section C] Stretch Goals:\n");
    stretch_relu(N_DEFAULT);
    stretch_warp_divergence(1 << 18);

    printf("\n========================================================\n");
    printf("  Exercise 1 complete! All [PASS] = ready for Ex02.\n");
    printf("========================================================\n\n");
    return 0;
}
