/*
 * ============================================================
 * CUDA DIY Exercise 2: Memory Hierarchy & Shared Memory
 * ============================================================
 * TOPIC        : Shared Memory Tiling, Reduction, Bank Conflicts
 * CUDA VERSION : 12.x
 *
 * Learning Objectives:
 *   1. Load from global -> shared memory and synchronize threads
 *   2. Implement parallel tree reduction with __syncthreads()
 *   3. Understand and avoid shared memory bank conflicts
 *   4. Use atomicAdd for safe concurrent writes (histogram)
 *   5. Implement warp-level reduction using __shfl_down_sync()
 *
 * Compile:
 *   nvcc -O2 -arch=sm_86 ex02_memory_hierarchy.cu -o ex02_memory_hierarchy
 *
 * Run:
 *   ./ex02_memory_hierarchy
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define THREADS 256
#define N_DEFAULT (1 << 20)

/* Verify two float arrays match */
int allclose_f(const float* a, const float* b, int N, float atol)
{
    for (int i = 0; i < N; i++)
        if (fabsf(a[i] - b[i]) > atol) return 0;
    return 1;
}


/* ================================================================
 * SECTION A — PROVIDED: Reference Implementations
 * ================================================================ */

/* ----- A1. Provided: Shared Memory Demo ----------------------- */
__global__ void smemDemo(const float* A, float* B, int N)
{
    __shared__ float tile[256];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    tile[threadIdx.x] = (i < N) ? A[i] : 0.0f;
    __syncthreads();              /* BARRIER: all loads must finish */
    if (i < N)
        B[i] = tile[threadIdx.x] * 2.0f;
}

/* ----- A2. Provided: Tree Reduction (sum) --------------------- */
__global__ void treeReduceSum(const float* input, float* output, int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    /* Tree reduction: halve active threads each step */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/* Host wrapper: two-pass (GPU reduces to partial sums, CPU sums those) */
float provided_tree_reduce(const float* d_data, int N)
{
    int threads = THREADS;
    int blocks  = (N + threads - 1) / threads;
    float *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    treeReduceSum<<<blocks, threads>>>(d_data, d_partial, N);

    float *h_partial = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float total = 0.0f;
    for (int b = 0; b < blocks; b++) total += h_partial[b];

    cudaFree(d_partial);
    free(h_partial);
    return total;
}

void run_provided_reduction(void)
{
    int N = N_DEFAULT;
    float *h_data = (float*)malloc(N * sizeof(float));
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX;
        cpu_sum  += h_data[i];
    }
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    float gpu_sum = provided_tree_reduce(d_data, N);
    printf("  [A2-TreeReduce] GPU=%.2f  CPU=%.2f  Match: %s\n",
           gpu_sum, (float)cpu_sum,
           fabsf(gpu_sum - (float)cpu_sum) < 100.0f ? "[PASS]" : "[FAIL]");

    cudaFree(d_data);
    free(h_data);
}


/* ================================================================
 * SECTION B — DIY EXERCISES
 * ================================================================ */

/* ----- B1. DIY: Shared Memory Copy ----------------------------
 * Load from global -> shared -> global (round-trip)
 * This teaches the load -> sync -> compute -> sync pattern.
 * -------------------------------------------------------------- */
__global__ void smemCopy(const float* input, float* output, int N)
{
    __shared__ float tile[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* ══════════════════════════════════════════════════════════
     * TODO (B1-Step1): Load element from global into shared tile.
     *   HINT: tile[threadIdx.x] = (i < N) ? input[i] : 0.0f;
     * ═════════════════════════════════════════════════════════= */
    tile[threadIdx.x] = (i < N) ? input[i] : 0.0f;

    /* ══════════════════════════════════════════════════════════
     * TODO (B1-Step2): Synchronize — wait for ALL threads in this
     *   block to finish loading before anyone reads from tile[].
     *   HINT: __syncthreads();
     * ══════════════════════════════════════════════════════════ */
    __syncthreads();

    /* ══════════════════════════════════════════════════════════
     * TODO (B1-Step3): Write from shared memory to output.
     *   HINT: if (i < N) output[i] = tile[threadIdx.x];
     * ══════════════════════════════════════════════════════════ */
    if (i < N) output[i] = tile[threadIdx.x];
}

void diy_smem_copy(void)
{
    int N = 1 << 16;
    size_t bytes = N * sizeof(float);
    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    smemCopy<<<blocks, threads>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_in, N, 1e-5f);
    printf("  [B1-SmemCopy] %s\n",
           ok ? "[PASS]" : "[FAIL] -- did you add __syncthreads()?");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
}


/* ----- B2. DIY: Block-level Max Reduction ---------------------
 * Find the maximum value across N elements using shared memory.
 * Key for softmax numerical stability (subtract max before exp).
 * -------------------------------------------------------------- */
__global__ void maxReduce(const float* input, float* output, int N)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    /* ══════════════════════════════════════════════════════════
     * TODO (B2-Step1): Load data into shared memory.
     *   Use -1e30f for out-of-bounds (identity for max).
     *   HINT: sdata[tid] = (i < N) ? input[i] : -1e30f;
     * ══════════════════════════════════════════════════════════ */
    sdata[tid] = (i < N) ? input[i] : -1e30f;
    __syncthreads();

    /* ══════════════════════════════════════════════════════════
     * TODO (B2-Step2): Tree reduction using fmaxf instead of +.
     *   HINT:
     *     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
     *         if (tid < s)
     *             sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
     *         __syncthreads();
     *     }
     * ══════════════════════════════════════════════════════════ */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

void diy_max_reduce(void)
{
    int N = 1 << 18;
    float *h_data = (float*)malloc(N * sizeof(float));
    float cpu_max = -1e30f;
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / RAND_MAX * 100.0f;
        if (h_data[i] > cpu_max) cpu_max = h_data[i];
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    float *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
    maxReduce<<<blocks, threads>>>(d_data, d_partial, N);

    float *h_partial = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float gpu_max = -1e30f;
    for (int b = 0; b < blocks; b++)
        if (h_partial[b] > gpu_max) gpu_max = h_partial[b];

    int ok = fabsf(gpu_max - cpu_max) < 0.01f;
    printf("  [B2-MaxReduce] GPU=%.4f  CPU=%.4f  %s\n",
           gpu_max, cpu_max, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_partial);
    free(h_data); free(h_partial);
}


/* ----- B3. DIY: Bank Conflict Demo ----------------------------
 * Shared memory has 32 banks. Strided access causes conflicts.
 * Measure kernel time for different access strides.
 * -------------------------------------------------------------- */
__global__ void bankConflictAccess(float* data, float* out, int stride, int N)
{
    __shared__ float smem[1024];
    int tid = threadIdx.x;

    /* Strided access — stride=1 is optimal, stride=32 is worst */
    smem[tid * stride % 1024] = (tid < N) ? data[tid] : 0.0f;
    __syncthreads();
    if (tid < N)
        out[tid] = smem[tid * stride % 1024] * 2.0f;
}

void diy_bank_conflict_demo(void)
{
    int strides[] = {1, 2, 4, 8, 16, 32};
    int n_strides = sizeof(strides) / sizeof(strides[0]);
    int N = 1024, REPS = 5000;

    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    printf("\n  [B3-BankConflictDemo] (lower = better)\n");
    printf("  %8s  %12s  Notes\n", "Stride", "Time (us)");
    printf("  %s\n", "--------------------------------------");

    float base_us = -1.0f;
    for (int s = 0; s < n_strides; s++) {
        int stride = strides[s];

        /* ══════════════════════════════════════════════════════
         * TODO (B3): Time the bankConflictAccess kernel.
         *   HINT:
         *     cudaEventRecord(t0);
         *     for (int r = 0; r < REPS; r++)
         *         bankConflictAccess<<<1, N>>>(d_data, d_out, stride, N);
         *     cudaEventRecord(t1);
         *     cudaEventSynchronize(t1);
         *     float ms; cudaEventElapsedTime(&ms, t0, t1);
         *     float us = ms * 1000.0f / REPS;  // average microseconds
         * ══════════════════════════════════════════════════════ */
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < REPS; r++)
            bankConflictAccess<<<1, N>>>(d_data, d_out, stride, N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        float us = ms * 1000.0f / REPS;

        if (base_us < 0.0f) base_us = us;
        printf("  %8d  %12.2f  %s\n", stride, us,
               stride == 1  ? "(sequential — best)" :
               stride == 32 ? "(32-way conflict — worst)" : "");
    }

    printf("  Baseline stride=1 average time: %.2f us\n", base_us);
    cudaFree(d_data); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}


/* ----- B4. DIY: Histogram with Atomics ------------------------
 * Each thread increments the bin corresponding to its data element.
 * Use atomicAdd to prevent data races between threads.
 * -------------------------------------------------------------- */
__global__ void histogram(const int* data, int* hist, int N, int num_bins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        /* ══════════════════════════════════════════════════════
         * TODO (B4): Atomically increment histogram bin.
         *   HINT: atomicAdd(&hist[data[i]], 1);
         *   Why atomic? Without it, threads race to write the
         *   same bin address simultaneously, producing wrong counts.
         * ══════════════════════════════════════════════════════ */
        atomicAdd(&hist[data[i]], 1);
    }
}

void diy_histogram(void)
{
    int N = 1 << 18, num_bins = 256;
    int *h_data  = (int*)malloc(N * sizeof(int));
    int *h_hist  = (int*)calloc(num_bins, sizeof(int));
    int *h_ref   = (int*)calloc(num_bins, sizeof(int));

    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % num_bins;
        h_ref[h_data[i]]++;
    }

    int *d_data, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    histogram<<<blocks, threads>>>(d_data, d_hist, N, num_bins);
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, num_bins * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int ok = 1;
    for (int b = 0; b < num_bins; b++)
        if (h_hist[b] != h_ref[b]) { ok = 0; break; }

    printf("  [B4-Histogram] N=%d bins=%d  %s\n",
           N, num_bins,
           ok ? "[PASS]" : "[FAIL] -- did you use atomicAdd?");

    cudaFree(d_data); cudaFree(d_hist);
    free(h_data); free(h_hist); free(h_ref);
}


/* ================================================================
 * SECTION C — STRETCH: Warp-level Reduction (No Shared Memory)
 * ================================================================ */

/* ----- C1. Stretch: Warp Shuffle Reduction --------------------
 * __shfl_down_sync passes register values between warp lanes.
 * This is faster than shared memory because it uses the warp
 * register file directly — no shared memory or __syncthreads needed.
 * -------------------------------------------------------------- */
__global__ void warpSumReduce(const float* data, float* out, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? data[i] : 0.0f;

    /* ══════════════════════════════════════════════════════════
     * STRETCH (C1): Butterfly warp reduction.
     *   HINT:
     *     for (int offset = 16; offset > 0; offset >>= 1)
     *         val += __shfl_down_sync(0xffffffff, val, offset);
     *   After the loop, lane 0 of each warp holds the warp sum.
     *   Use atomicAdd to accumulate across warps.
     * ══════════════════════════════════════════════════════════ */
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (threadIdx.x % 32 == 0) atomicAdd(out, val);
}

void stretch_warp_reduce(void)
{
    int N = 32;   /* single warp */
    float *h_data = (float*)malloc(N * sizeof(float));
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) { h_data[i] = (float)i; cpu_sum += h_data[i]; }

    float *d_data, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    warpSumReduce<<<1, 32>>>(d_data, d_out, N);

    float gpu_sum;
    CUDA_CHECK(cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    int ok = fabsf(gpu_sum - cpu_sum) < 0.01f;
    printf("  [C1-WarpReduce] GPU=%.1f  CPU=%.1f  %s\n",
           gpu_sum, cpu_sum, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_out);
    free(h_data);
}


/* ----- C2. Stretch: Shared-Memory Histogram (Less Contention) -
 * Each block keeps a private histogram in shared memory,
 * then merges to global using atomicAdd once per block.
 * -------------------------------------------------------------- */
__global__ void histogramSharedMem(const int* data, int* hist,
                                   int N, int num_bins)
{
    extern __shared__ int local_hist[];  /* size = num_bins */

    /* ══════════════════════════════════════════════════════════
     * STRETCH (C2): Implement two-phase histogram.
     *   Phase 1 — initialise local_hist to 0 for this block:
     *     for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
     *         local_hist[b] = 0;
     *     __syncthreads();
     *   Phase 2 — accumulate into shared histogram (less contention
     *   than global atomics because only 256 threads compete):
     *     int i = blockIdx.x * blockDim.x + threadIdx.x;
     *     if (i < N) atomicAdd(&local_hist[data[i]], 1);
     *     __syncthreads();
     *   Phase 3 — merge to global histogram:
     *     for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
     *         atomicAdd(&hist[b], local_hist[b]);
     * ══════════════════════════════════════════════════════════ */
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        local_hist[b] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        atomicAdd(&local_hist[data[i]], 1);
    __syncthreads();

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        atomicAdd(&hist[b], local_hist[b]);
}

void stretch_shared_histogram(void)
{
    int N = 1 << 20, num_bins = 256;
    int *h_data  = (int*)malloc(N * sizeof(int));
    int *h_hist  = (int*)calloc(num_bins, sizeof(int));
    int *h_ref   = (int*)calloc(num_bins, sizeof(int));
    for (int i = 0; i < N; i++) { h_data[i] = rand() % num_bins; h_ref[h_data[i]]++; }

    int *d_data, *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));

    int threads = THREADS, blocks = (N + threads - 1) / threads;
    int smem = num_bins * sizeof(int);
    histogramSharedMem<<<blocks, threads, smem>>>(d_data, d_hist, N, num_bins);
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, num_bins * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int ok = 1;
    for (int b = 0; b < num_bins; b++)
        if (h_hist[b] != h_ref[b]) { ok = 0; break; }

    printf("  [C2-SharedHistogram] N=%d bins=%d  %s\n",
           N, num_bins, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_data); cudaFree(d_hist);
    free(h_data); free(h_hist); free(h_ref);
}


/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 2: Memory Hierarchy & Shared Mem\n");
    printf("========================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  Shared mem/block: %zu KB\n\n",
           prop.name, prop.sharedMemPerBlock / 1024);

    printf("[Section A] Reference:\n");
    run_provided_reduction();

    printf("\n[Section B] DIY Exercises:\n");
    diy_smem_copy();
    diy_max_reduce();
    diy_bank_conflict_demo();
    diy_histogram();

    printf("\n[Section C] Stretch Goals:\n");
    stretch_warp_reduce();
    stretch_shared_histogram();

    printf("\n========================================================\n");
    printf("  All [PASS] = ready for Exercise 3!\n");
    printf("========================================================\n\n");
    return 0;
}
