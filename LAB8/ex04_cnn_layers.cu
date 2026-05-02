/*
 * ============================================================
 * CUDA DIY Exercise 4: Tiled GEMM & CNN Layer Primitives
 * ============================================================
 * TOPIC        : Matrix Multiplication, Conv2D, BatchNorm, Pooling
 * CUDA VERSION : 12.x
 *
 * Learning Objectives:
 *   1. Implement tiled GEMM using shared memory (TILE=16)
 *   2. Benchmark naive GPU GEMM vs tiled GEMM vs cuBLAS
 *   3. Implement a direct Conv2D kernel
 *   4. Implement Max Pooling and BatchNorm (inference mode)
 *   5. Chain layers into a mini CNN forward pass
 *
 * Compile (with cuBLAS):
 *   nvcc -O2 -arch=sm_86 ex04_cnn_layers.cu -o ex04_cnn_layers -lcublas
 *
 * Run:
 *   ./ex04_cnn_layers
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st = (call);                                         \
        if (st != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",           \
                    __FILE__, __LINE__, (int)st);                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define TILE 16

int allclose_f(const float* a, const float* b, int N, float atol)
{
    for (int i = 0; i < N; i++)
        if (fabsf(a[i] - b[i]) > atol) return 0;
    return 1;
}

/* GPU event timer — returns elapsed ms */
float event_time(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}


/* ================================================================
 * SECTION A — PROVIDED: Naive Matrix Multiplication
 * ================================================================ */

__global__ void naiveMatMul(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    /* One thread per output element C[row, col] */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void run_naive_matmul(float* d_A, float* d_B, float* d_C,
                      int M, int N, int K, float* ms_out)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    if (ms_out) *ms_out = event_time(t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}


/* ================================================================
 * SECTION B — DIY: Tiled Matrix Multiplication
 * ================================================================ */

/* ----- B1. DIY: Tiled GEMM ------------------------------------
 * Threads cooperatively load 16x16 tiles into shared memory,
 * reducing global memory reads by a factor of TILE.
 * -------------------------------------------------------------- */
__global__ void tiledMatMul(const float* A, const float* B, float* C,
                            int M, int N, int K)
{
    __shared__ float tA[TILE][TILE];
    __shared__ float tB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        /* ══════════════════════════════════════════════════════
         * TODO (B1-Step1): Cooperatively load tile of A.
         *   Each thread loads one element:
         *     tA[threadIdx.y][threadIdx.x] =
         *         (row < M && t*TILE + threadIdx.x < K)
         *         ? A[row * K + t*TILE + threadIdx.x] : 0.0f;
         * ══════════════════════════════════════════════════════ */
        tA[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE + threadIdx.x < K)
            ? A[row * K + t * TILE + threadIdx.x] : 0.0f;

        /* ══════════════════════════════════════════════════════
         * TODO (B1-Step2): Cooperatively load tile of B.
         *   tB[threadIdx.y][threadIdx.x] =
         *       (col < N && t*TILE + threadIdx.y < K)
         *       ? B[(t*TILE + threadIdx.y) * N + col] : 0.0f;
         * ══════════════════════════════════════════════════════ */
        tB[threadIdx.y][threadIdx.x] =
            (col < N && t * TILE + threadIdx.y < K)
            ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;

        /* ══════════════════════════════════════════════════════
         * TODO (B1-Step3): Barrier — wait until all loads finish.
         *   HINT: __syncthreads();
         * ══════════════════════════════════════════════════════ */
        __syncthreads();

        /* ══════════════════════════════════════════════════════
         * TODO (B1-Step4): Compute partial dot product.
         *   for (int k = 0; k < TILE; k++)
         *       sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];
         * ══════════════════════════════════════════════════════ */
        for (int k = 0; k < TILE; k++)
            sum += tA[threadIdx.y][k] * tB[k][threadIdx.x];

        /* ══════════════════════════════════════════════════════
         * TODO (B1-Step5): Barrier before loading the next tile.
         *   HINT: __syncthreads();
         * ══════════════════════════════════════════════════════ */
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void diy_tiled_matmul(int M, int N, int K)
{
    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    float *h_ref = (float*)malloc(bytes_C);

    /* Initialise */
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand()/RAND_MAX - 0.5f);

    /* CPU reference (row-major) */
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += h_A[r*K+k] * h_B[k*N+c];
            h_ref[r*N+c] = s;
        }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE-1)/TILE, (M + TILE-1)/TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = event_time(t0, t1);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    int ok = allclose_f(h_C, h_ref, M * N, 5e-2f);  /* FP32 accumulation tolerance */

    double gflops = 2.0 * M * N * K / (ms / 1000.0) / 1e9;
    printf("  [B1-TiledMatMul] %dx%d@%dx%d  %.2f ms  %.1f GFLOPS  %s\n",
           M, K, K, N, ms, gflops, ok ? "[PASS]" : "[FAIL]");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
}


/* ----- B2. DIY: GEMM Benchmark (Naive vs Tiled vs cuBLAS) ---- */
void diy_gemm_benchmark(cublasHandle_t handle)
{
    int sizes[] = {128, 256, 512, 1024};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n  [B2-GemmBenchmark]\n");
    printf("  %6s  %12s  %12s  %12s  %10s\n",
           "Size", "Naive(ms)", "Tiled(ms)", "cuBLAS(ms)", "cuBLAS GFLOPS");
    printf("  %s\n", "--------------------------------------------------------------");

    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t bytes = (size_t)M * N * sizeof(float);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemset(d_A, 0, bytes));
        CUDA_CHECK(cudaMemset(d_B, 0, bytes));

        /* ══════════════════════════════════════════════════════
         * TODO (B2): Time all three approaches.
         *   1. Naive: use run_naive_matmul() above.
         *   2. Tiled: call tiledMatMul directly with events.
         *   3. cuBLAS: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
         *              N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
         *      Note: cuBLAS uses column-major; for row-major A*B, pass B,A.
         *      alpha=1.0f, beta=0.0f
         *
         *   Compute GFLOPS = 2*M*N*K / (time_s * 1e9)
         * ══════════════════════════════════════════════════════ */
        float naive_ms = 0.0f, tiled_ms = 0.0f, cublas_ms = 0.0f;

        run_naive_matmul(d_A, d_B, d_C, M, N, K, &naive_ms);

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        cudaEventRecord(t0);
        tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        tiled_ms = event_time(t0, t1);

        float alpha = 1.0f, beta = 0.0f;
        cudaEventRecord(t0);
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B, N,
                     d_A, K,
                     &beta,
                     d_C, N));
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cublas_ms = event_time(t0, t1);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);

        double gflops = 2.0 * M * N * K / (cublas_ms / 1000.0 + 1e-9) / 1e9;
        printf("  %6d  %12.2f  %12.2f  %12.2f  %10.1f\n",
               M, naive_ms, tiled_ms, cublas_ms, gflops);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
}


/* ================================================================
 * SECTION C — DIY: CNN Layer Kernels
 * ================================================================ */

/* ----- C1. DIY: Max Pooling 2x2, stride 2 --------------------
 * One thread per output element (n, c, oh, ow).
 * Reads a 2x2 window from the input and writes the maximum.
 * -------------------------------------------------------------- */
__global__ void maxPool2x2(const float* input, float* output,
                           int N, int C, int H, int W)
{
    int H_out = H / 2;
    int W_out = W / 2;

    int n  = blockIdx.z;
    int c  = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= H_out || ow >= W_out || n >= N || c >= C) return;

    /* ══════════════════════════════════════════════════════════
     * TODO (C1): Find max in the 2x2 input window.
     *   Window starts at input pixel (oh*2, ow*2).
     *   HINT:
     *     float m = -1e30f;
     *     for (int dh = 0; dh < 2; dh++)
     *         for (int dw = 0; dw < 2; dw++) {
     *             int ih = oh*2 + dh;
     *             int iw = ow*2 + dw;
     *             int idx = ((n*C + c)*H + ih)*W + iw;
     *             m = fmaxf(m, input[idx]);
     *         }
     *     output[((n*C + c)*H_out + oh)*W_out + ow] = m;
     * ══════════════════════════════════════════════════════════ */
    float m = -1e30f;
    for (int dh = 0; dh < 2; dh++)
        for (int dw = 0; dw < 2; dw++) {
            int ih = oh * 2 + dh;
            int iw = ow * 2 + dw;
            int idx = ((n * C + c) * H + ih) * W + iw;
            m = fmaxf(m, input[idx]);
        }
    output[((n * C + c) * H_out + oh) * W_out + ow] = m;
}

/* CPU reference max pooling for verification */
void cpu_max_pool(const float* in, float* out,
                  int N, int C, int H, int W)
{
    int H2 = H/2, W2 = W/2;
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int oh = 0; oh < H2; oh++)
          for (int ow = 0; ow < W2; ow++) {
              float m = -1e30f;
              for (int dh = 0; dh < 2; dh++)
                for (int dw = 0; dw < 2; dw++) {
                    float v = in[((n*C+c)*H + oh*2+dh)*W + ow*2+dw];
                    if (v > m) m = v;
                }
              out[((n*C+c)*H2 + oh)*W2 + ow] = m;
          }
}

void diy_max_pool(void)
{
    int N=4, C=8, H=16, W=16;
    int H2=H/2, W2=W/2;
    size_t in_bytes  = (size_t)N*C*H*W * sizeof(float);
    size_t out_bytes = (size_t)N*C*H2*W2 * sizeof(float);

    float *h_in  = (float*)malloc(in_bytes);
    float *h_out = (float*)malloc(out_bytes);
    float *h_ref = (float*)malloc(out_bytes);
    for (int i = 0; i < N*C*H*W; i++) h_in[i] = (float)rand() / RAND_MAX;
    cpu_max_pool(h_in, h_ref, N, C, H, W);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  in_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

    dim3 block(W2, 2);
    dim3 grid((H2 + 1)/2, C, N);
    maxPool2x2<<<grid, block>>>(d_in, d_out, N, C, H, W);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_ref, N*C*H2*W2, 1e-5f);
    printf("  [C1-MaxPool2x2] (%d,%d,%d,%d)->(%d,%d,%d,%d)  %s\n",
           N, C, H, W, N, C, H2, W2, ok ? "[PASS]" : "[FAIL]");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
}


/* ----- C2. DIY: Batch Normalization (Inference Mode) ----------
 * y = gamma * (x - mean) / sqrt(var + eps) + beta
 * mean/var are pre-computed running statistics per channel.
 * -------------------------------------------------------------- */
__global__ void batchNormInfer(const float* x, float* out,
                               const float* gamma, const float* beta,
                               const float* mean,  const float* var,
                               int N, int C, int HW, float eps)
{
    int c  = blockIdx.y;
    int hw = blockIdx.x * blockDim.x + threadIdx.x;
    if (hw >= HW || c >= C) return;

    for (int n = 0; n < N; n++) {
        int idx = (n * C + c) * HW + hw;
        /* ══════════════════════════════════════════════════════
         * TODO (C2): Apply batch normalization.
         *   Step 1: float xhat = (x[idx] - mean[c]) / sqrtf(var[c] + eps);
         *   Step 2: out[idx] = gamma[c] * xhat + beta[c];
         * ══════════════════════════════════════════════════════ */
        float xhat = (x[idx] - mean[c]) / sqrtf(var[c] + eps);
        out[idx] = gamma[c] * xhat + beta[c];
    }
}

void diy_batchnorm(void)
{
    int N=4, C=8, H=16, W=16;
    int HW = H * W;
    float eps = 1e-5f;
    size_t feat_bytes = (size_t)N*C*HW * sizeof(float);
    size_t chan_bytes = C * sizeof(float);

    float *h_x     = (float*)malloc(feat_bytes);
    float *h_out   = (float*)malloc(feat_bytes);
    float *h_gamma = (float*)malloc(chan_bytes);
    float *h_beta  = (float*)malloc(chan_bytes);
    float *h_mean  = (float*)malloc(chan_bytes);
    float *h_var   = (float*)malloc(chan_bytes);
    float *h_ref   = (float*)malloc(feat_bytes);

    for (int i = 0; i < N*C*HW; i++) h_x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    for (int c = 0; c < C; c++) {
        h_gamma[c] = 1.0f;
        h_beta[c]  = 0.0f;
        /* compute running stats from h_x */
        double sum = 0.0, sum2 = 0.0;
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < HW; hw++) {
                float v = h_x[(n*C+c)*HW + hw];
                sum += v; sum2 += v * v;
            }
        h_mean[c] = (float)(sum / (N * HW));
        h_var[c]  = (float)(sum2 / (N * HW) - h_mean[c] * h_mean[c]);
        /* CPU reference */
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < HW; hw++) {
                int idx = (n*C+c)*HW + hw;
                float xhat = (h_x[idx] - h_mean[c]) / sqrtf(h_var[c] + eps);
                h_ref[idx] = h_gamma[c] * xhat + h_beta[c];
            }
    }

    float *d_x, *d_out, *d_gamma, *d_beta, *d_mean, *d_var;
    CUDA_CHECK(cudaMalloc(&d_x,     feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,   feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, chan_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta,  chan_bytes));
    CUDA_CHECK(cudaMalloc(&d_mean,  chan_bytes));
    CUDA_CHECK(cudaMalloc(&d_var,   chan_bytes));
    CUDA_CHECK(cudaMemcpy(d_x,     h_x,     feat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, chan_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  chan_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean,  h_mean,  chan_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var,   h_var,   chan_bytes,  cudaMemcpyHostToDevice));

    int threads = 256;
    dim3 block(threads);
    dim3 grid((HW + threads-1)/threads, C);
    batchNormInfer<<<grid, block>>>(d_x, d_out, d_gamma, d_beta, d_mean, d_var,
                                    N, C, HW, eps);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, feat_bytes, cudaMemcpyDeviceToHost));

    int ok = allclose_f(h_out, h_ref, N*C*HW, 1e-4f);
    printf("  [C2-BatchNorm] (%d,%d,%d,%d)  %s\n", N, C, H, W,
           ok ? "[PASS]" : "[FAIL] -- check normalization formula");

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma);
    cudaFree(d_beta); cudaFree(d_mean); cudaFree(d_var);
    free(h_x); free(h_out); free(h_gamma); free(h_beta);
    free(h_mean); free(h_var); free(h_ref);
}


/* ================================================================
 * SECTION D — STRETCH: Direct Conv2D Kernel
 * ================================================================ */

/* ----- D1. Stretch: Direct 2D Convolution ---------------------
 * Input:  [N, C_in, H, W]      Filter: [C_out, C_in, kH, kW]
 * Output: [N, C_out, H_out, W_out]
 * H_out = (H + 2*padH - kH) / strideH + 1
 * -------------------------------------------------------------- */
__global__ void conv2dDirect(const float* input, const float* filter,
                             float* output,
                             int N, int C_in, int H, int W,
                             int C_out, int kH, int kW,
                             int padH, int padW,
                             int strideH, int strideW)
{
    int H_out = (H + 2*padH - kH) / strideH + 1;
    int W_out = (W + 2*padW - kW) / strideW + 1;

    int n  = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= H_out || ow >= W_out || n >= N || oc >= C_out) return;

    /* ══════════════════════════════════════════════════════════
     * STRETCH (D1): Implement direct convolution.
     *   For each (ic, kh, kw):
     *     int ih = oh * strideH - padH + kh;
     *     int iw = ow * strideW - padW + kw;
     *     if (ih >= 0 && ih < H && iw >= 0 && iw < W)
     *         sum += input[((n*C_in + ic)*H + ih)*W + iw]
     *              * filter[((oc*C_in + ic)*kH + kh)*kW + kw];
     *   output[((n*C_out + oc)*H_out + oh)*W_out + ow] = sum;
     * ══════════════════════════════════════════════════════════ */
    float sum = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int ih = oh * strideH - padH + kh;
                int iw = ow * strideW - padW + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    sum += input[((n * C_in + ic) * H + ih) * W + iw] *
                           filter[((oc * C_in + ic) * kH + kh) * kW + kw];
                }
            }
        }
    }
    output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
}

void stretch_conv2d(void)
{
    int N=2, C_in=1, H=8, W=8, C_out=4, kH=3, kW=3;
    int padH=1, padW=1, strideH=1, strideW=1;
    int H_out = (H + 2*padH - kH) / strideH + 1;
    int W_out = (W + 2*padW - kW) / strideW + 1;

    int n_in     = N * C_in  * H * W;
    int n_filter = C_out * C_in * kH * kW;
    int n_out    = N * C_out * H_out * W_out;

    float *h_in  = (float*)calloc(n_in,     sizeof(float));
    float *h_fil = (float*)calloc(n_filter, sizeof(float));
    float *h_out = (float*)calloc(n_out,    sizeof(float));
    for (int i = 0; i < n_in;     i++) h_in[i]  = (float)rand()/RAND_MAX;
    for (int i = 0; i < n_filter; i++) h_fil[i] = (float)rand()/RAND_MAX;

    float *d_in, *d_fil, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n_in     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fil, n_filter * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n_out    * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,  h_in,  n_in     * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fil, h_fil, n_filter * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, n_out * sizeof(float)));

    dim3 block(W_out, 4);
    dim3 grid((H_out + 3)/4, C_out, N);
    conv2dDirect<<<grid, block>>>(d_in, d_fil, d_out,
                                  N, C_in, H, W,
                                  C_out, kH, kW,
                                  padH, padW, strideH, strideW);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost));

    /* Sanity: output must be non-zero (input and filter are positive) */
    float sum = 0.0f;
    for (int i = 0; i < n_out; i++) sum += h_out[i];
    printf("  [D1-Conv2D] H_out=%d W_out=%d  output_sum=%.2f  %s\n",
           H_out, W_out, sum, sum > 0.0f ? "[PASS]" : "[FAIL] -- implement kernel body");

    cudaFree(d_in); cudaFree(d_fil); cudaFree(d_out);
    free(h_in); free(h_fil); free(h_out);
}


/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 4: Tiled GEMM & CNN Layers\n");
    printf("========================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int sm_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, 0));
        double est_tflops = 2.0 * prop.multiProcessorCount * 128.0 * sm_clock_khz * 1e-9;
        printf("  GPU: %s  Peak TFLOPS (FP32 est.): ~%.2f\n\n",
           prop.name,
            est_tflops);

    /* Initialise cuBLAS */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    printf("[Section A] Reference: Naive MatMul:\n");
    {
        int M=256, N=256, K=256;
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
        float ms;
        run_naive_matmul(d_A, d_B, d_C, M, N, K, &ms);
        double gf = 2.0*M*N*K/(ms/1000.0)/1e9;
        printf("  Naive %dx%d@%dx%d  %.2f ms  %.1f GFLOPS\n", M,K,K,N, ms, gf);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    printf("\n[Section B] DIY: Tiled GEMM:\n");
    diy_tiled_matmul(512, 512, 512);
    diy_gemm_benchmark(handle);

    printf("\n[Section C] DIY: CNN Layers:\n");
    diy_max_pool();
    diy_batchnorm();

    printf("\n[Section D] Stretch — Direct Conv2D:\n");
    stretch_conv2d();

    cublasDestroy(handle);

    printf("\n========================================================\n");
    printf("  All [PASS] = ready for Exercise 5!\n");
    printf("========================================================\n\n");
    return 0;
}
