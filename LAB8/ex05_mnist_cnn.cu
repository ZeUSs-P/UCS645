/*
 * ============================================================
 * CUDA DIY Exercise 5: Full CNN Training Pipeline on MNIST
 * ============================================================
 * TOPIC        : cuDNN, cuBLAS, CUDA Streams, Full Training Loop
 * CUDA VERSION : 12.x  |  cuDNN 9.x  |  cuBLAS
 *
 * Learning Objectives:
 *   1. Use cuDNN to run Conv2D, BatchNorm, and Pooling forward passes
 *   2. Use cuBLAS for the fully-connected layer (SGEMM)
 *   3. Implement cross-entropy loss and softmax (custom kernel)
 *   4. Assemble a complete forward pass for LeNet on MNIST
 *   5. Use CUDA Streams to overlap data transfer with compute
 *   6. Implement the SGD optimizer kernel
 *
 * Architecture (LeNet-5 variant):
 *   Input [N,1,28,28]
 *   -> Conv(1->32, 5x5, pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
 *      [N,32,14,14]
 *   -> Conv(32->64, 5x5, pad=2) -> BatchNorm -> ReLU -> MaxPool(2)
 *      [N,64,7,7]
 *   -> Flatten [N, 3136]
 *   -> FC(3136->256) -> ReLU
 *   -> FC(256->10)
 *   -> Cross-Entropy Loss
 *
 * Compile:
 *   nvcc -O2 -arch=sm_86 ex05_mnist_cnn.cu -o ex05_mnist_cnn \
 *        -lcudnn -lcublas -lm
 *
 * MNIST binary files must be present in ./data/
 *   Download: http://yann.lecun.com/exdb/mnist/ or just google MNIST (hello world dataset of ML/DL)
 *     train-images-idx3-ubyte
 *     train-labels-idx1-ubyte
 *     t10k-images-idx3-ubyte
 *     t10k-labels-idx1-ubyte
 *
 * Run:
 *   ./ex05_mnist_cnn
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

/* ── Error macros ─────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                    \
    do { cudaError_t e=(call);                                              \
         if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",              \
         __FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)

#define CUDNN_CHECK(call)                                                   \
    do { cudnnStatus_t e=(call);                                            \
         if(e!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"cuDNN %s:%d %d\n",    \
         __FILE__,__LINE__,(int)e);exit(1);} } while(0)

#define CUBLAS_CHECK(call)                                                  \
    do { cublasStatus_t e=(call);                                           \
         if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d %d\n",  \
         __FILE__,__LINE__,(int)e);exit(1);} } while(0)

/* ── Hyperparameters ─────────────────────────────────────────────── */
#define BATCH_SIZE    256
#define LEARNING_RATE 0.01f
#define NUM_EPOCHS    10
#define MNIST_IMG     784     /* 28 * 28 */
#define NUM_CLASSES   10

/* ── Global handles ──────────────────────────────────────────────── */
cudnnHandle_t   cudnn;
cublasHandle_t  cublas;

/* Wall-clock ms helper */
static double wall_ms(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1e3 + t.tv_nsec*1e-6;
}


/* ================================================================
 * SECTION A — PROVIDED: MNIST Data Loader
 * ================================================================ */

/* Read big-endian 32-bit integer from file */
static int read_int(FILE* f)
{
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

typedef struct {
    float* images;   /* [n_samples, 784] float32 normalised to [0,1] */
    int*   labels;   /* [n_samples] int32 in [0,9] */
    int    n;
} MnistData;

MnistData load_mnist(const char* img_path, const char* lbl_path)
{
    FILE *fi = fopen(img_path, "rb");
    FILE *fl = fopen(lbl_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open MNIST files. Make sure these exist:\n"
                        "  %s\n  %s\n"
                        "Download from http://yann.lecun.com/exdb/mnist/\n",
                        img_path, lbl_path);
        exit(1);
    }
    read_int(fi); read_int(fl);           /* magic numbers */
    int n    = read_int(fi); read_int(fl);
    int rows = read_int(fi);
    int cols = read_int(fi);
    (void)rows; (void)cols;

    MnistData d;
    d.n      = n;
    d.images = (float*)malloc((size_t)n * MNIST_IMG * sizeof(float));
    d.labels = (int*)malloc(n * sizeof(int));

    unsigned char* buf = (unsigned char*)malloc(MNIST_IMG);
    for (int i = 0; i < n; i++) {
        fread(buf, 1, MNIST_IMG, fi);
        for (int j = 0; j < MNIST_IMG; j++)
            d.images[i * MNIST_IMG + j] = (buf[j] - 127.5f) / 127.5f;
        unsigned char lbl; fread(&lbl, 1, 1, fl);
        d.labels[i] = (int)lbl;
    }
    free(buf); fclose(fi); fclose(fl);
    printf("[✓] Loaded %d MNIST samples from %s\n", n, img_path);
    return d;
}


/* ================================================================
 * SECTION B — PROVIDED: cuDNN Descriptor Helpers
 * ================================================================ */

/* Create a 4D NCHW tensor descriptor */
cudnnTensorDescriptor_t make_tensor_desc(int N, int C, int H, int W)
{
    cudnnTensorDescriptor_t d;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&d));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT, N, C, H, W));
    return d;
}

/* Create a filter (weight) descriptor */
cudnnFilterDescriptor_t make_filter_desc(int k, int c, int h, int w)
{
    cudnnFilterDescriptor_t d;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&d));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(d, CUDNN_DATA_FLOAT,
                CUDNN_TENSOR_NCHW, k, c, h, w));
    return d;
}

/* Create a convolution descriptor */
cudnnConvolutionDescriptor_t make_conv_desc(int pad, int stride)
{
    cudnnConvolutionDescriptor_t d;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&d));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(d, pad, pad, stride, stride,
                1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    return d;
}


/* ================================================================
 * SECTION C — PROVIDED: Custom Kernels (Activations, Loss)
 * ================================================================ */

/* ReLU in-place */
__global__ void reluInPlace(float* x, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = fmaxf(0.0f, x[i]);
}

/* Numerically stable softmax + cross-entropy loss (one thread per sample) */
__global__ void softmaxCrossEntropy(const float* logits, const int* labels,
                                    float* probs, float* loss, int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row  = logits + n * C;
    float*       prow = probs  + n * C;

    float maxV = -1e30f;
    for (int c = 0; c < C; c++) maxV = fmaxf(maxV, row[c]);
    float sumE = 0.0f;
    for (int c = 0; c < C; c++) { prow[c] = expf(row[c] - maxV); sumE += prow[c]; }
    for (int c = 0; c < C; c++) prow[c] /= sumE;

    loss[n] = -logf(prow[labels[n]] + 1e-9f);
}


/* ================================================================
 * SECTION D — PROVIDED: SGD Optimizer Kernel
 * ================================================================ */

__global__ void sgdUpdate(float* w, const float* grad, float lr, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) w[i] -= lr * grad[i];
}

__global__ void add_bias(float* output, const float* bias, int N, int C);


/* ================================================================
 * SECTION E — DIY: cuDNN Convolution Forward Pass
 * ================================================================ */

/*
 * Wrapper that runs a cuDNN convolution forward pass.
 * You must fill in the cudnnFindConvolutionForwardAlgorithm call
 * and the cudnnConvolutionForward call.
 */
void diy_cudnn_conv_forward(
    cudnnTensorDescriptor_t     input_desc,   float* d_input,
    cudnnFilterDescriptor_t     filter_desc,  float* d_filter,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t     output_desc,  float* d_output)
{
    float alpha = 1.0f, beta = 0.0f;

    /* ══════════════════════════════════════════════════════════
     * TODO (E1): Find the best cuDNN conv algorithm for this shape.
     *   HINT:
     *     int nAlgo;
     *     cudnnConvolutionFwdAlgoPerf_t perfResult;
     *     CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
     *         cudnn,
     *         input_desc, filter_desc, conv_desc, output_desc,
     *         1, &nAlgo, &perfResult));
     *     cudnnConvolutionFwdAlgo_t algo = perfResult.algo;
     * ══════════════════════════════════════════════════════════ */
    int nAlgo = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResult;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_desc, filter_desc, conv_desc, output_desc,
        1, &nAlgo, &perfResult));
    cudnnConvolutionFwdAlgo_t algo = perfResult.algo;

    /* ══════════════════════════════════════════════════════════
     * TODO (E2): Allocate workspace memory required by cuDNN.
     *   HINT:
     *     size_t ws_bytes = 0;
     *     CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
     *         cudnn, input_desc, filter_desc, conv_desc, output_desc,
     *         algo, &ws_bytes));
     *     void* d_ws = NULL;
     *     if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));
     * ══════════════════════════════════════════════════════════ */
    size_t ws_bytes = 0;
    void*  d_ws     = NULL;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, input_desc, filter_desc, conv_desc, output_desc,
        algo, &ws_bytes));
    if (ws_bytes > 0) CUDA_CHECK(cudaMalloc(&d_ws, ws_bytes));

    /* ══════════════════════════════════════════════════════════
     * TODO (E3): Run the convolution forward pass.
     *   HINT:
     *     CUDNN_CHECK(cudnnConvolutionForward(
     *         cudnn,
     *         &alpha, input_desc, d_input,
     *         filter_desc, d_filter,
     *         conv_desc, algo, d_ws, ws_bytes,
     *         &beta, output_desc, d_output));
     * ══════════════════════════════════════════════════════════ */
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha, input_desc, d_input,
        filter_desc, d_filter,
        conv_desc, algo, d_ws, ws_bytes,
        &beta, output_desc, d_output));

    if (d_ws) cudaFree(d_ws);
}


/* ================================================================
 * SECTION F — DIY: cuDNN Pooling Forward Pass
 * ================================================================ */

void diy_cudnn_maxpool_forward(
    cudnnTensorDescriptor_t input_desc,  float* d_input,
    cudnnTensorDescriptor_t output_desc, float* d_output,
    int pool_h, int pool_w, int stride_h, int stride_w)
{
    /* ══════════════════════════════════════════════════════════
     * TODO (F1): Create a pooling descriptor.
     *   HINT:
     *     cudnnPoolingDescriptor_t pool_desc;
     *     CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
     *     CUDNN_CHECK(cudnnSetPooling2dDescriptor(
     *         pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
     *         pool_h, pool_w,
     *         0, 0,         // padding
     *         stride_h, stride_w));
     * ══════════════════════════════════════════════════════════ */
    cudnnPoolingDescriptor_t pool_desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
        pool_h, pool_w,
        0, 0,
        stride_h, stride_w));

    float alpha = 1.0f, beta = 0.0f;

    /* ══════════════════════════════════════════════════════════
     * TODO (F2): Call cudnnPoolingForward.
     *   HINT:
     *     CUDNN_CHECK(cudnnPoolingForward(
     *         cudnn, pool_desc, &alpha,
     *         input_desc, d_input,
     *         &beta, output_desc, d_output));
     *     CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
     * ══════════════════════════════════════════════════════════ */
    CUDNN_CHECK(cudnnPoolingForward(
        cudnn, pool_desc, &alpha,
        input_desc, d_input,
        &beta, output_desc, d_output));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
}


/* ================================================================
 * SECTION G — DIY: Fully-Connected Layer via cuBLAS
 * ================================================================ */

/*
 * Fully-connected layer: output = input * W^T + b
 * input:  [N, in_features]
 * weight: [out_features, in_features]
 * output: [N, out_features]
 *
 * cuBLAS is column-major. For row-major: output = input @ W^T
 * Using cublasSgemm: C = alpha * A * B + beta * C
 *   A = input   [N, in]    lda = in
 *   B = weight  [in, out]  ldb = out   (transposed = W^T)
 *   C = output  [N, out]   ldc = out
 */
void diy_fc_forward(float* d_input, float* d_weight, float* d_bias,
                    float* d_output, int N, int in_feat, int out_feat)
{
    float alpha = 1.0f, beta = 0.0f;

    /* ══════════════════════════════════════════════════════════
     * TODO (G1): Use cublasSgemm to compute the FC layer.
     *   cuBLAS expects column-major. For row-major input @ W^T:
     *   CUBLAS_CHECK(cublasSgemm(
     *       cublas,
     *       CUBLAS_OP_T,    // transpose W to get W^T
     *       CUBLAS_OP_N,    // input as-is
     *       out_feat,       // M: rows of output
     *       N,              // N: columns of output
     *       in_feat,        // K: inner dimension
     *       &alpha,
     *       d_weight, in_feat,   // W: [out_feat, in_feat]
     *       d_input,  in_feat,   // X: [N, in_feat]
     *       &beta,
     *       d_output, out_feat   // Y: [N, out_feat]
     *   ));
     * ══════════════════════════════════════════════════════════ */
    CUBLAS_CHECK(cublasSgemm(
        cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_feat,
        N,
        in_feat,
        &alpha,
        d_weight, in_feat,
        d_input,  in_feat,
        &beta,
        d_output, out_feat));

    /* ══════════════════════════════════════════════════════════
     * TODO (G2): Add bias to every row of d_output.
     *   HINT: use a simple kernel or cudnnAddTensor.
     *   Simple approach — one thread per output element:
     *     add_bias<<<grid, block>>>(d_output, d_bias, N, out_feat);
     * ══════════════════════════════════════════════════════════ */
    dim3 block(256);
    dim3 grid((out_feat + block.x - 1) / block.x, N);
    add_bias<<<grid, block>>>(d_output, d_bias, N, out_feat);
}

__global__ void add_bias(float* output, const float* bias, int N, int C)
{
    int n = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && c < C) output[n * C + c] += bias[c];
}


/* ================================================================
 * SECTION H — DIY: CUDA Streams for Async Data Transfer
 * ================================================================ */

/*
 * Use two streams to overlap H2D transfers with GPU computation.
 * While the GPU processes batch i, stream_transfer loads batch i+1.
 */
void diy_async_pipeline_demo(const float* h_images, int n_samples,
                             float* d_buf_A, float* d_buf_B)
{
    int batch = BATCH_SIZE;
    size_t bytes = (size_t)batch * MNIST_IMG * sizeof(float);

    /* ══════════════════════════════════════════════════════════
     * TODO (H1): Create two CUDA streams.
     *   HINT:
     *     cudaStream_t compute_stream, transfer_stream;
     *     CUDA_CHECK(cudaStreamCreate(&compute_stream));
     *     CUDA_CHECK(cudaStreamCreate(&transfer_stream));
     * ══════════════════════════════════════════════════════════ */
    cudaStream_t compute_stream, transfer_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&transfer_stream));

    /* ══════════════════════════════════════════════════════════
     * TODO (H2): Implement a double-buffered pipeline:
     *   - Preload batch 0 into d_buf_A
     *   - For each batch i:
     *       a. Start async load of batch (i+1) into d_buf_B
     *          on transfer_stream
     *       b. Process d_buf_A on compute_stream (e.g. call relu)
     *       c. cudaStreamSynchronize(transfer_stream)
     *       d. Swap d_buf_A and d_buf_B pointers
     *
     *   HINT for async copy:
     *     cudaMemcpyAsync(d_buf_B,
     *                     h_images + (i+1)*batch*MNIST_IMG,
     *                     bytes,
     *                     cudaMemcpyHostToDevice,
     *                     transfer_stream);
     *   Note: h_images must be pinned (cudaMallocHost) for async to work.
     * ══════════════════════════════════════════════════════════ */
    int n_batches = n_samples / batch;
    if (n_batches <= 0) {
        printf("  [H-AsyncPipeline] no full batches to process\n");
        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        CUDA_CHECK(cudaStreamDestroy(transfer_stream));
        return;
    }

    CUDA_CHECK(cudaMemcpyAsync(d_buf_A,
                               h_images,
                               bytes,
                               cudaMemcpyHostToDevice,
                               transfer_stream));
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

    for (int i = 0; i < n_batches; i++) {
        if (i + 1 < n_batches) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf_B,
                                       h_images + (size_t)(i + 1) * batch * MNIST_IMG,
                                       bytes,
                                       cudaMemcpyHostToDevice,
                                       transfer_stream));
        }

        int n_elems = batch * MNIST_IMG;
        int threads = 256;
        int blocks = (n_elems + threads - 1) / threads;
        reluInPlace<<<blocks, threads, 0, compute_stream>>>(d_buf_A, n_elems);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        if (i + 1 < n_batches) {
            CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
            float* tmp = d_buf_A;
            d_buf_A = d_buf_B;
            d_buf_B = tmp;
        }
    }
    printf("  [H-AsyncPipeline] processed %d batches with overlap pattern\n", n_batches);

    /* ══════════════════════════════════════════════════════════
     * TODO (H3): Synchronize and destroy streams.
     *   CUDA_CHECK(cudaDeviceSynchronize());
     *   CUDA_CHECK(cudaStreamDestroy(compute_stream));
     *   CUDA_CHECK(cudaStreamDestroy(transfer_stream));
     * ══════════════════════════════════════════════════════════ */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(transfer_stream));
}


/* ================================================================
 * SECTION I — DIY: Full Training Loop (ties everything together)
 * ================================================================ */

/*
 * One epoch of training using cuDNN + cuBLAS.
 * Provided scaffold — fill in the forward and backward stubs.
 */
void train_epoch(int epoch,
                 /* Weights */
                 float* d_conv1_w, float* d_conv2_w,
                 float* d_fc1_w,   float* d_fc1_b,
                 float* d_fc2_w,   float* d_fc2_b,
                 /* Feature maps — pre-allocated */
                 float* d_x,    /* Input [BATCH, 1, 28, 28]  */
                 float* d_c1,   /* After conv1 [BATCH, 32, 28, 28] */
                 float* d_p1,   /* After pool1 [BATCH, 32, 14, 14] */
                 float* d_c2,   /* After conv2 [BATCH, 64, 14, 14] */
                 float* d_p2,   /* After pool2 [BATCH, 64, 7, 7]   */
                 float* d_fc1,  /* After fc1   [BATCH, 256]        */
                 float* d_logit,/* After fc2   [BATCH, 10]         */
                 float* d_prob, /* Softmax     [BATCH, 10]         */
                 float* d_loss, /* Per-sample loss [BATCH]         */
                 /* cuDNN descriptors */
                 cudnnTensorDescriptor_t     x_desc,
                 cudnnFilterDescriptor_t     f1_desc,
                 cudnnConvolutionDescriptor_t c1_desc,
                 cudnnTensorDescriptor_t     c1_out_desc,
                 cudnnTensorDescriptor_t     p1_desc,
                 cudnnFilterDescriptor_t     f2_desc,
                 cudnnConvolutionDescriptor_t c2_desc,
                 cudnnTensorDescriptor_t     c2_out_desc,
                 cudnnTensorDescriptor_t     p2_desc,
                 /* Data */
                 const float* h_images, const int* h_labels,
                 int n_train)
{
    int n_batches = n_train / BATCH_SIZE;
    float total_loss = 0.0f;
    int   correct    = 0;

    for (int b = 0; b < n_batches; b++) {
        const float* batch_imgs   = h_images + (size_t)b * BATCH_SIZE * MNIST_IMG;
        const int*   batch_labels = h_labels + b * BATCH_SIZE;
        int*         d_labels;
        CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));

        /* H2D transfer */
        CUDA_CHECK(cudaMemcpy(d_x, batch_imgs,
                              (size_t)BATCH_SIZE * MNIST_IMG * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, batch_labels,
                              BATCH_SIZE * sizeof(int),
                              cudaMemcpyHostToDevice));

        /* ══════════════════════════════════════════════════════
         * TODO (I1): Forward pass — Conv1 + ReLU + Pool1.
         *   HINT:
         *     diy_cudnn_conv_forward(x_desc, d_x, f1_desc, d_conv1_w,
         *                            c1_desc, c1_out_desc, d_c1);
         *     int n_c1 = BATCH_SIZE * 32 * 28 * 28;
         *     reluInPlace<<<(n_c1+255)/256, 256>>>(d_c1, n_c1);
         *     diy_cudnn_maxpool_forward(c1_out_desc, d_c1, p1_desc, d_p1, 2, 2, 2, 2);
         * ══════════════════════════════════════════════════════ */
        diy_cudnn_conv_forward(x_desc, d_x, f1_desc, d_conv1_w,
                       c1_desc, c1_out_desc, d_c1);
        int n_c1 = BATCH_SIZE * 32 * 28 * 28;
        reluInPlace<<<(n_c1 + 255) / 256, 256>>>(d_c1, n_c1);
        diy_cudnn_maxpool_forward(c1_out_desc, d_c1, p1_desc, d_p1, 2, 2, 2, 2);

        /* ══════════════════════════════════════════════════════
         * TODO (I2): Forward pass — Conv2 + ReLU + Pool2.
         *   (same pattern as above but for layer 2)
         * ══════════════════════════════════════════════════════ */
        diy_cudnn_conv_forward(p1_desc, d_p1, f2_desc, d_conv2_w,
                       c2_desc, c2_out_desc, d_c2);
        int n_c2 = BATCH_SIZE * 64 * 14 * 14;
        reluInPlace<<<(n_c2 + 255) / 256, 256>>>(d_c2, n_c2);
        diy_cudnn_maxpool_forward(c2_out_desc, d_c2, p2_desc, d_p2, 2, 2, 2, 2);

        /* ══════════════════════════════════════════════════════
         * TODO (I3): Forward pass — FC1 + ReLU.
         *   Flatten d_p2: [BATCH, 64, 7, 7] -> [BATCH, 3136]
         *   (the memory is already contiguous — just treat it as [BATCH, 3136])
         *   HINT:
         *     diy_fc_forward(d_p2, d_fc1_w, d_fc1_b, d_fc1,
         *                    BATCH_SIZE, 64*7*7, 256);
         *     reluInPlace<<<(BATCH_SIZE*256+255)/256, 256>>>(d_fc1, BATCH_SIZE*256);
         * ══════════════════════════════════════════════════════ */
        diy_fc_forward(d_p2, d_fc1_w, d_fc1_b, d_fc1,
                   BATCH_SIZE, 64 * 7 * 7, 256);
        reluInPlace<<<(BATCH_SIZE * 256 + 255) / 256, 256>>>(d_fc1, BATCH_SIZE * 256);

        /* ══════════════════════════════════════════════════════
         * TODO (I4): Forward pass — FC2 (logits).
         *   diy_fc_forward(d_fc1, d_fc2_w, d_fc2_b, d_logit,
         *                  BATCH_SIZE, 256, NUM_CLASSES);
         * ══════════════════════════════════════════════════════ */
        diy_fc_forward(d_fc1, d_fc2_w, d_fc2_b, d_logit,
                   BATCH_SIZE, 256, NUM_CLASSES);

        /* Loss (provided) */
        int T = 256, B_loss = (BATCH_SIZE + T - 1) / T;
        softmaxCrossEntropy<<<B_loss, T>>>(d_logit, d_labels, d_prob,
                                           d_loss, BATCH_SIZE, NUM_CLASSES);

        /* Accumulate metrics */
        float h_loss_batch[BATCH_SIZE];
        CUDA_CHECK(cudaMemcpy(h_loss_batch, d_loss, BATCH_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < BATCH_SIZE; i++) total_loss += h_loss_batch[i];

        if (b % 50 == 0)
            printf("  Epoch %d  Batch [%d/%d]  AvgLoss=%.4f\n",
                   epoch, b, n_batches, total_loss / ((b+1)*BATCH_SIZE));

        cudaFree(d_labels);
    }

    printf("  --- Epoch %d Done  AvgLoss=%.4f ---\n",
           epoch, total_loss / (n_batches * BATCH_SIZE));
}


/* ================================================================
 * SECTION J — STRETCH: Mixed Precision with Tensor Cores
 * ================================================================ */

void stretch_fp16_matmul(int M, int N, int K)
{
    /*
     * Use cublasGemmEx with FP16 (half precision) to leverage Tensor Cores.
     * Tensor Cores are ~8x faster than FP32 CUDA cores on Ampere+.
     *
     * STRETCH (J1):
     *   Allocate __half arrays on device.
     *   Convert FP32 -> FP16 using a simple kernel.
     *   Call cublasGemmEx with:
     *     CUDA_R_16F for A, B types
     *     CUDA_R_32F for compute type (mixed precision accumulation)
     *     CUBLAS_GEMM_DEFAULT_TENSOR_OP for algo
     *   Compare timing to FP32 cublasSgemm.
     */
    printf("  [J-FP16-TensorCore] STRETCH: implement cublasGemmEx with CUDA_R_16F\n");
}


/* ================================================================
 * MAIN
 * ================================================================ */
int main(void)
{
    printf("\n========================================================\n");
    printf("  CUDA DIY Exercise 5: MNIST CNN (cuDNN + cuBLAS)\n");
    printf("========================================================\n");

    /* Print device info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s  Compute: %d.%d  VRAM: %.0f MB\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e6);

    /* Init handles */
    CUDNN_CHECK(cudnnCreate(&cudnn));
    CUBLAS_CHECK(cublasCreate(&cublas));

    /* ── Load MNIST ────────────────────────────────────────── */
    MnistData train = load_mnist(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte");
    MnistData test = load_mnist(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte");

    /* ── Allocate weights (random init) ────────────────────── */
    /* Conv1: [32, 1, 5, 5]  Conv2: [64, 32, 5, 5]             */
    /* FC1: [256, 3136]  FC1 bias: [256]                        */
    /* FC2: [10, 256]    FC2 bias: [10]                         */
    float *d_conv1_w, *d_conv2_w;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b;

    CUDA_CHECK(cudaMalloc(&d_conv1_w, 32*1*5*5   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv2_w, 64*32*5*5  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_w,   256*3136   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1_b,   256        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_w,   10*256     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc2_b,   10         * sizeof(float)));

    /* He initialisation on CPU, copy to GPU */
    {
        int lens[] = {32*1*5*5, 64*32*5*5, 256*3136, 256, 10*256, 10};
        float* ptrs_d[] = {d_conv1_w, d_conv2_w, d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b};
        for (int p = 0; p < 6; p++) {
            float* tmp = (float*)malloc(lens[p] * sizeof(float));
            float scale = sqrtf(2.0f / lens[p]);
            for (int i = 0; i < lens[p]; i++)
                tmp[i] = scale * (2.0f * (float)rand()/RAND_MAX - 1.0f);
            CUDA_CHECK(cudaMemcpy(ptrs_d[p], tmp, lens[p]*sizeof(float),
                                  cudaMemcpyHostToDevice));
            free(tmp);
        }
    }

    /* ── Allocate feature maps ──────────────────────────────── */
    float *d_x, *d_c1, *d_p1, *d_c2, *d_p2, *d_fc1, *d_logit, *d_prob, *d_loss;
    CUDA_CHECK(cudaMalloc(&d_x,     (size_t)BATCH_SIZE*1 *28*28 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1,    (size_t)BATCH_SIZE*32*28*28 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1,    (size_t)BATCH_SIZE*32*14*14 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2,    (size_t)BATCH_SIZE*64*14*14 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2,    (size_t)BATCH_SIZE*64*7 *7  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fc1,   (size_t)BATCH_SIZE*256       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_logit, (size_t)BATCH_SIZE*10         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_prob,  (size_t)BATCH_SIZE*10         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,  (size_t)BATCH_SIZE             * sizeof(float)));

    /* ── cuDNN Descriptors ──────────────────────────────────── */
    cudnnTensorDescriptor_t     x_desc      = make_tensor_desc(BATCH_SIZE, 1,  28, 28);
    cudnnTensorDescriptor_t     c1_out_desc = make_tensor_desc(BATCH_SIZE, 32, 28, 28);
    cudnnTensorDescriptor_t     p1_desc     = make_tensor_desc(BATCH_SIZE, 32, 14, 14);
    cudnnTensorDescriptor_t     c2_out_desc = make_tensor_desc(BATCH_SIZE, 64, 14, 14);
    cudnnTensorDescriptor_t     p2_desc     = make_tensor_desc(BATCH_SIZE, 64, 7,  7);

    cudnnFilterDescriptor_t     f1_desc     = make_filter_desc(32, 1,  5, 5);
    cudnnFilterDescriptor_t     f2_desc     = make_filter_desc(64, 32, 5, 5);

    cudnnConvolutionDescriptor_t c1_desc    = make_conv_desc(2, 1); /* pad=2 stride=1 */
    cudnnConvolutionDescriptor_t c2_desc    = make_conv_desc(2, 1);

    /* ── Training Loop ──────────────────────────────────────── */
    printf("\n[Training] Starting for %d epochs...\n\n", NUM_EPOCHS);

    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        double t0 = wall_ms();

        train_epoch(epoch,
                    d_conv1_w, d_conv2_w,
                    d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b,
                    d_x, d_c1, d_p1, d_c2, d_p2,
                    d_fc1, d_logit, d_prob, d_loss,
                    x_desc, f1_desc, c1_desc, c1_out_desc, p1_desc,
                    f2_desc, c2_desc, c2_out_desc, p2_desc,
                    train.images, train.labels, train.n);

        double epoch_ms = wall_ms() - t0;
        printf("  Epoch %d complete in %.1f s\n\n", epoch, epoch_ms / 1000.0);
    }

    /* ── Stretch ────────────────────────────────────────────── */
    printf("[Stretch] CUDA Streams async pipeline:\n");
    float *d_bufA, *d_bufB;
    CUDA_CHECK(cudaMalloc(&d_bufA, (size_t)BATCH_SIZE*MNIST_IMG*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bufB, (size_t)BATCH_SIZE*MNIST_IMG*sizeof(float)));
    diy_async_pipeline_demo(train.images, train.n, d_bufA, d_bufB);
    cudaFree(d_bufA); cudaFree(d_bufB);

    printf("\n[Stretch] FP16 Tensor Core GEMM:\n");
    stretch_fp16_matmul(1024, 1024, 1024);

    /* ── Cleanup ────────────────────────────────────────────── */
    cudaFree(d_conv1_w); cudaFree(d_conv2_w);
    cudaFree(d_fc1_w);   cudaFree(d_fc1_b);
    cudaFree(d_fc2_w);   cudaFree(d_fc2_b);
    cudaFree(d_x);  cudaFree(d_c1); cudaFree(d_p1);
    cudaFree(d_c2); cudaFree(d_p2); cudaFree(d_fc1);
    cudaFree(d_logit); cudaFree(d_prob); cudaFree(d_loss);
    free(train.images); free(train.labels);
    free(test.images);  free(test.labels);
    cudnnDestroy(cudnn);
    cublasDestroy(cublas);

    printf("\n========================================================\n");
    printf("  Exercise 5 complete!\n");
    printf("  Implement all TODOs to see MNIST training in action.\n");
    printf("========================================================\n\n");
    return 0;
}
