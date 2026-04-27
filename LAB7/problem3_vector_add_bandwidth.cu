#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

constexpr int N = 1 << 22;

__device__ float d_A[N];
__device__ float d_B[N];
__device__ float d_C[N];

__global__ void vectorAddKernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    std::vector<float> h_A(N), h_B(N), h_C(N);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    cudaMemcpyToSymbol(d_A, h_A.data(), N * sizeof(float));
    cudaMemcpyToSymbol(d_B, h_B.data(), N * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int iterations = 200;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    cudaMemcpyFromSymbol(h_C.data(), d_C, N * sizeof(float));

    int correct = 1;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            correct = 0;
            break;
        }
    }

    int memoryClockRateKHz = 0;
    int memoryBusWidthBits = 0;
    cudaDeviceGetAttribute(&memoryClockRateKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);

    double memory_clock_hz = static_cast<double>(memoryClockRateKHz) * 1000.0;
    double memory_bus_bytes = static_cast<double>(memoryBusWidthBits) / 8.0;
    double theoretical_bw_gbps = (memory_clock_hz * memory_bus_bytes * 2.0) / 1e9;

    const double read_bytes = static_cast<double>(N) * 2.0 * sizeof(float);
    const double write_bytes = static_cast<double>(N) * sizeof(float);
    const double total_bytes = read_bytes + write_bytes;
    const double kernel_sec = static_cast<double>(kernel_ms) / 1000.0;
    const double measured_bw_gbps = ((total_bytes * iterations) / kernel_sec) / 1e9;

    printf("Problem 3: Vector Add + Bandwidth Analysis\n");
    printf("-------------------------------------------\n");
    printf("Vector length: %d\n", N);
    printf("Threads/Block: %d, Blocks/Grid: %d\n", threadsPerBlock, blocksPerGrid);
    printf("Timed launches: %d\n", iterations);
    printf("Kernel execution time (total): %.6f ms\n", kernel_ms);
    printf("Result check: %s\n", correct ? "PASS" : "FAIL");
    printf("Theoretical memory bandwidth: %.3f GB/s\n", theoretical_bw_gbps);
    printf("Measured memory bandwidth:    %.3f GB/s\n", measured_bw_gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
