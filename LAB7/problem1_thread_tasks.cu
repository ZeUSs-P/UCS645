#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void sumTasksKernel(const int *d_input, unsigned long long *d_output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        unsigned long long iterative_sum = 0;
        for (int i = 0; i < n; ++i) {
            iterative_sum += static_cast<unsigned long long>(d_input[i]);
        }
        d_output[0] = iterative_sum;
    } else if (tid == 1) {
        unsigned long long direct_sum = (static_cast<unsigned long long>(n) * (n + 1ULL)) / 2ULL;
        d_output[1] = direct_sum;
    }
}

int main() {
    const int N = 1024;

    std::vector<int> h_input(N);
    unsigned long long h_output[2] = {0ULL, 0ULL};

    for (int i = 0; i < N; ++i) {
        h_input[i] = i + 1;
    }

    int *d_input = nullptr;
    unsigned long long *d_output = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_input), N * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_output), 2 * sizeof(unsigned long long));

    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((2 + threadsPerBlock.x - 1) / threadsPerBlock.x);

    sumTasksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Problem 1: Different thread tasks (N=%d)\n", N);
    printf("-----------------------------------------\n");
    printf("Task A (iterative sum): %llu\n", h_output[0]);
    printf("Task B (direct formula): %llu\n", h_output[1]);

    const unsigned long long expected = (static_cast<unsigned long long>(N) * (N + 1ULL)) / 2ULL;
    printf("Expected: %llu\n", expected);
    printf("Match check: iterative=%s, direct=%s\n",
           (h_output[0] == expected ? "PASS" : "FAIL"),
           (h_output[1] == expected ? "PASS" : "FAIL"));

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
