#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void mergePassKernel(const int *src, int *dst, int width, int n) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = pair_idx * (2 * width);

    if (left >= n) {
        return;
    }

    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);

    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }

    while (i < mid) {
        dst[k++] = src[i++];
    }

    while (j < right) {
        dst[k++] = src[j++];
    }
}

static void mergeRange(const std::vector<int> &src, std::vector<int> &dst, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (src[i] <= src[j]) {
            dst[k++] = src[i++];
        } else {
            dst[k++] = src[j++];
        }
    }

    while (i < mid) {
        dst[k++] = src[i++];
    }

    while (j < right) {
        dst[k++] = src[j++];
    }
}

static void pipelinedMergeSortCPU(std::vector<int> &arr) {
    const int n = static_cast<int>(arr.size());
    std::vector<int> tmp(n);

    std::vector<int> *src = &arr;
    std::vector<int> *dst = &tmp;

    for (int width = 1; width < n; width *= 2) {
        for (int left = 0; left < n; left += 2 * width) {
            int mid = std::min(left + width, n);
            int right = std::min(left + 2 * width, n);
            mergeRange(*src, *dst, left, mid, right);
        }
        std::swap(src, dst);
    }

    if (src != &arr) {
        arr = *src;
    }
}

static float cudaMergeSort(const std::vector<int> &h_input, std::vector<int> &h_output) {
    const int n = static_cast<int>(h_input.size());
    int *d_a = nullptr;
    int *d_b = nullptr;

    cudaMalloc(reinterpret_cast<void **>(&d_a), n * sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&d_b), n * sizeof(int));
    cudaMemcpy(d_a, h_input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int *src = d_a;
    int *dst = d_b;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int width = 1; width < n; width *= 2) {
        int pairs = (n + (2 * width) - 1) / (2 * width);
        int threads = 256;
        int blocks = (pairs + threads - 1) / threads;

        mergePassKernel<<<blocks, threads>>>(src, dst, width, n);
        std::swap(src, dst);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    h_output.resize(n);
    cudaMemcpy(h_output.data(), src, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);

    return ms;
}

static bool isSorted(const std::vector<int> &v) {
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i - 1] > v[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1000;

    std::vector<int> input(N);
    for (int i = 0; i < N; ++i) {
        input[i] = rand() % 10000;
    }

    std::vector<int> cpu_data = input;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    pipelinedMergeSortCPU(cpu_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    float cpu_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    std::vector<int> gpu_data;
    float gpu_ms = cudaMergeSort(input, gpu_data);

    printf("Problem 2: Parallel sorting comparison (N=%d)\n", N);
    printf("-------------------------------------------------------\n");
    printf("Method (a): CPU pipelined merge sort time  = %.6f ms\n", cpu_ms);
    printf("Method (b): CUDA parallel merge sort time  = %.6f ms\n", gpu_ms);
    printf("CPU sorted: %s\n", isSorted(cpu_data) ? "YES" : "NO");
    printf("GPU sorted: %s\n", isSorted(gpu_data) ? "YES" : "NO");

    if (!cpu_data.empty() && !gpu_data.empty()) {
        printf("First 10 sorted elements (CPU | GPU):\n");
        for (int i = 0; i < 10; ++i) {
            printf("%d | %d\n", cpu_data[i], gpu_data[i]);
        }
    }

    return 0;
}
