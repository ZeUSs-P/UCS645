#include <stdio.h>
#include <stdlib.h>

#define N 4096

__global__ void matrixAddKernel(int *d_A, int *d_B, int *d_C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        int idx = row * width + col;
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

int main() {
    size_t size = N * N * sizeof(int);
    
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }
    
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    int blockDims[] = {2, 4, 8, 16, 32};
    int numConfigs = sizeof(blockDims) / sizeof(blockDims[0]);
    
    printf("Matrix Addition Performance Profiling (N=%d)\n", N);
    printf("--------------------------------------------------\n");
    printf("Block Dim (2D) | Grid Dim (2D) | Execution Time (ms)\n");
    printf("--------------------------------------------------\n");
    
    for (int i = 0; i < numConfigs; i++) {
        dim3 threadsPerBlock(blockDims[i], blockDims[i]);
        dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
                           
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warmup
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
        
        // Time execution
        cudaEventRecord(start);
        matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("%6d x %-5d | %5d x %-5d | %f\n", 
               threadsPerBlock.x, threadsPerBlock.y, 
               blocksPerGrid.x, blocksPerGrid.y, 
               milliseconds);
               
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
