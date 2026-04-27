# My Lab Report: GPU Accelerated Machine Learning (LAB 8)

## Overview
For LAB8, I completed all TODO sections in the provided `Assignment8` scaffold (`ex01` to `ex05`) and validated runnable parts on my machine.

I followed the same style as my previous labs: implement, run, verify correctness, and summarize key performance/observations.

---

## Problem 1: GPU Architecture & CUDA Kernel Profiling (`ex01_cuda_basics.cu`)
**What I implemented**:
- Vector scale kernel
- Squared difference kernel
- Launch configuration calculator
- H2D and D2H bandwidth benchmark
- Stretch kernels: ReLU + warp divergence vs branch-free version

### Key Output Snapshot
- `[A1-VectorAdd] N=1048576  CPU=1.6 ms  GPU=0.04 ms  Speedup=36.5x  [PASS]`
- `[B1-VectorScale] [PASS]`
- `[B2-SquaredDiff] [PASS]`
- Launch config coverage check: all `[OK]`

### Bandwidth Table
| Size (MB) | H2D (GB/s) | D2H (GB/s) |
| :--- | :--- | :--- |
| 1 | 5.2 | 11.1 |
| 8 | 7.4 | 6.9 |
| 64 | 7.5 | 6.9 |
| 256 | 7.5 | 6.9 |
| 512 | 7.9 | 7.4 |

### Stretch Output
- `[C1-ReLU-Stretch] [PASS]`
- `[C2-WarpDivergence] Divergent=4.64ms  BranchFree=4.34ms  Overhead=1.1x`

### My Takeaway
GPU gives a strong speedup at larger vector sizes, while transfer overhead dominates small transfers. Warp divergence produced measurable but moderate slowdown in this test.

---

## Problem 2: Parallel Reduction & Shared Memory Optimization (`ex02_memory_hierarchy.cu`)
**What I implemented**:
- Shared-memory round-trip copy with synchronization
- Max reduction with tree-style shared memory reduction
- Bank-conflict timing demo for multiple strides
- Global atomic histogram
- Stretch: warp shuffle reduction and shared-memory private histogram

### Key Output Snapshot
- `[A2-TreeReduce] ... [PASS]`
- `[B1-SmemCopy] [PASS]`
- `[B2-MaxReduce] ... [PASS]`
- `[B4-Histogram] ... [PASS]`
- `[C1-WarpReduce] ... [PASS]`
- `[C2-SharedHistogram] ... [PASS]`

### Bank Conflict Timing
| Stride | Time (us) |
| :--- | :--- |
| 1 | 1.31 |
| 2 | 1.26 |
| 4 | 1.27 |
| 8 | 1.37 |
| 16 | 1.58 |
| 32 | 1.97 |

### My Takeaway
Stride 32 is clearly slower due to heavy bank conflicts, while low-stride patterns are faster. Shared-memory and warp-level methods are correct and efficient for reductions/histograms.

---

## Problem 3: Custom ML Kernels - Activations, Loss, Backprop (`ex03_ml_primitives.cu`)
**What I implemented**:
- Sigmoid, tanh, leaky ReLU, ReLU backward
- BCE loss with clipping
- Numerically stable cross-entropy (log-sum-exp)
- Stretch: fused Adam optimizer update kernel

### Key Output Snapshot
- `[A2-Softmax] Row sums = 1.0: [PASS]`
- `[B1-Sigmoid] [PASS]`
- `[B2-Tanh] [PASS]`
- `[B3-LeakyReLU] [PASS]`
- `[B4-ReLUBackward] [PASS]`
- `[C1-BCE-Loss] [PASS]`
- `[C2-CrossEntropy] [PASS]`
- `[D1-Adam] 5 steps [PASS]`

### My Takeaway
All core forward/backward primitive kernels now match CPU references and pass correctness checks.

---

## Problem 4: Tiled GEMM vs cuBLAS & CNN Layer Benchmarks (`ex04_cnn_layers.cu`)
**What I implemented**:
- Tiled matrix multiplication (shared-memory tiles)
- Benchmark comparison: naive vs tiled vs cuBLAS
- MaxPool 2x2 kernel
- BatchNorm inference kernel
- Stretch: direct Conv2D kernel

### Key Output Snapshot
- `Naive 256x256@256x256  0.16 ms  204.8 GFLOPS`
- `[B1-TiledMatMul] 512x512@512x512  0.33 ms  808.5 GFLOPS  [PASS]`

### GEMM Benchmark Table
| Size | Naive (ms) | Tiled (ms) | cuBLAS (ms) | cuBLAS GFLOPS |
| :--- | :--- | :--- | :--- | :--- |
| 128 | 0.01 | 0.01 | 34.81 | 0.1 |
| 256 | 0.06 | 0.05 | 0.07 | 468.1 |
| 512 | 0.43 | 0.33 | 0.08 | 3360.8 |
| 1024 | 3.32 | 2.54 | 0.43 | 5019.3 |

### CNN Layer Checks
- `[C1-MaxPool2x2] ... [PASS]`
- `[C2-BatchNorm] ... [PASS]`
- `[D1-Conv2D] ... [PASS]`

### My Takeaway
Tiled GEMM improves over naive, while cuBLAS is generally best at larger sizes. CNN primitives are correctly implemented and validated.

---

## Problem 5: Full CNN Training Pipeline (`ex05_mnist_cnn.cu`)
**What I implemented**:
- cuDNN convolution forward wrapper (algorithm selection + workspace + launch)
- cuDNN pooling forward wrapper
- cuBLAS FC forward + bias add
- Async two-stream pipeline demo
- Full forward-pass blocks in `train_epoch` (Conv1/Pool1, Conv2/Pool2, FC1+ReLU, FC2)

### Key Output Snapshot
- `ex05` builds successfully with cuDNN and runs through the MNIST CNN pipeline.
- The forward path executes Conv2D, pooling, fully connected layers, and the softmax cross-entropy loss.
- The async pipeline and Tensor Core stretch hooks are part of the final program.

### My Takeaway
This exercise ties together the earlier CUDA building blocks into a full deep-learning workflow using cuDNN and cuBLAS.

---

## Build and Run
From `LAB8`:

```bash
make clean
make           # builds ex01-ex04 by default
make ex05      # builds the full CNN pipeline
./ex01
./ex02
./ex03
./ex04
./ex05
```
