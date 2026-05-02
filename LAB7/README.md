# My Lab Report: CUDA Part II (LAB 7)

## Problem 1: Different Thread Tasks for Summation
**What I wanted to do**: I wrote `problem1_thread_tasks.cu` where different CUDA threads perform different tasks, exactly as asked:
- **Thread 0** computes the sum of first `N` integers using the **iterative approach**.
- **Thread 1** computes the sum using the **direct formula**.

I used `N = 1024`, created input/output arrays, moved data host-to-device, and launched a kernel with a normal grid/block configuration.

### Observed Output
| Metric | Value |
| :--- | :--- |
| N | 1024 |
| Iterative Sum | 524800 |
| Direct Formula Sum | 524800 |
| Expected Sum | 524800 |
| Verification | PASS |

### My Thoughts & Takeaways
- Both methods match the expected value exactly.
- This confirms that different threads in the same kernel can branch into separate logical tasks safely.
- Even though only two threads do useful work here (for demonstration), the pattern is useful for multi-task kernels.

---

## Problem 2: Parallel Sorting Comparison (Pipelining vs CUDA)
**What I wanted to do**: I implemented and compared two methods for sorting an array of size `n = 1000`:
- **(a) Pipelined merge-sort style on CPU**: iterative bottom-up merge passes.
- **(b) Parallel CUDA merge sort**: each thread merges one sorted pair per pass, with passes doubling merge width each iteration.

File used: `problem2_parallel_mergesort.cu`.

### Observed Output
| Metric | Value |
| :--- | :--- |
| Array Size | 1000 |
| Method (a) Time | 0.058724 ms |
| Method (b) Time | 0.302016 ms |
| CPU Sorted Check | YES |
| GPU Sorted Check | YES |

### Comparison
- For this small input size (`n=1000`), CPU method is faster in my run.
- CUDA still gives correct sorting, but GPU launch overhead dominates for small `n`.

### My Thoughts & Takeaways
- Parallel GPU sort becomes more beneficial as data size increases.
- This experiment clearly shows the difference between algorithmic parallelism and practical speedup boundaries.

---

## Problem 3: Vector Add + Profiling + Bandwidth Analysis
**What I wanted to do**: I implemented a basic vector addition kernel and completed all requested modifications in `problem3_vector_add_bandwidth.cu`:
1. Used **statically defined device global arrays** (no `cudaMalloc`).
2. Measured kernel timing using **CUDA events**.
3. Queried device attributes and computed **theoretical memory bandwidth**.
4. Computed **measured bandwidth** from bytes transferred and measured time.

### Key Implementation Notes
- Static device variables:
  - `__device__ float d_A[N];`
  - `__device__ float d_B[N];`
  - `__device__ float d_C[N];`
- Host-device transfer uses `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`.
- Theoretical bandwidth formula used:
  - `theoreticalBW = memoryClockRate * memoryBusWidth * 2` (with unit conversions to GB/s).
- Measured bandwidth formula used:
  - `measuredBW = (RBytes + WBytes) / t`

### Observed Output
| Metric | Value |
| :--- | :--- |
| Vector Length | 4,194,304 |
| Threads per Block | 256 |
| Blocks per Grid | 16,384 |
| Timed Kernel Launches | 200 |
| Total Kernel Time | 66.571358 ms |
| Correctness Check | PASS |
| Theoretical BW | 192.024 GB/s |
| Measured BW | 151.211 GB/s |

### My Thoughts & Takeaways
- Measured bandwidth is lower than theoretical, which is expected in real-world conditions.
- Theoretical bandwidth is an upper bound; measured values include practical limitations like scheduling and memory-access behavior.
- Averaging across many launches gives more stable profiling numbers.

---

## Build and Run
Use the following commands inside `LAB7`:

```bash
make clean
make
./problem1
./problem2
./problem3
```

## Optional nvprof Commands
(If your CUDA toolkit still includes `nvprof`):

```bash
nvprof ./problem3
nvprof --print-gpu-trace ./problem3
```
