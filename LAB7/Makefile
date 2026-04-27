NVCC = nvcc
ARCH ?= -arch=sm_89
CFLAGS = -O3

all: problem1 problem2 problem3

problem1: problem1_thread_tasks.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

problem2: problem2_parallel_mergesort.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

problem3: problem3_vector_add_bandwidth.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

clean:
	rm -f problem1 problem2 problem3
