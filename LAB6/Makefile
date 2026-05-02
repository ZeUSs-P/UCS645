NVCC = nvcc
# Compile for Ada Lovelace (RTX 4050)
ARCH = -arch=sm_89
CFLAGS = -O3

all: device_query array_sum matrix_add

device_query: device_query.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

array_sum: array_sum.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

matrix_add: matrix_add.cu
	$(NVCC) $(ARCH) $(CFLAGS) $< -o $@

clean:
	rm -f device_query array_sum matrix_add
