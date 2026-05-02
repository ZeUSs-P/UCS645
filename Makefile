# =============================================================================
#  Makefile – Weighted Voronoi CVT
#  MSYS2/UCRT64-compatible (use: make or make ENABLE_CUDA=1)
#
#  Build variants:
#    make                       -> Serial + OpenMP only   (g++ / UCRT64)
#    make ENABLE_CUDA=1         -> Serial + OpenMP + CUDA (nvcc + MSVC host)
#
#  GPU target: sm_86  (RTX 2050, Ampere GA107, CUDA 13.1)
#
#  Windows quirks resolved:
#    - NVCC on Windows requires MSVC (cl.exe) as host compiler.
#    - When ENABLE_CUDA=1, ALL .cpp objects are also compiled via nvcc -x c++
#      so that every object file is MSVC-ABI compatible for linking.
#    - MSYS2_ARG_CONV_EXCL=* must be set to stop MSYS2 sh from mangling
#      MSVC flags like /openmp into POSIX paths.
# =============================================================================

SHELL     := /bin/sh
# Prevent MSYS2 sh from converting MSVC /flags (e.g. /openmp) into POSIX paths
export MSYS2_ARG_CONV_EXCL := *
CUDA_ARCH ?= sm_86

# MSVC host compiler directory (required by nvcc on Windows)
NVCC_HOST_DIR := C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64

# ── CPU-only build (g++ / UCRT64) ──────────────────────────────────────────
CXX      := g++
CXXFLAGS := -O3 -std=c++17 -fopenmp -Iinclude -Ithird_party -Wall

# ── CUDA build: nvcc compiles EVERYTHING with MSVC as host ─────────────────
NVCC     := nvcc
# Common nvcc flags (compile and link)
NVCC_BASE := -std=c++17 -arch=$(CUDA_ARCH) \
             --compiler-bindir "$(NVCC_HOST_DIR)" \
             -allow-unsupported-compiler \
             -Iinclude -Ithird_party
# Flags for compiling .cu device code
NVCCFLAGS := $(NVCC_BASE) -O3 -Xcompiler /openmp,/O2
# Flags for compiling .cpp host-only code through nvcc
NVCC_CXX  := $(NVCC_BASE) -x c++ -O3 -Xcompiler /openmp,/O2
# Link flags
NVCC_LD   := $(NVCC_BASE) -Xcompiler /openmp

CPU_SRCS := src/main.cpp src/density.cpp src/output.cpp \
            src/voronoi_serial.cpp src/voronoi_omp.cpp

ifeq ($(ENABLE_CUDA),1)
  # Compile every .cpp via nvcc -x c++ so all objects share MSVC ABI
  CPU_OBJS  := $(CPU_SRCS:src/%.cpp=build/cuda_%.o)
  CUDA_OBJ  := build/voronoi_cuda.o
  EXTRA_DEF := -DENABLE_CUDA
  TARGET    := voronoi_cvt.exe
  ALL_OBJS  := $(CPU_OBJS) $(CUDA_OBJ)
else
  CPU_OBJS  := $(CPU_SRCS:src/%.cpp=build/%.o)
  CUDA_OBJ  :=
  EXTRA_DEF :=
  TARGET    := voronoi_cvt.exe
  ALL_OBJS  := $(CPU_OBJS)
endif

BENCH_BW   := bench_bandwidth.exe
BENCH_BLK  := bench_blocksize.exe

.PHONY: all clean setup run run-cuda run-stress scale plot help bw blk all-bench speedup

all: setup $(TARGET)

# ── CPU-only link ──────────────────────────────────────────────────────────
$(TARGET): $(ALL_OBJS)
ifeq ($(ENABLE_CUDA),1)
	$(NVCC) $(NVCC_LD) $(EXTRA_DEF) -o $@ $^ || true
else
	$(CXX) -fopenmp -o $@ $^
endif
	@test -f $@ && echo "Build complete: $(TARGET)" || (echo "ERROR: link failed" && exit 1)

# ── CPU objects (g++ when no CUDA) ────────────────────────────────────────
build/%.o: src/%.cpp | build third_party
	$(CXX) $(CXXFLAGS) $(EXTRA_DEF) -c -o $@ $<

# ── CPU objects compiled via nvcc -x c++ (CUDA build, MSVC ABI) ───────────
build/cuda_%.o: src/%.cpp | build third_party
	$(NVCC) $(NVCC_CXX) $(EXTRA_DEF) -c -o $@ $<

# ── CUDA device object ─────────────────────────────────────────────────────
build/voronoi_cuda.o: src/voronoi_cuda.cu | build third_party
	$(NVCC) $(NVCCFLAGS) $(EXTRA_DEF) -c -o $@ $<

build:
	mkdir -p build

third_party:
	mkdir -p third_party

results:
	mkdir -p results

data:
	mkdir -p data

setup: | third_party results data
	@test -f third_party/stb_image_write.h || \
	  (echo "Downloading stb_image_write.h..." && \
	   curl.exe -fsSL -o third_party/stb_image_write.h \
	     https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h && \
	   echo "Done.")

run: all
	./$(TARGET) --backend all --density gaussian \
	  --width 1024 --height 1024 --sites 256 --iter 100 --output results

run-cuda: all
	./$(TARGET) --backend all --density gaussian \
	  --width 1024 --height 1024 --sites 256 --iter 100 --output results

run-stress: all
	./$(TARGET) --backend all --density stress \
	  --width 1024 --height 1024 --sites 256 --iter 100 --output results

scale: all
	python scripts/scaling_analysis.py --binary $(TARGET) --output results \
	  $(if $(filter 1,$(ENABLE_CUDA)),--cuda,)

speedup: all
	python scripts/scaling_analysis.py --binary $(TARGET) --output results \
	  --cuda --speedup --plot-only

plot:
	python scripts/scaling_analysis.py --plot-only --output results

# ── Benchmark targets (require ENABLE_CUDA=1) ─────────────────────────────────
$(BENCH_BW): src/bench_bandwidth.cu | build
	$(NVCC) $(NVCCFLAGS) $(EXTRA_DEF) -o $@ $<

$(BENCH_BLK): src/bench_blocksize.cu | build
	$(NVCC) $(NVCCFLAGS) $(EXTRA_DEF) -o $@ $<

bw: setup $(BENCH_BW)
	./$(BENCH_BW) --output results
	python scripts/scaling_analysis.py --plot-only --output results

blk: setup $(BENCH_BLK)
	./$(BENCH_BLK) --output results
	python scripts/scaling_analysis.py --plot-only --output results

all-bench: bw blk

clean:
	rm -rf build
	rm -f $(TARGET) $(BENCH_BW) $(BENCH_BLK)

help:
	@echo "make                         -- build serial+OMP  (UCRT64 g++)"
	@echo "make ENABLE_CUDA=1           -- build with CUDA   (nvcc + MSVC host)"
	@echo "make run                     -- run all CPU backends"
	@echo "make run-cuda                -- run all backends  (CUDA)"
	@echo "make scale                   -- strong+weak scaling experiments"
	@echo "make ENABLE_CUDA=1 bw        -- PCIe bandwidth benchmark  (B4)"
	@echo "make ENABLE_CUDA=1 blk       -- block-size sweep (launch config)"
	@echo "make ENABLE_CUDA=1 all-bench -- run both benchmarks"
	@echo "make ENABLE_CUDA=1 speedup   -- speedup table (6 N values)"
	@echo "make clean                   -- remove build artefacts"
