# Heterogeneous Parallel Weighted Voronoi Tessellation

Adaptive sampling via **Centroidal Voronoi Tessellation (CVT)** with Lloyd's relaxation.
Parallelised with **OpenMP** (CPU) and **CUDA** (GPU, RTX 2050 / sm_86).

---
## Submitted By : -
Group - TryCatchers
Piyush Biswal - 102303607
Sarvagya Srivastava - 102303589
Krish Bajaj - 102303470

## Project Structure

```
voronoi_cvt/
├── Makefile
├── include/
│   ├── common.h          # Point2D, DensityGrid, RunResult
│   ├── timer.h           # std::chrono timer
│   ├── density.h         # Density generators + PGM loader
│   ├── voronoi.h         # Backend declarations
│   └── output.h          # PNG + CSV output
├── src/
│   ├── main.cpp          # CLI entry point
│   ├── density.cpp       # Gaussian / checkerboard / PGM density
│   ├── output.cpp        # stb_image_write PNG + CSV writer
│   ├── voronoi_serial.cpp
│   ├── voronoi_omp.cpp
│   └── voronoi_cuda.cu   # 3-kernel CUDA implementation
├── scripts/
│   └── scaling_analysis.py   # Strong + weak scaling runner & plotter
├── third_party/
│   └── stb_image_write.h     # Downloaded by `make setup`
├── data/                 # Place PGM density maps here
└── results/              # Output PNGs and CSVs land here
```

---

## Prerequisites

### Linux / WSL2 (recommended for CUDA)
```bash
# GCC with C++17 and OpenMP
sudo apt install g++ make

# CUDA toolkit  (if not already installed)
# → https://developer.nvidia.com/cuda-downloads

# Python for scaling graphs
pip install matplotlib numpy
```

### Windows (MSYS2 / MinGW64)
```bash
pacman -S mingw-w64-x86_64-gcc make
# CUDA toolkit must be installed system-wide (adds nvcc to PATH)
pip install matplotlib numpy
```

---

## Build

```bash
cd voronoi_cvt

# Serial + OpenMP only
make

# Serial + OpenMP + CUDA  (requires nvcc)
make ENABLE_CUDA=1

# Clean
make clean
```

**GPU architecture** defaults to `sm_86` (RTX 2050).  
Override if needed: `make ENABLE_CUDA=1 CUDA_ARCH=sm_75`

---

## Run

```bash
# Quick test – all backends, 1024×1024, 256 sites
make run                  # CPU only
make run-cuda             # includes CUDA

# Manual invocation
./voronoi_cvt --width 1024 --height 1024 --sites 256 \
              --backend all --density gaussian --output results

# Available backends: serial | omp | cuda | all
# Available densities: gaussian | check | pgm
# Use --input file.pgm with --density pgm

# Control OMP threads
./voronoi_cvt --backend omp --threads 8 ...
```

---

## Scaling Experiments

```bash
# Run strong + weak scaling and generate graphs
make ENABLE_CUDA=1 scale

# Or manually
python3 scripts/scaling_analysis.py --binary ./voronoi_cvt --output results --cuda

# Re-plot from saved CSVs
make plot
```

Outputs:
- `results/strong_scaling.csv`
- `results/weak_scaling.csv`
- `results/scaling_graphs.png`  ← 2×2 dark-themed figure

**Strong scaling** sweeps OMP threads over `{1, 2, 4, 6, 8, 10, 12}` on a fixed
`1024×1024` grid.  
**Weak scaling** keeps work-per-thread constant by scaling the grid area
proportionally (1→512², 2→~724², 4→1024², ..., 12→~1792²).

---

## Algorithm (Lloyd's Relaxation)

```
1. Load / generate density map D[W×H]  (float, normalised to [0,1])
2. Initialise N sites via inverse-CDF sampling of D (density-biased)
3. Repeat until convergence (max_displacement < eps) or max_iter:
   a. ASSIGN  – every pixel gets assigned to its nearest site
   b. CENTROID – new site = density-weighted centroid of its cell
   c. MOVE    – update sites; check convergence
4. Output false-colour Voronoi PNG + timing CSV
```

### Parallelisation summary

| Stage | Serial | OpenMP | CUDA |
|---|---|---|---|
| ASSIGN | nested `for` loop | `#pragma omp parallel for` over rows | 1 thread/pixel kernel |
| CENTROID | sequential accum. | per-thread private arrays + critical merge | atomicAdd per pixel |
| MOVE | sequential | sequential (N sites is small) | thread-per-site kernel |

---

## Output Files

| File | Description |
|---|---|
| `results/serial_1024x1024.png` | False-colour Voronoi (serial) |
| `results/omp_1024x1024.png`    | False-colour Voronoi (OMP) |
| `results/cuda_1024x1024.png`   | False-colour Voronoi (CUDA) |
| `results/results.csv`          | Timing per backend |
| `results/strong_scaling.csv`   | Strong scaling data |
| `results/weak_scaling.csv`     | Weak scaling data |
| `results/scaling_graphs.png`   | 4-panel speedup figure |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `stb_image_write.h` missing | Run `make setup` (downloads via curl/wget) |
| `nvcc` not found | Ensure CUDA toolkit is on PATH; build without CUDA: `make` |
| `sm_86` error | Override: `make ENABLE_CUDA=1 CUDA_ARCH=sm_75` |
| OMP threads not scaling | Check `OMP_PROC_BIND=close OMP_WAIT_POLICY=active` |
| PNG looks wrong | Make sure `--sites` ≤ 1024 (constant-memory limit) |
