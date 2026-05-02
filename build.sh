#!/usr/bin/env bash
# =============================================================================
#  build.sh  –  Build script for WSL2 Ubuntu
#  Run from Windows:  wsl -d Ubuntu -- bash build.sh [cuda]
#  Or from WSL:       bash build.sh [cuda]
#
#  Arg: "cuda"  -> compile with CUDA support (nvcc must be in PATH)
#       (none)  -> Serial + OpenMP only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENABLE_CUDA=0
if [[ "$1" == "cuda" ]]; then ENABLE_CUDA=1; fi

CUDA_ARCH="sm_86"
TARGET="voronoi_cvt"
CXXFLAGS="-O3 -std=c++17 -fopenmp -Iinclude -Ithird_party -Wall"

echo "========================================"
echo " Weighted Voronoi CVT – Build Script"
echo " CUDA: $ENABLE_CUDA   ARCH: $CUDA_ARCH"
echo "========================================"

# ── Directory setup ──────────────────────────────────────────────────────────
mkdir -p build results data

# ── stb_image_write.h ────────────────────────────────────────────────────────
if [[ ! -f third_party/stb_image_write.h ]]; then
    echo "[setup] Downloading stb_image_write.h ..."
    mkdir -p third_party
    curl -fsSL -o third_party/stb_image_write.h \
        https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    echo "[setup] Done."
fi

# ── Helper: compile a .cpp ───────────────────────────────────────────────────
compile_cpp() {
    local src="$1"
    local obj="build/$(basename "${src%.cpp}").o"
    local extra="${2:-}"
    echo "  [CXX] $(basename $src)"
    g++ $CXXFLAGS $extra -c -o "$obj" "$src"
    echo "$obj"
}

# ── CPU objects ──────────────────────────────────────────────────────────────
OBJS=()
for src in src/main.cpp src/density.cpp src/output.cpp \
           src/voronoi_serial.cpp src/voronoi_omp.cpp; do
    EXTRA=""
    [[ $ENABLE_CUDA -eq 1 ]] && EXTRA="-DENABLE_CUDA"
    obj="build/$(basename "${src%.cpp}").o"
    echo "  [CXX] $(basename $src)"
    g++ $CXXFLAGS $EXTRA -c -o "$obj" "$src"
    OBJS+=("$obj")
done

# ── CUDA object ──────────────────────────────────────────────────────────────
if [[ $ENABLE_CUDA -eq 1 ]]; then
    echo "  [NVCC] voronoi_cuda.cu"
    nvcc -O3 -std=c++17 -arch=$CUDA_ARCH \
         -Xcompiler "-fopenmp -O3" \
         -Iinclude -Ithird_party \
         -DENABLE_CUDA \
         -c -o build/voronoi_cuda.o src/voronoi_cuda.cu
    OBJS+=("build/voronoi_cuda.o")
fi

# ── Link ─────────────────────────────────────────────────────────────────────
echo "  [LINK] $TARGET"
if [[ $ENABLE_CUDA -eq 1 ]]; then
    nvcc -Xcompiler -fopenmp -o "$TARGET" "${OBJS[@]}"
else
    g++ -fopenmp -o "$TARGET" "${OBJS[@]}"
fi

echo ""
echo "  Build complete -> ./$TARGET"
echo ""
echo "  Quick test:"
echo "    ./$TARGET --backend serial --width 256 --height 256 --sites 64 --iter 20"
