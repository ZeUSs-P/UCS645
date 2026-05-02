#pragma once
#include "common.h"
#include <vector>

// Serial CPU baseline
RunResult run_serial(const DensityGrid&   grid,
                     std::vector<Point2D> sites,
                     int max_iter, float eps,
                     std::vector<int>&    out_assignment);

// OpenMP multi-core  (n_threads=0 → use OMP_NUM_THREADS / max)
RunResult run_omp   (const DensityGrid&   grid,
                     std::vector<Point2D> sites,
                     int max_iter, float eps,
                     std::vector<int>&    out_assignment,
                     int n_threads = 0);

#ifdef ENABLE_CUDA
// CUDA GPU
RunResult run_cuda  (const DensityGrid&   grid,
                     std::vector<Point2D> sites,
                     int max_iter, float eps,
                     std::vector<int>&    out_assignment);
#endif
