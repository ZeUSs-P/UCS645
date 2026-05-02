#pragma once
#include "common.h"
#include <vector>
#include <string>

// False-colour Voronoi PNG: each site gets a unique hue, overlaid on density
void write_voronoi_png(const std::string&        filename,
                       const DensityGrid&         grid,
                       const std::vector<Point2D>& sites,
                       const std::vector<int>&     assignment);

// CSV: backend, width, height, sites, threads, iter, time_ms, error
void write_results_csv(const std::string&          filename,
                       const std::vector<RunResult>& results);

// Pretty console speedup table
void print_benchmark_table(const std::vector<RunResult>& results);
