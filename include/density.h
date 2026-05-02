#pragma once
#include "common.h"
#include <string>
#include <vector>

DensityGrid generate_gaussian    (int width, int height);
DensityGrid generate_checkerboard(int width, int height, int n_cells = 8);
DensityGrid load_pgm             (const std::string& filename);
void        normalize_density    (DensityGrid& grid);

// Weighted random init: samples are biased toward high-density regions
std::vector<Point2D> init_sites_weighted(const DensityGrid& grid,
                                         int n_sites, unsigned seed = 42);

// Uniform random init: ignores density (baseline for quality comparison)
std::vector<Point2D> init_sites_uniform(const DensityGrid& grid,
                                        int n_sites, unsigned seed = 42);

// Kirsch analytical stress field: circular hole in plate under uniaxial tension
// Hot-spots (stress concentration factor ~3) appear at ±90° from hole edges.
DensityGrid generate_stress_field(int width, int height);

// CVT energy (density-weighted mean squared distance to nearest site).
// Lower = better.  CVT minimises this; use to compare sampling strategies.
double compute_cvt_energy(const DensityGrid&         grid,
                           const std::vector<Point2D>& sites,
                           const std::vector<int>&      assignment);
