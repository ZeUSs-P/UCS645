#pragma once
#include <vector>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// Basic 2D point (float precision)
// ---------------------------------------------------------------------------
struct Point2D {
    float x, y;
};

// ---------------------------------------------------------------------------
// Density grid: row-major float array [height × width] normalised to [0,1]
// ---------------------------------------------------------------------------
struct DensityGrid {
    std::vector<float> data;
    int width  = 0;
    int height = 0;

    float&       at(int x, int y)       { return data[y * width + x]; }
    const float& at(int x, int y) const { return data[y * width + x]; }
    int          total()           const { return width * height; }
};

// ---------------------------------------------------------------------------
// Result of one CVT run (returned by every backend)
// ---------------------------------------------------------------------------
struct RunResult {
    std::string backend;
    int    width       = 0;
    int    height      = 0;
    int    n_sites     = 0;
    int    threads     = 1;
    int    iterations  = 0;
    double time_ms     = 0.0;
    float  final_error = 0.0f;
};
