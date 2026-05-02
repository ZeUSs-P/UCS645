#include "voronoi.h"
#include "timer.h"
#include <cmath>
#include <algorithm>
#include <limits>

RunResult run_serial(const DensityGrid&   grid,
                     std::vector<Point2D> sites,
                     int max_iter, float eps,
                     std::vector<int>&    assignment) {
    const int W = grid.width, H = grid.height;
    const int N = static_cast<int>(sites.size());
    assignment.assign(W * H, 0);

    Timer timer; timer.start();
    int   iter  = 0;
    float error = std::numeric_limits<float>::max();

    while (iter < max_iter && error > eps) {

        // ── Assignment: each pixel → nearest site ─────────────────────────
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float best  = std::numeric_limits<float>::max();
                int   best_i = 0;
                for (int i = 0; i < N; ++i) {
                    float dx = x - sites[i].x, dy = y - sites[i].y;
                    float d  = dx*dx + dy*dy;
                    if (d < best) { best = d; best_i = i; }
                }
                assignment[y * W + x] = best_i;
            }
        }

        // ── Centroid: density-weighted centre per site ────────────────────
        std::vector<double> sum_wx(N, 0.0), sum_wy(N, 0.0), sum_w(N, 0.0);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int    i = assignment[y * W + x];
                double w = grid.data[y * W + x];
                sum_wx[i] += w * x;
                sum_wy[i] += w * y;
                sum_w [i] += w;
            }

        // ── Move sites + convergence check ───────────────────────────────
        error = 0.0f;
        for (int i = 0; i < N; ++i) {
            if (sum_w[i] < 1e-10) continue;
            float nx = static_cast<float>(sum_wx[i] / sum_w[i]);
            float ny = static_cast<float>(sum_wy[i] / sum_w[i]);
            float dx = nx - sites[i].x, dy = ny - sites[i].y;
            error    = std::max(error, std::sqrt(dx*dx + dy*dy));
            sites[i] = {nx, ny};
        }
        ++iter;
    }

    return {"serial", W, H, N, 1, iter, timer.elapsed_ms(), error};
}
