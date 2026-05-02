/*
 * output.cpp  –  PNG writer (via stb_image_write) + CSV + console table
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "output.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// HSV → RGB  (H in [0,360], S,V in [0,1])
// ---------------------------------------------------------------------------
static void hsv_to_rgb(float H, float S, float V,
                        uint8_t& r, uint8_t& g, uint8_t& b)
{
    float C = V * S;
    float X = C * (1.0f - std::fabs(std::fmod(H / 60.0f, 2.0f) - 1.0f));
    float m = V - C;
    float rr, gg, bb;
    if      (H < 60)  { rr=C; gg=X; bb=0; }
    else if (H < 120) { rr=X; gg=C; bb=0; }
    else if (H < 180) { rr=0; gg=C; bb=X; }
    else if (H < 240) { rr=0; gg=X; bb=C; }
    else if (H < 300) { rr=X; gg=0; bb=C; }
    else              { rr=C; gg=0; bb=X; }
    r = static_cast<uint8_t>((rr+m)*255.0f);
    g = static_cast<uint8_t>((gg+m)*255.0f);
    b = static_cast<uint8_t>((bb+m)*255.0f);
}

// ---------------------------------------------------------------------------
// write_voronoi_png
//   Each Voronoi cell gets a deterministic HSV colour driven by its index.
//   Brightness is modulated by the local density so structure is visible.
//   Each site is marked with a white 5×5 cross.
// ---------------------------------------------------------------------------
void write_voronoi_png(const std::string&         filename,
                       const DensityGrid&          grid,
                       const std::vector<Point2D>& sites,
                       const std::vector<int>&     assignment)
{
    const int W = grid.width, H = grid.height;
    const int N = static_cast<int>(sites.size());

    // Build per-site colour table
    struct RGB { uint8_t r, g, b; };
    std::vector<RGB> palette(N);
    for (int i = 0; i < N; ++i) {
        // golden-ratio hue spread for perceptual separation
        float hue = std::fmod(i * 137.508f, 360.0f);
        hsv_to_rgb(hue, 0.80f, 0.92f, palette[i].r, palette[i].g, palette[i].b);
    }

    std::vector<uint8_t> img(W * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int   site = assignment[y * W + x];
            float d    = 0.45f + 0.55f * grid.data[y * W + x];  // lighten by density
            int   idx  = (y * W + x) * 3;
            img[idx+0] = static_cast<uint8_t>(palette[site].r * d);
            img[idx+1] = static_cast<uint8_t>(palette[site].g * d);
            img[idx+2] = static_cast<uint8_t>(palette[site].b * d);
        }
    }

    // Draw white cross marker at each site
    auto draw_cross = [&](int sx, int sy) {
        for (int d = -3; d <= 3; ++d) {
            auto set = [&](int px, int py) {
                if (px>=0 && px<W && py>=0 && py<H) {
                    int i = (py*W+px)*3;
                    img[i] = img[i+1] = img[i+2] = 255;
                }
            };
            set(sx+d, sy); set(sx, sy+d);
        }
    };
    for (int i = 0; i < N; ++i)
        draw_cross(static_cast<int>(sites[i].x), static_cast<int>(sites[i].y));

    stbi_write_png(filename.c_str(), W, H, 3, img.data(), W * 3);
}

// ---------------------------------------------------------------------------
// write_results_csv
// ---------------------------------------------------------------------------
void write_results_csv(const std::string&           filename,
                       const std::vector<RunResult>& results)
{
    FILE* f = fopen(filename.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename.c_str()); return; }

    fprintf(f, "backend,width,height,n_sites,threads,iterations,time_ms,error\n");
    for (const auto& r : results)
        fprintf(f, "%s,%d,%d,%d,%d,%d,%.3f,%.6f\n",
                r.backend.c_str(), r.width, r.height, r.n_sites,
                r.threads, r.iterations, r.time_ms, r.final_error);
    fclose(f);
}

// ---------------------------------------------------------------------------
// print_benchmark_table
// ---------------------------------------------------------------------------
void print_benchmark_table(const std::vector<RunResult>& results)
{
    if (results.empty()) return;

    // Determine baseline (serial or first result)
    double base_ms = results[0].time_ms;
    for (const auto& r : results) if (r.backend == "serial") { base_ms = r.time_ms; break; }

    printf("\n%-10s %6s %9s %6s %8s %10s\n",
           "Backend", "Thread", "Time(ms)", "Iter", "Error", "Speedup");
    printf("%-10s %6s %9s %6s %8s %10s\n",
           "-------", "------", "--------", "----", "-----", "-------");

    for (const auto& r : results) {
        double sp = base_ms / r.time_ms;
        printf("%-10s %6d %9.1f %6d %8.2e %9.2fx\n",
               r.backend.c_str(), r.threads, r.time_ms,
               r.iterations, r.final_error, sp);
    }
    printf("\n");
}
