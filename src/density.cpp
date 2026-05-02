#include "density.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <stdexcept>
#include <numeric>

// ---------------------------------------------------------------------------
// Gaussian density: sum of 6 Gaussian lobes (normalised to [0,1])
// ---------------------------------------------------------------------------
DensityGrid generate_gaussian(int W, int H) {
    DensityGrid grid;
    grid.width = W; grid.height = H;
    grid.data.resize(W * H, 0.0f);

    struct Gauss { float cx, cy, sx, sy, amp; };
    const Gauss bumps[] = {
        {0.20f, 0.20f, 0.07f, 0.07f, 1.0f},
        {0.70f, 0.30f, 0.10f, 0.10f, 0.8f},
        {0.50f, 0.72f, 0.05f, 0.05f, 1.3f},
        {0.30f, 0.55f, 0.09f, 0.13f, 0.6f},
        {0.80f, 0.75f, 0.07f, 0.07f, 0.9f},
        {0.60f, 0.10f, 0.06f, 0.06f, 0.7f},
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float nx = (x + 0.5f) / W;
            float ny = (y + 0.5f) / H;
            float val = 0.03f;
            for (const auto& g : bumps) {
                float dx = (nx - g.cx) / g.sx;
                float dy = (ny - g.cy) / g.sy;
                val += g.amp * std::exp(-0.5f * (dx*dx + dy*dy));
            }
            grid.data[y * W + x] = val;
        }
    }
    normalize_density(grid);
    return grid;
}

// ---------------------------------------------------------------------------
// Checkerboard density
// ---------------------------------------------------------------------------
DensityGrid generate_checkerboard(int W, int H, int n_cells) {
    DensityGrid grid;
    grid.width = W; grid.height = H;
    grid.data.resize(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int cx = (x * n_cells) / W;
            int cy = (y * n_cells) / H;
            grid.data[y * W + x] = ((cx ^ cy) & 1) ? 0.9f : 0.1f;
        }
    return grid;
}

// ---------------------------------------------------------------------------
// PGM loader (P5 binary / P2 text)
// ---------------------------------------------------------------------------
DensityGrid load_pgm(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PGM: " + filename);

    std::string magic;
    f >> magic;
    if (magic != "P5" && magic != "P2")
        throw std::runtime_error("Unsupported PGM magic: " + magic);

    int W, H, maxval;
    f >> W >> H >> maxval;
    f.ignore(1);

    DensityGrid grid;
    grid.width = W; grid.height = H;
    grid.data.resize(W * H);

    if (magic == "P5") {
        std::vector<uint8_t> raw(W * H);
        f.read(reinterpret_cast<char*>(raw.data()), W * H);
        for (int i = 0; i < W * H; ++i)
            grid.data[i] = raw[i] / 255.0f;
    } else {
        for (int i = 0; i < W * H; ++i) {
            int v; f >> v;
            grid.data[i] = v / static_cast<float>(maxval);
        }
    }
    normalize_density(grid);
    return grid;
}

// ---------------------------------------------------------------------------
// Normalise density to [0,1]; if flat → set all to 1
// ---------------------------------------------------------------------------
void normalize_density(DensityGrid& grid) {
    if (grid.data.empty()) return;
    float mn = *std::min_element(grid.data.begin(), grid.data.end());
    float mx = *std::max_element(grid.data.begin(), grid.data.end());
    if (mx - mn < 1e-9f) { std::fill(grid.data.begin(), grid.data.end(), 1.0f); return; }
    for (auto& v : grid.data) v = (v - mn) / (mx - mn);
}

// ---------------------------------------------------------------------------
// Weighted random initialisation via inverse CDF sampling
// ---------------------------------------------------------------------------
std::vector<Point2D> init_sites_weighted(const DensityGrid& grid,
                                          int N, unsigned seed) {
    const int total = grid.total();
    std::vector<float> cdf(total);
    double sum = 0.0;
    for (int i = 0; i < total; ++i) sum += grid.data[i];
    double running = 0.0;
    for (int i = 0; i < total; ++i) {
        running   += grid.data[i];
        cdf[i]     = static_cast<float>(running / sum);
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<Point2D> sites(N);
    for (int i = 0; i < N; ++i) {
        float u   = dist(rng);
        int   idx = static_cast<int>(
                        std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        idx = std::min(idx, total - 1);
        sites[i] = { static_cast<float>(idx % grid.width)  + 0.5f,
                     static_cast<float>(idx / grid.width) + 0.5f };
    }
    return sites;
}

// ---------------------------------------------------------------------------
// Kirsch's analytical stress field
//
// Models a thin elastic plate with a circular hole of radius `a` at the
// centre, loaded by far-field uniaxial tension σ₀ in the Y-direction.
//
// In polar coordinates (r, θ) from the hole centre:
//   σ_rr = (σ₀/2)(1 − a²/r²) + (σ₀/2)(1 − 4a²/r² + 3a⁴/r⁴)cos(2θ)
//   σ_θθ = (σ₀/2)(1 + a²/r²) − (σ₀/2)(1 +         3a⁴/r⁴)cos(2θ)
//   τ_rθ = −(σ₀/2)(1 + 2a²/r² − 3a⁴/r⁴)sin(2θ)
//
// These are converted to Cartesian σ_xx, σ_yy, τ_xy and then combined into
// the von Mises equivalent stress for 2-D plane stress.
//
// Hot-spots: at θ = ±π/2 on the hole boundary  →  σ_VM ≈ 3σ₀
// ---------------------------------------------------------------------------
DensityGrid generate_stress_field(int W, int H) {
    DensityGrid grid;
    grid.width  = W;
    grid.height = H;
    grid.data.resize(W * H, 0.0f);

    const float cx    = W * 0.5f;
    const float cy    = H * 0.5f;
    const float a     = std::min(W, H) * 0.12f;   // hole radius = 12% of short side
    const float a2    = a * a;
    const float a4    = a2 * a2;
    const float sigma = 1.0f;                      // far-field stress magnitude

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float dx = x + 0.5f - cx;
            float dy = y + 0.5f - cy;
            float r2 = dx*dx + dy*dy;

            if (r2 < a2) {
                // Inside hole: no material → zero stress
                grid.data[y * W + x] = 0.0f;
                continue;
            }

            float r    = std::sqrt(r2);
            float r4   = r2 * r2;
            float cos2 = (dx*dx - dy*dy) / r2;   // cos(2θ) = (x²−y²)/r²
            float sin2 = 2.0f * dx * dy   / r2;  // sin(2θ) = 2xy/r²

            // Kirsch stress components in cylindrical coords
            float srr = (sigma/2.f) * (1.f - a2/r2)
                      + (sigma/2.f) * (1.f - 4.f*a2/r2 + 3.f*a4/r4) * cos2;
            float stt = (sigma/2.f) * (1.f + a2/r2)
                      - (sigma/2.f) * (1.f + 3.f*a4/r4) * cos2;
            float trt = -(sigma/2.f) * (1.f + 2.f*a2/r2 - 3.f*a4/r4) * sin2;

            // Convert cylindrical → Cartesian using rotation matrix
            float c  = dx / r, s = dy / r;
            float c2 = c*c, s2 = s*s, cs = c*s;

            float sxx = srr*c2 - 2.f*trt*cs + stt*s2;
            float syy = srr*s2 + 2.f*trt*cs + stt*c2;
            float txy = (srr - stt)*cs + trt*(c2 - s2);

            // Von Mises equivalent stress (2-D plane stress)
            float vm = std::sqrt(std::max(0.f,
                           sxx*sxx - sxx*syy + syy*syy + 3.f*txy*txy));
            grid.data[y * W + x] = vm;
        }
    }
    normalize_density(grid);
    return grid;
}

// ---------------------------------------------------------------------------
// Uniform random site initialisation (ignores density)
// Used as the weakest baseline in the quality comparison.
// ---------------------------------------------------------------------------
std::vector<Point2D> init_sites_uniform(const DensityGrid& grid,
                                         int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xd(0.5f, grid.width  - 0.5f);
    std::uniform_real_distribution<float> yd(0.5f, grid.height - 0.5f);
    std::vector<Point2D> sites(N);
    for (auto& s : sites) s = {xd(rng), yd(rng)};
    return sites;
}

// ---------------------------------------------------------------------------
// CVT energy – density-weighted mean squared distance to nearest site
//
//   E = [Σ_p ρ(p) · ‖p − s(p)‖²] / [Σ_p ρ(p)]
//
// where s(p) is the site assigned to pixel p.
// Lower is better.  Lloyd's algorithm monotonically decreases this value.
// ---------------------------------------------------------------------------
double compute_cvt_energy(const DensityGrid&          grid,
                           const std::vector<Point2D>& sites,
                           const std::vector<int>&      assignment) {
    const int W  = grid.width, H = grid.height;
    double sum_e = 0.0, sum_w = 0.0;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int   idx  = y * W + x;
            int   site = assignment[idx];
            float w    = grid.data[idx];
            if (site < 0 || site >= (int)sites.size()) continue;

            float ddx = x + 0.5f - sites[site].x;
            float ddy = y + 0.5f - sites[site].y;
            sum_e += w * (ddx*ddx + ddy*ddy);
            sum_w += w;
        }
    }
    return (sum_w > 1e-12) ? sum_e / sum_w : 0.0;
}

