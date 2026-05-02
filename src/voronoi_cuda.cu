/*
 * voronoi_cuda.cu  –  CUDA GPU implementation of weighted CVT
 *
 * Architecture target: sm_86 (RTX 2050, Ampere GA107)
 *
 * Kernels per Lloyd iteration:
 *   1. jfa_init_kernel          – zero-fill assignment buffer
 *   2. jfa_seed_kernel          – seed site positions into assignment
 *   3. jfa_flood_kernel         – one JFA pass at stride s (ping → pong)
 *      [repeated ceil(log2(max(W,H)))+1 times – 1+JFA variant]
 *   4. accumulate_centroids_shmem_kernel – block-local shared-mem atomics
 *   5. update_sites_kernel      – divide, compute displacement
 *
 * JFA replaces brute-force O(P×N) assign with O(P × log(max(W,H))) passes.
 * Shared-mem centroid replaces per-pixel global atomicAdd with per-block
 * accumulation: reduces global atomic traffic by ~blockDim.x (256×).
 */

#ifdef ENABLE_CUDA

#include "voronoi.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ─── Error macro ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("CUDA error in " __FILE__ " line ") +             \
                std::to_string(__LINE__) + ": " +                             \
                cudaGetErrorString(_e));                                        \
    } while (0)

// =============================================================================
// JFA – Jump Flooding Algorithm for Voronoi assignment
//
// Each pixel stores the index of the site it believes is nearest.
// -1 means "unknown".  Passes propagate information at decreasing strides.
// =============================================================================

// ─── K1a: initialise all pixels to "unknown" (-1) ────────────────────────────
__global__ void jfa_init_kernel(int* __restrict__ buf, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) buf[idx] = -1;
}

// ─── K1b: seed each site's pixel with its own index ──────────────────────────
//   sites_x/y are in pixel coordinates (float); we clamp to grid bounds.
__global__ void jfa_seed_kernel(int* __restrict__ buf,
                                const float* __restrict__ sites_x,
                                const float* __restrict__ sites_y,
                                int W, int H, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int px = max(0, min(W - 1, (int)roundf(sites_x[i])));
    int py = max(0, min(H - 1, (int)roundf(sites_y[i])));
    buf[py * W + px] = i;   // last write wins (fine for non-overlapping sites)
}

// ─── K1c: one JFA flood pass ─────────────────────────────────────────────────
//   Reads from `in`, writes to `out`.  Each pixel inspects 8 neighbours at
//   offset ±stride and adopts whichever neighbour provides the nearest site.
__global__ void jfa_flood_kernel(const int*   __restrict__ in,
                                       int*   __restrict__ out,
                                 const float* __restrict__ sites_x,
                                 const float* __restrict__ sites_y,
                                 int W, int H, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;

    int px = idx % W, py = idx / W;

    int   best_site = in[idx];
    float best_dist = 1e30f;

    if (best_site >= 0) {
        float dx = px - sites_x[best_site];
        float dy = py - sites_y[best_site];
        best_dist = dx*dx + dy*dy;
    }

    // 8 neighbours (+ self skipped) at distance `stride`
    for (int dy_off = -1; dy_off <= 1; ++dy_off) {
        for (int dx_off = -1; dx_off <= 1; ++dx_off) {
            if (dx_off == 0 && dy_off == 0) continue;
            int nx = px + dx_off * stride;
            int ny = py + dy_off * stride;
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;

            int nb_site = in[ny * W + nx];
            if (nb_site < 0) continue;

            float ddx = px - sites_x[nb_site];
            float ddy = py - sites_y[nb_site];
            float d   = ddx*ddx + ddy*ddy;
            if (d < best_dist) { best_dist = d; best_site = nb_site; }
        }
    }
    out[idx] = best_site;
}

// =============================================================================
// K2: Centroid accumulation with shared-memory block-local atomics
//
// Dynamic shared memory layout (3*N doubles):
//   sh_wx[0..N-1]  sh_wy[N..2N-1]  sh_w[2N..3N-1]
//
// Each block accumulates into on-chip shared memory (fast, low latency),
// then flushes one entry per thread into global memory.
// This reduces global atomic traffic by a factor of ~blockDim.x versus
// the original per-pixel global atomicAdd.
//
// Requires sm_60+ for double atomicAdd in shared memory (sm_86 = fine).
// Shared mem needed: 3 * N * 8 bytes; for N=1024 → 24 KB < 48 KB limit.
// =============================================================================
__global__ void accumulate_centroids_shmem_kernel(
    const float* __restrict__ density,
    const int*   __restrict__ assignment,
    double* __restrict__ sum_wx,
    double* __restrict__ sum_wy,
    double* __restrict__ sum_w,
    int W, int H, int N)
{
    extern __shared__ double sh[];   // allocated by host: 3*N*sizeof(double)
    double* sh_wx = sh;
    double* sh_wy = sh + N;
    double* sh_wt = sh + 2*N;

    // Zero the shared accumulators
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        sh_wx[k] = 0.0;
        sh_wy[k] = 0.0;
        sh_wt[k] = 0.0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < W * H) {
        int   site = assignment[idx];
        float w    = density[idx];
        int   px   = idx % W, py = idx / W;

        // atomicAdd into shared memory – same operation as before but
        // on-chip: ~100× lower latency, bank-conflict-free for spread sites
        atomicAdd(&sh_wx[site], (double)(w * px));
        atomicAdd(&sh_wy[site], (double)(w * py));
        atomicAdd(&sh_wt[site], (double)w);
    }
    __syncthreads();

    // Each thread flushes a contiguous slice of sites → global memory
    // One global atomicAdd per (block, site) pair instead of one per pixel
    for (int k = threadIdx.x; k < N; k += blockDim.x) {
        if (sh_wt[k] > 0.0) {
            atomicAdd(&sum_wx[k], sh_wx[k]);
            atomicAdd(&sum_wy[k], sh_wy[k]);
            atomicAdd(&sum_w [k], sh_wt[k]);
        }
    }
}

// =============================================================================
// K3: update site positions and record displacement (unchanged)
// =============================================================================
__global__ void update_sites_kernel(float* sites_x, float* sites_y,
                                    const double* sum_wx, const double* sum_wy,
                                    const double* sum_w,
                                    float* displacement, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float nx = sites_x[i], ny = sites_y[i];
    if (sum_w[i] > 1e-10) {
        nx = static_cast<float>(sum_wx[i] / sum_w[i]);
        ny = static_cast<float>(sum_wy[i] / sum_w[i]);
    }
    float dx = nx - sites_x[i], dy = ny - sites_y[i];
    displacement[i] = sqrtf(dx*dx + dy*dy);
    sites_x[i] = nx;
    sites_y[i] = ny;
}

// =============================================================================
// Host entry point
// =============================================================================
RunResult run_cuda(const DensityGrid&   grid,
                   std::vector<Point2D> sites,
                   int max_iter, float eps,
                   std::vector<int>&    assignment)
{
    const int W = grid.width, H = grid.height;
    const int N = static_cast<int>(sites.size());
    const int pixels = W * H;
    assignment.resize(pixels);

    // ── Separate SoA on host ──────────────────────────────────────────────────
    std::vector<float> hx(N), hy(N);
    for (int i = 0; i < N; ++i) { hx[i] = sites[i].x; hy[i] = sites[i].y; }

    // ── Device allocations ────────────────────────────────────────────────────
    float  *d_density, *d_sites_x, *d_sites_y, *d_disp;
    int    *d_assign_a, *d_assign_b;          // ping-pong buffers for JFA
    double *d_sum_wx, *d_sum_wy, *d_sum_w;

    CUDA_CHECK(cudaMalloc(&d_density,  pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sites_x,  N      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sites_y,  N      * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_assign_a, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_assign_b, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum_wx,   N      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_wy,   N      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum_w,    N      * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_disp,     N      * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_density, grid.data.data(), pixels * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sites_x, hx.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sites_y, hy.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── Kernel launch parameters ──────────────────────────────────────────────
    const int BLK    = 256;
    const int GBLK   = (pixels + BLK - 1) / BLK;
    const int GBLK_N = (N + BLK - 1) / BLK;

    // JFA pass count: ceil(log2(max(W,H)))
    const int max_dim  = std::max(W, H);
    const int jfa_steps = static_cast<int>(std::ceil(std::log2(static_cast<float>(max_dim))));

    // Shared memory size for centroid kernel: 3 * N doubles
    const size_t shmem_bytes = 3 * N * sizeof(double);

    // ── Start timing after transfers ──────────────────────────────────────────
    Timer timer; timer.start();

    int   iter  = 0;
    float error = 1e30f;
    std::vector<float> h_disp(N);

    while (iter < max_iter && error > eps) {

        // ── Upload current site positions to device ───────────────────────────
        CUDA_CHECK(cudaMemcpy(d_sites_x, hx.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sites_y, hy.data(), N * sizeof(float),
                              cudaMemcpyHostToDevice));

        // ── JFA: initialise + seed ────────────────────────────────────────────
        jfa_init_kernel<<<GBLK, BLK>>>(d_assign_a, pixels);
        CUDA_CHECK(cudaGetLastError());

        jfa_seed_kernel<<<GBLK_N, BLK>>>(d_assign_a, d_sites_x, d_sites_y,
                                          W, H, N);
        CUDA_CHECK(cudaGetLastError());

        // ── JFA: flood passes (1+JFA: standard passes then one extra stride=1)
        int*  ping = d_assign_a;
        int*  pong = d_assign_b;

        for (int k = jfa_steps - 1; k >= 0; --k) {
            int stride = 1 << k;
            jfa_flood_kernel<<<GBLK, BLK>>>(ping, pong,
                                             d_sites_x, d_sites_y,
                                             W, H, stride);
            CUDA_CHECK(cudaGetLastError());
            std::swap(ping, pong);
        }
        // Extra stride=1 pass (1+JFA correction — fixes most boundary errors)
        jfa_flood_kernel<<<GBLK, BLK>>>(ping, pong,
                                         d_sites_x, d_sites_y,
                                         W, H, 1);
        CUDA_CHECK(cudaGetLastError());
        std::swap(ping, pong);

        // `ping` now holds the final assignment for this iteration
        int* d_assign_final = ping;

        // ── Centroid accumulation (shared-memory) ─────────────────────────────
        CUDA_CHECK(cudaMemset(d_sum_wx, 0, N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_wy, 0, N * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w,  0, N * sizeof(double)));

        accumulate_centroids_shmem_kernel<<<GBLK, BLK, shmem_bytes>>>(
            d_density, d_assign_final,
            d_sum_wx, d_sum_wy, d_sum_w,
            W, H, N);
        CUDA_CHECK(cudaGetLastError());

        // ── Update site positions ─────────────────────────────────────────────
        update_sites_kernel<<<GBLK_N, BLK>>>(
            d_sites_x, d_sites_y,
            d_sum_wx, d_sum_wy, d_sum_w,
            d_disp, N);
        CUDA_CHECK(cudaGetLastError());

        // Synchronise once per iteration; copy back N floats for convergence
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hx.data(), d_sites_x, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), d_sites_y, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_disp.data(), d_disp, N * sizeof(float),
                              cudaMemcpyDeviceToHost));

        error = *std::max_element(h_disp.begin(), h_disp.end());
        ++iter;

        // Keep track of which buffer holds the result for final copy-back
        // (copy final assignment after loop ends)
        // Store final assignment pointer alias so we can copy after the loop.
        // Since ping/pong swap each pass and we track `ping` = current result,
        // we must copy from d_assign_final BEFORE the next iteration overwrites it.
        // So we copy the assignment every iter here:
        if (iter == max_iter || error <= eps) {
            CUDA_CHECK(cudaMemcpy(assignment.data(), d_assign_final,
                                  pixels * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }
    }

    double ms = timer.elapsed_ms();

    // Free resources
    cudaFree(d_density);  cudaFree(d_sites_x);  cudaFree(d_sites_y);
    cudaFree(d_assign_a); cudaFree(d_assign_b);
    cudaFree(d_sum_wx);   cudaFree(d_sum_wy);   cudaFree(d_sum_w);
    cudaFree(d_disp);

    return {"cuda", W, H, N, 1, iter, ms, error};
}

#endif  // ENABLE_CUDA
