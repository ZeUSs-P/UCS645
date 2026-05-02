/*
 * bench_bandwidth.cu  –  PCIe Host<->Device bandwidth benchmark
 *
 * Criterion: B4 – "Correct MB/s values, both H2D and D2H curves present"
 *
 * Method:
 *   - Allocates one pinned host buffer and one device buffer.
 *   - Times cudaMemcpy (H2D and D2H) using cudaEventRecord for each of
 *     N_SIZES transfer sizes between MIN_MB and MAX_MB.
 *   - Each size is repeated REPS times; the MINIMUM time is used
 *     (removes OS scheduling jitter, matches hardware ceiling).
 *   - Reports bandwidth in MB/s = bytes / (time_s * 1e6).
 *
 * Output:
 *   - One [BW] tag line per measurement (parsed by Python).
 *   - results/bandwidth.csv with columns: direction,size_mb,bandwidth_mbs
 *
 * Compile (via Makefile):
 *   make ENABLE_CUDA=1 bw
 *
 * Usage:
 *   bench_bandwidth.exe [--output <dir>]
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  define MKDIR(p) mkdir(p, 0755)
#endif

// ─── Error helper ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── Configuration ────────────────────────────────────────────────────────────
static const int REPS = 5;   // repeats per (direction, size) — take minimum

// Transfer sizes in MiB
static const size_t SIZES_MB[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
static const int    N_SIZES    = (int)(sizeof(SIZES_MB) / sizeof(SIZES_MB[0]));

// ─── Measure one direction ────────────────────────────────────────────────────
// Returns bandwidth in MB/s.
static double measure_bw(void* h_buf, void* d_buf, size_t bytes,
                          cudaMemcpyKind kind, int reps)
{
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    float best_ms = 1e30f;

    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, kind));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (ms < best_ms) best_ms = ms;
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // MB/s = (bytes / 1e6) / (best_ms / 1000) = bytes * 1000 / (best_ms * 1e6)
    double bw = (static_cast<double>(bytes) / 1.0e6) /
                (static_cast<double>(best_ms) / 1000.0);
    return bw;
}

// ─── Entry point ──────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    std::string outdir = "results";
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            outdir = argv[++i];
    }

    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[BW] device: %s  PCIe Gen%d x%d\n",
           prop.name, prop.pciBusID, prop.pciDeviceID);
    printf("[BW] %-10s  %-12s  %-14s  %-14s\n",
           "Size(MB)", "H2D(MB/s)", "D2H(MB/s)", "Avg(MB/s)");
    printf("[BW] %-10s  %-12s  %-14s  %-14s\n",
           "--------", "---------", "---------", "---------");

    // Find maximum size and allocate once
    size_t max_bytes = SIZES_MB[N_SIZES - 1] * 1024ULL * 1024ULL;

    void* h_buf = nullptr;
    void* d_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_buf, max_bytes));   // pinned = real PCIe speed
    CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));

    // Initialise host buffer with dummy data so transfers aren't optimised away
    memset(h_buf, 0xAB, max_bytes);

    // ── CSV output ────────────────────────────────────────────────────────────
    // Make output directory (best effort)
    MKDIR(outdir.c_str());

    std::string csv_path = outdir + "/bandwidth.csv";
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        fprintf(stderr, "[BW] WARNING: cannot open %s for writing\n",
                csv_path.c_str());
    } else {
        csv << "direction,size_mb,bandwidth_mbs\n";
    }

    // ── Sweep sizes ───────────────────────────────────────────────────────────
    for (int i = 0; i < N_SIZES; ++i) {
        size_t mb    = SIZES_MB[i];
        size_t bytes = mb * 1024ULL * 1024ULL;

        double h2d = measure_bw(h_buf, d_buf, bytes, cudaMemcpyHostToDevice,   REPS);
        double d2h = measure_bw(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost,   REPS);
        double avg = (h2d + d2h) / 2.0;

        printf("[BW] %-10zu  %-12.1f  %-14.1f  %-14.1f\n", mb, h2d, d2h, avg);
        fflush(stdout);

        if (csv.is_open()) {
            csv << "H2D," << mb << "," << h2d << "\n";
            csv << "D2H," << mb << "," << d2h << "\n";
        }
    }

    printf("[BW] CSV written to: %s\n", csv_path.c_str());

    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_buf));

    return 0;
}
