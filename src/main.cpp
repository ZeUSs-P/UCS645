/*
 * main.cpp  –  Entry point for voronoi_cvt
 *
 * Usage:  voronoi_cvt [options]
 *   --width   N    Grid width            (default 1024)
 *   --height  N    Grid height           (default 1024)
 *   --sites   N    Number of sites       (default 256)
 *   --iter    N    Max Lloyd iterations  (default 100)
 *   --eps     F    Convergence threshold (default 1e-3)
 *   --threads N    OMP thread count  (0=auto)
 *   --density T    gaussian|check|pgm
 *   --input   F    PGM file (used with --density pgm)
 *   --backend T    serial|omp|cuda|all   (default all)
 *   --output  D    Output directory      (default results)
 *   --seed    N    RNG seed              (default 42)
 *   --no-img       Skip PNG output (faster for scaling runs)
 */

#include "common.h"
#include "density.h"
#include "voronoi.h"
#include "output.h"
#include <fstream>
#include <iomanip>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef _WIN32
#  include <direct.h>
#  define MKDIR(p) _mkdir(p)
#else
#  include <sys/stat.h>
#  include <sys/types.h>
#  define MKDIR(p) mkdir(p, 0755)
#endif

static void make_dir(const std::string& path) {
    MKDIR(path.c_str());   // silently ignore if already exists
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct Args {
    int         width    = 1024;
    int         height   = 1024;
    int         sites    = 256;
    int         max_iter = 100;
    float       eps      = 1e-3f;
    int         threads  = 0;
    std::string density  = "gaussian";
    std::string input;
    std::string backend  = "all";
    std::string output   = "results";
    unsigned    seed     = 42;
    bool        no_img   = false;
    bool        quality  = false;  // run sampling quality comparison
};

static void print_help(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --width   N    Grid width            (default 1024)\n"
           "  --height  N    Grid height           (default 1024)\n"
           "  --sites   N    Sample sites          (default 256)\n"
           "  --iter    N    Max iterations        (default 100)\n"
           "  --eps     F    Convergence epsilon   (default 1e-3)\n"
           "  --threads N    OMP threads (0=auto)  (default 0)\n"
           "  --density T    gaussian|check|pgm|stress (default gaussian)\n"
           "  --input   F    PGM file  (density=pgm)\n"
           "  --backend T    serial|omp|cuda|all   (default all)\n"
           "  --output  D    Output directory      (default results)\n"
           "  --seed    N    RNG seed              (default 42)\n"
           "  --no-img       Skip PNG output\n"
           "  --quality      Run uniform/importance/CVT quality comparison\n"
           "                 (auto-enabled when --density stress)\n", prog);
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", k.c_str()); exit(1);
            }
            return argv[++i];
        };
        if      (k == "--width")   a.width    = std::stoi(next());
        else if (k == "--height")  a.height   = std::stoi(next());
        else if (k == "--sites")   a.sites    = std::stoi(next());
        else if (k == "--iter")    a.max_iter = std::stoi(next());
        else if (k == "--eps")     a.eps      = std::stof(next());
        else if (k == "--threads") a.threads  = std::stoi(next());
        else if (k == "--density") a.density  = next();
        else if (k == "--input")   a.input    = next();
        else if (k == "--backend") a.backend  = next();
        else if (k == "--output")  a.output   = next();
        else if (k == "--seed")    a.seed     = (unsigned)std::stoi(next());
        else if (k == "--no-img")  a.no_img   = true;
        else if (k == "--quality") a.quality  = true;
        else if (k == "--help")    { print_help(argv[0]); exit(0); }
        else { fprintf(stderr, "Unknown option: %s\n", k.c_str()); exit(1); }
    }
    return a;
}

// ---------------------------------------------------------------------------
// Run one backend + optional PNG
// ---------------------------------------------------------------------------
static RunResult dispatch(const std::string&          backend,
                           const DensityGrid&           grid,
                           const std::vector<Point2D>& init_sites,
                           const Args&                  a,
                           const std::string&           out_prefix)
{
    std::vector<int> assignment;
    RunResult r;

    printf("[voronoi_cvt] backend=%-6s  %dx%d  sites=%d ...\n",
           backend.c_str(), a.width, a.height, a.sites);
    fflush(stdout);

    if (backend == "serial") {
        r = run_serial(grid, init_sites, a.max_iter, a.eps, assignment);
    } else if (backend == "omp") {
        r = run_omp(grid, init_sites, a.max_iter, a.eps, assignment, a.threads);
#ifdef ENABLE_CUDA
    } else if (backend == "cuda") {
        r = run_cuda(grid, init_sites, a.max_iter, a.eps, assignment);
#endif
    } else {
        fprintf(stderr, "Unknown backend: %s\n", backend.c_str());
        exit(1);
    }

    // Machine-parseable timing line (read by scaling_analysis.py)
    printf("[RESULT] backend=%s width=%d height=%d sites=%d threads=%d "
           "iter=%d time_ms=%.3f error=%.6f\n",
           r.backend.c_str(), r.width, r.height, r.n_sites, r.threads,
           r.iterations, r.time_ms, r.final_error);
    fflush(stdout);

    if (!a.no_img) {
        std::string png = out_prefix + ".png";
        write_voronoi_png(png, grid, init_sites, assignment);
        printf("  -> wrote %s\n", png.c_str());
    }
    return r;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Args a = parse_args(argc, argv);

    make_dir(a.output);

    // ── Load / generate density ──────────────────────────────────────────
    DensityGrid grid;
    try {
        if (a.density == "gaussian") {
            grid = generate_gaussian(a.width, a.height);
        } else if (a.density == "check") {
            grid = generate_checkerboard(a.width, a.height);
        } else if (a.density == "pgm") {
            if (a.input.empty()) {
                fprintf(stderr, "Error: --density pgm requires --input <file.pgm>\n");
                return 1;
            }
            grid = load_pgm(a.input);
            a.width  = grid.width;
            a.height = grid.height;
        } else if (a.density == "stress") {
            grid     = generate_stress_field(a.width, a.height);
            a.quality = true;   // auto-enable quality comparison for stress runs
        } else {
            fprintf(stderr, "Unknown density: %s\n", a.density.c_str());
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading density: %s\n", e.what());
        return 1;
    }

    printf("[voronoi_cvt] Density '%s'  grid=%dx%d  sites=%d\n",
           a.density.c_str(), a.width, a.height, a.sites);

    // ── Init sites ───────────────────────────────────────────────────────
    std::vector<Point2D> init_sites =
        init_sites_weighted(grid, a.sites, a.seed);

    // ── Dispatch ─────────────────────────────────────────────────────────
    auto out_path = [&](const std::string& tag) {
        return a.output + "/" + tag +
               "_" + std::to_string(a.width) +
               "x" + std::to_string(a.height);
    };

    std::vector<RunResult> results;

    if (a.backend == "all") {
        results.push_back(dispatch("serial", grid, init_sites, a, out_path("serial")));
        results.push_back(dispatch("omp",    grid, init_sites, a, out_path("omp")));
#ifdef ENABLE_CUDA
        results.push_back(dispatch("cuda",   grid, init_sites, a, out_path("cuda")));
#endif
    } else {
        results.push_back(dispatch(a.backend, grid, init_sites, a,
                                   out_path(a.backend)));
    }

    // ── Output ───────────────────────────────────────────────────────────
    print_benchmark_table(results);

    std::string csv = a.output + "/results.csv";
    write_results_csv(csv, results);
    printf("[voronoi_cvt] CSV -> %s\n", csv.c_str());

    // ── Quality benchmark (uniform vs importance-sampled vs CVT) ─────────────
    if (a.quality) {
        printf("\n[voronoi_cvt] === Sampling Quality Comparison ===\n");
        printf("  Metric: density-weighted mean squared distance to nearest site\n");
        printf("  Lower energy = better adaptive placement.\n\n");

        std::vector<Point2D> usites  = init_sites_uniform (grid, a.sites, a.seed + 1);
        std::vector<Point2D> isites  = init_sites_weighted(grid, a.sites, a.seed);
        auto                 cvt_sites = init_sites_weighted(grid, a.sites, a.seed);

        std::vector<int> uassign, iassign, cassign;

        // 1 iteration with huge eps = pure assign step (no site movement)
        run_serial(grid, usites,    1,          1e30f,   uassign);
        run_serial(grid, isites,    1,          1e30f,   iassign);
        run_serial(grid, cvt_sites, a.max_iter, a.eps,   cassign);

        double eu = compute_cvt_energy(grid, usites,    uassign);
        double ei = compute_cvt_energy(grid, isites,    iassign);
        double ec = compute_cvt_energy(grid, cvt_sites, cassign);

        printf("[QUALITY] method=uniform     energy=%12.4f\n", eu);
        printf("[QUALITY] method=importance  energy=%12.4f   (%.2fx vs uniform)\n",
               ei, (ec > 0 ? eu / ei : 0.0));
        printf("[QUALITY] method=cvt         energy=%12.4f   (%.2fx vs uniform)\n",
               ec, (ec > 0 ? eu / ec : 0.0));

        std::string qcsv = a.output + "/quality_comparison.csv";
        std::ofstream f(qcsv);
        f << "method,energy,improvement_vs_uniform\n";
        f << std::fixed << std::setprecision(6);
        f << "uniform,"    << eu << "," << 1.0           << "\n";
        f << "importance," << ei << "," << (ei > 0 ? eu/ei : 0.0) << "\n";
        f << "cvt,"        << ec << "," << (ec > 0 ? eu/ec : 0.0) << "\n";
        printf("[voronoi_cvt] Quality CSV -> %s\n", qcsv.c_str());
    }

    return 0;
}

