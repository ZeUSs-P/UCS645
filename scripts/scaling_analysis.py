#!/usr/bin/env python3
"""
scaling_analysis.py  –  Run strong/weak scaling experiments and plot results.

Usage:
    # Run experiments + plot
    python3 scripts/scaling_analysis.py --binary ./voronoi_cvt --output results [--cuda]

    # Re-plot only (from saved CSVs)
    python3 scripts/scaling_analysis.py --plot-only --output results
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")          # non-interactive backend for WSL / headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Styling ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#3a3d5c",
    "axes.labelcolor":  "#c8cde0",
    "xtick.color":      "#c8cde0",
    "ytick.color":      "#c8cde0",
    "text.color":       "#e8eaf0",
    "grid.color":       "#2a2d42",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
    "lines.linewidth":  2.2,
    "lines.markersize": 7,
})

COLORS = {
    "serial": "#ff6b6b",
    "omp":    "#4ecdc4",
    "cuda":   "#ffe66d",
    "ideal":  "#a8e6cf",
}

# ── Thread sweep (matching your 12-logical-core machine) ──────────────────────
STRONG_THREADS = [1, 2, 4, 6, 8, 10, 12]

# Weak scaling: base area per thread = 512×512
BASE_W = 512
BASE_H = 512
WEAK_THREADS = [1, 2, 4, 6, 8, 12]

SITES    = 256
MAX_ITER = 100
REPS     = 3     # repeat each config REPS times, take minimum (warm cache)

# =============================================================================
# Run experiment: return wall-clock time in ms
# =============================================================================
def run_one(binary: str, backend: str, width: int, height: int,
             threads: int, sites: int = SITES, max_iter: int = MAX_ITER) -> float:
    """Run binary once and return time_ms parsed from stdout."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)

    cmd = [binary,
           "--backend",  backend,
           "--width",    str(width),
           "--height",   str(height),
           "--sites",    str(sites),
           "--iter",     str(max_iter),
           "--density",  "gaussian",
           "--no-img",
           "--threads",  str(threads)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {' '.join(cmd)}", file=sys.stderr)
        return float("nan")

    for line in result.stdout.splitlines():
        if line.startswith("[RESULT]"):
            parts = dict(tok.split("=") for tok in line.split() if "=" in tok)
            return float(parts.get("time_ms", "nan"))

    print(f"  [WARN] No [RESULT] line from: {' '.join(cmd)}", file=sys.stderr)
    print(f"  stdout: {result.stdout[:300]}", file=sys.stderr)
    return float("nan")


def best_of(binary, backend, width, height, threads, reps=REPS, **kwargs):
    times = [run_one(binary, backend, width, height, threads, **kwargs) for _ in range(reps)]
    valid = [t for t in times if not np.isnan(t)]
    return min(valid) if valid else float("nan")


# =============================================================================
# Strong scaling
# =============================================================================
def run_strong(binary: str, output: str, cuda: bool):
    print("\n=== Strong Scaling (1024×1024, 256 sites) ===")
    rows = []

    # Serial baseline (1 thread)
    t_serial = best_of(binary, "serial", 1024, 1024, 1)
    print(f"  serial   t={t_serial:.1f} ms")
    rows.append({"backend":"serial","threads":1,"time_ms":t_serial})

    # OMP sweep
    for T in STRONG_THREADS:
        t = best_of(binary, "omp", 1024, 1024, T)
        print(f"  omp T={T:2d}  t={t:.1f} ms  speedup={t_serial/t:.2f}x")
        rows.append({"backend":"omp","threads":T,"time_ms":t})

    # CUDA
    if cuda:
        t = best_of(binary, "cuda", 1024, 1024, 1)
        print(f"  cuda     t={t:.1f} ms  speedup={t_serial/t:.2f}x")
        rows.append({"backend":"cuda","threads":1,"time_ms":t})

    csv_path = os.path.join(output, "strong_scaling.csv")
    _write_csv(csv_path, ["backend","threads","time_ms"], rows)
    print(f"  -> {csv_path}")
    return rows


# =============================================================================
# Weak scaling
# =============================================================================
def run_weak(binary: str, output: str, cuda: bool):
    print("\n=== Weak Scaling (area ∝ threads, sites ∝ threads) ===")
    rows = []

    for T in WEAK_THREADS:
        # Scale area with threads; keep aspect ratio square
        scale = int(round((T ** 0.5) * BASE_W))
        scale = _round_power2(scale)
        sites = SITES * T
        t = best_of(binary, "omp", scale, scale, T)
        print(f"  omp T={T:2d}  grid={scale}x{scale}  sites={sites}  t={t:.1f} ms")
        rows.append({"threads":T,"width":scale,"sites":sites,"time_ms":t})

    csv_path = os.path.join(output, "weak_scaling.csv")
    _write_csv(csv_path, ["threads","width","sites","time_ms"], rows)
    print(f"  -> {csv_path}")
    return rows


def _round_power2(n):
    powers = [64,128,256,512,768,1024,1280,1536,1792,2048]
    return min(powers, key=lambda p: abs(p-n))


# =============================================================================
# Plotting
# =============================================================================
def plot_all(output: str):
    strong_csv = os.path.join(output, "strong_scaling.csv")
    weak_csv   = os.path.join(output, "weak_scaling.csv")

    if not os.path.exists(strong_csv) or not os.path.exists(weak_csv):
        print("CSV files not found – run without --plot-only first.")
        return

    strong = _read_csv(strong_csv)
    weak   = _read_csv(weak_csv)

    # ── Extract data ─────────────────────────────────────────────────────────
    serial_row = [r for r in strong if r["backend"]=="serial"]
    t_serial   = float(serial_row[0]["time_ms"]) if serial_row else 1.0

    omp_rows  = [r for r in strong if r["backend"]=="omp"]
    cuda_rows = [r for r in strong if r["backend"]=="cuda"]

    omp_T  = [int(r["threads"])   for r in omp_rows]
    omp_ms = [float(r["time_ms"]) for r in omp_rows]
    omp_su = [t_serial / t        for t in omp_ms]
    omp_eff= [s / T * 100         for s, T in zip(omp_su, omp_T)]

    # ── Figure layout: 2×2 ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])   # strong: wall time
    ax2 = fig.add_subplot(gs[0, 1])   # strong: speedup
    ax3 = fig.add_subplot(gs[1, 0])   # strong: efficiency
    ax4 = fig.add_subplot(gs[1, 1])   # weak: wall time

    fig.suptitle("CVT Performance Scaling  –  RTX 2050 / 12-core CPU",
                 fontsize=15, fontweight="bold", color="#e8eaf0", y=0.98)

    # ── Plot 1: Strong scaling – wall time ────────────────────────────────────
    ax1.plot(omp_T, omp_ms, "o-", color=COLORS["omp"],  label="OpenMP")
    ax1.axhline(t_serial, color=COLORS["serial"], linestyle="--",
                linewidth=1.5, label=f"Serial ({t_serial:.0f} ms)")
    if cuda_rows:
        t_cuda = float(cuda_rows[0]["time_ms"])
        ax1.axhline(t_cuda, color=COLORS["cuda"], linestyle="-.",
                    linewidth=1.8, label=f"CUDA ({t_cuda:.0f} ms)")
    _style_ax(ax1, "Strong Scaling – Wall Time",
              "OMP Thread Count", "Time (ms)")
    ax1.legend(framealpha=0.2, edgecolor="#444")

    # ── Plot 2: Strong scaling – speedup ─────────────────────────────────────
    ax2.plot(omp_T, omp_su, "s-", color=COLORS["omp"], label="OpenMP")
    ax2.plot(omp_T, omp_T,  "--", color=COLORS["ideal"], alpha=0.7, label="Ideal")
    if cuda_rows:
        cuda_su = t_serial / float(cuda_rows[0]["time_ms"])
        ax2.axhline(cuda_su, color=COLORS["cuda"], linestyle="-.",
                    linewidth=1.8, label=f"CUDA ({cuda_su:.1f}×)")
    _style_ax(ax2, "Strong Scaling – Speedup",
              "OMP Thread Count", "Speedup (×)")
    ax2.legend(framealpha=0.2, edgecolor="#444")
    _annotate_points(ax2, omp_T, omp_su, COLORS["omp"])

    # ── Plot 3: Strong scaling – parallel efficiency ──────────────────────────
    ax3.bar(omp_T, omp_eff, color=COLORS["omp"], alpha=0.85, width=0.7)
    ax3.axhline(100, color=COLORS["ideal"], linestyle="--",
                linewidth=1.5, label="Ideal (100%)")
    _style_ax(ax3, "Strong Scaling – Parallel Efficiency",
              "OMP Thread Count", "Efficiency (%)")
    ax3.set_ylim(0, 120)
    ax3.legend(framealpha=0.2, edgecolor="#444")
    for T, eff in zip(omp_T, omp_eff):
        ax3.text(T, eff + 2, f"{eff:.0f}%", ha="center", fontsize=9,
                 color="#e8eaf0")

    # ── Plot 4: Weak scaling – wall time ─────────────────────────────────────
    w_T  = [int(r["threads"])   for r in weak]
    w_ms = [float(r["time_ms"]) for r in weak]
    w_lbl= [f'{r["width"]}²'    for r in weak]
    ideal_ms = w_ms[0]

    ax4.plot(w_T, w_ms, "D-", color=COLORS["omp"], label="OpenMP (actual)")
    ax4.axhline(ideal_ms, color=COLORS["ideal"], linestyle="--",
                linewidth=1.5, label=f"Ideal (flat, {ideal_ms:.0f} ms)")
    ax4.set_xticks(w_T)
    ax4.set_xticklabels([f"T={t}\n{l}" for t, l in zip(w_T, w_lbl)], fontsize=9)
    _style_ax(ax4, "Weak Scaling – Wall Time (area ∝ threads)",
              "Thread Count (grid size)", "Time (ms)")
    ax4.legend(framealpha=0.2, edgecolor="#444")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(output, "scaling_graphs.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  -> Scaling graphs saved to {out_path}")


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_points(ax, xs, ys, color):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=8.5, color=color)


# =============================================================================
# CSV helpers
# =============================================================================
def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Speedup table with crossover analysis  (criterion: Speedup table + Crossover)
# =============================================================================
SPEEDUP_N_VALUES = [64, 128, 256, 512, 1024, 2048]   # 6 site counts
SPEEDUP_W, SPEEDUP_H = 512, 512                       # fixed grid


SPEEDUP_OMP_THREADS = 12    # logical cores on i5-12500H

def run_speedup_table(binary: str, output: str):
    """Run serial vs OMP vs CUDA across 6 N values; detect crossover; write CSV."""
    print("\n=== Speedup Table (512x512 grid, varying N sites) ===")
    print(f"  {'N':>6}  {'Serial(ms)':>12}  {'OMP(ms)':>10}  {'CUDA(ms)':>10}  {'OMP-SU':>8}  {'CUDA-SU':>8}")
    print(f"  {'------':>6}  {'----------':>12}  {'-------':>10}  {'--------':>10}  {'------':>8}  {'-------':>8}")

    rows = []
    crossover_n = None

    for n in SPEEDUP_N_VALUES:
        t_serial = best_of(binary, "serial", SPEEDUP_W, SPEEDUP_H, 1,
                           sites=n, max_iter=50)
        t_omp    = best_of(binary, "omp",    SPEEDUP_W, SPEEDUP_H,
                           SPEEDUP_OMP_THREADS, sites=n, max_iter=50)
        t_cuda   = best_of(binary, "cuda",   SPEEDUP_W, SPEEDUP_H, 1,
                           sites=n, max_iter=50)

        su_omp  = (t_serial / t_omp)  if (not np.isnan(t_omp)  and t_omp  > 0) else float("nan")
        su_cuda = (t_serial / t_cuda) if (not np.isnan(t_cuda) and t_cuda > 0) else float("nan")

        if not np.isnan(su_cuda) and su_cuda >= 1.0 and crossover_n is None:
            crossover_n = n

        print(f"  {n:>6}  {t_serial:>12.1f}  {t_omp:>10.1f}  {t_cuda:>10.1f}  {su_omp:>8.2f}x  {su_cuda:>8.2f}x")
        rows.append({"n_sites": n, "serial_ms": round(t_serial,1),
                     "omp_ms": round(t_omp,1), "cuda_ms": round(t_cuda,1),
                     "su_omp": round(su_omp,2), "su_cuda": round(su_cuda,2)})

    if crossover_n:
        print(f"\n  CUDA crossover at N={crossover_n}: above this GPU outpaces serial.")

    csv_path = os.path.join(output, "speedup_table.csv")
    _write_csv(csv_path, ["n_sites","serial_ms","omp_ms","cuda_ms","su_omp","su_cuda"], rows)
    print(f"  -> {csv_path}")


def plot_speedup_table(output):
    csv_path = os.path.join(output, "speedup_table.csv")
    if not os.path.exists(csv_path):
        return
    rows    = _read_csv(csv_path)
    ns      = [int(r["n_sites"]) for r in rows]
    su_omp  = [float(r.get("su_omp",  r.get("speedup", "nan"))) for r in rows]
    su_cuda = [float(r.get("su_cuda", r.get("speedup", "nan"))) for r in rows]
    x = np.arange(len(ns))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.bar(x - w/2, su_omp,  w, color=COLORS["omp"],  alpha=0.88, label="OpenMP (12T)")
    ax.bar(x + w/2, su_cuda, w, color=COLORS["cuda"], alpha=0.88, label="CUDA (JFA)")
    ax.axhline(1.0, color=COLORS["ideal"], linestyle="--", linewidth=1.4, label="Breakeven (1x)")
    for xi, (so, sc) in enumerate(zip(su_omp, su_cuda)):
        if so == so:
            ax.text(xi - w/2, so + 1.0, f"{int(so)}x", ha="center", fontsize=8, color="#e8eaf0")
        if sc == sc:
            ax.text(xi + w/2, sc + 1.0, f"{int(sc)}x", ha="center", fontsize=8, color="#e8eaf0")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n2) for n2 in ns])
    _style_ax(ax, "Serial vs OpenMP vs CUDA Speedup  (512x512 grid, 50 iters)",
              "Number of Sites (N)", "Speedup vs Serial (x)")
    ax.legend(framealpha=0.2, edgecolor="#444")
    out_path = os.path.join(output, "speedup_table.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> Speedup table plot saved to {out_path}")


CROSSOVER_GRIDS = [128, 256, 512, 768, 1024]
CROSSOVER_N     = 256

def run_gridsize_crossover(binary, output):
    print("\n=== Grid-Size Crossover Sweep (N=256 fixed, 50 iters) ===")
    rows = []
    crossover_grid = None
    for W in CROSSOVER_GRIDS:
        t_serial = best_of(binary, "serial", W, W, 1, sites=CROSSOVER_N, max_iter=50)
        t_cuda   = best_of(binary, "cuda",   W, W, 1, sites=CROSSOVER_N, max_iter=50)
        su = (t_serial / t_cuda) if (t_cuda == t_cuda and t_cuda > 0) else float("nan")
        if su == su and su >= 1.0 and crossover_grid is None:
            crossover_grid = W
        marker = " <- CROSSOVER" if W == crossover_grid else ""
        print(f"  {W}x{W}  pixels={W*W}  serial={t_serial:.1f}ms  cuda={t_cuda:.1f}ms  speedup={su:.2f}x{marker}")
        rows.append({"grid": f"{W}x{W}", "pixels": W*W,
                     "serial_ms": round(t_serial,1), "cuda_ms": round(t_cuda,1),
                     "speedup": round(su,2)})
    if crossover_grid:
        print(f"\n  PCIe crossover at {crossover_grid}x{crossover_grid} ({crossover_grid**2} pixels).")
    csv_path = os.path.join(output, "gridsize_crossover.csv")
    _write_csv(csv_path, ["grid","pixels","serial_ms","cuda_ms","speedup"], rows)
    print(f"  -> {csv_path}")
    return rows, crossover_grid


# =============================================================================
# PCIe Bandwidth plot  (criterion: B4 bandwidth plot)
# =============================================================================
def plot_bandwidth(output: str):
    csv_path = os.path.join(output, "bandwidth.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] bandwidth.csv not found; run: make ENABLE_CUDA=1 bw")
        return

    rows = _read_csv(csv_path)
    h2d_rows = [r for r in rows if r["direction"] == "H2D"]
    d2h_rows = [r for r in rows if r["direction"] == "D2H"]

    h2d_mb = [float(r["size_mb"])       for r in h2d_rows]
    h2d_bw = [float(r["bandwidth_mbs"]) for r in h2d_rows]
    d2h_mb = [float(r["size_mb"])       for r in d2h_rows]
    d2h_bw = [float(r["bandwidth_mbs"]) for r in d2h_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")

    ax.semilogx(h2d_mb, h2d_bw, "o-", color="#4fc3f7", label="H2D (Host->Device)",
                linewidth=2.2, markersize=7)
    ax.semilogx(d2h_mb, d2h_bw, "s-", color="#ffb74d", label="D2H (Device->Host)",
                linewidth=2.2, markersize=7)

    # Annotate peak values
    if h2d_bw:
        peak_h2d = max(h2d_bw)
        ax.axhline(peak_h2d, color="#4fc3f7", linestyle=":", alpha=0.5,
                   label=f"Peak H2D: {peak_h2d:.0f} MB/s")
    if d2h_bw:
        peak_d2h = max(d2h_bw)
        ax.axhline(peak_d2h, color="#ffb74d", linestyle=":", alpha=0.5,
                   label=f"Peak D2H: {peak_d2h:.0f} MB/s")

    _style_ax(ax, "PCIe Bandwidth  –  H2D and D2H (Pinned Memory)",
              "Transfer Size (MB, log scale)", "Bandwidth (MB/s)")
    ax.legend(framealpha=0.2, edgecolor="#444")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    out_path = os.path.join(output, "bandwidth.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> Bandwidth plot saved to {out_path}")


# =============================================================================
# Block size table plot  (criterion: Launch config + Timing/optimal)
# =============================================================================
def plot_blocksize(output: str):
    csv_path = os.path.join(output, "blocksize_table.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] blocksize_table.csv not found; run: make ENABLE_CUDA=1 blk")
        return

    rows = _read_csv(csv_path)
    blk_sizes  = [int(r["block_size"])   for r in rows]
    median_ms  = [float(r["median_ms"])  for r in rows]
    grid_blks  = [int(r["grid_blocks"])  for r in rows]
    occ        = [float(r["occupancy_pct"]) for r in rows]
    is_optimal = [r["optimal"] == "yes"  for r in rows]

    fig, (ax_t, ax_o) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Launch Configuration Sweep  –  JFA Flood Kernel (1024×1024)",
                 fontsize=14, fontweight="bold", color="#e8eaf0")

    bar_colors = [COLORS["cuda"] if opt else COLORS["omp"]
                  for opt in is_optimal]

    # Left: kernel time
    bars = ax_t.bar([str(b) for b in blk_sizes], median_ms,
                    color=bar_colors, alpha=0.88, width=0.6)
    for bar, ms, gb, opt in zip(bars, median_ms, grid_blks, is_optimal):
        label = f"{ms:.3f} ms\n({gb} blocks)"
        color = COLORS["cuda"] if opt else "#e8eaf0"
        ax_t.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + max(median_ms)*0.01,
                  label, ha="center", va="bottom", fontsize=8, color=color)
    _style_ax(ax_t, "Median Kernel Time vs Block Size",
              "Threads per Block", "Median Time (ms)")
    ax_t.set_ylim(0, max(median_ms) * 1.25)

    # Right: occupancy
    bars2 = ax_o.bar([str(b) for b in blk_sizes], occ,
                     color=bar_colors, alpha=0.88, width=0.6)
    ax_o.axhline(100, color=COLORS["ideal"], linestyle="--",
                 linewidth=1.5, label="Theoretical max (100%)")
    for bar, o in zip(bars2, occ):
        ax_o.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + 1.5,
                  f"{o:.1f}%", ha="center", va="bottom", fontsize=9,
                  color="#e8eaf0")
    _style_ax(ax_o, "Warp Occupancy vs Block Size",
              "Threads per Block", "Occupancy (%)")
    ax_o.set_ylim(0, 120)
    ax_o.legend(framealpha=0.2, edgecolor="#444")

    # Legend patch for optimal
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=COLORS["cuda"], label="Optimal config"),
        Patch(color=COLORS["omp"],  label="Other configs"),
    ]
    ax_t.legend(handles=legend_patches, framealpha=0.2, edgecolor="#444", fontsize=9)

    for ax in (ax_t, ax_o):
        ax.set_facecolor("#1a1d2e")

    plt.tight_layout()
    out_path = os.path.join(output, "blocksize.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> Block size plot saved to {out_path}")


# =============================================================================
# Entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="CVT scaling analysis")
    ap.add_argument("--binary",    default="./voronoi_cvt", help="Path to binary")
    ap.add_argument("--output",    default="results",       help="Output directory")
    global REPS
    ap.add_argument("--cuda",      action="store_true",     help="Include CUDA backend")
    ap.add_argument("--plot-only", action="store_true",     help="Skip experiments, just plot")
    ap.add_argument("--reps",      type=int, default=REPS,  help="Repetitions per config")
    ap.add_argument("--speedup",   action="store_true",
                    help="Run speedup table (serial vs OMP vs CUDA, 6 N values)")
    ap.add_argument("--crossover", action="store_true",
                    help="Run grid-size crossover sweep (fixed N=256)")
    args = ap.parse_args()

    REPS = args.reps

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        run_strong(args.binary, args.output, args.cuda)
        run_weak(args.binary, args.output, args.cuda)
        if args.speedup and args.cuda:
            run_speedup_table(args.binary, args.output)
        if args.crossover and args.cuda:
            run_gridsize_crossover(args.binary, args.output)

    plot_all(args.output)
    plot_quality(args.output)
    plot_bandwidth(args.output)
    plot_speedup_table(args.output)
    plot_blocksize(args.output)


# =============================================================================
# Quality comparison plot
# =============================================================================
def plot_quality(output: str):
    qcsv = os.path.join(output, "quality_comparison.csv")
    if not os.path.exists(qcsv):
        return   # only runs if --density stress was used

    rows = _read_csv(qcsv)

    methods      = [r["method"]                   for r in rows]
    energies     = [float(r["energy"])             for r in rows]
    improvements = [float(r["improvement_vs_uniform"]) for r in rows]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Sampling Quality Comparison  –  Kirsch Stress Field",
                 fontsize=14, fontweight="bold", color="#e8eaf0", y=1.02)

    bar_colors = [COLORS["serial"], COLORS["omp"], COLORS["cuda"]]

    # Left panel: absolute energy (lower = better)
    bars = ax_e.barh(methods, energies, color=bar_colors, alpha=0.88, height=0.5)
    ax_e.set_xlabel("CVT Energy  (lower = better)")
    ax_e.set_title("Density-Weighted Mean Squared Distance\nto Nearest Site",
                   fontsize=11, fontweight="bold")
    ax_e.set_xscale("log")
    for bar, val in zip(bars, energies):
        ax_e.text(val * 1.04, bar.get_y() + bar.get_height()/2,
                  f"{val:.1f}", va="center", fontsize=9, color="#e8eaf0")

    # Right panel: improvement vs uniform (higher = better)
    bars2 = ax_i.barh(methods, improvements, color=bar_colors, alpha=0.88, height=0.5)
    ax_i.set_xlabel("Improvement vs Uniform Sampling  (×)")
    ax_i.set_title("Relative Improvement in CVT Energy\nvs Random Uniform Baseline",
                   fontsize=11, fontweight="bold")
    ax_i.axvline(1.0, color=COLORS["ideal"], linestyle="--",
                 linewidth=1.4, label="Uniform baseline (1×)")
    for bar, val in zip(bars2, improvements):
        ax_i.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                  f"{val:.2f}×", va="center", fontsize=9, color="#e8eaf0")
    ax_i.legend(framealpha=0.2, edgecolor="#444", fontsize=9)

    for ax in (ax_e, ax_i):
        ax.set_facecolor("#1a1d2e")
        ax.tick_color   = "#c8cde0"
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3d5c")
        ax.tick_params(colors="#c8cde0")
        ax.xaxis.label.set_color("#c8cde0")
        ax.yaxis.label.set_color("#c8cde0")
        ax.title.set_color("#e8eaf0")

    plt.tight_layout()
    out_path = os.path.join(output, "quality_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  -> Quality comparison plot saved to {out_path}")


if __name__ == "__main__":
    main()


