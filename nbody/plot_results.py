#!/usr/bin/env python3
"""
plot_results.py — Generate benchmark charts from N-body results.

Produces (in ui/):
  scaling_chart.png     — time vs N + speedup bars  (original)
  tile_experiment.png   — CUDA time vs tile size
"""

import json, sys, os, math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UI = os.path.join(SCRIPT_DIR, "ui")
os.makedirs(UI, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────────
BG = "#0d0d1f"
GRID = "#2a2a4a"
TEXT = "#cbd5e1"
ACC = "#a855f7"  # accent purple


def dark_fig(w=12, h=6, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=BG)
    for ax in axes if ncols > 1 else [axes]:
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT, labelsize=10)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    return fig, axes


def save(fig, name):
    path = os.path.join(UI, name)
    plt.tight_layout(pad=2.0)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"[plot] {path}")


# ── 1. scaling_chart.png ───────────────────────────────────────────────────────
METHODS = [
    ("seq", "Sequential", "#ef4444", "o", 2.5),
    ("omp2", "OpenMP 2T", "#16a34a", "s", 2.0),
    ("omp4", "OpenMP 4T", "#22c55e", "s", 2.0),
    ("omp8", "OpenMP 8T", "#4ade80", "s", 2.0),
    ("mpi2", "MPI 2R", "#2563eb", "^", 2.0),
    ("mpi4", "MPI 4R", "#3b82f6", "^", 2.0),
    ("mpi8", "MPI 8R", "#60a5fa", "^", 2.0),
    ("hyb22", "Hybrid 2R X 2T", "#f59e0b", "D", 2.5),
    ("hyb24", "Hybrid 2R X 4T", "#fbbf24", "D", 2.5),
    ("hyb42", "Hybrid 4R X 2T", "#fde68a", "D", 2.5),
    ("cuda", "CUDA GPU", "#a855f7", "P", 2.5),
]

rj = os.path.join(UI, "results.json")
if os.path.exists(rj):
    with open(rj) as f:
        data = json.load(f)
    configs = data if isinstance(data, list) else data.get("configs", data)
    Ns = [c["n"] for c in configs]
    N0, t0r = Ns[0], configs[0].get("seq", 1.0)
    ref_n2 = [t0r * (n / N0) ** 2 for n in Ns]
    ref_nlogn = [t0r * (n * math.log2(n)) / (N0 * math.log2(N0)) for n in Ns]

    fig, axes = dark_fig(16, 7, 2)
    ax = axes[0]
    ax.set_title("Execution Time vs Problem Size (log scale)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([f"N={n:,}" for n in Ns])
    for key, label, color, marker, lw in METHODS:
        times = [c.get(key) for c in configs]
        if any(t and t > 0 for t in times):
            ax.plot(Ns, times, color=color, marker=marker, linewidth=lw, markersize=7, label=label, zorder=3)
            last = next((t for t in reversed(times) if t), None)
            if last:
                ax.annotate(f"{last:.2f}s", xy=(Ns[-1], last), xytext=(5, 0), textcoords="offset points", color=color, fontsize=7.5, va="center")
    ax.plot(Ns, ref_n2, color="#ef4444", linestyle=":", linewidth=1.2, label="Ref O(N²)", alpha=0.6)
    ax.plot(Ns, ref_nlogn, color="#34d399", linestyle=":", linewidth=1.2, label="Ref O(N log N)", alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.2f}s" if v < 1 else f"{v:.1f}s"))
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor="#1a1a30", edgecolor=GRID, loc="upper left")

    ax = axes[1]
    ax.set_title("Speedup vs Sequential (largest N)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Speedup (×)", fontsize=11)
    last_cfg = configs[-1]
    lbs, vals, cols = [], [], []
    for key, label, color, *_ in METHODS:
        spd = last_cfg.get(f"s_{key}") or (1.0 if key == "seq" else None)
        if spd is not None:
            lbs.append(label)
            vals.append(float(spd))
            cols.append(color)
    x_pos = np.arange(len(lbs))
    bars = ax.bar(x_pos, vals, color=cols, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lbs, rotation=35, ha="right", fontsize=9)
    ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, alpha=0.6, label="Baseline (1×)")
    ax.set_ylim(0, max(vals) * 1.2 + 1)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, f"{val:.1f}×", ha="center", va="bottom", color=TEXT, fontsize=8.5)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT, facecolor="#1a1a30", edgecolor=GRID)

    fig.suptitle("N-Body Galaxy Simulation — Benchmark Results (Velocity Verlet)", fontsize=15, fontweight="bold", color=TEXT, y=1.01)
    fig.text(0.5, -0.02, "Galaxy disk init • Velocity Verlet integrator • G=6.674×10⁻¹¹ N·m²/kg²  •  5 timesteps", ha="center", fontsize=9, color="#64748b")
    save(fig, "scaling_chart.png")

# ── 2. tile_experiment.png ─────────────────────────────────────────────────────
tj = os.path.join(UI, "tile_results.json")
if os.path.exists(tj):
    with open(tj) as f:
        td = json.load(f)
    tiles = [d["tile"] for d in td]
    times = [float(d.get("time", 0)) for d in td]
    best_t = min(t for t in times if t > 0)
    colors = ["#22c55e" if t == best_t else ACC for t in times]

    fig, ax = dark_fig(8, 5)
    ax.set_title("CUDA Tile Size Experiment (N=30000)\nRuntime tile — analogous to OMP_NUM_THREADS", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Shared-Memory Tile Size", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    bars = ax.bar([str(t) for t in tiles], times, color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, val, tile in zip(bars, times, tiles):
        label = f"{val:.3f}s" + (" ← best" if val == best_t else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005, label, ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_ylim(0, max(times) * 1.25)
    fig.suptitle("CUDA Tile Size Sensitivity", fontsize=14, fontweight="bold", color=TEXT)
    fig.text(0.5, -0.02, "Tile size passed as argv[2] — no recompilation needed (dynamic shared memory)", ha="center", fontsize=9, color="#64748b")
    save(fig, "tile_experiment.png")

print("[plot_results] Done.")
