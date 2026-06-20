"""
Concurrency benchmark visualizer.

Reads measured results from ../plot.data (JSON, written by each language's
benchmark) and renders:
  - one horizontal-bar panel per language (native variant names, no forced
    cross-language equivalences)
  - a final normalized throughput panel (million iterations / sec), which
    accounts for languages run with different term counts (e.g. Python 50M
    vs 500M for the rest).

DATA FORMAT (../plot.data)
--------------------------
A JSON object keyed by language name. Each language block:
  {
    "color":     "#dd4444",         # bar color for this language
    "n_terms":   500000000,         # terms per task in this run
    "n_tasks":   3,                 # number of tasks
    "serial_ms": 485.0,             # single-task baseline (throughput basis)
    "variants": {                   # ordered map: variant name -> wall ms
      "thread::spawn": 485.2,
      "rayon":         484.1,
      ...
    }
  }

Run:  python plot_bench.py   ->   writes bench_comparison.png
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Load measured data from ../plot.data (relative to this script, not the cwd)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, os.pardir, "plot.data")

# Preferred display order; any languages in the file but not listed here are
# appended afterwards (alphabetically) so new languages still show up.
PREFERRED_ORDER = ["rust", "go", "node", "java", "python"]
# Fallback palette for any language whose block omits "color".
FALLBACK_COLORS = ["#dd4444", "#3b9c5a", "#4477dd", "#d98c1f", "#8855cc",
                   "#11999e", "#c0392b", "#7f8c8d"]


def load_data(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"error: {path} not found. Run the benchmarks first so each "
                 f"language writes its block into plot.data.")
    except json.JSONDecodeError as e:
        sys.exit(f"error: {path} is not valid JSON: {e}")
    if not isinstance(data, dict) or not data:
        sys.exit(f"error: {path} did not contain any language blocks.")
    return data


def ordered_languages(data):
    known = [lang for lang in PREFERRED_ORDER if lang in data]
    extra = sorted(k for k in data if k not in PREFERRED_ORDER)
    return known + extra


def color_for(lang, block, idx):
    return block.get("color") or FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]


def throughput_mips(block):
    """Single-task throughput in millions of iterations/sec, from serial_ms."""
    serial_ms = block.get("serial_ms")
    n_terms = block.get("n_terms")
    if not serial_ms or not n_terms:
        return None
    return n_terms / (serial_ms * 1000.0)  # n_terms / (serial_ms/1000) / 1e6


# ---------------------------------------------------------------------------
# Panel drawing (shared by the combined grid and the individual figures)
# ---------------------------------------------------------------------------
def draw_language_panel(ax, lang, block, idx):
    """Render one language's variant bars onto the given axis."""
    variants = block.get("variants", {})
    names = list(variants.keys())          # JSON preserves file order
    times = [variants[n] for n in names]
    color = color_for(lang, block, idx)

    y = np.arange(len(names))
    ax.barh(y, times, color=color, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()                      # first variant on top
    ax.set_xlabel("wall time (ms)", fontsize=8.5)

    terms = block.get("n_terms")
    tasks = block.get("n_tasks")
    subtitle = ""
    if terms and tasks:
        subtitle = f"  ({terms/1e6:g}M terms x {tasks} tasks)"
    ax.set_title(f"{lang}{subtitle}", fontsize=10.5, fontweight="bold")

    for yi, t in zip(y, times):
        ax.text(t, yi, f" {t:.0f}", va="center", fontsize=8, color="#333")
    ax.margins(x=0.18)
    ax.grid(axis="x", alpha=0.25)


def draw_throughput_panel(ax, data, langs):
    """Render the normalized single-task throughput bars onto the given axis."""
    tput_langs, tput_vals, tput_colors = [], [], []
    for idx, lang in enumerate(langs):
        mips = throughput_mips(data[lang])
        if mips is not None:
            tput_langs.append(lang)
            tput_vals.append(mips)
            tput_colors.append(color_for(lang, data[lang], idx))

    order = np.argsort(tput_vals)[::-1]
    tput_langs = [tput_langs[i] for i in order]
    tput_vals = [tput_vals[i] for i in order]
    tput_colors = [tput_colors[i] for i in order]

    yb = np.arange(len(tput_langs))
    ax.barh(yb, tput_vals, color=tput_colors, alpha=0.85)
    ax.set_yticks(yb)
    ax.set_yticklabels(tput_langs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("millions of iterations / sec (single-task)", fontsize=8.5)
    ax.set_title("Single-task throughput (raw compute speed)", fontsize=10.5, fontweight="bold")
    for yi, v in zip(yb, tput_vals):
        ax.text(v, yi, f" {v:.0f}", va="center", fontsize=8, color="#333")
    ax.margins(x=0.18)
    ax.grid(axis="x", alpha=0.25)


# ---------------------------------------------------------------------------
# Individual per-plot figures
# ---------------------------------------------------------------------------
def save_individual_plots(data, langs):
    """Write each language panel + the throughput panel as standalone PNGs."""
    written = []
    for idx, lang in enumerate(langs):
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        draw_language_panel(ax, lang, data[lang], idx)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() else "_" for c in lang).strip("_").lower()
        out = os.path.join(HERE, f"bench_{safe}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    draw_throughput_panel(ax, data, langs)
    fig.tight_layout()
    out = os.path.join(HERE, "bench_throughput.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    written.append(out)

    for p in written:
        print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main():
    data = load_data(DATA_PATH)
    langs = ordered_languages(data)

    n_panels = len(langs) + 1            # one per language + throughput panel
    ncols = 3
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    # --- per-language panels --------------------------------------------------
    for idx, lang in enumerate(langs):
        draw_language_panel(axes[idx], lang, data[lang], idx)

    # --- throughput panel -----------------------------------------------------
    draw_throughput_panel(axes[len(langs)], data, langs)

    # hide any unused cells
    for j in range(len(langs) + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Cross-language concurrency benchmark (Leibniz π)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "Per-language panels show within-language variant differences. "
             "Throughput normalizes for differing term counts. "
             "Parallel wall time reflects the longest task, not the sum.",
             ha="center", fontsize=8.5, color="#666")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(HERE, "bench_comparison.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")

    save_individual_plots(data, langs)


if __name__ == "__main__":
    main()