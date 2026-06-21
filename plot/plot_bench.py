"""
CPU concurrency benchmark visualizer (Leibniz π).

Reads measured results from ../plot.data (JSON, written by each language's
benchmark) and renders three things, all tuned for vertical reading on dev.to:

  1. One STANDALONE PNG per language  -> bench_<lang>.png
       horizontal variant bars, native variant names, no forced equivalences.
  2. One COMBINED figure, languages stacked VERTICALLY -> bench_comparison.png
       one language panel per row (single column), readable top-to-bottom.
  3. One COMBINED-SERIAL figure -> bench_serial_comparison.png
       the `serial` variant of every language side by side, p50 bars with
       p99/p999 drawn as whiskers. LINEAR x-axis (per request): Python's
       interpreter tax (~37 ms) dwarfs the compiled languages (~0.2-0.9 ms),
       so per-bar value labels are added to keep the small bars legible.

WHAT THIS BENCHMARK ACTUALLY MEASURES
-------------------------------------
At n_terms = 10_000 each task is a few-µs FP loop, so this stresses each
runtime's DISPATCH / SCHEDULER layer, not the FP divider. `serial` is the
reference cost of the raw compute; a parallel variant BELOW serial is real
speedup, a variant ABOVE serial (e.g. Rust thread::spawn, async tokio,
Python asyncio-pure) is dispatch overhead exceeding the work itself.

DATA FORMAT (../plot.data)
--------------------------
A JSON object keyed by language name. Each language block:
  {
    "color":    "#dd4444",
    "n_terms":  10000,
    "n_tasks":  100,
    "samples":  1000,
    "variants": {                  # ordered map: variant name -> stats
      "serial":      {"ms_p50": .., "ms_p99": .., "ms_p999": .., ...},
      "rayon":       {"ms_p50": .., ...},
      ...
    }
  }
Variant stats may also carry gc_cycles / gc_pause_ms (the "Go (alloc)" block);
those are ignored by these panels.

Run:  python plot_bench.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Load measured data from ../plot.data (relative to this script, not the cwd)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, os.pardir, "cpu", "plot.data")

# Preferred display order; any languages in the file but not listed here are
# appended afterwards (alphabetically) so new languages still show up.
PREFERRED_ORDER = ["Rust", "Go", "Go (alloc)", "Node.js", "Java", "python"]
# Fallback palette for any language whose block omits "color".
FALLBACK_COLORS = ["#dd4444", "#3b9c5a", "#4477dd", "#d98c1f", "#8855cc",
                   "#11999e", "#c0392b", "#7f8c8d"]

DPI = 150

# The data files key this variant "serial"; we DISPLAY it as "sequential" and
# always pin it to the top bar of every panel. Data keys are never rewritten.
SERIAL_KEY = "serial"
SERIAL_LABEL = "sequential"

def display_label(name):
    return SERIAL_LABEL if name == SERIAL_KEY else name


def ordered_variant_names(variants):
    """Variant names with the serial baseline forced first; rest keep file order."""
    names = list(variants.keys())
    if SERIAL_KEY in names:
        names.remove(SERIAL_KEY)
        names.insert(0, SERIAL_KEY)
    return names

# ---------------------------------------------------------------------------
# Loading / ordering helpers
# ---------------------------------------------------------------------------
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


def variant_p50(stats):
    """Tolerant p50 read: schema uses ms_p50, fall back to a bare number."""
    if isinstance(stats, dict):
        return stats.get("ms_p50", 0.0)
    return float(stats)  # legacy {name: ms} shape


def subtitle_for(block):
    terms = block.get("n_terms")
    tasks = block.get("n_tasks")
    if terms and tasks:
        return f"  ({terms/1e6:g}M terms x {tasks} tasks)"
    return ""


def safe_name(lang):
    return "".join(c if c.isalnum() else "_" for c in lang).strip("_").lower()


# ---------------------------------------------------------------------------
# Panel: one language's variant bars (horizontal)
# ---------------------------------------------------------------------------
def draw_language_panel(ax, lang, block, idx):
    variants = block.get("variants", {})
    names = ordered_variant_names(variants)        # serial baseline forced first
    times = [variant_p50(variants[n]) for n in names]
    color = color_for(lang, block, idx)

    y = np.arange(len(names))
    ax.barh(y, times, color=color, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([display_label(n) for n in names], fontsize=8.5)
    ax.invert_yaxis()                             # first variant (sequential) on top
    ax.set_xlabel("p50 wall time (ms)", fontsize=8.5)
    ax.set_title(f"{lang}{subtitle_for(block)}", fontsize=10.5, fontweight="bold")

    for yi, t in zip(y, times):
        ax.text(t, yi, f" {t:g}", va="center", fontsize=8, color="#333")
    ax.margins(x=0.18)
    ax.grid(axis="x", alpha=0.25)


# ---------------------------------------------------------------------------
# Panel: combined serial across languages (p50 bars + p99/p999 whiskers)
# ---------------------------------------------------------------------------
def draw_serial_comparison_panel(ax, data, langs):
    rows = []
    for idx, lang in enumerate(langs):
        variants = data[lang].get("variants", {})
        s = variants.get("serial")
        if not isinstance(s, dict):
            continue
        p50 = s.get("ms_p50", 0.0)
        p99 = s.get("ms_p99", p50)
        p999 = s.get("ms_p999", p99)
        rows.append((lang, p50, p99, p999, color_for(lang, data[lang], idx)))

    # slowest at top so the eye lands on the interpreter-tax outlier first
    rows.sort(key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    p50s = [r[1] for r in rows]
    p99s = [r[2] for r in rows]
    p999s = [r[3] for r in rows]
    colors = [r[4] for r in rows]

    y = np.arange(len(rows))
    ax.barh(y, p50s, color=colors, alpha=0.85)

    # whiskers: p50 -> p999, with a tick at p99. Drawn as horizontal lines so
    # the distribution tail is visible without a second axis.
    for yi, p50, p99, p999 in zip(y, p50s, p99s, p999s):
        ax.plot([p50, p999], [yi, yi], color="#222", lw=1.0, alpha=0.7)
        ax.plot([p999, p999], [yi - 0.18, yi + 0.18], color="#222", lw=1.0)  # p999 cap
        ax.plot([p99, p99], [yi - 0.12, yi + 0.12], color="#666", lw=1.0)     # p99 tick

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("sequential wall time (ms) — bar = p50, whisker = p99 / p999",
                  fontsize=8.5)
    ax.set_title("Single-task sequential cost across languages",
                 fontsize=11, fontweight="bold")

    # value labels: on a linear axis the compiled langs are tiny, so print the
    # p50 next to every bar to keep them legible despite Python's scale.
    for yi, p50 in zip(y, p50s):
        ax.text(p50, yi, f" {p50:g} ms", va="center", fontsize=8, color="#333")
    ax.margins(x=0.22)
    ax.grid(axis="x", alpha=0.25)


# ---------------------------------------------------------------------------
# (1) Per-language standalone PNGs
# ---------------------------------------------------------------------------
def save_individual_plots(data, langs):
    written = []
    for idx, lang in enumerate(langs):
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        draw_language_panel(ax, lang, data[lang], idx)
        fig.tight_layout()
        out = os.path.join(HERE, f"bench_{safe_name(lang)}.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# (2) Combined, languages stacked VERTICALLY (one column)
# ---------------------------------------------------------------------------
def save_combined_vertical(data, langs):
    n = len(langs)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 3.2 * n))
    axes = np.atleast_1d(axes).ravel()
    for idx, lang in enumerate(langs):
        draw_language_panel(axes[idx], lang, data[lang], idx)

    fig.suptitle("Cross-language concurrency benchmark (Leibniz π) — per-language variants",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005,
             "Each panel shows within-language variant spread (p50). "
             "Distance below `sequential` is real parallel speedup; a variant "
             "above it is dispatch overhead exceeding the work.",
             ha="center", fontsize=8.5, color="#777")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = os.path.join(HERE, "bench_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# (3) Combined serial across languages
# ---------------------------------------------------------------------------
def save_serial_comparison(data, langs):
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(langs) + 1.8))
    draw_serial_comparison_panel(ax, data, langs)
    fig.tight_layout()
    out = os.path.join(HERE, "bench_serial_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data(DATA_PATH)
    langs = ordered_languages(data)

    written = []
    written.append(save_combined_vertical(data, langs))
    written.append(save_serial_comparison(data, langs))
    written.extend(save_individual_plots(data, langs))

    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()