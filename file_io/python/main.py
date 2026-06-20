import gc
import json
import math
import mmap
import os
import tracemalloc
from pathlib import Path

# DCE sink: CPython does no dead-code elimination, so this isn't strictly
# needed, but we keep it for parity and to actually touch every line.
sink = 0

# ───────────────────────── the four variants ─────────────────────────────
# Contract mirrors Rust/Go/Java: return list[str] of all lines, each OWNED,
# terminators stripped, no trailing empty. In CPython every line is a full
# str PyObject (~49B compact-ASCII header + 1B/char) plus an 8B pointer in the
# list, so the per-line overhead is ~2-3x Rust's String / Go's string. That
# overhead, not I/O, is the result. The fastest path is whichever keeps the
# decode+split loop in C rather than bytecode.

#  1. read_text + splitlines — slurp the whole ~45 MB file into one str (full
#     UTF-8 decode up front), then split. Decode AND split run entirely in C,
#     so this is the fastest idiomatic Python path. Peak holds the whole-file
#     str AND the line list simultaneously. splitlines() emits no trailing
#     empty (matches Rust .lines()); it also splits on extra Unicode separators
#     (\v \f NEL LS PS), moot on the ASCII fixture.
def read_text_splitlines(path):
    return path.read_text(encoding="utf-8").splitlines()


#  2. buffered line iteration — the TextIOWrapper idiom: incremental C decode +
#     C readline, but the rstrip is Python-level per line. (readlines() is pure
#     C but KEEPS the '\n', diverging from the stripped contract — so we iterate
#     and strip instead, accepting the small Python-loop cost.)
def buffered_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    return lines


#  3. manual chunk — you own the 64 KiB reads, the newline scan, and the
#     cross-chunk carry. The scan runs as bytecode, so this is expected to be
#     among the SLOWEST in Python (inverse of the compiled languages, where the
#     hand loop can win). bytes.decode copies -> owned str.
def manual_chunk(path):
    CHUNK = 64 * 1024
    lines = []
    carry = bytearray()
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            start = 0
            n = len(buf)
            for i in range(n):
                if buf[i] == 0x0A:  # '\n'
                    if not carry:
                        lines.append(buf[start:i].decode("utf-8"))
                    else:
                        carry += buf[start:i]
                        lines.append(carry.decode("utf-8"))
                        del carry[:]
                    start = i + 1
            carry += buf[start:n]  # partial line spans chunks
    if carry:
        lines.append(carry.decode("utf-8"))
    return lines


#  4. mmap (stdlib) — map the file, scan with mm.find(b'\n') which runs in C
#     (a byte-indexed Python loop here would be ~100x slower and a strawman).
#     mm[a:b] still copies bytes and decode copies again, so time ≈ manual
#     chunk; the mapping is OS memory (not tracemalloc-visible), so heap_peak is
#     the lowest of the four and the pages surface in rss_peak instead.
def mmap_scan(path):
    lines = []
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            return lines
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = 0
            while True:
                i = mm.find(b"\n", start)
                if i == -1:
                    break
                lines.append(mm[start:i].decode("utf-8"))
                start = i + 1
            if start < size:  # no trailing empty for a '\n'-terminated file
                lines.append(mm[start:size].decode("utf-8"))
        finally:
            mm.close()
    return lines


# ───────────────────────── per-variant peak RSS (Linux) ──────────────────
def reset_rss_peak():
    # writing "5" to clear_refs resets VmHWM to current VmRSS, so each variant
    # gets its own peak rather than a process-wide high-water.
    if os.name == "posix" and os.path.exists("/proc/self/clear_refs"):
        try:
            with open("/proc/self/clear_refs", "w") as f:
                f.write("5")
        except OSError:
            pass


def rss_peak_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except OSError:
        pass
    return 0


# ───────────────────────── harness ───────────────────────────────────────
def checksum(lines):
    return len(lines), sum(len(l) for l in lines)


def pct(sorted_times, p):
    if not sorted_times:
        return 0.0
    rank = p * (len(sorted_times) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_times[lo]
    f = rank - lo
    return sorted_times[lo] * (1.0 - f) + sorted_times[hi] * f


def round3(x):
    return round(x * 1000.0) / 1000.0


def sat_sub(a, b):
    return 0 if a < b else a - b


def measure(name, samples, build):
    global sink

    # warm-up: page cache + size-classes (1 build, matching the other harnesses
    # exactly — no extra JIT-style warm-up; CPython has none anyway).
    v = build()
    c, b = checksum(v)
    sink += c + b
    del v

    # timing pass — GC left at default (generational cyclic GC enabled), so
    # p99/p999 honestly carry any collector tail. tracemalloc is OFF here; it
    # roughly halves allocation throughput and must never wrap a timed region.
    times = []
    lines = 0
    for _ in range(samples):
        t = time.perf_counter_ns()
        v = build()
        dt = (time.perf_counter_ns() - t) / 1e6
        c, b = checksum(v)
        sink += c + b
        lines = c
        times.append(dt)
        del v
    times.sort()

    # memory pass — single representative build. tracemalloc is the allocator
    # analog of Rust's counting GlobalAlloc; it counts bytes through CPython's
    # allocator (NOT the mmap mapping — OS memory, like Rust's mmap bypassing
    # GlobalAlloc). gc.disable() mirrors Go's SetGCPercent(-1) so nothing is
    # reclaimed mid-build.
    gc.collect()
    gc.disable()
    tracemalloc.start()
    tracemalloc.reset_peak()
    base_cur, _ = tracemalloc.get_traced_memory()

    v = build()

    cur, peak = tracemalloc.get_traced_memory()
    heap_peak = sat_sub(peak, base_cur)

    gc.collect()  # result stays live across this collection
    live_cur, _ = tracemalloc.get_traced_memory()
    heap_live = sat_sub(live_cur, base_cur)

    rss = rss_peak_kb()
    c, b = checksum(v)
    sink += c + b  # keep v live until after the reads

    tracemalloc.stop()
    gc.enable()
    del v

    s = {
        "ms_p50": round3(pct(times, 0.50)),
        "ms_p99": round3(pct(times, 0.99)),
        "ms_p999": round3(pct(times, 0.999)),
        "heap_peak_kb": heap_peak // 1024,
        "heap_live_kb": heap_live // 1024,
        "rss_peak_kb": rss,
        "lines": lines,
    }
    print(
        f"[{name}]  p50={s['ms_p50']:.3f}ms p99={s['ms_p99']:.3f}ms "
        f"p999={s['ms_p999']:.3f}ms  heap_peak={s['heap_peak_kb']}KB "
        f"heap_live={s['heap_live_kb']}KB rss_peak={s['rss_peak_kb']}KB  "
        f"({s['lines']} lines)"
    )
    return s


# ───────────────────────── data file (dict merge) ────────────────────────
# Read the existing tree, replace only the "Python ..." key, write back. A
# plain dict preserves insertion order (3.7+), so the other languages' blocks
# keep their order and content — the analog of Go's RawMessage / Rust's
# preserve_order / Java's LinkedHashMap-backed ObjectNode.
def save(data_path, lang, color, file_bytes, n_lines, samples, variants):
    root = {}
    if data_path.exists():
        try:
            root = json.loads(data_path.read_text())
        except (ValueError, OSError):
            root = {}
    root[lang] = {
        "color": color,
        "file_bytes": file_bytes,
        "n_lines": n_lines,
        "samples": samples,
        "variants": {name: s for name, s in variants},
    }
    data_path.write_text(json.dumps(root, indent=2))


import time  # placed after the perf-critical defs; stdlib import cost is irrelevant

if __name__ == "__main__":
    # Anchor to file_io/ (one up from this script), so the fixture and data file
    # resolve regardless of cwd — the CARGO_MANIFEST_DIR / runtime.Caller analog.
    base = Path(__file__).resolve().parent.parent
    path = base / "text-examples.txt"
    data = base / "file_io.data"

    file_bytes = path.stat().st_size  # raises if the fixture is missing

    SAMPLES = 100
    s1 = measure("read_text + splitlines", SAMPLES, lambda: read_text_splitlines(path))
    s2 = measure("buffered lines",         SAMPLES, lambda: buffered_lines(path))
    s3 = measure("manual chunk",           SAMPLES, lambda: manual_chunk(path))
    s4 = measure("mmap (stdlib)",          SAMPLES, lambda: mmap_scan(path))

    n_lines = s1["lines"]
    save(data, "Python 3.14", "#FFD43B", file_bytes, n_lines, SAMPLES, [
        ("read_text + splitlines", s1),
        ("buffered lines", s2),
        ("manual chunk", s3),
        ("mmap (stdlib)", s4),
    ])

    print(f"sink={sink}")  # defeat any doubt the builds ran; do not remove