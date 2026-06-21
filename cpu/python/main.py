# cpu/python/main.py
# Cross-language concurrency benchmark (Leibniz π) — Python port.
#
# Runtime: CPython 3.14t FREE-THREADED (no GIL), PGO+LTO, on macOS/Darwin.
#   HARDWARE: <FILL IN: chip + core count, e.g. "Apple M3 Pro, 11 cores">
#
# What this actually stresses: at N_TERMS=10_000 each task is a few-µs FP loop
# in NATIVE Rust, but in CPython it's thousands of bytecode dispatches — so the
# interpreter eval-loop dominates, not the FP divider. Concurrency here measures
# Python's DISPATCH layer (thread spawn, pool reuse, fork/spawn+pickle IPC),
# which is the whole point of the comparison.
#
# WHY this differs from the Rust file (deliberately, not a blind port):
#   - No DCE sink needed: CPython does no dead-code elimination. We still drop
#     worker()'s print — stdout in the timed region serializes everything and is
#     the exact trap Rust's SINK fix removed.
#   - Persistent pools: Python pools are HEAVY (threads, or spawn+pickle procs).
#     Rust rebuilds per sample cheaply; here we build ONCE outside the timed loop
#     and measure only dispatch — otherwise we'd be timing spawn, not compute.
#   - Pool sizing: max_workers OMITTED so the defaults self-limit
#     (TPE -> min(32, cpu+4); Process pools -> cpu_count). "Let the runtime decide."
#   - `threading` deliberately stays 100 RAW threads as the unbounded-spawn
#     contrast (the analog of Rust's thread::spawn vs rayon).
#
# macOS note: start method is `spawn` (the only safe one on Darwin), so process
# variants pay full interpreter-respawn + pickle cost even when the pool is warm.
# Expect them to LOSE badly at this tiny payload — that's the interesting result.

import json
import multiprocessing
import asyncio
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

N_TERMS = 10_000
N_TASKS = 100
SAMPLES = 1000


# The shared payload — identical naive FP loop to the other four languages, so
# only the dispatch layer differs. No print: stdout in a timed region serializes
# parallel tasks on the I/O lock (Rust folds into SINK for the same reason).
def compute_pi(n_terms):
    pi = 0.0
    sign = 1.0
    k = 1.0
    for _ in range(n_terms):
        pi += sign / k
        sign = -sign
        k += 2.0
    return 4.0 * pi


def worker(n_terms):
    return compute_pi(n_terms)


# ───────────────────────── the six variants ──────────────────────────────
# Each runs one full N_TASKS fan-out. Pool variants receive their persistent
# pool/executor and only DISPATCH into it — pool construction is outside timing.

# Serial baseline — N_TASKS back to back on one thread. Zero dispatch overhead;
# the honest reference the parallel variants are measured against.
def compute_serial():
    for _ in range(N_TASKS):
        worker(N_TERMS)


# Raw thread-per-task: 100 fresh OS threads/sample, started then joined. On 3.14t
# they run truly in parallel, but creation tax + oversubscription (100 threads on
# ~10 cores) is the cost this variant exists to expose. NOT queued by us — the OS
# scheduler time-slices them. Predicted: parallel but slower than the pool.
def compute_with_threads():
    threads = [threading.Thread(target=worker, args=(N_TERMS,)) for _ in range(N_TASKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# Persistent ThreadPoolExecutor (default size ~cpu+4). 100 tasks QUEUE through a
# bounded, reused thread set — no per-sample spawn. On free-threaded Python this
# is the predicted HEADLINE WINNER (true parallelism, near-zero dispatch cost),
# the analog of Rust's rayon.
def compute_with_thread_pool(pool):
    list(pool.map(worker, [N_TERMS] * N_TASKS))


# Persistent ProcessPoolExecutor (default size cpu_count). Real OS parallelism,
# but every dispatch pickles args/results across the spawn boundary. At this tiny
# payload the IPC dwarfs the compute — predicted heavy LOSER on Darwin.
def compute_with_process_pool(pool):
    list(pool.map(worker, [N_TERMS] * N_TASKS))


# Persistent multiprocessing.Pool — same idea, classic API. Kept alongside
# ProcessPoolExecutor for breadth (the two process-dispatch models), like Rust
# keeping threadpool beside futures-cpupool.
def compute_with_mp_pool(pool):
    pool.map(worker, [N_TERMS] * N_TASKS)


# asyncio dispatching CPU work onto a thread pool via to_thread. Included because
# it's the WRONG tool — it documents what pushing blocking compute through the
# event loop costs (the analog of Rust's spawn_blocking-on-tokio variant), so the
# misuse has a measured number instead of an assumption.
async def _async_fanout():
    await asyncio.gather(*(asyncio.to_thread(worker, N_TERMS) for _ in range(N_TASKS)))


def compute_with_asyncio():
    asyncio.run(_async_fanout())


# Pure asyncio.gather over plain coroutines — NO executor. The CPU loop never
# yields, so the event loop runs the tasks strictly SERIALLY with coroutine +
# scheduling overhead piled on top. This is the real "wrong tool" result:
# predicted to land at or WORSE than serial. The contrast with the to_thread
# variant above is the lesson — async parallelizes I/O, never CPU, on its own.
async def _async_pure_fanout():
    async def task():
        return worker(N_TERMS)
    await asyncio.gather(*(task() for _ in range(N_TASKS)))


def compute_with_asyncio_pure():
    asyncio.run(_async_pure_fanout())


# ───────────────────────── harness ───────────────────────────────────────
# Mirrors the Rust/file_io harness: 1 warm-up, then SAMPLES timed full fan-outs,
# sorted, reported as p50/p99/p999 via linear-interpolated percentile. Monotonic
# ns clock. GC left at default so p99/p999 honestly carry any collector tail.
# At SAMPLES=1000, p999 lands on rank ~998 — a real 1-in-1000 tail.

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


def measure(name, build):
    build()  # one warm-up (primes import caches; for pools, first dispatch)

    times = []
    for _ in range(SAMPLES):
        t = time.perf_counter_ns()
        build()
        times.append((time.perf_counter_ns() - t) / 1e6)
    times.sort()

    s = {
        "ms_p50": round3(pct(times, 0.50)),
        "ms_p99": round3(pct(times, 0.99)),
        "ms_p999": round3(pct(times, 0.999)),
        # CPU benchmark has no file/memory dimension — written as 0 (n/a),
        # mirroring the Rust CPU block's schema exactly.
        "heap_peak_kb": 0,
        "heap_live_kb": 0,
        "rss_peak_kb": 0,
        "lines": 0,
    }
    print(
        f"[{name}]  p50={s['ms_p50']:.3f}ms p99={s['ms_p99']:.3f}ms "
        f"p999={s['ms_p999']:.3f}ms"
    )
    return s


# ───────────────────────── data file (file_io-style schema) ──────────────
# Read the existing tree, replace ONLY the "python" key, write back. The Rust
# plot.data is just a template seeded with Rust's block — we MERGE our values in
# and must not clobber other languages' blocks. A plain dict preserves insertion
# order (3.7+), so the other blocks keep their order/content.
def save(data_path, lang, color, samples, variants):
    entry = {
        "color": color,
        "file_bytes": 0,
        "n_lines": 0,
        "n_terms": N_TERMS,
        "n_tasks": N_TASKS,
        "samples": samples,
        "variants": {name: s for name, s in variants},
    }
    root = {}
    try:
        with open(data_path) as f:
            root = json.load(f)
    except (ValueError, OSError):
        root = {}
    root[lang] = entry
    with open(data_path, "w") as f:
        json.dump(root, f, indent=2)


if __name__ == "__main__":
    # Persistent pools — built ONCE, outside the timed region. Building per sample
    # would measure spawn/pickle, not dispatch. max_workers omitted on purpose so
    # the defaults self-limit (your chosen option A).
    with ThreadPoolExecutor() as tpe, \
         ProcessPoolExecutor() as ppe, \
         multiprocessing.Pool() as mpp:

        variants = [
            ("serial", measure("serial", compute_serial)),
            ("threading", measure("threading", compute_with_threads)),
            ("ThreadPoolExecutor", measure("ThreadPoolExecutor", lambda: compute_with_thread_pool(tpe))),
            ("ProcessPoolExecutor", measure("ProcessPoolExecutor", lambda: compute_with_process_pool(ppe))),
            ("multiprocessing.Pool", measure("multiprocessing.Pool", lambda: compute_with_mp_pool(mpp))),
            ("asyncio (to_thread)", measure("asyncio (to_thread)", compute_with_asyncio)),
            ("asyncio (pure gather)", measure("asyncio (pure gather)", compute_with_asyncio_pure)),
        ]

    save("../plot.data", "Python (no GIL)", "#FFD43B", SAMPLES, variants)