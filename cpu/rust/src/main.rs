use futures::Future;
use futures_cpupool::CpuPool;
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use threadpool::ThreadPool;

const N_TERMS: u32 = 10_000;
const N_TASKS: usize = 100;
const SAMPLES: usize = 1000;

// DCE sink: every worker folds its result's bits in here via a relaxed atomic
// XOR. This makes the computed value escape to a global so --release can't
// delete the loop, and it removes all stdout from the timed region (the old
// println in worker would otherwise be measured SAMPLES*N_TASKS times and the
// stdout lock would serialize the parallel tasks, burying the dispatch signal).
static SINK: AtomicU64 = AtomicU64::new(0);

/// The shared CPU payload for every variant: a tight, branch-free FP loop whose
/// loop-carried dependency on `pi` isolates raw FP-divider throughput. Kept naive
/// (no Kahan summation, SIMD, or unrolling) on purpose, so the benchmark measures
/// the runtime and dispatch layer rather than who optimizes the math best, and so
/// the same hot path ports cleanly to the other four languages.
fn compute_pi(n_terms: u32) -> f64 {
    let mut pi = 0.0;
    let mut sign = 1.0;
    let mut k = 1.0;
    for _ in 0..n_terms {
        pi += sign / k;
        sign = -sign;
        k += 2.0;
    }
    4.0 * pi
}

/// The single unit of work each concurrency primitive dispatches, giving every
/// variant an identical payload so only scheduling differs. Folds its result into
/// the global SINK rather than printing: this keeps stdout out of the timed region
/// (a per-task println would be measured SAMPLES×N_TASKS times and serialize
/// parallel tasks on the stdout lock) while still letting the optimizer see the
/// result escape, so the loop survives --release.
fn worker(n_terms: u32) {
    let r = compute_pi(n_terms);
    SINK.fetch_xor(r.to_bits(), Ordering::Relaxed);
}

/// The no-parallelism baseline — N_TASKS run back to back on one thread. Exists so
/// every parallel variant has an honest reference point: it carries zero dispatch
/// overhead and the tightest distribution, making the cost (and tail) that threading
/// adds visible by contrast.
fn compute_serial() {
    for _ in 0..N_TASKS {
        worker(N_TERMS);
    }
}

/// Thread-per-task via the OS scheduler, the most direct mapping of "run these
/// concurrently." Chosen to expose the cost of unbounded spawning at scale: with
/// N_TASKS fresh threads per run it shows creation overhead and oversubscription on
/// a fixed core count, which is the point of comparison against the pooled variants.
fn compute_with_threads() {
    let handles: Vec<_> = (0..N_TASKS)
        .map(|_| std::thread::spawn(|| worker(N_TERMS)))
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// The idiomatic data-parallel path: a work-stealing pool sized to the core count,
/// with tasks as items in per-thread deques. Included as the expected best case —
/// near-zero per-task overhead and good load balance — and as the bar the other
/// pool-based approaches are measured against.
fn compute_with_rayon() {
    (0..N_TASKS).into_par_iter().for_each(|_| worker(N_TERMS));
}

/// A classic fixed-size pool fed through a shared job queue. Chosen to contrast with
/// rayon's per-thread deques: here all workers contend on one central queue, so it
/// shows what that contention costs relative to work-stealing at the same core count.
fn compute_with_threadpool() {
    let pool = ThreadPool::new(num_cpus::get());
    for _ in 0..N_TASKS {
        pool.execute(|| worker(N_TERMS));
    }
    pool.join();
}

/// CPU work forced through an async runtime via spawn_blocking. Included precisely
/// because it's the wrong tool — it documents what happens when blocking compute is
/// pushed onto an async executor's blocking pool, so the misuse has a measured number
/// rather than being assumed.
fn compute_with_async() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let mut handles = Vec::new();
        for _ in 0..N_TASKS {
            handles.push(tokio::task::spawn_blocking(|| worker(N_TERMS)));
        }
        for h in handles {
            h.await.unwrap();
        }
    });
}

/// An older futures-based pool, kept for breadth across the ecosystem's concurrency
/// abstractions. Sits alongside threadpool as another bounded-pool dispatch model so
/// the comparison isn't limited to the two modern primitives.
fn compute_with_cpupool() {
    let pool = CpuPool::new_num_cpus();
    let futs: Vec<_> = (0..N_TASKS)
        .map(|_| {
            pool.spawn_fn(|| {
                worker(N_TERMS);
                Ok::<(), ()>(())
            })
        })
        .collect();
    for f in futs {
        f.wait().unwrap();
    }
}

// ───────────────────────── harness ───────────────────────────────────────
struct Stats {
    p50: f64,
    p99: f64,
    p999: f64,
}

/// Linear-interpolated percentile over a pre-sorted slice. Chosen over nearest-rank
/// so p50/p99/p999 are stable across sample counts and match the file_io harness
/// exactly, keeping percentile semantics identical across the whole suite.
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = p * (sorted.len() as f64 - 1.0);
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    if lo == hi {
        sorted[lo]
    } else {
        let f = rank - lo as f64;
        sorted[lo] * (1.0 - f) + sorted[hi] * f
    }
}

/// Rounds to whole microseconds (3 decimal ms). Exists only to keep stored and
/// printed numbers readable; sub-microsecond digits at this scale are scheduler
/// noise, not signal.
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

// Each sample is one full N_TASKS fan-out through the variant's primitive.
// We collect SAMPLES wall times, sort, and report p50/p99/p999.
// NOTE: at SAMPLES=1000, p999 lands on rank ~0.999*999 ≈ 998 — a genuine
// 1-in-1000 tail, not just "the worst run."
fn measure(name: &str, build: impl Fn()) -> Stats {
    // one warm-up run (primes branch predictor / i-cache; for rayon it also
    // forces the global pool to be built so it isn't charged to sample 0).
    build();

    let mut times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t = Instant::now();
        build();
        times.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let s = Stats {
        p50: round3(pct(&times, 0.50)),
        p99: round3(pct(&times, 0.99)),
        p999: round3(pct(&times, 0.999)),
    };
    println!(
        "[{name}]  p50={:.3}ms p99={:.3}ms p999={:.3}ms",
        s.p50, s.p99, s.p999
    );
    s
}

// ───────────────────────── data file (file_io-style schema) ──────────────
// Mirrors file_io/file_io.data: block = {color, file_bytes, n_lines, samples,
// variants:{name:{ms_p50,ms_p99,ms_p999,heap_peak_kb,heap_live_kb,rss_peak_kb,
// lines}}}. The CPU benchmark has no file/memory dimension, so file_bytes,
// n_lines, and all memory fields are written as 0 (n/a).
fn save(
    data: &Path,
    lang: &str,
    color: &str,
    samples: usize,
    variants: &[(&str, Stats)],
) {
    let mut vmap = Map::new();
    for (name, s) in variants {
        vmap.insert(
            (*name).to_string(),
            json!({
                "ms_p50": s.p50, "ms_p99": s.p99, "ms_p999": s.p999,
                "heap_peak_kb": 0, "heap_live_kb": 0, "rss_peak_kb": 0,
                "lines": 0
            }),
        );
    }
    let entry = json!({
        "color": color,
        "file_bytes": 0,
        "n_lines": 0,
        "n_terms": N_TERMS,
        "n_tasks": N_TASKS,
        "samples": samples,
        "variants": Value::Object(vmap),
    });

    let mut root: Value = fs::read_to_string(data)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    root[lang] = entry;

    fs::write(data, serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

fn main() {
    let variants = vec![
        ("serial", measure("serial", compute_serial)),
        ("thread::spawn", measure("thread::spawn", compute_with_threads)),
        ("rayon", measure("rayon", compute_with_rayon)),
        ("threadpool", measure("threadpool", compute_with_threadpool)),
        ("async (tokio)", measure("async (tokio)", compute_with_async)),
        ("futures-cpupool", measure("futures-cpupool", compute_with_cpupool)),
    ];

    save(Path::new("../plot.data"), "Rust", "#CE422B", SAMPLES, &variants);

    // defeat DCE; do not remove
    println!("sink={}", SINK.load(Ordering::Relaxed));
}