use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ───────────────────────── counting global allocator ─────────────────────
// Byte-exact, profiler-free. Counts every allocation routed through the
// global allocator. realloc is overridden so Vec growth uses the *real*
// System.realloc (no forced copy) — timing stays representative.
struct Counting;
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let np = System.realloc(p, l, new);
        if !np.is_null() {
            if new >= l.size() {
                let now = CURRENT.fetch_add(new - l.size(), Ordering::Relaxed) + (new - l.size());
                PEAK.fetch_max(now, Ordering::Relaxed);
            } else {
                CURRENT.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        np
    }
}
#[global_allocator]
static GA: Counting = Counting;

fn cur() -> usize {
    CURRENT.load(Ordering::Relaxed)
}
fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}
fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

// ───────────────────────── per-variant peak RSS (Linux) ──────────────────
#[cfg(target_os = "linux")]
fn reset_rss_peak() {
    // Linux >= 4.0: writing "5" to clear_refs resets VmHWM to current VmRSS,
    // so each variant gets its own peak instead of a process-wide high-water.
    let _ = fs::write("/proc/self/clear_refs", "5");
}
#[cfg(target_os = "linux")]
fn rss_peak_kb() -> u64 {
    fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("VmHWM:")
                    .and_then(|r| r.split_whitespace().next())
                    .and_then(|v| v.parse().ok())
            })
        })
        .unwrap_or(0)
}
#[cfg(not(target_os = "linux"))]
fn reset_rss_peak() {}
#[cfg(not(target_os = "linux"))]
fn rss_peak_kb() -> u64 {
    0
}

// ───────────────────────── the three variants ────────────────────────────
// Contract: each returns Vec<String> of all lines (owned, ready to process).

// 1. slurp + split — one big read, one bulk UTF-8 validation, then carve.
//    Peak holds the whole-file String AND the line Vec simultaneously.
fn slurp_split(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).expect("read_to_string");
    text.lines().map(|l| l.to_owned()).collect()
}

// 2. buffered line iteration — the idiom; 8 KiB BufReader, one String/line.
fn buffered_lines(path: &Path) -> Vec<String> {
    let reader = BufReader::new(File::open(path).expect("open"));
    reader.lines().map(|l| l.expect("line")).collect()
}

// 3. manual chunk + hand-split — you own the 64 KiB reads, the newline scan,
//    and the UTF-8 boundary: trailing bytes carry into the next chunk before
//    decoding, so a multi-byte char straddling a chunk is handled correctly.
fn manual_chunk(path: &Path) -> Vec<String> {
    const CHUNK: usize = 64 * 1024;
    let mut f = File::open(path).expect("open");
    let mut buf = vec![0u8; CHUNK];
    let mut carry: Vec<u8> = Vec::new();
    let mut lines: Vec<String> = Vec::new();
    loop {
        let n = f.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        let mut start = 0;
        for i in 0..n {
            if buf[i] == b'\n' {
                if carry.is_empty() {
                    lines.push(String::from_utf8_lossy(&buf[start..i]).into_owned());
                } else {
                    carry.extend_from_slice(&buf[start..i]);
                    lines.push(String::from_utf8_lossy(&carry).into_owned());
                    carry.clear();
                }
                start = i + 1;
            }
        }
        carry.extend_from_slice(&buf[start..n]); // partial line + partial char
    }
    if !carry.is_empty() {
        lines.push(String::from_utf8_lossy(&carry).into_owned());
    }
    lines
}

// ───────────────────────── harness ───────────────────────────────────────
struct Stats {
    p50: f64,
    p99: f64,
    p999: f64,
    heap_peak_kb: u64,
    heap_live_kb: u64,
    rss_peak_kb: u64,
    lines: usize,
}

// DCE sink: fold the lines so --release can't delete the build.
fn checksum(lines: &[String]) -> (usize, usize) {
    (lines.len(), lines.iter().map(|l| l.len()).sum())
}

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
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn measure<F: Fn() -> Vec<String>>(name: &str, samples: usize, build: F) -> Stats {
    // warm-up: cache + allocator arenas
    {
        let v = build();
        std::hint::black_box(checksum(&v));
    }

    // timing pass — one build per sample, dropped each time
    let mut times = Vec::with_capacity(samples);
    let mut lines = 0usize;
    for _ in 0..samples {
        let t = Instant::now();
        let v = build();
        let dt = t.elapsed().as_secs_f64() * 1000.0;
        let (c, b) = checksum(&v);
        std::hint::black_box((c, b));
        lines = c;
        times.push(dt);
        drop(v);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // memory pass — single representative build, held live while we read
    reset_rss_peak();
    reset_peak();
    let base = cur();
    let v = build();
    let heap_peak = peak().saturating_sub(base);
    let heap_live = cur().saturating_sub(base);
    std::hint::black_box(checksum(&v));
    let rss_peak = rss_peak_kb();
    drop(v);

    let s = Stats {
        p50: round3(pct(&times, 0.50)),
        p99: round3(pct(&times, 0.99)),
        p999: round3(pct(&times, 0.999)),
        heap_peak_kb: (heap_peak / 1024) as u64,
        heap_live_kb: (heap_live / 1024) as u64,
        rss_peak_kb: rss_peak,
        lines,
    };
    println!(
        "[{name}]  p50={:.3}ms p99={:.3}ms p999={:.3}ms  \
              heap_peak={}KB heap_live={}KB rss_peak={}KB  ({} lines)",
        s.p50, s.p99, s.p999, s.heap_peak_kb, s.heap_live_kb, s.rss_peak_kb, s.lines
    );
    s
}

fn save(
    data: &Path,
    lang: &str,
    color: &str,
    file_bytes: u64,
    n_lines: usize,
    samples: usize,
    variants: &[(&str, Stats)],
) {
    use serde_json::{json, Map, Value};
    let mut vmap = Map::new();
    for (name, s) in variants {
        vmap.insert(
            (*name).to_string(),
            json!({
                "ms_p50": s.p50, "ms_p99": s.p99, "ms_p999": s.p999,
                "heap_peak_kb": s.heap_peak_kb, "heap_live_kb": s.heap_live_kb,
                "rss_peak_kb": s.rss_peak_kb, "lines": s.lines
            }),
        );
    }
    let entry = json!({
        "color": color, "file_bytes": file_bytes,
        "n_lines": n_lines, "samples": samples, "variants": Value::Object(vmap)
    });
    let mut root: Value = fs::read_to_string(data)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    root[lang] = entry;
    fs::write(data, serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

const DATA: &str = "../file_io.data";
const SAMPLES: usize = 100;

fn main() {
    // Anchor to the crate dir (file_io/rust) at compile time, so the fixture
    // and data file resolve correctly no matter where cargo is invoked from.
    // Both live one level up, in file_io/.
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = base.join("../text-examples.txt");
    let data = base.join("../file_io.data");

    let file_bytes = fs::metadata(&path)
        .expect("stat fixture — expected file_io/text-examples.txt")
        .len();

    let s1 = measure("slurp + split", SAMPLES, || slurp_split(&path));
    let s2 = measure("buffered lines", SAMPLES, || buffered_lines(&path));
    let s3 = measure("manual chunk", SAMPLES, || manual_chunk(&path));

    let n_lines = s1.lines;
    save(
        &data,
        "Rust",
        "#CE422B",
        file_bytes,
        n_lines,
        SAMPLES,
        &[
            ("slurp + split", s1),
            ("buffered lines", s2),
            ("manual chunk", s3),
        ],
    );
}
