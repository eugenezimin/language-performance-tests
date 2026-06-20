use futures::Future;
use futures_cpupool::CpuPool;
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::fs;
use threadpool::ThreadPool;

const N_TERMS: u32 = 500_000_000; // tune this
const N_TASKS: usize = 3;

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

fn worker(n_terms: u32) {
    println!("  π[{}] = {}", n_terms, compute_pi(n_terms));
}

// 1. threading -> std::thread::spawn
fn compute_with_threads() {
    let handles: Vec<_> = (0..N_TASKS)
        .map(|_| std::thread::spawn(|| worker(N_TERMS)))
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

// 2. multiprocessing analog -> rayon (real data parallelism)
fn compute_with_rayon() {
    (0..N_TASKS).into_par_iter().for_each(|_| worker(N_TERMS));
}

// 3. ThreadPoolExecutor -> threadpool
fn compute_with_threadpool() {
    let pool = ThreadPool::new(num_cpus::get());
    for _ in 0..N_TASKS {
        pool.execute(|| worker(N_TERMS));
    }
    pool.join();
}

// 4. asyncio -> tokio (CPU work via spawn_blocking)
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

// 5. ProcessPoolExecutor analog -> futures-cpupool
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

fn save_plot_data(lang: &str, color: &str, serial_ms: f64, variants: &[(&str, f64)]) {
    let mut vmap = Map::new();
    for (name, ms) in variants {
        vmap.insert((*name).to_string(), json!((ms * 10.0).round() / 10.0));
    }
    let entry = json!({
        "color": color,
        "n_terms": N_TERMS,
        "n_tasks": N_TASKS,
        "serial_ms": (serial_ms * 10.0).round() / 10.0,
        "variants": Value::Object(vmap),
    });

    let mut root: Value = fs::read_to_string("../plot.data")
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({}));
    root[lang] = entry;

    fs::write("../plot.data", serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

fn bench(name: &str, f: impl FnOnce()) -> f64 {
    let start = std::time::Instant::now();
    println!("[{}]", name);
    f();
    let elapsed = start.elapsed();
    println!("  Computation time: {:.2?}\n-------\n", elapsed);
    elapsed.as_secs_f64() * 1000.0
}

fn main() {
    let serial_ms = bench("serial", || {
        for _ in 0..N_TASKS {
            worker(N_TERMS);
        }
    });

    let variants = vec![
        (
            "thread::spawn",
            bench("thread::spawn", compute_with_threads),
        ),
        ("rayon", bench("rayon", compute_with_rayon)),
        ("threadpool", bench("threadpool", compute_with_threadpool)),
        ("async (tokio)", bench("async (tokio)", compute_with_async)),
        (
            "futures-cpupool",
            bench("futures-cpupool", compute_with_cpupool),
        ),
    ];

    save_plot_data("Rust", "#CE422B", serial_ms, &variants);
}
