const { Worker, isMainThread, parentPort } = require("worker_threads");
const os = require("os");
const path = require("path");
const fs = require("fs");

const N_TERMS = 10_000;
const N_TASKS = 100;
const SAMPLES = 1000;
// Leave the main thread a core; the pool gets the rest.
const POOL_SIZE = Math.max(1, os.cpus().length - 1);

// The shared CPU payload — identical naive FP loop to the other four languages,
// so only the dispatch layer differs. Defined at top level so BOTH the worker
// side and the (single-threaded) main-side variants call the exact same code.
// Naive on purpose (no Kahan / SIMD / unrolling): this measures the runtime and
// dispatch layer, not who optimizes the math best.
function computePi(nTerms) {
  let pi = 0;
  let sign = 1;
  let k = 1;
  for (let i = 0; i < nTerms; i++) {
    pi += sign / k;
    sign = -sign;
    k += 2;
  }
  return 4 * pi;
}

// Fold a float64's raw bits to a BigInt, the JS analog of Rust's f64::to_bits().
// Lets us XOR results into a sink so V8 can't prove the compute dead, and keeps
// stdout OUT of every timed region (a per-task console.log would be measured
// SAMPLES*N_TASKS times and serialize parallel workers on the I/O lock — the
// exact trap Rust's SINK and Go's sink removed).
const fbuf = new ArrayBuffer(8);
const f64 = new Float64Array(fbuf);
const u64 = new BigUint64Array(fbuf);
function bitsOf(x) {
  f64[0] = x;
  return u64[0];
}

// ───────────────────────── worker side ───────────────────────────────────
// Each persistent worker is its OWN V8 isolate: it re-parses and re-JITs
// computePi from cold (no JIT sharing across the thread boundary — the Node
// fact that makes this different from Rust/Go thread pools). It receives a term
// count, computes, folds to bits locally, and posts the bits back. The main
// thread XORs them into its sink. Only an integer goes out and a BigInt comes
// back, so structured-clone cost is minimal but real.
if (!isMainThread) {
  parentPort.on("message", (nTerms) => {
    const r = computePi(nTerms);
    parentPort.postMessage(bitsOf(r));
  });
  return; // worker stops here; never runs the harness
}

// ───────────────────────── persistent worker pool ────────────────────────
// Built ONCE, before any timing. Per sample we dispatch a full N_TASKS fan-out
// across the idle workers and resolve when all N_TASKS results land — we measure
// DISPATCH (MessagePort round-trips), never spawn. This is the Python lesson:
// build the heavy pool outside the timed loop or you measure construction.
//
// CAVEAT made explicit: at N_TERMS=10_000 the compute is a few microseconds but
// each task is a MessagePort round-trip. The pool's p50 is therefore a port-
// latency floor, NOT compute — expect it at or ABOVE serial. If it comes in
// BELOW serial, suspect the dispatcher collapsed the fan-out (a harness bug),
// not a real win.
class WorkerPool {
  constructor(size) {
    this.workers = [];
    for (let i = 0; i < size; i++) {
      const w = new Worker(__filename);
      w.on("error", (e) => {
        throw e;
      });
      this.workers.push(w);
    }
  }

  // Run one full fan-out of `count` identical tasks; resolve after all return.
  // Each worker carries ONE in-flight task at a time: post a task, on its reply
  // fold the bits, then either pull the next queued task or go idle. The promise
  // resolves when `completed === count`. A single message listener per worker is
  // installed for the fan-out's duration and removed at the end — no per-task
  // listener churn (which would leak across 1000 samples and inflate p999).
  runFanout(count, nTerms) {
    return new Promise((resolve) => {
      let remaining = count;   // tasks not yet dispatched
      let completed = 0;       // tasks whose result has come back
      const listeners = [];

      const pump = (w) => {
        if (remaining > 0) {
          remaining--;
          w.postMessage(nTerms);
        }
      };

      for (const w of this.workers) {
        const onMsg = (bits) => {
          sink ^= bits;        // fold result; defeats DCE
          completed++;
          if (completed === count) {
            for (let i = 0; i < this.workers.length; i++) {
              this.workers[i].off("message", listeners[i]);
            }
            resolve();
            return;
          }
          pump(w);             // give this worker the next queued task
        };
        listeners.push(onMsg);
        w.on("message", onMsg);
      }

      // Prime every worker with its first task.
      for (const w of this.workers) pump(w);
    });
  }

  async terminate() {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}

// ───────────────────────── the three variants ────────────────────────────

// 1. serial baseline — N_TASKS back to back on one thread. Zero dispatch
//    overhead; the honest reference the parallel variant is measured against.
//    Single isolate, fully JIT'd after warm-up.
function computeSerial() {
  for (let i = 0; i < N_TASKS; i++) {
    sink ^= bitsOf(computePi(N_TERMS));
  }
}

// 2. Promise.all — the idiom people WRONGLY reach for to "parallelize." The CPU
//    loop never yields, so the microtask queue runs all N_TASKS strictly
//    SERIALLY on the main thread with promise/scheduling overhead piled on top.
//    Predicted at or slightly WORSE than serial — the "async parallelizes I/O,
//    never CPU" lesson, the analog of Python's pure asyncio.gather.
async function computeWithPromiseAll() {
  await Promise.all(
    Array.from({ length: N_TASKS }, () =>
      Promise.resolve().then(() => {
        sink ^= bitsOf(computePi(N_TERMS));
      })
    )
  );
}

// 3. worker pool — the genuine idiomatic Node parallel path. Persistent isolates,
//    tasks dispatched over MessagePort. True OS-thread parallelism, but every
//    task crosses the port boundary. See the WorkerPool caveat: at this payload
//    the round-trip floor likely puts it AT or ABOVE serial.
function makeComputeWithWorkerPool(pool) {
  return () => pool.runFanout(N_TASKS, N_TERMS);
}

// ───────────────────────── harness ───────────────────────────────────────
// Lifted from cpu/rust and cpu/python so percentile semantics match the suite:
// linear-interpolated percentile over a pre-sorted slice, 1 warm-up, SAMPLES
// timed full fan-outs, monotonic ns clock. GC left at default — CPU variants
// allocate ~nothing, so p99/p999 honestly carry any scavenge tail.
function pct(sorted, p) {
  if (sorted.length === 0) return 0;
  const rank = p * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  const f = rank - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}

const round3 = (x) => Math.round(x * 1000) / 1000;

async function measure(name, build) {
  await build(); // one warm-up: primes Ignition->TurboFan; for the pool, first
  //               dispatch so isolate JIT isn't charged to sample 0.

  const times = new Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) {
    const t = process.hrtime.bigint();
    await build();
    times[i] = Number(process.hrtime.bigint() - t) / 1e6;
  }
  times.sort((a, b) => a - b);

  const s = {
    ms_p50: round3(pct(times, 0.5)),
    ms_p99: round3(pct(times, 0.99)),
    ms_p999: round3(pct(times, 0.999)),
    // CPU benchmark has no file/memory dimension — written as 0 (n/a),
    // mirroring the Rust/Go/Python CPU blocks' schema exactly.
    heap_peak_kb: 0,
    heap_live_kb: 0,
    rss_peak_kb: 0,
    lines: 0,
  };
  console.log(
    `[${name}]  p50=${s.ms_p50.toFixed(3)}ms p99=${s.ms_p99.toFixed(3)}ms ` +
      `p999=${s.ms_p999.toFixed(3)}ms`
  );
  return s;
}

// ───────────────────────── data file (file_io-style schema) ──────────────
// Read the existing tree, replace ONLY the "Node.js" key, write back.
// JSON.parse/stringify preserve insertion order for non-numeric string keys, so
// the other languages' blocks keep their order/content — the analog of Go's
// RawMessage / Rust's preserve_order / Python's insertion-ordered dict.
function save(dataPath, lang, color, samples, variants) {
  const vmap = {};
  for (const [vname, s] of variants) vmap[vname] = s;
  const block = {
    color,
    file_bytes: 0,
    n_lines: 0,
    n_terms: N_TERMS,
    n_tasks: N_TASKS,
    samples,
    variants: vmap,
  };
  let root = {};
  try {
    root = JSON.parse(fs.readFileSync(dataPath, "utf8"));
  } catch {}
  root[lang] = block;
  fs.writeFileSync(dataPath, JSON.stringify(root, null, 2));
}

// DCE sink: every variant folds its results here via XOR; printed once at the
// end so the optimizer sees the work escape. Never read inside a timed region.
let sink = 0n;

(async () => {
  // Anchor to cpu/ (one up from this script) so plot.data resolves regardless
  // of cwd — the CARGO_MANIFEST_DIR / runtime.Caller analog.
  const dataPath = path.join(__dirname, "..", "plot.data");

  const pool = new WorkerPool(POOL_SIZE);

  const variants = [
    ["serial", await measure("serial", computeSerial)],
    ["Promise.all", await measure("Promise.all", computeWithPromiseAll)],
    ["worker pool", await measure("worker pool", makeComputeWithWorkerPool(pool))],
  ];

  await pool.terminate();

  save(dataPath, "Node.js", "#3C873A", SAMPLES, variants);

  console.log("sink=" + sink.toString()); // defeat DCE; do not remove
})();