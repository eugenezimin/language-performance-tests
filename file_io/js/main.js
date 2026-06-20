const fs = require("fs");
const path = require("path");
const readline = require("readline");

// DCE sink: V8's escape analysis will delete a build whose result never
// escapes, so we fold every checksum into a module-level counter and print it
// at the end. The List also escapes into checksum(), so the loop can't be DCE'd.
let sink = 0;

// ───────────────────────── the three variants ────────────────────────────
// Contract mirrors Rust/Go/Java/Python: return string[] of all lines, each
// OWNED, terminators stripped, no trailing empty. CAVEAT: variant 1's split
// yields V8 SlicedStrings (lines >= 13 code units) that SHARE the parent's
// backing store — so it PINS the whole-file string and is NOT semantically the
// owned contract, the exact analog of Go's ReadFile+Split. That divergence is
// the result. The fixture is non-Latin1, so every decoded string is UTF-16
// (2 bytes/char) — heap numbers run ~2x Go/Rust for identical lines.

//  1. readFileSync + split — slurp the whole ~45 MB file, full UTF-8->UTF-16
//     decode into one V8 string, then split. Decode + split both run in C++,
//     so this is the fastest idiomatic path. Peak holds the whole-file string;
//     long lines pin it via SlicedString, short (<13) lines are copied out.
async function readFileSplit(p) {
  const s = fs.readFileSync(p, "utf8");
  const lines = s.split("\n");
  // split yields a trailing "" when the file ends in '\n'; .lines() does not.
  // (LF-only fixture; CRLF would leave a trailing '\r' — changes byte sums, not counts.)
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

//  2. readline over createReadStream — the idiomatic streaming line reader
//     (analog of bufio.Scanner / readAllLines / buffered lines). Strips the
//     terminator and emits no trailing empty. Event-emitter + closure per line:
//     expect this to be the SLOWEST by a wide margin — Node's idiom tax.
function readlineLines(p) {
  return new Promise((resolve, reject) => {
    const lines = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(p, { encoding: "utf8" }),
      crlfDelay: Infinity, // \r\n counts as one break (moot on the LF fixture)
    });
    rl.on("line", (line) => lines.push(line)); // owned, terminator stripped
    rl.on("close", () => resolve(lines));
    rl.on("error", reject);
  });
}

//  3. manual chunk — you own the 64 KiB reads, the newline scan, and the
//     cross-chunk carry. buf.toString('utf8', a, b) copies -> owned flat string.
//     Cutting only at 0x0A is always a valid UTF-8 char boundary (0x0A never
//     appears mid-sequence); a line straddling a chunk is carried as raw BYTES
//     and decoded once joined, so multibyte chars across boundaries are intact.
//     The per-byte scan is JIT'd JS (not C++ split), so slower than variant 1 —
//     but nowhere near Python's bytecode-scan penalty.
async function manualChunk(p) {
  const CHUNK = 64 * 1024;
  const buf = Buffer.allocUnsafe(CHUNK); // reused; fully overwritten each read
  const lines = [];
  let carry = null; // raw bytes of a partial line spanning chunks
  const fd = fs.openSync(p, "r");
  try {
    let n;
    while ((n = fs.readSync(fd, buf, 0, CHUNK, null)) > 0) {
      let start = 0;
      for (let i = 0; i < n; i++) {
        if (buf[i] === 0x0a) {
          if (carry === null) {
            lines.push(buf.toString("utf8", start, i));
          } else {
            // subarray shares `buf`; concat copies, so decode is safe here
            lines.push(Buffer.concat([carry, buf.subarray(start, i)]).toString("utf8"));
            carry = null;
          }
          start = i + 1;
        }
      }
      if (start < n) {
        // copy out of the reused buffer before the next read overwrites it
        const tail = Buffer.from(buf.subarray(start, n));
        carry = carry === null ? tail : Buffer.concat([carry, tail]);
      }
    }
  } finally {
    fs.closeSync(fd);
  }
  // No trailing empty for a '\n'-terminated file (carry === null here).
  if (carry !== null && carry.length > 0) lines.push(carry.toString("utf8"));
  return lines;
}

// ───────────────────────── per-variant peak RSS (Linux) ──────────────────
function resetRssPeak() {
  if (process.platform === "linux") {
    try {
      fs.writeFileSync("/proc/self/clear_refs", "5"); // VmHWM -> current VmRSS
    } catch (_) {}
  }
}
function rssPeakKb() {
  if (process.platform !== "linux") return 0;
  try {
    const status = fs.readFileSync("/proc/self/status", "utf8");
    for (const line of status.split("\n")) {
      if (line.startsWith("VmHWM:")) {
        const f = line.trim().split(/\s+/);
        if (f.length >= 2) return parseInt(f[1], 10);
      }
    }
  } catch (_) {}
  return 0;
}

// ───────────────────────── harness ───────────────────────────────────────
// checksum uses .length = UTF-16 code units, NOT bytes. On the multibyte
// fixture this diverges from Go/Rust byte sums (same caveat Java flags); it is
// only a DCE sink + line count, and line COUNTS still match across languages.
function checksum(lines) {
  let total = 0;
  for (let i = 0; i < lines.length; i++) total += lines[i].length;
  return [lines.length, total];
}

function pct(sorted, p) {
  if (sorted.length === 0) return 0;
  const rank = p * (sorted.length - 1);
  const lo = Math.floor(rank), hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  const f = rank - lo;
  return sorted[lo] * (1 - f) + sorted[hi] * f;
}
const round3 = (x) => Math.round(x * 1000) / 1000;
const satSub = (a, b) => (a < b ? 0 : a - b);

async function measure(name, samples, build) {
  // warm-up: page cache + size-classes + a first JIT pass (1 build, matching
  // the other harnesses exactly — no extra warm-up rounds to game steady state)
  {
    const v = await build();
    const [c, b] = checksum(v);
    sink += c + b;
  }

  // timing pass — monotonic ns clock. V8 GC cannot be frozen (no
  // SetGCPercent(-1)/gc.disable() analog), so p99/p999 honestly carry the
  // scavenge/mark tail. The `await` adds a sub-microsecond microtask turn to
  // the sync variants — uniform and negligible at ms scale.
  const times = new Array(samples);
  let lines = 0;
  for (let i = 0; i < samples; i++) {
    const t = process.hrtime.bigint();
    const v = await build();
    const dt = Number(process.hrtime.bigint() - t) / 1e6;
    const [c, b] = checksum(v);
    sink += c + b;
    lines = c;
    times[i] = dt;
  }
  times.sort((a, b) => a - b);

  // memory pass — single build held live. heap = heapUsed + arrayBuffers
  // (captures the off-heap slurp Buffer, the Go/Rust slurp-alloc analog).
  // heap_live (forced GC, result live) is the comparable number; heap_peak is
  // a soft LOWER bound because V8 may have already scavenged transient garbage.
  if (typeof global.gc === "function") global.gc();
  const base = process.memoryUsage();
  const baseHeap = base.heapUsed + base.arrayBuffers;
  resetRssPeak();

  const v = await build();

  const after = process.memoryUsage();
  const heapPeak = satSub(after.heapUsed + after.arrayBuffers, baseHeap);

  if (typeof global.gc === "function") global.gc(); // result must stay live
  const liveU = process.memoryUsage();
  const heapLive = satSub(liveU.heapUsed + liveU.arrayBuffers, baseHeap);

  const rss = rssPeakKb();
  const [c, b] = checksum(v);
  sink += c + b; // keep v reachable across the gc + reads

  const s = {
    ms_p50: round3(pct(times, 0.5)),
    ms_p99: round3(pct(times, 0.99)),
    ms_p999: round3(pct(times, 0.999)),
    heap_peak_kb: Math.floor(heapPeak / 1024),
    heap_live_kb: Math.floor(heapLive / 1024),
    rss_peak_kb: rss,
    lines,
  };
  console.log(
    `[${name}]  p50=${s.ms_p50.toFixed(3)}ms p99=${s.ms_p99.toFixed(3)}ms ` +
      `p999=${s.ms_p999.toFixed(3)}ms  heap_peak=${s.heap_peak_kb}KB ` +
      `heap_live=${s.heap_live_kb}KB rss_peak=${s.rss_peak_kb}KB  (${s.lines} lines)`
  );
  return s;
}

// ───────────────────────── data file (object merge) ──────────────────────
// JSON.parse/stringify preserve insertion order for non-numeric string keys,
// so other languages' blocks keep their order/content (the analog of Go's
// RawMessage / Rust's preserve_order / Java's LinkedHashMap-backed ObjectNode).
function save(dataPath, lang, color, fileBytes, nLines, samples, variants) {
  const vmap = {};
  for (const [vname, s] of variants) vmap[vname] = s;
  const block = {
    color,
    file_bytes: fileBytes,
    n_lines: nLines,
    samples,
    variants: vmap,
  };
  let root = {};
  try {
    root = JSON.parse(fs.readFileSync(dataPath, "utf8"));
  } catch (_) {}
  root[lang] = block;
  fs.writeFileSync(dataPath, JSON.stringify(root, null, 2));
}

(async () => {
  if (typeof global.gc !== "function") {
    console.warn(
      "WARNING: run with `node --expose-gc main.js` — heap_live/heap_peak are unreliable without forced GC"
    );
  }

  // Anchor to file_io/ (one up from this script) so fixture/data resolve
  // regardless of cwd — the CARGO_MANIFEST_DIR / runtime.Caller analog.
  const baseDir = path.join(__dirname, "..");
  const filePath = path.join(baseDir, "text-examples.txt");
  const dataPath = path.join(baseDir, "file_io.data");

  const fileBytes = fs.statSync(filePath).size; // throws if fixture missing

  const SAMPLES = 100;
  const s1 = await measure("readFileSync + split", SAMPLES, () => readFileSplit(filePath));
  const s2 = await measure("readline (stream)", SAMPLES, () => readlineLines(filePath));
  const s3 = await measure("manual chunk", SAMPLES, () => manualChunk(filePath));

  const nLines = s1.lines;
  save(dataPath, "Node.js", "#3C873A", fileBytes, nLines, SAMPLES, [
    ["readFileSync + split", s1],
    ["readline (stream)", s2],
    ["manual chunk", s3],
  ]);

  console.log("sink=" + sink); // defeat DCE; do not remove
})();