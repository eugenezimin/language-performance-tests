const { Worker, isMainThread, parentPort } = require("worker_threads");
const { fork } = require("child_process");
const { performance } = require("perf_hooks");

const fs = require("fs");

const N_TERMS = 500_000_000; // tune this
const N_TASKS = 3;
const POOL_SIZE = 2; // workers reused in the pool variant

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

function worker(nTerms) {
  console.log(`  \u03C0[${nTerms}] = ${computePi(nTerms)}`);
}

if (!isMainThread) {
  parentPort.on("message", (nTerms) => {
    worker(nTerms);
    parentPort.postMessage("done");
  });
  return; // stop here; do not run main()
}

if (process.env.PI_CHILD) {
  worker(Number(process.env.PI_CHILD));
  process.exit(0);
}

function computeSerial() {
  for (let i = 0; i < N_TASKS; i++) worker(N_TERMS);
}

async function computeWithAsync() {
  await Promise.all(
    Array.from({ length: N_TASKS }, () =>
      Promise.resolve().then(() => worker(N_TERMS))
    )
  );
}

function computeWithWorkerThreads() {
  return Promise.all(
    Array.from({ length: N_TASKS }, () =>
      new Promise((resolve) => {
        const w = new Worker(__filename);
        w.once("message", () => w.terminate().then(resolve));
        w.postMessage(N_TERMS);
      })
    )
  );
}

function computeWithWorkerPool() {
  const tasks = Array.from({ length: N_TASKS }, () => N_TERMS);
  const size = Math.min(POOL_SIZE, tasks.length);
  return new Promise((resolve) => {
    let next = 0;
    let done = 0;
    const workers = [];
    const assign = (w) => {
      if (next < tasks.length) w.postMessage(tasks[next++]);
    };
    for (let i = 0; i < size; i++) {
      const w = new Worker(__filename);
      w.on("message", () => {
        if (++done === tasks.length) {
          workers.forEach((x) => x.terminate());
          resolve();
        } else {
          assign(w);
        }
      });
      workers.push(w);
      assign(w);
    }
  });
}

function computeWithChildProcess() {
  return Promise.all(
    Array.from({ length: N_TASKS }, () =>
      new Promise((resolve) => {
        const cp = fork(__filename, [], {
          env: { ...process.env, PI_CHILD: String(N_TERMS) },
        });
        cp.on("exit", () => resolve());
      })
    )
  );
}

const round1 = (x) => Math.round(x * 10) / 10;

function savePlotData(lang, color, serialMs, variants) {
  const entry = {
    color,
    n_terms: N_TERMS,
    n_tasks: N_TASKS,
    serial_ms: round1(serialMs),
    variants: Object.fromEntries(variants.map(([name, ms]) => [name, round1(ms)])),
  };

  let root = {};
  try {
    root = JSON.parse(fs.readFileSync("../plot.data", "utf8"));
  } catch {}
  root[lang] = entry;

  fs.writeFileSync("../plot.data", JSON.stringify(root, null, 2));
}

async function bench(name, fn) {
  const start = performance.now();
  console.log(`[${name}]`);
  await fn();
  const ms = performance.now() - start;
  console.log(`  Computation time: ${ms.toFixed(2)}ms\n-------\n`);
  return ms;
}

(async () => {
  const serialMs = await bench("serial", computeSerial);
  const variants = [
    ["serial", serialMs],
    ["async/await", await bench("async/await", computeWithAsync)],
    ["worker_threads", await bench("worker_threads", computeWithWorkerThreads)],
    ["worker pool", await bench("worker pool", computeWithWorkerPool)],
    ["child_process", await bench("child_process", computeWithChildProcess)],
  ];
  savePlotData("Node.js", "#3C873A", serialMs, variants);
})();