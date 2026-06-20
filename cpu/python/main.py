import json
import os
import threading
import multiprocessing
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

N_TERMS = 500_000_000   # tune this; your Rust used up to 10_000_000
N_TASKS = 3


def compute_pi(n_terms):
    pi = 0.0
    sign = 1.0
    k = 1.0
    for _ in range(n_terms):
        pi += sign / k
        sign = -sign
        k += 2.0
    return 4 * pi


def worker(n_terms):
    result = compute_pi(n_terms)
    print(f"  π[{n_terms}] = {result}")
    return result


def compute_with_threads():
    threads = [threading.Thread(target=worker, args=(N_TERMS,)) for _ in range(N_TASKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def compute_with_processes():
    procs = [multiprocessing.Process(target=worker, args=(N_TERMS,)) for _ in range(N_TASKS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


async def _async_main():
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=N_TASKS) as pool:
        tasks = [loop.run_in_executor(pool, worker, N_TERMS) for _ in range(N_TASKS)]
        await asyncio.gather(*tasks)


def compute_with_asyncio():
    asyncio.run(_async_main())


def compute_with_thread_pool():
    with ThreadPoolExecutor(max_workers=N_TASKS) as pool:
        list(pool.map(worker, [N_TERMS] * N_TASKS))


def compute_with_process_pool():
    with ProcessPoolExecutor(max_workers=N_TASKS) as pool:
        list(pool.map(worker, [N_TERMS] * N_TASKS))

def save_plot_data(lang, color, serial_ms, variants):
    entry = {
        "color": color,
        "n_terms": N_TERMS,
        "n_tasks": N_TASKS,
        "serial_ms": round(serial_ms, 1),
        "variants": {name: round(ms, 1) for name, ms in variants.items()},
    }

    root = {}
    if os.path.exists("../plot.data"):
        try:
            with open("../plot.data") as f:
                root = json.load(f)
        except (ValueError, OSError):
            root = {}
    root[lang] = entry

    with open("../plot.data", "w") as f:
        json.dump(root, f, indent=2)

def bench(name, fn):
    start = time.time()
    print(f"[{name}]")
    fn()
    ms = (time.time() - start) * 1000.0
    print(f"  Computation time: {ms / 1000.0:.2f}s\n-------\n")
    return ms


if __name__ == "__main__":
    serial_ms = bench("serial", lambda: [worker(N_TERMS) for _ in range(N_TASKS)])

    variants = {}
    variants["threading"]           = bench("threading", compute_with_threads)
    variants["multiprocessing"]     = bench("multiprocessing", compute_with_processes)
    variants["asyncio"]             = bench("asyncio", compute_with_asyncio)
    variants["ThreadPoolExecutor"]  = bench("ThreadPoolExecutor", compute_with_thread_pool)
    variants["ProcessPoolExecutor"] = bench("ProcessPoolExecutor", compute_with_process_pool)

    save_plot_data("Python 3.14t", "#FFD43B", serial_ms, variants)