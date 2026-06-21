//go:build !gc

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

const (
	nTerms  = 10_000
	nTasks  = 100
	samples = 1000
)

// DCE sink: fold every result's bits into a package-level atomic via XOR, the
// analog of Rust's SINK. This makes the computed value escape to a global so the
// compiler can't prove the loop dead, AND removes all stdout from the timed
// region — the old per-task fmt.Printf serialized parallel goroutines on the
// stdout lock and buried the dispatch signal (the exact trap Rust's SINK fixed).
var sink uint64

// The shared CPU payload for every variant: a tight, naive FP loop whose
// loop-carried dependency on pi isolates raw FP-divider throughput. Kept naive
// (no Kahan / SIMD / unrolling) on purpose so the benchmark measures the
// scheduler and dispatch layer, not who optimizes the math best.
func computePi(n int) float64 {
	pi := 0.0
	sign := 1.0
	k := 1.0
	for i := 0; i < n; i++ {
		pi += sign / k
		sign = -sign
		k += 2.0
	}
	return 4 * pi
}

// The single unit of work each primitive dispatches. Folds its result into the
// global sink rather than printing — keeps stdout out of the timed region while
// still letting the result escape so the loop survives compilation.
func worker() {
	r := computePi(nTerms)
	atomic.AddUint64(&sink, math.Float64bits(r))
}

// ───────────────────────── the five variants ─────────────────────────────

//  1. serial baseline — nTasks back to back on one goroutine. Zero dispatch
//     overhead; the honest reference every parallel variant is measured against.
func computeSerial() {
	for i := 0; i < nTasks; i++ {
		worker()
	}
}

//  2. goroutine-per-task + WaitGroup — Go's idiomatic "run these concurrently."
//     Cheap ~2KB-stack G's, work-stolen across P's. Unlike Rust's thread::spawn
//     (OS threads, slow), Go's spawn is so cheap this is predicted at or near
//     the headline — the rayon-vs-spawn gap mostly COLLAPSES here. That
//     collapse is the cross-language result.
func computeWithGoroutines() {
	var wg sync.WaitGroup
	wg.Add(nTasks)
	for i := 0; i < nTasks; i++ {
		go func() {
			defer wg.Done()
			worker()
		}()
	}
	wg.Wait()
}

//  3. worker pool sized to GOMAXPROCS — channel-fed, fixed goroutines reused.
//     The classic Go pool idiom. Contrasts central-queue contention against
//     variant 2's direct spawn at the same effective core count.
func computeWithWorkerPool() {
	size := runtime.NumCPU()
	if size > nTasks {
		size = nTasks
	}
	tasks := make(chan struct{}, nTasks)
	var wg sync.WaitGroup
	wg.Add(size)
	for i := 0; i < size; i++ {
		go func() {
			defer wg.Done()
			for range tasks {
				worker()
			}
		}()
	}
	for i := 0; i < nTasks; i++ {
		tasks <- struct{}{}
	}
	close(tasks)
	wg.Wait()
}

//  4. errgroup with SetLimit(NumCPU) — the MODERN idiomatic bounded-fanout
//     primitive (golang.org/x/sync). The Go-native replacement for Rust's
//     threadpool / futures-cpupool, not a forced analog. SetLimit caps live
//     goroutines; Go() blocks when the limit is hit, so this self-throttles.
func computeWithErrgroup() {
	var g errgroup.Group
	g.SetLimit(runtime.NumCPU())
	for i := 0; i < nTasks; i++ {
		g.Go(func() error {
			worker()
			return nil
		})
	}
	_ = g.Wait()
}

//  5. chunked over GOMAXPROCS goroutines — partition nTasks into NumCPU
//     contiguous ranges, one goroutine per range. Minimizes scheduler traffic
//     to one G per core: least dispatch of all variants, so predicted FASTEST
//     at this granularity. This is the genuinely-fastest-idiomatic headline path.
func computeWithChunked() {
	size := runtime.NumCPU()
	if size > nTasks {
		size = nTasks
	}
	var wg sync.WaitGroup
	wg.Add(size)
	base, rem := nTasks/size, nTasks%size
	start := 0
	for c := 0; c < size; c++ {
		count := base
		if c < rem {
			count++ // spread the remainder over the first `rem` chunks
		}
		go func(n int) {
			defer wg.Done()
			for i := 0; i < n; i++ {
				worker()
			}
		}(count)
		start += count
	}
	wg.Wait()
}

// ───────────────────────── harness ───────────────────────────────────────
// Lifted from the file_io Go harness so percentile semantics match the whole
// suite: linear-interpolated percentile over a pre-sorted slice.

type Stats struct {
	p50, p99, p999 float64
	numGC          uint32 // GC cycles observed during this variant's window
	pauseTotalNs   uint64 // STW pause time (ns) accumulated during the window
}

func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := p * float64(len(sorted)-1)
	lo, hi := int(math.Floor(rank)), int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	f := rank - float64(lo)
	return sorted[lo]*(1.0-f) + sorted[hi]*f
}

func round3(x float64) float64 { return math.Round(x*1000.0) / 1000.0 }

// Each sample is one full nTasks fan-out through the variant's primitive.
// 1 warm-up, then `samples` timed runs, sorted, reported p50/p99/p999.
// At samples=1000, p999 lands on rank ~0.999*999 ≈ 998 — a real 1-in-1000 tail.
func measure(name string, build func()) Stats {
	build() // warm-up: primes the scheduler / first-touch goroutine stacks

	// GC snapshot BEFORE the timed loop. ReadMemStats briefly STWs to get a
	// consistent snapshot, so it lives strictly OUTSIDE the timed region — the
	// per-sample timings never see it. We read process-global cumulative
	// counters and diff them; the delta is GC activity during this window.
	var msBefore runtime.MemStats
	runtime.ReadMemStats(&msBefore)

	times := make([]float64, 0, samples)
	for i := 0; i < samples; i++ {
		t := time.Now()
		build()
		times = append(times, float64(time.Since(t).Nanoseconds())/1e6)
	}

	var msAfter runtime.MemStats
	runtime.ReadMemStats(&msAfter)

	sort.Float64s(times)

	s := Stats{
		p50:          round3(pct(times, 0.50)),
		p99:          round3(pct(times, 0.99)),
		p999:         round3(pct(times, 0.999)),
		numGC:        msAfter.NumGC - msBefore.NumGC,
		pauseTotalNs: msAfter.PauseTotalNs - msBefore.PauseTotalNs,
	}
	fmt.Printf("[%s]  p50=%.3fms p99=%.3fms p999=%.3fms  gc=%d pause=%.3fms\n",
		name, s.p50, s.p99, s.p999,
		s.numGC, float64(s.pauseTotalNs)/1e6)
	return s
}

// ───────────────────────── data file (file_io-style schema) ──────────────
// New schema mirrors file_io.data: block = {color, file_bytes, n_lines,
// n_terms, n_tasks, samples, variants:{name:{ms_p50,ms_p99,ms_p999,
// heap_peak_kb,heap_live_kb,rss_peak_kb,lines}}}. The CPU benchmark has no
// file/memory dimension, so those fields are written as 0 (n/a).
//
// Read the existing tree as RawMessage so other languages' blocks (Rust's
// preserve_order layout, etc.) are kept byte-for-byte and not re-sorted —
// the same merge strategy as the Go file_io implementation.
type namedStats struct {
	name string
	s    Stats
}

func save(dataPath, lang, color string, variants []namedStats) {
	vmap := map[string]interface{}{}
	for _, ns := range variants {
		vmap[ns.name] = map[string]interface{}{
			"ms_p50": ns.s.p50, "ms_p99": ns.s.p99, "ms_p999": ns.s.p999,
			"heap_peak_kb": 0, "heap_live_kb": 0, "rss_peak_kb": 0,
			"lines": 0,
		}
	}
	block := map[string]interface{}{
		"color":      color,
		"file_bytes": 0,
		"n_lines":    0,
		"n_terms":    nTerms,
		"n_tasks":    nTasks,
		"samples":    samples,
		"variants":   vmap,
	}

	root := map[string]json.RawMessage{}
	if b, err := os.ReadFile(dataPath); err == nil {
		_ = json.Unmarshal(b, &root)
	}
	gb, _ := json.MarshalIndent(block, "  ", "  ")
	root[lang] = gb

	out, _ := json.MarshalIndent(root, "", "  ")
	_ = os.WriteFile(dataPath, out, 0o644)
}

func main() {
	// Anchor to the source dir (cpu/go) at compile time, like Rust's
	// CARGO_MANIFEST_DIR, so plot.data resolves regardless of cwd.
	_, thisFile, _, _ := runtime.Caller(0)
	baseDir := filepath.Dir(thisFile)
	dataPath := filepath.Join(baseDir, "..", "plot.data")

	variants := []namedStats{
		{"serial", measure("serial", computeSerial)},
		{"goroutine-per-task", measure("goroutine-per-task", computeWithGoroutines)},
		{"worker pool", measure("worker pool", computeWithWorkerPool)},
		{"errgroup (limit=NumCPU)", measure("errgroup (limit=NumCPU)", computeWithErrgroup)},
		{"chunked (NumCPU)", measure("chunked (NumCPU)", computeWithChunked)},
	}

	save(dataPath, "Go", "#00ADD8", variants)

	fmt.Printf("sink=%d\n", atomic.LoadUint64(&sink)) // defeat DCE; do not remove
}
