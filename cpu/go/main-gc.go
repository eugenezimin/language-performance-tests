//go:build gc

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

// DCE sink — same role as the dispatch build: fold results into a global atomic
// so the compiler can't prove the work dead and stdout stays out of timing.
var sink uint64

// ───────────────────────── allocating payload ────────────────────────────
// THE DIFFERENCE FROM main.go: this worker builds a []float64 of every partial
// sum (length nTerms ≈ 80KB) and RETURNS it. The caller retains it in a
// pre-sized [][]float64 indexed by task id, so the slice ESCAPES to the heap
// (escape analysis can't stack-allocate it — retention is what guarantees the
// allocation is real) and stays live for the whole fan-out. 100 tasks => ~8MB
// live simultaneously, which crosses GOGC=100 several times per sample. The
// GC delta is now the headline signal, not a near-zero diagnostic.
func computePiAlloc(n int) []float64 {
	partials := make([]float64, n) // escapes via the returned slice -> heap
	pi := 0.0
	sign := 1.0
	k := 1.0
	for i := 0; i < n; i++ {
		pi += sign / k
		sign = -sign
		k += 2.0
		partials[i] = pi
	}
	atomic.AddUint64(&sink, math.Float64bits(4*pi))
	return partials
}

// ───────────────────────── the five variants ─────────────────────────────
// Each variant writes its task's result into out[idx]. Writing a UNIQUE index
// per task makes the shared backing array lock-free (no two goroutines touch
// the same slot). The task index is passed INTO the closure as a parameter —
// NOT closed over the loop variable — to avoid the classic capture bug.

//  1. serial baseline — allocates the same ~8MB live set, no dispatch. Isolates
//     "what does the allocation+GC cost on its own," so the parallel variants'
//     GC deltas are read against an allocating, not an empty, reference.
func computeSerial(out [][]float64) {
	for i := 0; i < nTasks; i++ {
		out[i] = computePiAlloc(nTerms)
	}
}

//  2. goroutine-per-task — 100 goroutines, each retaining ~80KB. This variant
//     holds the LARGEST simultaneous live set (all 100 results in flight at
//     once), so it should show the most GC cycles / longest pause tail. The
//     contrast against chunked is the experiment.
func computeWithGoroutines(out [][]float64) {
	var wg sync.WaitGroup
	wg.Add(nTasks)
	for i := 0; i < nTasks; i++ {
		go func(idx int) {
			defer wg.Done()
			out[idx] = computePiAlloc(nTerms)
		}(i)
	}
	wg.Wait()
}

//  3. worker pool sized to GOMAXPROCS — at most NumCPU results allocating
//     concurrently, but ALL 100 are retained in `out`, so peak live set is the
//     same ~8MB; the difference vs #2 is allocation CONCURRENCY, not live size.
func computeWithWorkerPool(out [][]float64) {
	size := runtime.NumCPU()
	if size > nTasks {
		size = nTasks
	}
	type job struct{ idx int }
	tasks := make(chan job, nTasks)
	var wg sync.WaitGroup
	wg.Add(size)
	for w := 0; w < size; w++ {
		go func() {
			defer wg.Done()
			for j := range tasks {
				out[j.idx] = computePiAlloc(nTerms)
			}
		}()
	}
	for i := 0; i < nTasks; i++ {
		tasks <- job{i}
	}
	close(tasks)
	wg.Wait()
}

//  4. errgroup with SetLimit(NumCPU) — modern bounded fan-out; same retention,
//     same ~8MB peak, NumCPU concurrent allocators. Closure captures idx by
//     parameter through the loop (errgroup.Go takes no arg, so we shadow i).
func computeWithErrgroup(out [][]float64) {
	var g errgroup.Group
	g.SetLimit(runtime.NumCPU())
	for i := 0; i < nTasks; i++ {
		idx := i // shadow: capture a fresh var, not the loop variable
		g.Go(func() error {
			out[idx] = computePiAlloc(nTerms)
			return nil
		})
	}
	_ = g.Wait()
}

//  5. chunked over GOMAXPROCS goroutines — one goroutine per core handles a
//     contiguous range. Fewest concurrent allocators, least scheduler traffic;
//     still retains all 100 results so peak live set matches the others. If its
//     GC delta is LOWER than #2 despite the same live set, that's allocation
//     RATE (not live size) driving GC — the subtle finding.
func computeWithChunked(out [][]float64) {
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
			count++
		}
		go func(lo, n int) {
			defer wg.Done()
			for i := 0; i < n; i++ {
				out[lo+i] = computePiAlloc(nTerms)
			}
		}(start, count)
		start += count
	}
	wg.Wait()
}

// ───────────────────────── harness ───────────────────────────────────────
type Stats struct {
	p50, p99, p999 float64
	numGC          uint32
	pauseTotalNs   uint64
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

// build() takes the retention slice so each sample reuses one backing array —
// we DON'T allocate a fresh out[][] per sample (that would add 100*8B*1000 of
// harness garbage and muddy the payload's own GC signal). The inner []float64
// results are overwritten each sample and become garbage — that's the intended
// pressure. out itself is allocated ONCE in measure().
func measure(name string, build func(out [][]float64)) Stats {
	out := make([][]float64, nTasks)
	build(out) // warm-up

	var msBefore runtime.MemStats
	runtime.ReadMemStats(&msBefore)

	times := make([]float64, 0, samples)
	for i := 0; i < samples; i++ {
		t := time.Now()
		build(out)
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
		name, s.p50, s.p99, s.p999, s.numGC, float64(s.pauseTotalNs)/1e6)
	return s
}

// ───────────────────────── data file ─────────────────────────────────────
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
			"lines":     0,
			"gc_cycles": ns.s.numGC, "gc_pause_ms": round3(float64(ns.s.pauseTotalNs) / 1e6),
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

	// Distinct key so the allocation scenario coexists with the dispatch block
	// in the same plot.data — the A-vs-B contrast is visible side by side.
	save(dataPath, "Go (alloc)", "#5DC9E2", variants)

	fmt.Printf("sink=%d\n", atomic.LoadUint64(&sink))
}
