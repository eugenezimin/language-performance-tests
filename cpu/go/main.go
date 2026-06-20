package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

const (
	nTerms   = 500_000_000 // tune this
	nTasks   = 3
	poolSize = 2 // goroutines reused in the fixed-pool variant
)

type kv struct {
	name string
	ms   float64
}

func computePi(nTerms int) float64 {
	pi := 0.0
	sign := 1.0
	for k := 0; k < nTerms; k++ {
		pi += sign / float64(2*k+1)
		sign = -sign
	}
	return 4 * pi
}

func worker() {
	fmt.Printf("  \u03C0[%d] = %.10f\n", nTerms, computePi(nTerms))
}

// 1. serial baseline (one goroutine, runs tasks back to back)
func computeSerial() {
	for i := 0; i < nTasks; i++ {
		worker()
	}
}

//  2. goroutines + WaitGroup -> one goroutine per task, REAL parallelism
//     (analog of thread::spawn / threading.Thread / Thread.start;
//     unlike Python threads there is no GIL, so CPU work truly overlaps)
func computeWithGoroutines() {
	var wg sync.WaitGroup
	for i := 0; i < nTasks; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			worker()
		}()
	}
	wg.Wait()
}

//  3. worker pool sized to NumCPU -> channel-fed, fixed goroutines reused
//     (analog of threadpool / ThreadPoolExecutor / ProcessPoolExecutor;
//     the Go scheduler work-steals across these like rayon does)
func computeWithWorkerPool() {
	tasks := make(chan int, nTasks)
	var wg sync.WaitGroup
	size := runtime.NumCPU()
	if size > nTasks {
		size = nTasks
	}
	for i := 0; i < size; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range tasks {
				worker()
			}
		}()
	}
	for i := 0; i < nTasks; i++ {
		tasks <- nTerms
	}
	close(tasks)
	wg.Wait()
}

//  4. fan-out / fan-in -> goroutines push results back over a channel
//     (analog of async-gather / CompletableFuture / futures-cpupool:
//     dispatch work, then collect the computed values)
func computeWithFanIn() {
	results := make(chan float64, nTasks)
	for i := 0; i < nTasks; i++ {
		go func() { results <- computePi(nTerms) }()
	}
	for i := 0; i < nTasks; i++ {
		fmt.Printf("  \u03C0[%d] = %.10f\n", nTerms, <-results)
	}
}

//  5. fixed worker pool (size=2) -> reused goroutines, demonstrates queuing
//     (analog of the JS / Java worker pool: 3 tasks through 2 workers,
//     so wall time visibly reflects the extra queued task)
func computeWithFixedPool() {
	tasks := make(chan int, nTasks)
	var wg sync.WaitGroup
	size := poolSize
	if size > nTasks {
		size = nTasks
	}
	for i := 0; i < size; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range tasks {
				worker()
			}
		}()
	}
	for i := 0; i < nTasks; i++ {
		tasks <- nTerms
	}
	close(tasks)
	wg.Wait()
}

func round1(f float64) float64 { return math.Round(f*10) / 10 }

func savePlotData(lang, color string, serialMs float64, variants []kv) {
	vmap := make(map[string]float64)
	for _, v := range variants {
		vmap[v.name] = round1(v.ms)
	}
	entry := map[string]interface{}{
		"color":     color,
		"n_terms":   nTerms,
		"n_tasks":   nTasks,
		"serial_ms": round1(serialMs),
		"variants":  vmap,
	}

	root := map[string]interface{}{}
	if data, err := os.ReadFile("../plot.data"); err == nil {
		json.Unmarshal(data, &root)
	}
	root[lang] = entry

	out, _ := json.MarshalIndent(root, "", "  ")
	os.WriteFile("../plot.data", out, 0644)
}

func bench(name string, fn func()) float64 {
	start := time.Now()
	fmt.Printf("[%s]\n", name)
	fn()
	elapsed := time.Since(start)
	fmt.Printf("  Computation time: %v\n-------\n\n", elapsed)
	return float64(elapsed.Nanoseconds()) / 1e6
}

func main() {
	serialMs := bench("serial", computeSerial)
	variants := []kv{
		{"serial", serialMs},
		{"goroutines", bench("goroutines", computeWithGoroutines)},
		{"worker pool", bench("worker pool", computeWithWorkerPool)},
		{"fan-in", bench("fan-in", computeWithFanIn)},
		{"fixed pool (size=2)", bench("fixed pool (size=2)", computeWithFixedPool)},
	}
	savePlotData("Go", "#00ADD8", serialMs, variants)
}
