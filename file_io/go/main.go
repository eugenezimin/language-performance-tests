package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// DCE sink: fold checksums into a package-level var so the compiler can't
// prove the build dead. Go's DCE is weak across calls, but this is free.
var sink uint64

// ───────────────────────── the three variants ────────────────────────────
// Contract mirrors Rust: return []string of all lines. NOTE the ownership
// difference flagged on variant 1.

//  1. ReadFile + Split — one stat-sized read, one 45 MB string() copy, then
//     strings.Split. ZERO-COPY per line: the returned elements are sub-slices
//     that share the backing array of `s`, so this pins the whole file in
//     memory and is NOT semantically equal to Rust's owned-per-line contract.
//     It is the genuinely fastest idiomatic Go path; that divergence is the result.
func readFileSplit(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	s := string(data) // 45 MB copy; the ReadFile []byte becomes transient garbage
	lines := strings.Split(s, "\n")
	// Split yields a trailing "" when the file ends in '\n'; Rust's .lines()
	// does not. Drop it so line counts match across all three variants.
	// (Assumes LF endings — the generated fixture is \n-only. CRLF would leave
	// a trailing '\r' that Rust's .lines() strips; would change byte sums, not counts.)
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	return lines
}

//  2. bufio.Scanner — the idiom. Default ScanLines strips trailing '\r' and
//     yields no trailing empty (matches Rust .lines()). Text() COPIES each line
//     into its own string, so this DOES honor Rust's owned contract.
func bufioScanner(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f) // 64 KiB max token — fine for short lines
	var lines []string
	for sc.Scan() {
		lines = append(lines, sc.Text()) // copy -> owned line
	}
	if err := sc.Err(); err != nil {
		panic(err)
	}
	return lines
}

//  3. manual chunk — you own the 64 KiB reads, the newline scan, and the
//     cross-chunk carry. string(buf[a:b]) copies (owned). Unlike Rust's
//     from_utf8_lossy this does no UTF-8 validation/replacement; for the
//     ASCII-ish fixture that's moot.
func manualChunk(path string) []string {
	const chunk = 64 * 1024
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	buf := make([]byte, chunk)
	var carry []byte
	var lines []string
	for {
		n, err := f.Read(buf)
		if n > 0 {
			start := 0
			for i := 0; i < n; i++ {
				if buf[i] == '\n' {
					if len(carry) == 0 {
						lines = append(lines, string(buf[start:i]))
					} else {
						carry = append(carry, buf[start:i]...)
						lines = append(lines, string(carry))
						carry = carry[:0]
					}
					start = i + 1
				}
			}
			carry = append(carry, buf[start:n]...) // partial line spans chunks
		}
		if err != nil { // includes io.EOF; n>0 already processed above
			break
		}
	}
	if len(carry) > 0 {
		lines = append(lines, string(carry))
	}
	return lines
}

//  4. mmap (syscall) — map the file into the address space, scan bytes in
//     place, string(seg[a:b]) copies each line owned (matches manualChunk).
//     The mapping is OS memory, NOT on the Go heap, so ReadMemStats HeapAlloc
//     (heap_peak/heap_live) reflects only the line slice + strings — no 45 MB
//     heap buffer; the faulted pages surface in rss_peak instead. Expect time
//     ≈ manual chunk (copy dominates), heap_peak the LOWEST of the four.
func mmapScan(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		panic(err)
	}
	size := int(fi.Size())
	if size == 0 {
		return []string{}
	}
	seg, err := syscall.Mmap(int(f.Fd()), 0, size,
		syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		panic(err)
	}
	defer syscall.Munmap(seg)

	var lines []string
	start := 0
	for i := 0; i < size; i++ {
		if seg[i] == '\n' {
			lines = append(lines, string(seg[start:i]))
			start = i + 1
		}
	}
	// No trailing empty for a '\n'-terminated file (start == size here),
	// matching bufio.Scanner and manualChunk.
	if start < size {
		lines = append(lines, string(seg[start:]))
	}
	return lines
}

// ───────────────────────── per-variant peak RSS (Linux) ──────────────────
func resetRSSPeak() {
	if runtime.GOOS == "linux" {
		// writing "5" to clear_refs resets VmHWM to current VmRSS, so each
		// variant gets its own peak instead of a process-wide high-water.
		_ = os.WriteFile("/proc/self/clear_refs", []byte("5"), 0o644)
	}
}

func rssPeakKB() uint64 {
	if runtime.GOOS != "linux" {
		return 0
	}
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				v, _ := strconv.ParseUint(f[1], 10, 64)
				return v
			}
		}
	}
	return 0
}

// ───────────────────────── harness ───────────────────────────────────────
type Stats struct {
	p50, p99, p999                    float64
	heapPeakKB, heapLiveKB, rssPeakKB uint64
	lines                             int
}

func checksum(lines []string) (int, int) {
	total := 0
	for _, l := range lines {
		total += len(l)
	}
	return len(lines), total
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

func satSub(a, b uint64) uint64 {
	if a < b {
		return 0
	}
	return a - b
}

func measure(name string, samples int, build func() []string) Stats {
	// warm-up: page cache + allocator size-classes
	{
		v := build()
		c, b := checksum(v)
		sink += uint64(c) + uint64(b)
		runtime.KeepAlive(v)
	}

	// timing pass — GC left at default (GOGC=100); p99/p999 honestly carry
	// the collector tail, which is a real part of Go's runtime cost here.
	times := make([]float64, 0, samples)
	lines := 0
	for i := 0; i < samples; i++ {
		t := time.Now()
		v := build()
		dt := float64(time.Since(t).Nanoseconds()) / 1e6
		c, b := checksum(v)
		sink += uint64(c) + uint64(b)
		lines = c
		times = append(times, dt)
		runtime.KeepAlive(v)
		v = nil
	}
	sort.Float64s(times)

	// memory pass — single build. Go has no GlobalAlloc counter, so we use
	// ReadMemStats. GC is frozen during the build so nothing is reclaimed:
	//   heap_peak = HeapAlloc right after build (true high-water of this build)
	//   heap_live = HeapAlloc after a forced GC with the result still live
	// For variant 1 the gap exposes the transient ReadFile []byte vs the
	// pinned string copy; for the copying variants the two should be close.
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)
	base := m0.HeapAlloc
	resetRSSPeak()
	prev := debug.SetGCPercent(-1)

	v := build()

	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	heapPeak := satSub(m1.HeapAlloc, base)

	runtime.GC() // forced collection still runs while auto-GC is off
	var m2 runtime.MemStats
	runtime.ReadMemStats(&m2)
	heapLive := satSub(m2.HeapAlloc, base)

	rss := rssPeakKB()
	c, b := checksum(v)
	sink += uint64(c) + uint64(b)
	runtime.KeepAlive(v)
	debug.SetGCPercent(prev)

	s := Stats{
		p50:        round3(pct(times, 0.50)),
		p99:        round3(pct(times, 0.99)),
		p999:       round3(pct(times, 0.999)),
		heapPeakKB: heapPeak / 1024,
		heapLiveKB: heapLive / 1024,
		rssPeakKB:  rss,
		lines:      lines,
	}
	fmt.Printf("[%s]  p50=%.3fms p99=%.3fms p999=%.3fms  "+
		"heap_peak=%dKB heap_live=%dKB rss_peak=%dKB  (%d lines)\n",
		name, s.p50, s.p99, s.p999, s.heapPeakKB, s.heapLiveKB, s.rssPeakKB, s.lines)
	return s
}

// ───────────────────────── data file ─────────────────────────────────────
type namedStats struct {
	name string
	s    Stats
}

func save(dataPath, lang, color string, fileBytes uint64, nLines, samples int, variants []namedStats) {
	vmap := map[string]interface{}{}
	for _, ns := range variants {
		s := ns.s
		vmap[ns.name] = map[string]interface{}{
			"ms_p50": s.p50, "ms_p99": s.p99, "ms_p999": s.p999,
			"heap_peak_kb": s.heapPeakKB, "heap_live_kb": s.heapLiveKB,
			"rss_peak_kb": s.rssPeakKB, "lines": s.lines,
		}
	}
	goBlock := map[string]interface{}{
		"color": color, "file_bytes": fileBytes,
		"n_lines": nLines, "samples": samples, "variants": vmap,
	}

	// Read the existing file as RawMessage so other languages' blocks (e.g.
	// Rust's preserve_order layout) are kept byte-for-byte and not re-sorted.
	root := map[string]json.RawMessage{}
	if b, err := os.ReadFile(dataPath); err == nil {
		_ = json.Unmarshal(b, &root)
	}
	gb, _ := json.MarshalIndent(goBlock, "  ", "  ")
	root[lang] = gb

	out, _ := json.MarshalIndent(root, "", "  ")
	_ = os.WriteFile(dataPath, out, 0o644)
}

func main() {
	// Anchor to the source dir (file_io/go) at compile time, like Rust's
	// CARGO_MANIFEST_DIR, so the fixture/data resolve regardless of cwd.
	_, thisFile, _, _ := runtime.Caller(0)
	baseDir := filepath.Dir(thisFile)
	path := filepath.Join(baseDir, "..", "text-examples.txt")
	dataPath := filepath.Join(baseDir, "..", "file_io.data")

	fi, err := os.Stat(path)
	if err != nil {
		panic("stat fixture — expected file_io/text-examples.txt: " + err.Error())
	}
	fileBytes := uint64(fi.Size())

	const samples = 100
	s1 := measure("ReadFile + Split", samples, func() []string { return readFileSplit(path) })
	s2 := measure("bufio.Scanner", samples, func() []string { return bufioScanner(path) })
	s3 := measure("manual chunk", samples, func() []string { return manualChunk(path) })
	s4 := measure("mmap (syscall)", samples, func() []string { return mmapScan(path) })

	save(dataPath, "Go", "#00ADD8", fileBytes, s1.lines, samples,
		[]namedStats{
			{"ReadFile + Split", s1},
			{"bufio.Scanner", s2},
			{"manual chunk", s3},
			{"mmap (syscall)", s4},
		})
}
