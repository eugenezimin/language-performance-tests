package com.test.fileio;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.lang.ref.Reference;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

public final class Main {

    // DCE sink: C2 is more aggressive than Go's cross-call DCE, so we fold every
    // checksum into a volatile static and print it at the end. Combined with the
    // List escaping into measure(), the decode loop can't be proven dead.
    static volatile long sink = 0;

    // ───────────────────────── the four variants ─────────────────────────────
    // Contract mirrors Rust: return List<String> of all lines, each OWNED.
    // NOTE on the zero-copy asymmetry: Java's String has copied on substring
    // since 7u6, so there is NO Go-style zero-copy slurp here — every variant,
    // including readString+lines(), produces freshly-allocated owned Strings.
    // Java's slurp therefore behaves like Rust's owned slurp_split, NOT like
    // Go's pinned-backing-array ReadFile+Split. That divergence is the result.

    //  1. Files.readString + String.lines() — slurp the whole file into one
    //     ~45 MB String (full UTF-8 decode up front), then owned substrings.
    //     Peak holds the whole-file String AND the line List simultaneously.
    static List<String> readStringLines(Path p) {
        try {
            String s = Files.readString(p, StandardCharsets.UTF_8);
            // String.lines() splits on line terminators (\n, \r, \r\n) and emits
            // NO trailing empty — matches Rust .lines(). Each element is a copy.
            List<String> lines = new ArrayList<>();
            s.lines().forEach(lines::add);
            return lines;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    //  2. Files.readAllLines — the one-call idiom. BufferedReader over an
    //     InputStreamReader(UTF-8), readLine loop, ArrayList<String>. Owned.
    static List<String> readAllLines(Path p) {
        try {
            return Files.readAllLines(p, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    //  3. manual chunk — you own the 64 KiB reads, the newline scan, and the
    //     cross-chunk carry. new String(buf, off, len, UTF_8) copies (owned).
    //     Splits on '\n' only (does not strip a trailing '\r'); the LF-only
    //     fixture makes that moot, matching Rust/Go manual_chunk.
    static List<String> manualChunk(Path p) {
        final int CHUNK = 64 * 1024;
        List<String> lines = new ArrayList<>();
        byte[] buf = new byte[CHUNK];
        ByteArrayOutputStream carry = new ByteArrayOutputStream(); // reset() keeps capacity
        try (InputStream in = Files.newInputStream(p)) {
            int n;
            while ((n = in.read(buf)) != -1) {
                int start = 0;
                for (int i = 0; i < n; i++) {
                    if (buf[i] == '\n') {
                        if (carry.size() == 0) {
                            lines.add(new String(buf, start, i - start, StandardCharsets.UTF_8));
                        } else {
                            carry.write(buf, start, i - start);
                            lines.add(new String(carry.toByteArray(), StandardCharsets.UTF_8));
                            carry.reset();
                        }
                        start = i + 1;
                    }
                }
                carry.write(buf, start, n - start); // partial line spans chunks
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (carry.size() > 0) {
            lines.add(new String(carry.toByteArray(), StandardCharsets.UTF_8));
        }
        return lines;
    }

    //  4. mmap (FFM MemorySegment) — map the file into the address space via a
    //     confined Arena (deterministic unmap on close, unlike MappedByteBuffer's
    //     Cleaner-at-GC), scan bytes in place, decode each line owned. The point:
    //     mmap removes the read() syscalls and the file->heap byte[] copy, but we
    //     STILL allocate a transient byte[] + a String per line. So this mostly
    //     proves the COPY, not the read, is the cost — expect it near manual chunk,
    //     not dramatically faster.
    static List<String> mmapScan(Path p) {
        List<String> lines = new ArrayList<>();
        try (Arena arena = Arena.ofConfined();
             FileChannel ch = FileChannel.open(p, StandardOpenOption.READ)) {
            long size = ch.size();
            MemorySegment seg = ch.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
            long start = 0;
            for (long i = 0; i < size; i++) {
                if (seg.get(ValueLayout.JAVA_BYTE, i) == (byte) '\n') {
                    int len = (int) (i - start);
                    byte[] tmp = new byte[len];
                    MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, start, tmp, 0, len);
                    lines.add(new String(tmp, StandardCharsets.UTF_8));
                    start = i + 1;
                }
            }
            if (start < size) {
                int len = (int) (size - start);
                byte[] tmp = new byte[len];
                MemorySegment.copy(seg, ValueLayout.JAVA_BYTE, start, tmp, 0, len);
                lines.add(new String(tmp, StandardCharsets.UTF_8));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return lines;
    }

    // ───────────────────────── memory instrumentation ────────────────────────
    // The JVM has no GlobalAlloc hook (Rust) and no ReadMemStats freeze (Go's
    // SetGCPercent(-1)). The closest analogs:
    //   heap_peak -> sum of heap MemoryPool peak usages after resetPeakUsage().
    //                CAVEAT: pool peaks can occur at different instants, so the
    //                sum slightly OVER-estimates a single-instant high-water if a
    //                young GC fired mid-build. It is the conventional JVM measure.
    //   heap_live -> forced System.gc() with the result held live, then heap used.
    // We also print thread-allocated-bytes ("churn") — a bulletproof, GC-free
    // cumulative-allocation counter (TLAB-based) — as a diagnostic only.
    static final MemoryMXBean MEM = ManagementFactory.getMemoryMXBean();
    static final List<MemoryPoolMXBean> HEAP_POOLS =
            ManagementFactory.getMemoryPoolMXBeans().stream()
                    .filter(b -> b.getType() == MemoryType.HEAP)
                    .toList();
    static final com.sun.management.ThreadMXBean TMX =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    static long heapUsedBytes() {
        return MEM.getHeapMemoryUsage().getUsed();
    }

    static long sumHeapPoolPeak() {
        long total = 0;
        for (MemoryPoolMXBean pool : HEAP_POOLS) total += pool.getPeakUsage().getUsed();
        return total;
    }

    static long threadAllocatedBytes() {
        return TMX.getThreadAllocatedBytes(Thread.currentThread().threadId());
    }

    // ───────────────────────── per-variant peak RSS (Linux) ──────────────────
    static boolean isLinux() {
        return System.getProperty("os.name", "").toLowerCase().contains("linux");
    }

    static void resetRssPeak() {
        if (!isLinux()) return;
        // writing "5" to clear_refs resets VmHWM to current VmRSS, so each
        // variant gets its own peak rather than a process-wide high-water.
        // (On the JVM, RSS is dominated by committed heap + metaspace + code
        // cache, so this column reflects JVM overhead far more than line data.)
        try {
            Files.write(Paths.get("/proc/self/clear_refs"),
                    "5".getBytes(StandardCharsets.US_ASCII), StandardOpenOption.WRITE);
        } catch (IOException ignored) {
        }
    }

    static long rssPeakKB() {
        if (!isLinux()) return 0;
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    String[] f = line.trim().split("\\s+");
                    if (f.length >= 2) return Long.parseLong(f[1]);
                }
            }
        } catch (Exception ignored) {
        }
        return 0;
    }

    // ───────────────────────── harness ───────────────────────────────────────
    static final class Stats {
        double p50, p99, p999;
        long heapPeakKB, heapLiveKB, rssPeakKB;
        int lines;
    }

    record NamedStats(String name, Stats s) {}

    // Checksum uses char count. For the ASCII-ish fixture char == byte, so this
    // matches Rust's l.len() / Go's len(l) byte sums; on multibyte input it would
    // diverge. We avoid getBytes() here so the sink itself allocates nothing.
    static long[] checksum(List<String> lines) {
        long total = 0;
        for (String l : lines) total += l.length();
        return new long[]{lines.size(), total};
    }

    static double pct(double[] sorted, double p) {
        if (sorted.length == 0) return 0;
        double rank = p * (sorted.length - 1);
        int lo = (int) Math.floor(rank), hi = (int) Math.ceil(rank);
        if (lo == hi) return sorted[lo];
        double f = rank - lo;
        return sorted[lo] * (1.0 - f) + sorted[hi] * f;
    }

    static double round3(double x) {
        return Math.round(x * 1000.0) / 1000.0;
    }

    static long satSub(long a, long b) {
        return a < b ? 0 : a - b;
    }

    static Stats measure(String name, int samples, Supplier<List<String>> build) {
        // warm-up: page cache + size-classes + a first C1/C2 pass (1 build,
        // matching the Rust/Go harness exactly — we do NOT add JIT warm-up
        // rounds to game steady state).
        {
            List<String> v = build.get();
            long[] cs = checksum(v);
            sink += cs[0] + cs[1];
        }

        // timing pass — GC at default (pinned G1); p99/p999 honestly carry the
        // collector tail. p999 over 100 samples is effectively the worst sample,
        // which on the JVM is a young/mixed pause, not a decode.
        double[] times = new double[samples];
        int lines = 0;
        for (int i = 0; i < samples; i++) {
            long t = System.nanoTime();
            List<String> v = build.get();
            double dt = (System.nanoTime() - t) / 1e6;
            long[] cs = checksum(v);
            sink += cs[0] + cs[1];
            lines = (int) cs[0];
            times[i] = dt;
        }
        double coldMs = round3(times[0]); // first timed sample — the cold-ish cost
        Arrays.sort(times);

        // memory pass — single representative build.
        System.gc();
        long base = heapUsedBytes();
        resetRssPeak();
        for (MemoryPoolMXBean pool : HEAP_POOLS) pool.resetPeakUsage();
        long allocBefore = threadAllocatedBytes();

        List<String> v = build.get();

        long heapPeak = satSub(sumHeapPoolPeak(), base);
        long allocDuring = satSub(threadAllocatedBytes(), allocBefore);

        System.gc(); // result must stay live across this collection
        long heapLive = satSub(heapUsedBytes(), base);

        long rss = rssPeakKB();
        long[] cs = checksum(v);
        sink += cs[0] + cs[1];
        Reference.reachabilityFence(v); // keep v live until after the gc + read

        Stats s = new Stats();
        s.p50 = round3(pct(times, 0.50));
        s.p99 = round3(pct(times, 0.99));
        s.p999 = round3(pct(times, 0.999));
        s.heapPeakKB = heapPeak / 1024;
        s.heapLiveKB = heapLive / 1024;
        s.rssPeakKB = rss;
        s.lines = lines;

        System.out.printf(
                "[%s]  p50=%.3fms p99=%.3fms p999=%.3fms  cold=%.3fms  "
                        + "heap_peak=%dKB heap_live=%dKB rss_peak=%dKB  churn=%dKB  (%d lines)%n",
                name, s.p50, s.p99, s.p999, coldMs,
                s.heapPeakKB, s.heapLiveKB, s.rssPeakKB, allocDuring / 1024, s.lines);
        return s;
    }

    // ───────────────────────── data file (Jackson merge) ─────────────────────
    // Read the existing tree, replace only the "Java" key, write back. ObjectNode
    // is LinkedHashMap-backed, so the other languages' blocks keep their order
    // and content (the analog of Go's RawMessage / Rust's preserve_order).
    static void save(Path dataPath, String lang, String color, long fileBytes,
                     int nLines, int samples, List<NamedStats> variants) {
        ObjectMapper om = new ObjectMapper();
        ObjectNode root;
        try {
            if (Files.exists(dataPath)) {
                JsonNode existing = om.readTree(dataPath.toFile());
                root = existing.isObject() ? (ObjectNode) existing : om.createObjectNode();
            } else {
                root = om.createObjectNode();
            }
        } catch (IOException e) {
            root = om.createObjectNode();
        }

        ObjectNode vmap = om.createObjectNode();
        for (NamedStats ns : variants) {
            Stats s = ns.s();
            ObjectNode vn = om.createObjectNode();
            vn.put("ms_p50", s.p50);
            vn.put("ms_p99", s.p99);
            vn.put("ms_p999", s.p999);
            vn.put("heap_peak_kb", s.heapPeakKB);
            vn.put("heap_live_kb", s.heapLiveKB);
            vn.put("rss_peak_kb", s.rssPeakKB);
            vn.put("lines", s.lines);
            vmap.set(ns.name(), vn);
        }

        ObjectNode block = om.createObjectNode();
        block.put("color", color);
        block.put("file_bytes", fileBytes);
        block.put("n_lines", nLines);
        block.put("samples", samples);
        block.set("variants", vmap);

        root.set(lang, block);

        try {
            om.writerWithDefaultPrettyPrinter().writeValue(dataPath.toFile(), root);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static void main(String[] args) {
        if (TMX.isThreadAllocatedMemorySupported()) {
            TMX.setThreadAllocatedMemoryEnabled(true);
        }

        // Resolve fixture/data relative to file_io/ (one up from this project),
        // via the bench.base system property Gradle sets to projectDir.
        String base = System.getProperty("bench.base");
        Path projectDir = (base != null) ? Paths.get(base) : Paths.get("").toAbsolutePath();
        Path fileIoDir = projectDir.getParent();
        Path path = fileIoDir.resolve("text-examples.txt");
        Path data = fileIoDir.resolve("file_io.data");

        long fileBytes;
        try {
            fileBytes = Files.size(path);
        } catch (IOException e) {
            throw new UncheckedIOException(
                    "stat fixture — expected file_io/text-examples.txt", e);
        }

        final int SAMPLES = 100;
        Stats s1 = measure("Files.readString + lines()", SAMPLES, () -> readStringLines(path));
        Stats s2 = measure("Files.readAllLines",          SAMPLES, () -> readAllLines(path));
        Stats s3 = measure("manual chunk",                SAMPLES, () -> manualChunk(path));
        Stats s4 = measure("mmap (MemorySegment)",        SAMPLES, () -> mmapScan(path));

        int nLines = s1.lines;
        save(data, "Java", "#E76F00", fileBytes, nLines, SAMPLES, List.of(
                new NamedStats("Files.readString + lines()", s1),
                new NamedStats("Files.readAllLines", s2),
                new NamedStats("manual chunk", s3),
                new NamedStats("mmap (MemorySegment)", s4)
        ));

        System.out.println("sink=" + sink); // defeat DCE; do not remove
    }

    private Main() {}
}