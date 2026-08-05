// cta_dep_bench.cu — the workhorse microbenchmark for CTA-level dependency evaluation.
//
// One producer kernel -> one consumer kernel, with a PARAMETERIZED CTA-level dependency
// between them. Sweeping its knobs produces:
//   Tier 1.1  benefit map            : --sweep degree x grid, structure held fixed
//   Tier 1.2  tail/prologue map      : --tail / --prologue sweep
//   Tier 2.1  sync protocol shootout : --wait {cta-spin,cta-backoff,cta-counter,cta-exact}
//   Tier 2.3  encoding cost          : --wait cta-exact vs cta-spin (interval over-approx cost)
//   Tier 0.3  occupancy cost curve   : --smem-kb sweep
//
// FOUR-POINT BRACKET (see ../EXPERIMENT_PLAN.md §1). Each run reports:
//   Floor    = --wait grid          (griddepcontrol.wait, whole-grid all-or-nothing)
//   Impl     = --wait cta-*         (per-CTA software protocol)
//   Ceiling  = --wait none          (dependency removed; RESULTS ARE WRONG, timing only)
// Ideal is computed offline by injecting the overheads measured by overhead_probe.
//
// CRITICAL EXPERIMENT-DESIGN NOTE
// -------------------------------
// --structure and --degree are INDEPENDENT axes on purpose. BlockMaestro Fig.12 grew both
// together (n-group fully connected), so its "degree > 32 => no benefit" threshold cannot
// tell "too many edges" from "too complex a shape". LLM FFN GEMM chains and DSA
// indexer->topk both have HIGH degree but CONTIGUOUS structure, and would be wrongly
// excluded by that threshold. Always sweep one axis with the other pinned.
//
// Build: ./build.sh      Run: ./cta_dep_bench --help

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"
#include "common/dep_pattern.cuh"
#include "common/dep_wait.cuh"

// ---------------------------------------------------------------- kernels

// Producer: writes one value per CTA, then publishes completion, then burns `tail` cycles.
// The trigger sits BEFORE the tail so the consumer may launch during it (that is the whole
// point of PDL). Completion publication happens BEFORE the tail as well, because the data
// the consumer needs is already written -- the tail models independent trailing work.
__global__ void producerK(float* __restrict__ out,
                          int* __restrict__ done,
                          unsigned long long* __restrict__ counter,
                          int n_producer,
                          unsigned long long tail,
                          CtaTrace trace,
                          int use_pdl) {
    CtaTraceLocal tr = ctatrace_begin(trace);
    const int cta = blockIdx.x;

    if (cta < n_producer) {
        // Body: the data the consumer will read. One element per CTA keeps the dependency
        // structure (not memory bandwidth) the thing under test.
        if (threadIdx.x == 0) out[cta] = (float)cta * 2.0f + 1.0f;
    }

    dep_publish(done, counter, cta);          // release store -> visible to consumers

    if (use_pdl) cudaTriggerProgrammaticLaunchCompletion();

    spin_cycles(tail);                        // independent trailing work
    ctatrace_end(tr);
}

// Consumer: burns `prologue` cycles of independent work, waits on its dependency, then
// reads what its parents produced.
__global__ void consumerK(float* __restrict__ out,
                          const float* __restrict__ pin,
                          const int* __restrict__ done,
                          const unsigned long long* __restrict__ counter,
                          DepPattern pat,
                          int wait_mode,
                          unsigned long long prologue,
                          CtaTrace trace) {
    CtaTraceLocal tr = ctatrace_begin(trace);
    const int cta = blockIdx.x;

    spin_cycles(prologue);                    // overlappable independent prologue

    dep_wait(wait_mode, done, counter, pat, cta);
    ctatrace_mark_dep(tr);

    // Epilogue: read every parent's output. This is what makes a missed dependency show up
    // as a wrong answer rather than silently passing.
    if (cta < pat.n_consumer && threadIdx.x == 0) {
        float acc = 0.0f;
        int d = dep_degree(pat, cta);
        for (int k = 0; k < d; ++k) {
            int p = dep_parent(pat, cta, k);
            if (p >= 0) acc += pin[p];
        }
        out[cta] = acc;
    }
    ctatrace_end(tr);
}

// ---------------------------------------------------------------- host

struct Cfg {
    int  n_producer = 1024, n_consumer = 1024;
    int  structure  = DEP_INTERVAL;
    int  degree     = 1;
    int  threads    = 128;
    int  smem_kb    = 0;
    unsigned long long tail = 200000, prologue = 200000;
    int  repeats    = 30;
    int  wait_mode  = WAIT_SPIN;
    int  use_pdl    = 1;
    int  trace      = 0;
    const char* trace_path = "cta_trace.csv";
    const char* tag = "run";
};

static float timeOnce(const Cfg& cfg, const DepPattern& pat,
                      float* d_pout, float* d_out, int* d_done, unsigned long long* d_counter,
                      cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1,
                      const CtaTraceBuffer& tb, bool traceOn) {
    CUDA_CHECK(cudaMemsetAsync(d_done, 0, (size_t)cfg.n_producer * sizeof(int), s));
    CUDA_CHECK(cudaMemsetAsync(d_counter, 0, sizeof(unsigned long long), s));
    if (traceOn) CUDA_CHECK(cudaMemsetAsync(tb.d_recs, 0,
                    (size_t)tb.total * sizeof(CtaRecord), s));

    CtaTrace tr0 = traceOn ? ctatrace_slice(tb, 0) : ctatrace_disabled();
    CtaTrace tr1 = traceOn ? ctatrace_slice(tb, 1) : ctatrace_disabled();

    CUDA_CHECK(cudaEventRecord(e0, s));

    // Producer: plain launch (the trigger is inside the kernel).
    producerK<<<cfg.n_producer, cfg.threads, smemBytes({cfg.smem_kb, cfg.threads}), s>>>(
        d_pout, d_done, d_counter, cfg.n_producer, cfg.tail, tr0, cfg.use_pdl);
    CUDA_CHECK(cudaGetLastError());

    // Consumer: same stream. PDL stream-serialization lets it start before the producer
    // finishes; without it the stream keeps them strictly ordered.
    if (cfg.use_pdl) {
        cudaLaunchAttribute a{};
        a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t lc{};
        lc.gridDim = dim3(cfg.n_consumer);
        lc.blockDim = dim3(cfg.threads);
        lc.dynamicSmemBytes = smemBytes({cfg.smem_kb, cfg.threads});
        lc.stream = s;
        lc.attrs = &a; lc.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&lc, consumerK, d_out, (const float*)d_pout,
                                      (const int*)d_done,
                                      (const unsigned long long*)d_counter,
                                      pat, cfg.wait_mode, cfg.prologue, tr1));
    } else {
        consumerK<<<cfg.n_consumer, cfg.threads, smemBytes({cfg.smem_kb, cfg.threads}), s>>>(
            d_out, d_pout, d_done, d_counter, pat, cfg.wait_mode, cfg.prologue, tr1);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(e1, s));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

static bool verify(const Cfg& cfg, const DepPattern& pat, const float* d_out) {
    std::vector<float> h(cfg.n_consumer);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, (size_t)cfg.n_consumer * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < cfg.n_consumer; ++j) {
        double expect = 0.0;
        int d = dep_degree(pat, j);
        for (int k = 0; k < d; ++k) {
            int p = dep_parent(pat, j, k);
            if (p >= 0) expect += (double)p * 2.0 + 1.0;
        }
        if (fabs((double)h[j] - expect) > 1e-2 * (1.0 + fabs(expect))) {
            fprintf(stderr, "  verify FAIL at consumer %d: got %f expect %f\n", j, h[j], expect);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    Args A(argc, argv);
    if (A.has("--help")) {
        printf(
"usage: cta_dep_bench [options]\n"
"  --producers N     producer grid size in CTAs (default 1024)\n"
"  --consumers N     consumer grid size in CTAs (default = producers)\n"
"  --structure S     interval|grouped|strided|random|self|all|none (default interval)\n"
"  --degree D        parents per consumer CTA (default 1)\n"
"  --threads T       threads per CTA (default 128)\n"
"  --smem-kb K       dynamic shared memory per CTA in KB (default 0) [B2 occupancy sweep]\n"
"  --tail CYC        producer trailing cycles (default 200000)\n"
"  --prologue CYC    consumer independent prologue cycles (default = tail)\n"
"  --wait MODE       none(ceiling)|grid(griddepcontrol)|cta-spin|cta-backoff|cta-counter|cta-exact\n"
"  --all-waits       run every wait mode and print the bracket\n"
"  --no-pdl          disable PDL stream serialization\n"
"  --repeats N       timed repeats (default 30)\n"
"  --trace PATH      dump per-CTA timeline CSV (primitive 1)\n"
"  --tag STR         tag written into SUMMARY / trace rows\n");
        return 0;
    }

    DeviceInfo dev = queryDevice();

    Cfg cfg;
    cfg.n_producer = (int)A.ll("--producers", 1024);
    cfg.n_consumer = (int)A.ll("--consumers", cfg.n_producer);
    cfg.degree     = (int)A.ll("--degree", 1);
    cfg.threads    = (int)A.ll("--threads", 128);
    cfg.smem_kb    = (int)A.ll("--smem-kb", 0);
    cfg.tail       = (unsigned long long)A.ll("--tail", 200000);
    cfg.prologue   = (unsigned long long)A.ll("--prologue", (long long)cfg.tail);
    cfg.repeats    = (int)A.ll("--repeats", 30);
    cfg.use_pdl    = A.has("--no-pdl") ? 0 : 1;
    cfg.tag        = A.str("--tag", "run");
    cfg.trace_path = A.str("--trace", nullptr);
    cfg.trace      = cfg.trace_path ? 1 : 0;

    const char* sname = A.str("--structure", "interval");
    cfg.structure = depStructureFromName(sname);
    if (cfg.structure < 0) { fprintf(stderr, "unknown --structure %s\n", sname); return 1; }

    const char* wname = A.str("--wait", "cta-spin");
    cfg.wait_mode = waitModeFromName(wname);
    if (cfg.wait_mode < 0) { fprintf(stderr, "unknown --wait %s\n", wname); return 1; }

    DepPattern pat{cfg.structure, cfg.degree, cfg.n_producer, cfg.n_consumer, 12345u};

    printDeviceBanner(dev);
    double tightness = dep_interval_tightness(pat);
    double effdeg    = dep_effective_degree(pat);
    printf("Pattern: structure=%s degree=%d producers=%d consumers=%d | "
           "interval_tightness=%.3f effective_degree=%.1f\n",
           depStructureName(cfg.structure), cfg.degree, cfg.n_producer, cfg.n_consumer,
           tightness, effdeg);
    printf("Config : threads=%d smem=%dKB tail=%llu prologue=%llu repeats=%d pdl=%d\n\n",
           cfg.threads, cfg.smem_kb, cfg.tail, cfg.prologue, cfg.repeats, cfg.use_pdl);

    // ---- allocate ----
    float *d_pout = nullptr, *d_out = nullptr;
    int   *d_done = nullptr;
    unsigned long long* d_counter = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pout, (size_t)cfg.n_producer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  (size_t)cfg.n_consumer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_done, (size_t)cfg.n_producer * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));

    CtaTraceBuffer tb{};
    if (cfg.trace) {
        unsigned int per = (unsigned)std::max(cfg.n_producer, cfg.n_consumer);
        CUDA_CHECK(ctatrace_alloc(&tb, 2, per));
    }

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    // Dynamic shared-memory launches above CUDA's default 48-KB ceiling require
    // an explicit opt-in.  Do this before the occupancy query as well as launch,
    // otherwise a valid 64-KB B200/B300 configuration is reported as 0 CTAs/SM.
    if (cfg.smem_kb > 0) {
        int requested = (int)smemBytes({cfg.smem_kb, cfg.threads});
        CUDA_CHECK(cudaFuncSetAttribute(producerK, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        requested));
        CUDA_CHECK(cudaFuncSetAttribute(consumerK, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        requested));
    }

    int occ = ctasPerSM(consumerK, {cfg.smem_kb, cfg.threads});
    printf("Occupancy: consumer CTAs resident per SM = %d (device has %d SMs => %d concurrent CTAs)\n\n",
           occ, dev.sms, occ * dev.sms);

    // ---- run one or all wait modes ----
    int modes[WAIT_NMODES]; int nmodes = 0;
    if (A.has("--all-waits")) {
        modes[nmodes++] = WAIT_NONE;
        modes[nmodes++] = WAIT_GRID;
        modes[nmodes++] = WAIT_SPIN;
        modes[nmodes++] = WAIT_BACKOFF;
        modes[nmodes++] = WAIT_COUNTER;
        modes[nmodes++] = WAIT_EXACT;
    } else {
        modes[nmodes++] = cfg.wait_mode;
    }

    double med[WAIT_NMODES] = {0}, best[WAIT_NMODES] = {0};
    bool   ok[WAIT_NMODES];
    for (int i = 0; i < WAIT_NMODES; ++i) { ok[i] = true; med[i] = 0; }

    printf("%-24s %12s %12s %10s\n", "wait mode", "median(ms)", "min(ms)", "correct");
    for (int mi = 0; mi < nmodes; ++mi) {
        Cfg c = cfg; c.wait_mode = modes[mi];
        for (int w = 0; w < 3; ++w)
            (void)timeOnce(c, pat, d_pout, d_out, d_done, d_counter, s, e0, e1, tb, false);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> t;
        for (int r = 0; r < c.repeats; ++r) {
            bool traceThis = (cfg.trace && r == c.repeats - 1);
            t.push_back(timeOnce(c, pat, d_pout, d_out, d_done, d_counter, s, e0, e1, tb, traceThis));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        med[modes[mi]]  = medianOf(t);
        best[modes[mi]] = minOf(t);
        ok[modes[mi]]   = waitIsCorrect(modes[mi]) ? verify(c, pat, d_out) : true;

        printf("%-24s %12.4f %12.4f %10s\n", waitModeName(modes[mi]),
               med[modes[mi]], best[modes[mi]],
               waitIsCorrect(modes[mi]) ? (ok[modes[mi]] ? "PASS" : "FAIL") : "n/a");

        if (cfg.trace && modes[mi] == cfg.wait_mode) {
            char path[512];
            snprintf(path, sizeof(path), "%s", cfg.trace_path);
            if (ctatrace_dump_csv(tb, path, cfg.tag))
                printf("  [trace] per-CTA timeline -> %s\n", path);
        }
    }

    // ---- bracket ----
    double floorMs   = med[WAIT_GRID];
    double ceilMs    = med[WAIT_NONE];
    double implMs    = med[cfg.wait_mode] > 0 ? med[cfg.wait_mode] : 0.0;
    double space     = (floorMs > 0 && ceilMs > 0) ? (floorMs - ceilMs) / floorMs : 0.0;
    double captured  = (floorMs > 0 && implMs > 0) ? (floorMs - implMs) / floorMs : 0.0;

    if (A.has("--all-waits")) {
        printf("\nBRACKET: floor(grid)=%.4f ms  ceiling(none)=%.4f ms  => total space=%.1f%%\n",
               floorMs, ceilMs, space * 100.0);
        printf("         best software impl captured %.1f%% of floor "
               "(= %.1f%% of the available space)\n",
               captured * 100.0, space > 0 ? captured / space * 100.0 : 0.0);
    }

    // Single machine-parsable line for the sweep driver.
    printf("SUMMARY tag=%s structure=%s degree=%d eff_degree=%.2f tightness=%.4f "
           "producers=%d consumers=%d threads=%d smem_kb=%d occ_per_sm=%d sms=%d "
           "tail=%llu prologue=%llu pdl=%d "
           "floor_ms=%.5f ceiling_ms=%.5f spin_ms=%.5f backoff_ms=%.5f counter_ms=%.5f exact_ms=%.5f "
           "space_pct=%.3f\n",
           cfg.tag, depStructureName(cfg.structure), cfg.degree, effdeg, tightness,
           cfg.n_producer, cfg.n_consumer, cfg.threads, cfg.smem_kb, occ, dev.sms,
           cfg.tail, cfg.prologue, cfg.use_pdl,
           med[WAIT_GRID], med[WAIT_NONE], med[WAIT_SPIN], med[WAIT_BACKOFF],
           med[WAIT_COUNTER], med[WAIT_EXACT], space * 100.0);

    ctatrace_free(&tb);
    cudaFree(d_pout); cudaFree(d_out); cudaFree(d_done); cudaFree(d_counter);
    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
