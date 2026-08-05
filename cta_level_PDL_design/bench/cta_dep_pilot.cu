// cta_dep_pilot.cu — corrected, bounded CTA-level PDL screening pilot.
//
// This pilot exists because cta_dep_bench publishes every done[] flag before
// the PDL trigger.  Since a dependent grid becomes eligible only after every
// producer CTA triggers, those waits are all already satisfied and the claimed
// CTA-granular benefit map degenerates into protocol overhead.
//
// Corrected execution:
//   producer: trigger at entry -> readiness work -> release done[cta] -> tail
//   consumer: independent prologue -> dependency wait -> dependent epilogue
//
// Timed epilogue work is O(1) in dependency degree.  A separate, untimed
// validation launch checks every exact parent flag and datum immediately after
// each supposedly-correct wait.  The unsound global completion-counter protocol
// is deliberately excluded.
//
// Grid size may exceed SM count (multi-wave, EXPERIMENT_PLAN.md §5.3). Per-CTA
// publish-after-ready still holds when later waves are not yet resident.

#include "common/bench_util.cuh"
#include "common/dep_pattern.cuh"

#include <cuda/atomic>
#include <limits>

enum PilotWait {
    PILOT_NONE = 0,
    PILOT_GRID = 1,
    PILOT_INTERVAL_SPIN = 2,
    PILOT_INTERVAL_BACKOFF = 3,
    PILOT_EXACT_BACKOFF = 4,
    PILOT_NMODES = 5,
};

static const char* pilotWaitName(int mode) {
    switch (mode) {
        case PILOT_NONE:             return "none";
        case PILOT_GRID:             return "grid";
        case PILOT_INTERVAL_SPIN:    return "interval-spin";
        case PILOT_INTERVAL_BACKOFF: return "interval-backoff";
        case PILOT_EXACT_BACKOFF:    return "exact-backoff";
        default:                     return "?";
    }
}

__device__ __forceinline__ void pilot_wait_one(const int* done, int parent,
                                                bool backoff) {
    cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[parent]);
    unsigned int ns = 32;
    while (flag.load(cuda::memory_order_acquire) == 0) {
        if (backoff) {
            __nanosleep(ns);
            ns = ns < 1024 ? ns * 2 : 1024;
        }
    }
}

__device__ __forceinline__ void pilot_wait_interval(const int* done,
                                                     const DepPattern& pat,
                                                     int child,
                                                     bool backoff) {
    if (threadIdx.x == 0) {
        int lo, hi;
        dep_interval(pat, child, &lo, &hi);
        for (int p = lo; p <= hi; ++p) pilot_wait_one(done, p, backoff);
    }
}

__device__ __forceinline__ void pilot_wait_exact(const int* done,
                                                  const DepPattern& pat,
                                                  int child) {
    if (threadIdx.x == 0) {
        int degree = dep_degree(pat, child);
        for (int k = 0; k < degree; ++k) {
            int p = dep_parent(pat, child, k);
            if (p >= 0) pilot_wait_one(done, p, true);
        }
    }
}

__global__ void pilotProducer(float* data,
                              int* done,
                              int nproducer,
                              int wait_mode,
                              unsigned long long ready_cycles,
                              unsigned long long tail_cycles,
                              int skew_bins,
                              unsigned int seed) {
    // Proposed CTA modes and the no-wait reference need launch eligibility
    // before data readiness.  The production Floor keeps standard PDL trigger
    // placement after the required write.
    if (wait_mode != PILOT_GRID) cudaTriggerProgrammaticLaunchCompletion();

    int cta = (int)blockIdx.x;
    unsigned int h = dep_hash((unsigned int)cta ^ seed);
    unsigned int bucket = skew_bins > 1 ? h % (unsigned int)skew_bins : 0u;
    unsigned long long delay = ready_cycles;
    if (skew_bins > 1)
        delay += (ready_cycles * (unsigned long long)bucket) /
                 (unsigned long long)skew_bins;
    spin_cycles(delay);

    if (threadIdx.x == 0 && cta < nproducer)
        data[cta] = (float)cta * 2.0f + 1.0f;
    __syncthreads();

    bool software_wait = wait_mode == PILOT_INTERVAL_SPIN ||
                         wait_mode == PILOT_INTERVAL_BACKOFF ||
                         wait_mode == PILOT_EXACT_BACKOFF;
    if (software_wait && threadIdx.x == 0 && cta < nproducer) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(done[cta]);
        flag.store(1, cuda::memory_order_release);
    }
    if (wait_mode == PILOT_GRID) cudaTriggerProgrammaticLaunchCompletion();

    // Work independent of the published datum.  CTA-level consumers may run
    // dependent work during this tail; griddepcontrol.wait cannot.
    spin_cycles(tail_cycles);
}

__global__ void pilotConsumer(float* out,
                              const float* data,
                              const int* done,
                              DepPattern pat,
                              int wait_mode,
                              unsigned long long prologue_cycles,
                              unsigned long long epilogue_cycles,
                              int validate,
                              int* error) {
    int child = (int)blockIdx.x;
    spin_cycles(prologue_cycles);

    switch (wait_mode) {
        case PILOT_NONE:
            break;
        case PILOT_GRID:
            if (threadIdx.x == 0) cudaGridDependencySynchronize();
            break;
        case PILOT_INTERVAL_SPIN:
            pilot_wait_interval(done, pat, child, false);
            break;
        case PILOT_INTERVAL_BACKOFF:
            pilot_wait_interval(done, pat, child, true);
            break;
        case PILOT_EXACT_BACKOFF:
            pilot_wait_exact(done, pat, child);
            break;
    }
    __syncthreads();

    // Validation is intentionally outside timed samples.  It checks every true
    // dependency without adding O(degree) ordinary work to the benefit sweep.
    if (validate && threadIdx.x == 0) {
        int degree = dep_degree(pat, child);
        for (int k = 0; k < degree; ++k) {
            int p = dep_parent(pat, child, k);
            if (p < 0) continue;
            bool software_wait = wait_mode == PILOT_INTERVAL_SPIN ||
                                 wait_mode == PILOT_INTERVAL_BACKOFF ||
                                 wait_mode == PILOT_EXACT_BACKOFF;
            int ready = 1;
            if (software_wait) {
                cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[p]);
                ready = flag.load(cuda::memory_order_acquire);
            }
            float expected = (float)p * 2.0f + 1.0f;
            if (!ready || data[p] != expected) atomicExch(error, 1);
        }
    }
    __syncthreads();

    spin_cycles(epilogue_cycles);

    // Constant timed post-wait memory work: one representative parent read,
    // independent of degree.  Full validation above is a separate launch.
    if (threadIdx.x == 0 && child < pat.n_consumer) {
        int degree = dep_degree(pat, child);
        int p = degree > 0 ? dep_parent(pat, child, degree - 1) : -1;
        out[child] = p >= 0 ? data[p] : 0.0f;
    }
}

struct PilotCfg {
    int nproducer = 1024;
    int nconsumer = 1024;
    int structure = DEP_INTERVAL;
    int degree = 1;
    int threads = 128;
    int repeats = 30;
    int skew_bins = 8;
    unsigned int seed = 12345u;
    unsigned long long ready = 400000ull;
    unsigned long long tail = 1000000ull;
    unsigned long long prologue = 200000ull;
    unsigned long long epilogue = 1000000ull;
    const char* tag = "pilot";
};

struct PilotCtx {
    float* data = nullptr;
    float* out = nullptr;
    int* done = nullptr;
    int* error = nullptr;
    cudaStream_t stream{};
    cudaEvent_t begin{}, end{};
};

static float pilotOnce(const PilotCfg& cfg,
                       const DepPattern& pat,
                       PilotCtx& ctx,
                       int mode,
                       bool validate) {
    CUDA_CHECK(cudaMemsetAsync(ctx.data, 0xff,
                              (size_t)cfg.nproducer * sizeof(float), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.out, 0xff,
                              (size_t)cfg.nconsumer * sizeof(float), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.done, 0,
                              (size_t)cfg.nproducer * sizeof(int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
    CUDA_CHECK(cudaEventRecord(ctx.begin, ctx.stream));

    pilotProducer<<<cfg.nproducer, cfg.threads, 0, ctx.stream>>>(
        ctx.data, ctx.done, cfg.nproducer, mode, cfg.ready, cfg.tail,
        cfg.skew_bins, cfg.seed);
    CUDA_CHECK(cudaGetLastError());

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(cfg.nconsumer);
    launch.blockDim = dim3(cfg.threads);
    launch.stream = ctx.stream;
    launch.attrs = &attr;
    launch.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &launch, pilotConsumer, ctx.out, (const float*)ctx.data,
        (const int*)ctx.done, pat, mode, cfg.prologue, cfg.epilogue,
        validate ? 1 : 0, ctx.error));

    CUDA_CHECK(cudaEventRecord(ctx.end, ctx.stream));
    CUDA_CHECK(cudaEventSynchronize(ctx.end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx.begin, ctx.end));
    return ms;
}

static bool validateMode(const PilotCfg& cfg,
                         const DepPattern& pat,
                         PilotCtx& ctx,
                         int mode) {
    if (mode == PILOT_NONE) return true;
    (void)pilotOnce(cfg, pat, ctx, mode, true);
    int error = 1;
    CUDA_CHECK(cudaMemcpy(&error, ctx.error, sizeof(int), cudaMemcpyDeviceToHost));
    return error == 0;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf(
            "usage: cta_dep_pilot [options]\n"
            "  --producers N --consumers N   (may exceed SM count; multi-wave is §5.3)\n"
            "  --structure interval|grouped|strided|self\n"
            "  --degree D --repeats N --threads N\n"
            "  --ready CYC --tail CYC --prologue CYC --epilogue CYC\n"
            "  --skew-bins N --seed N --tag STR\n");
        return 0;
    }

    PilotCfg cfg;
    cfg.nproducer = (int)args.ll("--producers", cfg.nproducer);
    cfg.nconsumer = (int)args.ll("--consumers", cfg.nproducer);
    cfg.degree = (int)args.ll("--degree", cfg.degree);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.skew_bins = (int)args.ll("--skew-bins", cfg.skew_bins);
    cfg.seed = (unsigned int)args.ll("--seed", cfg.seed);
    cfg.ready = (unsigned long long)args.ll("--ready", cfg.ready);
    cfg.tail = (unsigned long long)args.ll("--tail", cfg.tail);
    cfg.prologue = (unsigned long long)args.ll("--prologue", cfg.prologue);
    cfg.epilogue = (unsigned long long)args.ll("--epilogue", cfg.epilogue);
    cfg.tag = args.str("--tag", cfg.tag);
    cfg.structure = depStructureFromName(args.str("--structure", "interval"));

    if (cfg.nproducer <= 0 || cfg.nconsumer <= 0 || cfg.degree <= 0 ||
        cfg.threads <= 0 || cfg.threads > 1024 || cfg.repeats <= 0 ||
        cfg.skew_bins <= 0 || cfg.structure < 0) {
        fprintf(stderr, "invalid pilot configuration\n");
        return 2;
    }
    if (cfg.structure == DEP_RANDOM || cfg.structure == DEP_ALL ||
        cfg.structure == DEP_NONE) {
        fprintf(stderr, "pilot excludes random/all/none structures; use interval/grouped/strided/self\n");
        return 2;
    }

    DeviceInfo dev = queryDevice();
    // Multi-wave (P,C > SM) is required by EXPERIMENT_PLAN.md §5.3. Trigger semantics
    // already satisfy §3.1: software modes trigger at entry then publish done[cta] only
    // after this CTA's readiness work; Floor triggers after readiness. Later producer
    // waves cannot publish early — they have not run yet. Do not re-introduce a P,C<=SM
    // cap here; that was the project's top measurement gap, not a correctness requirement.
    const int grid_max = cfg.nproducer > cfg.nconsumer ? cfg.nproducer : cfg.nconsumer;
    if (grid_max > 64 * dev.sms) {
        fprintf(stderr,
                "refusing producers/consumers=%d > 64*SM (%d); plan §5.3 tops out at 32*SM\n",
                grid_max, 64 * dev.sms);
        return 2;
    }
    const char* wave_regime = "underfilled";
    if (grid_max > dev.sms) wave_regime = "multi";
    else if (grid_max == dev.sms) wave_regime = "single_full";

    int producer_occ = ctasPerSM(pilotProducer, {0, cfg.threads});
    int consumer_occ = ctasPerSM(pilotConsumer, {0, cfg.threads});
    if (producer_occ < 2 || consumer_occ < 2) {
        fprintf(stderr, "pilot requires >=2 resident CTAs/SM; got producer=%d consumer=%d\n",
                producer_occ, consumer_occ);
        return 2;
    }
    printDeviceBanner(dev);
    DepPattern pat{cfg.structure, cfg.degree, cfg.nproducer, cfg.nconsumer, cfg.seed};
    double tightness = dep_interval_tightness(pat);
    double effdegree = dep_effective_degree(pat);
    printf("Pilot semantics=2 tag=%s structure=%s degree=%d effective_degree=%.2f "
           "tightness=%.4f grid=%d wave=%s ready=%llu tail=%llu prologue=%llu "
           "epilogue=%llu skew_bins=%d repeats=%d producer_occ=%d consumer_occ=%d\n",
           cfg.tag, depStructureName(cfg.structure), cfg.degree, effdegree,
           tightness, cfg.nproducer, wave_regime, cfg.ready, cfg.tail, cfg.prologue,
           cfg.epilogue, cfg.skew_bins, cfg.repeats, producer_occ, consumer_occ);
    if (grid_max > dev.sms) {
        printf("NOTE: multi-wave grid (max(P,C)=%d > SM=%d). Later producer waves wait for "
               "earlier ones to retire; overlap structure differs from single-wave. "
               "Trigger/publish still per-CTA after readiness (§3.1).\n",
               grid_max, dev.sms);
    }

    PilotCtx ctx;
    CUDA_CHECK(cudaMalloc(&ctx.data, (size_t)cfg.nproducer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.out, (size_t)cfg.nconsumer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.done, (size_t)cfg.nproducer * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.error, sizeof(int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&ctx.begin));
    CUDA_CHECK(cudaEventCreate(&ctx.end));

    double med[PILOT_NMODES] = {};
    double low[PILOT_NMODES] = {};
    bool valid[PILOT_NMODES] = {};
    bool any_failure = false;

    printf("%-20s %12s %12s %10s\n", "mode", "median_ms", "min_ms", "valid");
    for (int mode = 0; mode < PILOT_NMODES; ++mode) {
        for (int warm = 0; warm < 3; ++warm)
            (void)pilotOnce(cfg, pat, ctx, mode, false);

        std::vector<float> samples;
        samples.reserve((size_t)cfg.repeats);
        for (int rep = 0; rep < cfg.repeats; ++rep) {
            float ms = pilotOnce(cfg, pat, ctx, mode, false);
            samples.push_back(ms);
            printf("SAMPLE tag=%s mode=%s rep=%d ms=%.6f\n",
                   cfg.tag, pilotWaitName(mode), rep, ms);
        }
        med[mode] = medianOf(samples);
        low[mode] = minOf(samples);
        valid[mode] = validateMode(cfg, pat, ctx, mode);
        if (mode != PILOT_NONE && !valid[mode]) any_failure = true;
        printf("%-20s %12.5f %12.5f %10s\n", pilotWaitName(mode),
               med[mode], low[mode],
               mode == PILOT_NONE ? "n/a" : (valid[mode] ? "PASS" : "FAIL"));
    }

    double floor_ms = med[PILOT_GRID];
    double ceiling_ms = med[PILOT_NONE];
    const int impl_mode = PILOT_INTERVAL_BACKOFF;
    double impl_ms = med[impl_mode];
    double space_pct = floor_ms > 0.0
        ? 100.0 * (floor_ms - ceiling_ms) / floor_ms : 0.0;
    double captured_pct = floor_ms > 0.0 && valid[impl_mode]
        ? 100.0 * (floor_ms - impl_ms) / floor_ms : 0.0;
    double of_space_pct = space_pct > 0.0
        ? 100.0 * captured_pct / space_pct : 0.0;

    printf("BRACKET floor=%.5f ceiling=%.5f space=%.3f%% "
           "impl=%s impl_ms=%.5f captured=%.3f%% of_space=%.1f%%\n",
           floor_ms, ceiling_ms, space_pct, pilotWaitName(impl_mode), impl_ms,
           captured_pct, of_space_pct);
    printf("SUMMARY_PILOT semantics=2 tag=%s structure=%s degree=%d eff_degree=%.2f "
           "tightness=%.4f producers=%d consumers=%d threads=%d sms=%d "
           "wave=%s producer_occ=%d consumer_occ=%d trigger_floor=ready "
           "trigger_impl=entry trigger_ceiling=entry "
           "ready=%llu tail=%llu prologue=%llu epilogue=%llu skew_bins=%d "
           "repeats=%d floor_ms=%.6f ceiling_ms=%.6f interval_spin_ms=%.6f "
           "interval_backoff_ms=%.6f exact_backoff_ms=%.6f impl=%s "
           "impl_ms=%.6f space_pct=%.4f captured_pct=%.4f "
           "of_space_pct=%.3f valid=%d\n",
           cfg.tag, depStructureName(cfg.structure), cfg.degree, effdegree,
           tightness, cfg.nproducer, cfg.nconsumer, cfg.threads, dev.sms,
           wave_regime, producer_occ, consumer_occ,
           cfg.ready, cfg.tail, cfg.prologue, cfg.epilogue, cfg.skew_bins,
           cfg.repeats, floor_ms, ceiling_ms, med[PILOT_INTERVAL_SPIN],
           med[PILOT_INTERVAL_BACKOFF], med[PILOT_EXACT_BACKOFF],
           pilotWaitName(impl_mode), impl_ms, space_pct, captured_pct, of_space_pct,
           any_failure ? 0 : 1);

    cudaEventDestroy(ctx.begin);
    cudaEventDestroy(ctx.end);
    cudaStreamDestroy(ctx.stream);
    cudaFree(ctx.data);
    cudaFree(ctx.out);
    cudaFree(ctx.done);
    cudaFree(ctx.error);
    return any_failure ? 1 : 0;
}
