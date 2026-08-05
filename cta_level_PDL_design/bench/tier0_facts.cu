// tier0_facts.cu — Tier 0 base facts. Run this FIRST on B300.
//
// Nothing here evaluates a design option; it establishes the ground truth that every later
// experiment's interpretation depends on. See ../EXPERIMENT_PLAN.md §4.
//
//   0.1  How many kernels of a same-stream chain actually overlap on B300?
//        -> decides which B3 window-depth options are even REACHABLE on this hardware.
//   0.3  Occupancy cost of a CTA that waits while resident.
//        -> the ONLY pricing basis for dimension B2 ("what would pre-dispatch gating buy?").
//   0.5  Fence scope cost calibration (.cta / .gpu / .sys).
//        -> feeds the "Ideal" point of the four-point bracket.
//
// 0.2 (PDL eager-cross-stream behaviour) is covered by re-running the existing
// cross_stream_PDL_survey/bench/pdl_bench on B300; 0.4 (CLC try_cancel) needs sm_100+ PTX and is
// split into clc_probe.cu.
//
// Build: ./build.sh   Run: ./tier0_facts [--repeats N]

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"

#include <cstdint>
#include <random>
#include <tuple>

// ---------------------------------------------------------------- 0.1 chain overlap depth

// One record per stage/CTA.  Tier 0.1 needs more than launch/dep/end: the trace must also
// prove that the data write precedes the PDL trigger.  Keeping this schema local avoids
// changing the common CtaRecord contract used by the other benchmarks.
struct ChainTraceRecord {
    unsigned long long t_launch;
    unsigned long long t_dep_satisfied;
    unsigned long long t_value_ready;
    unsigned long long t_trigger;
    unsigned long long t_end;
    unsigned int block_id;
    unsigned short stage;
    unsigned short sm_id;
};
static_assert(sizeof(ChainTraceRecord) == 48, "ChainTraceRecord layout changed");

__host__ __device__ __forceinline__ unsigned int chainMix(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ __forceinline__ unsigned int chainInitial(unsigned int epoch,
                                                               unsigned int block) {
    return chainMix(epoch ^ 0xa5a5a5a5u ^ (0x9e3779b9u * (block + 1u)));
}

__host__ __device__ __forceinline__ unsigned int chainStep(unsigned int prior,
                                                            unsigned int epoch,
                                                            unsigned int stage,
                                                            unsigned int block) {
    // Non-commutative recurrence: a stale, skipped or reordered predecessor changes every
    // downstream checkpoint.  Validation retains every stage output, not only the final sum.
    return chainMix(prior ^ epoch ^ (0x85ebca6bu * (stage + 1u))
                          ^ (0xc2b2ae35u * (block + 1u)));
}

__global__ void initChainState(unsigned int* state, unsigned int epoch) {
    if (threadIdx.x == 0)
        state[blockIdx.x] = chainInitial(epoch, blockIdx.x);
}

// A link in a K1->K2->...->Kn chain.  The post-write barrier is load-bearing: every thread
// reaches cudaTriggerProgrammaticLaunchCompletion only after the CTA's value is ready.
__global__ void chainK(unsigned int* __restrict__ state,
                       unsigned int* __restrict__ checkpoints,
                       int stage, int max_stages, unsigned int epoch,
                       unsigned long long work, ChainTraceRecord* trace, int use_pdl) {
    ChainTraceRecord rec{};
    if (threadIdx.x == 0) {
        rec.t_launch = ctatrace_globaltimer();
        rec.block_id = blockIdx.x;
        rec.stage = (unsigned short)stage;
        rec.sm_id = (unsigned short)ctatrace_smid();
    }

    spin_cycles(work / 2);                        // independent prologue
    if (use_pdl) cudaGridDependencySynchronize(); // wait for predecessor grid retirement
    if (threadIdx.x == 0) rec.t_dep_satisfied = ctatrace_globaltimer();

    if (threadIdx.x == 0) {
        unsigned int value = chainStep(state[blockIdx.x], epoch,
                                       (unsigned int)stage, blockIdx.x);
        state[blockIdx.x] = value;
        if (checkpoints)
            checkpoints[(size_t)stage * gridDim.x + blockIdx.x] = value;
        rec.t_value_ready = ctatrace_globaltimer();
    }
    __syncthreads();

    if (use_pdl) {
        if (threadIdx.x == 0) rec.t_trigger = ctatrace_globaltimer();
        cudaTriggerProgrammaticLaunchCompletion();
    }
    spin_cycles(work / 2);                        // overlappable tail
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        trace[(size_t)stage * gridDim.x + blockIdx.x] = rec;
    }
    (void)max_stages;
}

struct ChainTraceMetrics {
    bool complete = false;
    int peak_active_ctas = 0;
    int peak_active_grids = 0;
    int early_links = 0;
    int dependency_safe_links = 0;
    int serial_links = 0;
    double makespan_ms = 0.0;
};

struct ChainStats {
    double median = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
};

struct ChainSample {
    double makespan_ms = 0.0;
    ChainTraceMetrics trace;
    bool valid = false;
};

static double chainMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    return (n & 1u) ? values[n / 2]
                    : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// SplitMix64 makes the bootstrap stream language-independent.  The Python validator can
// therefore reproduce every printed CI exactly from raw SAMPLE records.
static unsigned long long chainRandom(unsigned long long* state) {
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static ChainStats chainBootstrapMedian(const std::vector<double>& values,
                                       unsigned long long seed) {
    ChainStats out{};
    if (values.empty()) return out;
    out.median = chainMedian(values);
    if (values.size() == 1) {
        out.ci_low = out.ci_high = out.median;
        return out;
    }
    constexpr int kBootstrap = 2000;
    std::vector<double> medians;
    std::vector<double> sample(values.size());
    medians.reserve(kBootstrap);
    unsigned long long state = seed;
    for (int b = 0; b < kBootstrap; ++b) {
        for (size_t i = 0; i < values.size(); ++i)
            sample[i] = values[(size_t)(chainRandom(&state) % values.size())];
        medians.push_back(chainMedian(sample));
    }
    std::sort(medians.begin(), medians.end());
    out.ci_low = medians[(size_t)(0.025 * kBootstrap)];
    out.ci_high = medians[(size_t)(0.975 * kBootstrap)];
    return out;
}

static double modelImpliedChainDepth(double speedup, int stages) {
    return (speedup > 0.0 && speedup < 2.0)
         ? speedup / (2.0 - speedup) : (double)stages;
}

static ChainTraceMetrics analyzeChainTrace(const std::vector<ChainTraceRecord>& rows,
                                           int stages, int blocks, bool use_pdl) {
    ChainTraceMetrics out{};
    if ((int)rows.size() < stages * blocks) return out;

    std::vector<unsigned long long> first((size_t)stages, ~0ull);
    std::vector<unsigned long long> last((size_t)stages, 0ull);
    std::vector<std::tuple<unsigned long long, int, int>> events;
    events.reserve((size_t)stages * blocks * 2);
    unsigned long long global_first = ~0ull, global_last = 0ull;
    bool complete = true;
    for (int stage = 0; stage < stages; ++stage) {
        for (int block = 0; block < blocks; ++block) {
            const ChainTraceRecord& r = rows[(size_t)stage * blocks + block];
            bool ordered = r.block_id == (unsigned)block && r.stage == (unsigned)stage
                && r.sm_id < 65535u && r.t_launch > 0
                && r.t_launch <= r.t_dep_satisfied
                && r.t_dep_satisfied <= r.t_value_ready
                && r.t_value_ready <= r.t_end;
            if (use_pdl)
                ordered = ordered && r.t_value_ready <= r.t_trigger
                                  && r.t_trigger <= r.t_end;
            else
                ordered = ordered && r.t_trigger == 0;
            complete = complete && ordered;
            first[(size_t)stage] = std::min(first[(size_t)stage], r.t_launch);
            last[(size_t)stage] = std::max(last[(size_t)stage], r.t_end);
            global_first = std::min(global_first, r.t_launch);
            global_last = std::max(global_last, r.t_end);
            events.emplace_back(r.t_launch, +1, stage);
            events.emplace_back(r.t_end, -1, stage);
        }
    }

    // End events sort before starts at equal timestamps: intervals are [start,end).
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    });
    std::vector<int> active_by_stage((size_t)stages, 0);
    int active = 0;
    for (const auto& event : events) {
        int delta = std::get<1>(event), stage = std::get<2>(event);
        active += delta;
        active_by_stage[(size_t)stage] += delta;
        int grids = 0;
        for (int count : active_by_stage) if (count > 0) ++grids;
        out.peak_active_ctas = std::max(out.peak_active_ctas, active);
        out.peak_active_grids = std::max(out.peak_active_grids, grids);
    }

    for (int stage = 1; stage < stages; ++stage) {
        unsigned long long predecessor_end = last[(size_t)stage - 1];
        if (first[(size_t)stage] < predecessor_end) ++out.early_links;
        if (first[(size_t)stage] >= predecessor_end) ++out.serial_links;
        bool safe = true;
        for (int block = 0; block < blocks; ++block) {
            const auto& r = rows[(size_t)stage * blocks + block];
            safe = safe && r.t_dep_satisfied >= predecessor_end;
        }
        if (safe) ++out.dependency_safe_links;
    }
    if (!global_last || global_first == ~0ull || global_last <= global_first)
        complete = false;
    else
        out.makespan_ms = (double)(global_last - global_first) / 1.0e6;

    if (use_pdl) {
        // Early launch is the quantity this experiment measures. CUDA documents PDL
        // overlap as opportunistic, so a valid serial/partial-overlap observation must
        // stay in the sample set rather than become a semantic failure or retry target.
        complete = complete && out.dependency_safe_links == stages - 1;
    } else {
        complete = complete && out.serial_links == stages - 1;
    }
    out.complete = complete;
    return out;
}

static bool launchChain(cudaStream_t stream, unsigned int* state,
                        unsigned int* checkpoints, ChainTraceRecord* trace,
                        int stages, int blocks, int threads, unsigned int epoch,
                        unsigned long long work, bool use_pdl) {
    for (int stage = 0; stage < stages; ++stage) {
        if (use_pdl) {
            cudaLaunchAttribute attribute{};
            attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute.val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t launch{};
            launch.gridDim = dim3(blocks);
            launch.blockDim = dim3(threads);
            launch.stream = stream;
            launch.attrs = &attribute;
            launch.numAttrs = 1;
            CUDA_CHECK(cudaLaunchKernelEx(&launch, chainK, state, checkpoints,
                                          stage, stages, epoch, work, trace, 1));
        } else {
            chainK<<<blocks, threads, 0, stream>>>(state, checkpoints, stage, stages,
                                                   epoch, work, trace, 0);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    return true;
}

static ChainSample runChainSample(cudaStream_t stream, unsigned int* state,
                                  ChainTraceRecord* trace,
                                  std::vector<ChainTraceRecord>* host_trace,
                                  int stages, int max_stages, int blocks, int threads,
                                  unsigned int epoch, unsigned long long work,
                                  bool use_pdl) {
    ChainSample sample{};
    initChainState<<<blocks, 1, 0, stream>>>(state, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(trace, 0,
        (size_t)max_stages * blocks * sizeof(ChainTraceRecord), stream));
    // The epoch seed is the per-invocation poison.  Keep initialization outside the trace
    // makespan so the headline is max(t_end)-min(t_launch), sourced only from %globaltimer.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    launchChain(stream, state, nullptr, trace, stages, blocks, threads, epoch, work, use_pdl);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_trace->data(), trace,
        (size_t)max_stages * blocks * sizeof(ChainTraceRecord), cudaMemcpyDeviceToHost));
    sample.trace = analyzeChainTrace(*host_trace, stages, blocks, use_pdl);
    sample.makespan_ms = sample.trace.makespan_ms;
    sample.valid = sample.trace.complete && sample.makespan_ms > 0.0;
    return sample;
}

static unsigned long long digestValue(unsigned long long digest, unsigned int value) {
    digest ^= value;
    digest *= 1099511628211ull;
    return digest;
}

static bool validateChainConfig(cudaStream_t stream, unsigned int* state,
                                unsigned int* checkpoints, ChainTraceRecord* trace,
                                std::vector<unsigned int>* host_state,
                                std::vector<unsigned int>* host_checkpoints,
                                std::vector<ChainTraceRecord>* host_trace,
                                int stages, int max_stages, int blocks, int threads,
                                unsigned int epoch, unsigned long long work,
                                bool use_pdl) {
    initChainState<<<blocks, 1, 0, stream>>>(state, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(checkpoints, 0xa5,
        (size_t)max_stages * blocks * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaMemsetAsync(trace, 0,
        (size_t)max_stages * blocks * sizeof(ChainTraceRecord), stream));
    launchChain(stream, state, checkpoints, trace, stages, blocks, threads,
                epoch, work, use_pdl);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host_state->data(), state,
        (size_t)blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_checkpoints->data(), checkpoints,
        (size_t)max_stages * blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_trace->data(), trace,
        (size_t)max_stages * blocks * sizeof(ChainTraceRecord), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    unsigned long long observed_digest = 1469598103934665603ull;
    unsigned long long expected_digest = 1469598103934665603ull;
    unsigned long long observed_final_digest = 1469598103934665603ull;
    unsigned long long expected_final_digest = 1469598103934665603ull;
    for (int block = 0; block < blocks; ++block) {
        unsigned int expected = chainInitial(epoch, (unsigned int)block);
        for (int stage = 0; stage < stages; ++stage) {
            expected = chainStep(expected, epoch, (unsigned int)stage, (unsigned int)block);
            unsigned int observed = (*host_checkpoints)[(size_t)stage * blocks + block];
            observed_digest = digestValue(observed_digest, observed);
            expected_digest = digestValue(expected_digest, expected);
            if (observed != expected) {
                if (mismatches < 4)
                    fprintf(stderr,
                            "chain checkpoint mismatch mode=%s stages=%d stage=%d block=%d "
                            "got=%u expected=%u\n",
                            use_pdl ? "pdl_on" : "pdl_off", stages, stage, block,
                            observed, expected);
                ++mismatches;
            }
        }
        unsigned int observed_final = (*host_state)[(size_t)block];
        observed_final_digest = digestValue(observed_final_digest, observed_final);
        expected_final_digest = digestValue(expected_final_digest, expected);
        if (observed_final != expected) ++mismatches;
    }
    ChainTraceMetrics metrics = analyzeChainTrace(*host_trace, stages, blocks, use_pdl);
    bool valid = mismatches == 0 && observed_digest == expected_digest
              && observed_final_digest == expected_final_digest && metrics.complete;
    printf("VALIDATION_TIER0_CHAIN semantics=3 tag=t01_s%d stages=%d mode=%s epoch=%u "
           "checked_edges=%d checked_stage_outputs=%d checked_final_outputs=%d "
           "mismatches=%d observed_digest=%llu expected_digest=%llu "
           "observed_final_digest=%llu expected_final_digest=%llu trace_complete=%d "
           "early_links=%d dependency_safe_links=%d serial_links=%d valid=%d\n",
           stages, stages, use_pdl ? "pdl_on" : "pdl_off", epoch,
           (stages - 1) * blocks, stages * blocks, blocks, mismatches,
           observed_digest, expected_digest, observed_final_digest,
           expected_final_digest, metrics.complete ? 1 : 0,
           metrics.early_links, metrics.dependency_safe_links, metrics.serial_links,
           valid ? 1 : 0);
    return valid;
}

static bool dumpChainTrace(const char* path, const char* tag, int rep, const char* mode,
                           unsigned int epoch,
                           const std::vector<ChainTraceRecord>& rows,
                           int stages, int blocks, bool append) {
    FILE* f = fopen(path, append ? "a" : "w");
    if (!f) return false;
    bool ok = true;
    if (!append)
        ok = fprintf(f, "tag,rep,mode,epoch,stage,block_id,sm_id,t_launch,t_dep_satisfied,"
                        "t_value_ready,t_trigger,t_end\n") >= 0;
    for (int stage = 0; stage < stages; ++stage) {
        for (int block = 0; block < blocks; ++block) {
            const auto& r = rows[(size_t)stage * blocks + block];
            if (fprintf(f, "%s,%d,%s,%u,%d,%u,%u,%llu,%llu,%llu,%llu,%llu\n",
                        tag, rep, mode, epoch, stage, r.block_id, (unsigned)r.sm_id,
                        r.t_launch, r.t_dep_satisfied, r.t_value_ready,
                        r.t_trigger, r.t_end) < 0)
                ok = false;
        }
    }
    if (ferror(f)) ok = false;
    if (fclose(f) != 0) ok = false;
    return ok;
}

static bool chainOverlap(const DeviceInfo& dev, int repeats, int warmup,
                         const char* trace_path, bool allow_short) {
    constexpr int kMaxStages = 6;
    const int blocks = dev.sms;
    const int threads = 128;
    const unsigned long long work = 2000000ull;
    if (repeats < 31 && !allow_short) {
        fprintf(stderr,
                "formal Tier 0.1 requires >=31 paired repeats; use --allow-short only for FAST\n");
        return false;
    }

    unsigned int* d_state = nullptr;
    unsigned int* d_checkpoints = nullptr;
    ChainTraceRecord* d_trace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_state, (size_t)blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_checkpoints,
                          (size_t)kMaxStages * blocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_trace,
                          (size_t)kMaxStages * blocks * sizeof(ChainTraceRecord)));
    std::vector<unsigned int> h_state((size_t)blocks);
    std::vector<unsigned int> h_checkpoints((size_t)kMaxStages * blocks);
    std::vector<ChainTraceRecord> h_trace((size_t)kMaxStages * blocks);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    printf("\n=== 0.1 same-stream chain overlap depth ===\n");
    printf("CONFIG_TIER0_CHAIN semantics=3 device=\"%s\" sms=%d blocks=%d threads=%d "
           "stages_max=%d repeats=%d warmup=%d work_cycles=%llu prologue_cycles=%llu "
           "tail_cycles=%llu pairing=adjacent_alternating timer=globaltimer_makespan "
           "trace_timer=globaltimer trigger_pdl=after_value_ready "
           "poison=epoch_seed_every_invocation epoch_schedule=monotonic_all_invocations "
           "validation=independent_all_edges "
           "allow_short=%d\n",
           dev.name, dev.sms, blocks, threads, kMaxStages, repeats, warmup, work,
           work / 2, work / 2, allow_short ? 1 : 0);

    unsigned int epoch = 1u;
    bool trace_started = false;
    for (int stages = 1; stages <= kMaxStages; ++stages) {
        // One independent, untimed validation invocation for every configuration/mode.
        for (int mode = 0; mode < 2; ++mode) {
            if (!validateChainConfig(stream, d_state, d_checkpoints, d_trace,
                    &h_state, &h_checkpoints, &h_trace, stages, kMaxStages,
                    blocks, threads, epoch++, work, mode == 1)) {
                cudaFree(d_state); cudaFree(d_checkpoints); cudaFree(d_trace);
                CUDA_CHECK(cudaStreamDestroy(stream));
                return false;
            }
        }

        // Warmups are paired and alternate order exactly like timed repetitions.
        for (int w = 0; w < warmup; ++w) {
            bool on_first = (w & 1) != 0;
            for (int order = 0; order < 2; ++order) {
                bool use_pdl = on_first ? order == 0 : order == 1;
                ChainSample sample = runChainSample(stream, d_state, d_trace, &h_trace,
                    stages, kMaxStages, blocks, threads, epoch++, work, use_pdl);
                if (!sample.valid) {
                    fprintf(stderr, "Tier 0.1 warmup trace failed stages=%d mode=%s\n",
                            stages, use_pdl ? "pdl_on" : "pdl_off");
                    cudaFree(d_state); cudaFree(d_checkpoints); cudaFree(d_trace);
                    CUDA_CHECK(cudaStreamDestroy(stream));
                    return false;
                }
            }
        }

        std::vector<ChainSample> off((size_t)repeats), on((size_t)repeats);
        for (int rep = 0; rep < repeats; ++rep) {
            bool on_first = (rep & 1) != 0;
            for (int order = 0; order < 2; ++order) {
                bool use_pdl = on_first ? order == 0 : order == 1;
                unsigned int sample_epoch = epoch++;
                ChainSample sample = runChainSample(stream, d_state, d_trace, &h_trace,
                    stages, kMaxStages, blocks, threads, sample_epoch, work, use_pdl);
                (use_pdl ? on : off)[(size_t)rep] = sample;
                printf("SAMPLE_TIER0_CHAIN semantics=3 tag=t01_s%d stages=%d rep=%d "
                       "order=%d mode=%s epoch=%u makespan_ms=%.6f peak_active_ctas=%d "
                       "peak_active_grids=%d early_links=%d dependency_safe_links=%d "
                       "serial_links=%d trace_complete=%d valid=%d\n",
                       stages, stages, rep, order, use_pdl ? "pdl_on" : "pdl_off",
                       sample_epoch,
                       sample.makespan_ms, sample.trace.peak_active_ctas,
                       sample.trace.peak_active_grids, sample.trace.early_links,
                       sample.trace.dependency_safe_links, sample.trace.serial_links,
                       sample.trace.complete ? 1 : 0, sample.valid ? 1 : 0);
                if (!sample.valid) {
                    cudaFree(d_state); cudaFree(d_checkpoints); cudaFree(d_trace);
                    CUDA_CHECK(cudaStreamDestroy(stream));
                    return false;
                }
                if (trace_path && stages == kMaxStages && rep == repeats - 1) {
                    if (!dumpChainTrace(trace_path, "t01_s6", rep,
                                        use_pdl ? "pdl_on" : "pdl_off",
                                        sample_epoch,
                                        h_trace, stages, blocks, trace_started)) {
                        fprintf(stderr, "failed to write Tier 0.1 trace %s\n", trace_path);
                        cudaFree(d_state); cudaFree(d_checkpoints); cudaFree(d_trace);
                        CUDA_CHECK(cudaStreamDestroy(stream));
                        return false;
                    }
                    trace_started = true;
                }
            }
        }

        std::vector<double> off_ms, on_ms, speedup, model_depth;
        std::vector<double> on_peak_ctas, on_peak_grids;
        off_ms.reserve((size_t)repeats); on_ms.reserve((size_t)repeats);
        speedup.reserve((size_t)repeats); model_depth.reserve((size_t)repeats);
        for (int rep = 0; rep < repeats; ++rep) {
            off_ms.push_back(off[(size_t)rep].makespan_ms);
            on_ms.push_back(on[(size_t)rep].makespan_ms);
            double paired = on[(size_t)rep].makespan_ms > 0.0
                          ? off[(size_t)rep].makespan_ms / on[(size_t)rep].makespan_ms : 0.0;
            speedup.push_back(paired);
            model_depth.push_back(modelImpliedChainDepth(paired, stages));
            on_peak_ctas.push_back((double)on[(size_t)rep].trace.peak_active_ctas);
            on_peak_grids.push_back((double)on[(size_t)rep].trace.peak_active_grids);
        }
        unsigned long long seed = 0xc7100000ull + (unsigned long long)stages * 0x100ull;
        ChainStats off_stats = chainBootstrapMedian(off_ms, seed + 1);
        ChainStats on_stats = chainBootstrapMedian(on_ms, seed + 2);
        ChainStats speed_stats = chainBootstrapMedian(speedup, seed + 3);
        ChainStats depth_stats = chainBootstrapMedian(model_depth, seed + 4);
        ChainStats cta_stats = chainBootstrapMedian(on_peak_ctas, seed + 5);
        ChainStats grid_stats = chainBootstrapMedian(on_peak_grids, seed + 6);
        double grid_max = *std::max_element(on_peak_grids.begin(), on_peak_grids.end());
        double cta_max = *std::max_element(on_peak_ctas.begin(), on_peak_ctas.end());

        printf("SUMMARY tier0=chain semantics=3 tag=t01_s%d stages=%d blocks=%d threads=%d "
               "sms=%d work_cycles=%llu prologue_cycles=%llu tail_cycles=%llu warmup=%d "
               "repeats=%d pairing=adjacent_alternating timer=globaltimer_makespan "
               "trace_timer=globaltimer trigger_pdl=after_value_ready "
               "poison=epoch_seed_every_invocation epoch_schedule=monotonic_all_invocations "
               "validation=independent_all_edges "
               "pdl_off_ms=%.6f pdl_off_ci_low=%.6f pdl_off_ci_high=%.6f "
               "pdl_on_ms=%.6f pdl_on_ci_low=%.6f pdl_on_ci_high=%.6f "
               "paired_speedup=%.6f paired_speedup_ci_low=%.6f "
               "paired_speedup_ci_high=%.6f model_implied_chain_depth=%.6f "
               "model_implied_chain_depth_ci_low=%.6f "
               "model_implied_chain_depth_ci_high=%.6f "
               "pdl_on_peak_active_grids_median=%.1f "
               "pdl_on_peak_active_grids_ci_low=%.1f "
               "pdl_on_peak_active_grids_ci_high=%.1f "
               "pdl_on_peak_active_grids_max=%.1f "
               "pdl_on_peak_active_ctas_median=%.1f "
               "pdl_on_peak_active_ctas_ci_low=%.1f "
               "pdl_on_peak_active_ctas_ci_high=%.1f "
               "pdl_on_peak_active_ctas_max=%.1f valid=1\n",
               stages, stages, blocks, threads, dev.sms, work, work / 2, work / 2,
               warmup, repeats,
               off_stats.median, off_stats.ci_low, off_stats.ci_high,
               on_stats.median, on_stats.ci_low, on_stats.ci_high,
               speed_stats.median, speed_stats.ci_low, speed_stats.ci_high,
               depth_stats.median, depth_stats.ci_low, depth_stats.ci_high,
               grid_stats.median, grid_stats.ci_low, grid_stats.ci_high, grid_max,
               cta_stats.median, cta_stats.ci_low, cta_stats.ci_high, cta_max);
    }

    if (trace_path)
        printf("TRACE_TIER0_CHAIN semantics=3 path=%s tag=t01_s6 rep=%d "
               "modes=pdl_off,pdl_on timer=globaltimer epoch_in_rows=1\n",
               trace_path, repeats - 1);
    cudaFree(d_state);
    cudaFree(d_checkpoints);
    cudaFree(d_trace);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return true;
}

// ---------------------------------------------------------------- 0.3 occupancy cost of waiting

// A CTA that simply occupies its slot for `wait` cycles, holding `smem` shared memory.
// Models a consumer CTA that is resident but blocked on a dependency.
extern __shared__ char g_smem[];
__global__ void waiterK(float* __restrict__ sink, unsigned long long wait, int touch_smem) {
    if (threadIdx.x == 0 && touch_smem)
        g_smem[0] = (char)blockIdx.x;   // touch smem so it is not elided
    spin_cycles(wait);
    if (threadIdx.x == 0)
        sink[blockIdx.x] = touch_smem ? (float)g_smem[0] : (float)blockIdx.x;
}

static void occupancyCost(const DeviceInfo& dev, int repeats) {
    printf("\n=== 0.3 occupancy cost of a resident-but-waiting CTA ===\n");
    printf("This is the pricing basis for B2: how much does 'wait while holding a slot' cost,\n");
    printf("as a function of the resources the waiting CTA holds.\n\n");
    printf("%-10s %-10s %14s %16s %14s\n",
           "smem_kb", "threads", "ctas_per_sm", "concurrent_ctas", "median(ms)");

    const int smemKB[]   = {0, 8, 16, 32, 64};
    const int threadSet[] = {128, 256};
    const unsigned long long wait = 1000000ull;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, (size_t)(dev.sms * 64) * sizeof(float)));

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    // CUDA defaults to a 48-KB dynamic-shared-memory launch ceiling.  The 64-KB
    // point is legal on B200/B300 but must be explicitly opted in before both
    // occupancy calculation and launch.
    CUDA_CHECK(cudaFuncSetAttribute(waiterK, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    64 * 1024));

    for (int ti = 0; ti < 2; ++ti) {
        for (int si = 0; si < 5; ++si) {
            int threads = threadSet[ti];
            size_t smem = (size_t)smemKB[si] * 1024;

            int occ = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)waiterK,
                                                          threads, smem);
            if (occ <= 0) {
                printf("%-10d %-10d %14s %16s %14s\n", smemKB[si], threads, "n/a", "n/a", "skip");
                continue;
            }
            int blocks = occ * dev.sms;   // exactly fill the machine

            std::vector<float> v;
            for (int r = 0; r < repeats + 3; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, s));
                waiterK<<<blocks, threads, smem, s>>>(d_sink, wait, smem > 0 ? 1 : 0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(e1, s));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (r >= 3) v.push_back(ms);
            }
            double med = medianOf(v);
            printf("%-10d %-10d %14d %16d %14.4f\n", smemKB[si], threads, occ, blocks, med);
            printf("SUMMARY tier0=occupancy smem_kb=%d threads=%d ctas_per_sm=%d concurrent_ctas=%d median_ms=%.5f\n",
                   smemKB[si], threads, occ, blocks, med);
        }
    }
    cudaFree(d_sink);
    CUDA_CHECK(cudaStreamDestroy(s));
}

// ---------------------------------------------------------------- 0.5 fence scope cost

__global__ void fenceK(float* __restrict__ buf, int scope, int iters) {
    float acc = 0.f;
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x == 0) buf[blockIdx.x] = (float)i;
        switch (scope) {
            case 0: __threadfence_block(); break;   // .cta
            case 1: __threadfence();       break;   // .gpu
            case 2: __threadfence_system();break;   // .sys
            default: break;                         // none
        }
        acc += buf[blockIdx.x];
    }
    if (threadIdx.x == 0 && acc == 1e30f) buf[blockIdx.x] = acc;   // keep the loop alive
}

static void fenceCost(const DeviceInfo& dev, int repeats) {
    printf("\n=== 0.5 fence scope cost ===\n");
    const char* names[] = {"cta", "gpu", "sys", "none"};
    const int iters = 2000;
    const int blocks = dev.sms, threads = 128;

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, (size_t)blocks * sizeof(float)));
    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    double base = 0.0;
    printf("%-8s %14s %16s\n", "scope", "median(ms)", "ns_per_fence");
    for (int sc = 3; sc >= 0; --sc) {   // 'none' first to get the baseline
        std::vector<float> v;
        for (int r = 0; r < repeats + 3; ++r) {
            CUDA_CHECK(cudaEventRecord(e0, s));
            fenceK<<<blocks, threads, 0, s>>>(d_buf, sc, iters);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(e1, s));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 3) v.push_back(ms);
        }
        double med = medianOf(v);
        if (sc == 3) base = med;
        double nsPer = (med - base) * 1e6 / iters;
        printf("%-8s %14.4f %16.2f\n", names[sc], med, sc == 3 ? 0.0 : nsPer);
        printf("SUMMARY tier0=fence scope=%s median_ms=%.5f ns_per_fence=%.3f\n",
               names[sc], med, sc == 3 ? 0.0 : nsPer);
    }
    cudaFree(d_buf);
    CUDA_CHECK(cudaStreamDestroy(s));
}

int main(int argc, char** argv) {
    Args A(argc, argv);
    if (A.has("--help")) {
        printf("usage: tier0_facts [--repeats N] [--warmup N] [--trace PATH] [--allow-short]\n"
               "  Runs Tier 0 base-fact probes 0.1 / 0.3 / 0.5.\n"
               "  Formal Tier 0.1 requires >=31 adjacent paired repeats; --allow-short is\n"
               "  accepted only for FAST/smoke plumbing.\n"
               "  0.2 = re-run cross_stream_PDL_survey/bench/pdl_bench on this device.\n"
               "  0.4 = ./clc_probe (needs sm_100+).\n");
        return 0;
    }
    int repeats = (int)A.ll("--repeats", 31);
    int warmup = (int)A.ll("--warmup", 3);
    bool allow_short = A.has("--allow-short");
    const char* trace_path = A.str("--trace", "tier0_chain_trace.csv");
    if (repeats <= 0 || warmup < 0) {
        fprintf(stderr, "invalid Tier 0 repeat/warmup count\n");
        return 2;
    }

    DeviceInfo dev = queryDevice();
    printDeviceBanner(dev);
    printf("SUMMARY tier0=device name=\"%s\" sms=%d cc=%d.%d ghz=%.3f\n",
           dev.name, dev.sms, dev.major, dev.minor, dev.ghz);

    if (!chainOverlap(dev, repeats, warmup, trace_path, allow_short)) return 2;
    occupancyCost(dev, repeats);
    fenceCost(dev, repeats);

    printf("\nDone. Feed the SUMMARY lines to tools/parse_summary.py.\n");
    return 0;
}
