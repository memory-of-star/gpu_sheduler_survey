// tier0_background.cu -- Tier 0.3 productive-background pricing for B2.
//
// The original tier0_facts occupancy probe only answers how many waiting CTAs fit.  It
// does not say how much useful work those resident CTAs displace.  This benchmark keeps
// the producer and productive background identical, then compares two adjacent modes:
//
//   deferred_gate : producer + an ordinarily stream-serialized waiter + background
//   resident_wait : producer + a PDL-dependent resident waiter + the same background
//
// Both modes launch the same waiter grid with the same kernel, resources and validation.
// deferred_gate omits the PSS launch attribute, so ordinary same-stream ordering delays every
// waiter until the producer has finished.  resident_wait enables PSS: the producer triggers at
// kernel entry and continues readiness work, allowing dependent CTAs to become resident and
// block in cudaGridDependencySynchronize().  Per-CTA %globaltimer records prove both sides.
//
// Every repeat poisons and validates the complete background output, producer output and
// dependent output.  The background performs a fixed number of auditable LCG updates; its
// effective throughput includes dispatch delay from the earliest producer/background activity.
//
// Register tiers use compile-time live-across-wait arrays plus __launch_bounds__.  Runtime
// output reports cudaFuncGetAttributes().numRegs/localSizeBytes, which is authoritative;
// requested tier names alone are never treated as evidence of register allocation.

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>

static constexpr int kThreads = 128;
static constexpr int kRegSeedWords = 128;
// This experiment prices the resources held by the dependent grid, not producer
// saturation.  A single-CTA predecessor keeps the PDL dependency genuinely pending while
// leaving SMs on which even the 64 KiB/high-register waiter can become resident.  Filling
// every SM with producer CTAs makes that extreme start only as producer CTAs retire, so the
// early-wait interval collapses to nondeterministic grid-tail skew.
static constexpr int kProducerBlocks = 1;

struct ProbeRecord {
    unsigned long long t_start;
    unsigned long long t_wait_enter;
    unsigned long long t_wait_exit;
    unsigned long long t_end;
    unsigned int sm_id;
    unsigned int reserved;
};
static_assert(sizeof(ProbeRecord) == 40, "ProbeRecord layout changed");

__host__ __device__ __forceinline__ unsigned int rotl32(unsigned int x, int r) {
    r &= 31;
    return r ? ((x << r) | (x >> (32 - r))) : x;
}

__host__ __device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ __forceinline__ unsigned int producerValue(unsigned int epoch,
                                                                unsigned int block) {
    return mix32(epoch ^ (0x9e3779b9u * (block + 1u)));
}

__host__ __device__ __forceinline__ unsigned int backgroundSeed(unsigned int epoch,
                                                                 unsigned int tid) {
    return mix32(epoch ^ 0xa511e9b3u ^ (0x6d2b79f5u * (tid + 1u)));
}

__global__ void readinessProducer(unsigned int* data,
                                  unsigned int epoch,
                                  unsigned long long readiness_cycles,
                                  ProbeRecord* trace) {
    ProbeRecord rec{};
    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.sm_id = ctatrace_smid();
    }

    // All producer CTAs announce launch eligibility before readiness work.  The dependent
    // grid may therefore become resident and block in griddepcontrol.wait while this work
    // is still in flight.
    cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(readiness_cycles);

    if (threadIdx.x == 0)
        data[blockIdx.x] = producerValue(epoch, blockIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        trace[blockIdx.x] = rec;
    }
}

__global__ void productiveBackground(unsigned int* out,
                                     unsigned int epoch,
                                     unsigned int iters,
                                     ProbeRecord* trace) {
    ProbeRecord rec{};
    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.sm_id = ctatrace_smid();
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = backgroundSeed(epoch, tid);
    for (unsigned int i = 0; i < iters; ++i) {
        x = x * 1664525u + 1013904223u;
        // Keep every update in the executed dependency chain.  The host validator uses a
        // closed-form jump-ahead, so validation cost is O(outputs), not O(work).
        asm volatile("" : "+r"(x));
    }
    out[tid] = x;

    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_end = ctatrace_globaltimer();
        trace[blockIdx.x] = rec;
    }
}

template <int RegWords>
__device__ __forceinline__ unsigned int heldRegisterChecksum(
        const unsigned int (&held)[RegWords]) {
    unsigned int x = 0x243f6a88u;
#pragma unroll
    for (int i = 0; i < RegWords; ++i) {
        x ^= rotl32(held[i] + 0x9e3779b9u * (unsigned int)(i + 1), (i % 31) + 1);
        x = x * 747796405u + 2891336453u;
    }
    return x;
}

template <int RegWords, int MinBlocksPerSM>
__global__ __launch_bounds__(kThreads, MinBlocksPerSM)
void dependencyWaiter(unsigned int* out,
                    const unsigned int* producer_data,
                    int nproducer,
                    const volatile unsigned int* reg_seed,
                    unsigned int epoch,
                    int touch_smem,
                    ProbeRecord* trace) {
    extern __shared__ unsigned char held_smem[];
    unsigned int held[RegWords];
    ProbeRecord rec{};

    if (threadIdx.x == 0) {
        rec.t_start = ctatrace_globaltimer();
        rec.sm_id = ctatrace_smid();
#pragma unroll
        for (int i = 0; i < RegWords; ++i)
            held[i] = reg_seed[i];       // volatile loads must occur before the wait
        if (touch_smem)
            held_smem[0] = (unsigned char)(epoch + blockIdx.x);
        rec.t_wait_enter = ctatrace_globaltimer();
        cudaGridDependencySynchronize();
    }

    // Keep thread participation symmetric: the leader performs the dependency operation,
    // then this barrier propagates its memory visibility to the complete CTA.
    __syncthreads();
    if (threadIdx.x == 0) {
        rec.t_wait_exit = ctatrace_globaltimer();
        unsigned int value = producer_data[blockIdx.x % nproducer];
        value ^= heldRegisterChecksum(held);
        if (touch_smem)
            value ^= (unsigned int)held_smem[0];
        out[blockIdx.x] = value;
        rec.t_end = ctatrace_globaltimer();
        trace[blockIdx.x] = rec;
    }
}

struct Config {
    int smem_kb = 0;
    int repeats = 31;
    int warmup = 3;
    int bg_waves = 8;
    unsigned int bg_iters = 1000000u;
    unsigned long long producer_cycles = 4000000ull;
    bool allow_short = false;
    const char* reg_tier = "low";
    const char* tag = "tier0_bg";
    const char* trace_path = nullptr;
};

struct RunContext {
    int nproducer = 0;
    int nwaiter = 0;
    int nbackground = 0;
    size_t background_threads = 0;

    unsigned int* d_producer = nullptr;
    unsigned int* d_waiter = nullptr;
    unsigned int* d_background = nullptr;
    unsigned int* d_reg_seed = nullptr;
    ProbeRecord* d_producer_trace = nullptr;
    ProbeRecord* d_waiter_trace = nullptr;
    ProbeRecord* d_background_trace = nullptr;
    cudaStream_t dependency_stream{};
    cudaStream_t background_stream{};

    std::vector<unsigned int> h_producer;
    std::vector<unsigned int> h_waiter;
    std::vector<unsigned int> h_background;
    std::vector<unsigned int> h_reg_seed;
    std::vector<ProbeRecord> h_producer_trace;
    std::vector<ProbeRecord> h_waiter_trace;
    std::vector<ProbeRecord> h_background_trace;
};

struct Sample {
    double background_active_ms = 0.0;
    double background_effective_ms = 0.0;
    double background_gupdates_s = 0.0;
    double end_to_end_ms = 0.0;
    double waiter_median_us = 0.0;
    int deferred_waiters = 0;
    int early_waiters = 0;
    int peak_waiters = 0;
    int peak_background_ctas = 0;
    bool valid = false;
};

struct Stats {
    double median = 0.0;
    double ci_low = 0.0;
    double ci_high = 0.0;
};

struct LcgJump {
    unsigned int mult;
    unsigned int plus;
};

static LcgJump lcgJump(unsigned int delta) {
    // Jump ahead in x <- a*x+c modulo 2^32.  Unsigned overflow supplies the modulus.
    unsigned int cur_mult = 1664525u;
    unsigned int cur_plus = 1013904223u;
    unsigned int acc_mult = 1u;
    unsigned int acc_plus = 0u;
    while (delta > 0) {
        if (delta & 1u) {
            acc_mult = acc_mult * cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1u) * cur_plus;
        cur_mult = cur_mult * cur_mult;
        delta >>= 1;
    }
    return {acc_mult, acc_plus};
}

template <int RegWords>
static unsigned int hostRegisterChecksum(const std::vector<unsigned int>& seed) {
    unsigned int held[RegWords];
    for (int i = 0; i < RegWords; ++i) held[i] = seed[(size_t)i];
    unsigned int x = 0x243f6a88u;
    for (int i = 0; i < RegWords; ++i) {
        x ^= rotl32(held[i] + 0x9e3779b9u * (unsigned int)(i + 1), (i % 31) + 1);
        x = x * 747796405u + 2891336453u;
    }
    return x;
}

static double medianDouble(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n & 1u) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static Stats bootstrapMedian(const std::vector<double>& values, unsigned long long seed) {
    Stats out{};
    if (values.empty()) return out;
    out.median = medianDouble(values);
    if (values.size() == 1) {
        out.ci_low = out.ci_high = out.median;
        return out;
    }

    constexpr int kBootstrap = 2000;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, values.size() - 1);
    std::vector<double> medians;
    std::vector<double> sample(values.size());
    medians.reserve(kBootstrap);
    for (int b = 0; b < kBootstrap; ++b) {
        for (size_t i = 0; i < values.size(); ++i) sample[i] = values[pick(rng)];
        medians.push_back(medianDouble(sample));
    }
    std::sort(medians.begin(), medians.end());
    out.ci_low = medians[(size_t)(0.025 * kBootstrap)];
    out.ci_high = medians[(size_t)(0.975 * kBootstrap)];
    return out;
}

static int peakConcurrent(const std::vector<std::pair<unsigned long long,
                                                       unsigned long long>>& intervals) {
    std::vector<std::pair<unsigned long long, int>> events;
    events.reserve(intervals.size() * 2);
    for (const auto& interval : intervals) {
        if (interval.second <= interval.first) continue;
        events.emplace_back(interval.first, +1);
        events.emplace_back(interval.second, -1);
    }
    // -1 sorts before +1 at an equal timestamp, giving half-open [start,end) intervals.
    std::sort(events.begin(), events.end());
    int active = 0, peak = 0;
    for (const auto& event : events) {
        active += event.second;
        if (active > peak) peak = active;
    }
    return peak;
}

static bool initContext(RunContext* ctx, int producer_blocks, int background_sms,
                        int waiter_blocks, int bg_waves) {
    ctx->nproducer = producer_blocks;
    ctx->nwaiter = waiter_blocks;
    ctx->nbackground = background_sms * bg_waves;
    ctx->background_threads = (size_t)ctx->nbackground * kThreads;

    CUDA_CHECK(cudaMalloc(&ctx->d_producer, (size_t)ctx->nproducer * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_waiter, (size_t)ctx->nwaiter * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_background,
                          ctx->background_threads * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_reg_seed, kRegSeedWords * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_producer_trace,
                          (size_t)ctx->nproducer * sizeof(ProbeRecord)));
    CUDA_CHECK(cudaMalloc(&ctx->d_waiter_trace,
                          (size_t)ctx->nwaiter * sizeof(ProbeRecord)));
    CUDA_CHECK(cudaMalloc(&ctx->d_background_trace,
                          (size_t)ctx->nbackground * sizeof(ProbeRecord)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->dependency_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx->background_stream, cudaStreamNonBlocking));

    ctx->h_reg_seed.resize(kRegSeedWords);
    for (int i = 0; i < kRegSeedWords; ++i)
        ctx->h_reg_seed[(size_t)i] = mix32(0x6a09e667u ^ (unsigned int)i);
    CUDA_CHECK(cudaMemcpy(ctx->d_reg_seed, ctx->h_reg_seed.data(),
                          kRegSeedWords * sizeof(unsigned int), cudaMemcpyHostToDevice));

    ctx->h_producer.resize((size_t)ctx->nproducer);
    ctx->h_waiter.resize((size_t)ctx->nwaiter);
    ctx->h_background.resize(ctx->background_threads);
    ctx->h_producer_trace.resize((size_t)ctx->nproducer);
    ctx->h_waiter_trace.resize((size_t)ctx->nwaiter);
    ctx->h_background_trace.resize((size_t)ctx->nbackground);
    return true;
}

static void freeContext(RunContext* ctx) {
    cudaFree(ctx->d_producer);
    cudaFree(ctx->d_waiter);
    cudaFree(ctx->d_background);
    cudaFree(ctx->d_reg_seed);
    cudaFree(ctx->d_producer_trace);
    cudaFree(ctx->d_waiter_trace);
    cudaFree(ctx->d_background_trace);
    cudaStreamDestroy(ctx->dependency_stream);
    cudaStreamDestroy(ctx->background_stream);
}

static bool dumpTrace(const char* path, const char* tag, const char* mode,
                      const RunContext& ctx, bool append) {
    FILE* f = fopen(path, append ? "a" : "w");
    if (!f) return false;
    if (!append)
        fprintf(f, "tag,mode,kind,block_id,sm_id,t_start,t_wait_enter,t_wait_exit,t_end\n");
    auto emit = [&](const char* kind, const std::vector<ProbeRecord>& rows) {
        for (size_t i = 0; i < rows.size(); ++i) {
            const ProbeRecord& r = rows[i];
            if (!r.t_end) continue;
            fprintf(f, "%s,%s,%s,%zu,%u,%llu,%llu,%llu,%llu\n", tag, mode, kind, i,
                    r.sm_id, r.t_start, r.t_wait_enter, r.t_wait_exit, r.t_end);
        }
    };
    emit("producer", ctx.h_producer_trace);
    emit("waiter", ctx.h_waiter_trace);
    emit("background", ctx.h_background_trace);
    fclose(f);
    return true;
}

template <int RegWords, int MinBlocksPerSM>
static Sample runOnce(const Config& cfg, RunContext& ctx, bool resident,
                      unsigned int epoch, const char* trace_path, bool trace_append) {
    Sample sample{};
    const size_t smem_bytes = (size_t)cfg.smem_kb * 1024;

    CUDA_CHECK(cudaMemsetAsync(ctx.d_producer, 0xa5,
                              (size_t)ctx.nproducer * sizeof(unsigned int),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_waiter, 0xa5,
                              (size_t)ctx.nwaiter * sizeof(unsigned int),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_producer_trace, 0,
                              (size_t)ctx.nproducer * sizeof(ProbeRecord),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_waiter_trace, 0,
                              (size_t)ctx.nwaiter * sizeof(ProbeRecord),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_background, 0xa5,
                              ctx.background_threads * sizeof(unsigned int),
                              ctx.background_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_background_trace, 0,
                              (size_t)ctx.nbackground * sizeof(ProbeRecord),
                              ctx.background_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.dependency_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.background_stream));

    // Enqueue the dependency path first.  This gives eligible waiters the chance to become
    // resident before the independent background is dispatched.  The trace, not enqueue
    // order, is what decides whether a sample contained a real resident wait.
    readinessProducer<<<ctx.nproducer, kThreads, 0, ctx.dependency_stream>>>(
        ctx.d_producer, epoch, cfg.producer_cycles, ctx.d_producer_trace);
    CUDA_CHECK(cudaGetLastError());

    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(ctx.nwaiter);
    launch.blockDim = dim3(kThreads);
    launch.dynamicSmemBytes = smem_bytes;
    launch.stream = ctx.dependency_stream;
    cudaLaunchAttribute attr{};
    if (resident) {
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        launch.attrs = &attr;
        launch.numAttrs = 1;
    }
    // With no PSS attribute, ordinary same-stream serialization makes this the deferred-gate
    // control.  The host launch path and every kernel argument remain identical.
    CUDA_CHECK(cudaLaunchKernelEx(
        &launch, dependencyWaiter<RegWords, MinBlocksPerSM>, ctx.d_waiter,
        (const unsigned int*)ctx.d_producer, ctx.nproducer,
        (const volatile unsigned int*)ctx.d_reg_seed, epoch,
        smem_bytes ? 1 : 0, ctx.d_waiter_trace));

    productiveBackground<<<ctx.nbackground, kThreads, 0, ctx.background_stream>>>(
        ctx.d_background, epoch, cfg.bg_iters, ctx.d_background_trace);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(ctx.dependency_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.background_stream));

    CUDA_CHECK(cudaMemcpy(ctx.h_producer.data(), ctx.d_producer,
                          (size_t)ctx.nproducer * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_background.data(), ctx.d_background,
                          ctx.background_threads * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_producer_trace.data(), ctx.d_producer_trace,
                          (size_t)ctx.nproducer * sizeof(ProbeRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_background_trace.data(), ctx.d_background_trace,
                          (size_t)ctx.nbackground * sizeof(ProbeRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_waiter.data(), ctx.d_waiter,
                          (size_t)ctx.nwaiter * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx.h_waiter_trace.data(), ctx.d_waiter_trace,
                          (size_t)ctx.nwaiter * sizeof(ProbeRecord),
                          cudaMemcpyDeviceToHost));

    bool valid = true;
    for (int i = 0; i < ctx.nproducer; ++i) {
        if (ctx.h_producer[(size_t)i] != producerValue(epoch, (unsigned int)i)) {
            fprintf(stderr, "producer validation failed block=%d\n", i);
            valid = false;
            break;
        }
    }
    LcgJump jump = lcgJump(cfg.bg_iters);
    for (size_t tid = 0; tid < ctx.background_threads; ++tid) {
        unsigned int expect = jump.mult * backgroundSeed(epoch, (unsigned int)tid)
                            + jump.plus;
        if (ctx.h_background[tid] != expect) {
            fprintf(stderr, "background validation failed tid=%zu got=%u expect=%u\n",
                    tid, ctx.h_background[tid], expect);
            valid = false;
            break;
        }
    }

    unsigned long long producer_start = ~0ull, producer_end = 0;
    for (const ProbeRecord& r : ctx.h_producer_trace) {
        if (!r.t_start || !r.t_end || r.t_end <= r.t_start) valid = false;
        producer_start = std::min(producer_start, r.t_start);
        producer_end = std::max(producer_end, r.t_end);
    }
    unsigned long long bg_start = ~0ull, bg_end = 0;
    std::vector<std::pair<unsigned long long, unsigned long long>> bg_intervals;
    for (const ProbeRecord& r : ctx.h_background_trace) {
        if (!r.t_start || !r.t_end || r.t_end <= r.t_start) valid = false;
        bg_start = std::min(bg_start, r.t_start);
        bg_end = std::max(bg_end, r.t_end);
        bg_intervals.emplace_back(r.t_start, r.t_end);
    }
    sample.peak_background_ctas = peakConcurrent(bg_intervals);

    unsigned long long anchor = std::min(producer_start, bg_start);
    unsigned long long composite_end = std::max(producer_end, bg_end);
    if (!producer_end || !bg_end || bg_end <= anchor) valid = false;
    sample.background_active_ms = (double)(bg_end - bg_start) / 1.0e6;
    sample.background_effective_ms = (double)(bg_end - anchor) / 1.0e6;
    sample.end_to_end_ms = (double)(composite_end - anchor) / 1.0e6;
    double updates = (double)ctx.background_threads * (double)cfg.bg_iters;
    sample.background_gupdates_s = updates / (double)(bg_end - anchor);

    unsigned int reg_checksum = hostRegisterChecksum<RegWords>(ctx.h_reg_seed);
    std::vector<double> wait_us;
    std::vector<std::pair<unsigned long long, unsigned long long>> wait_intervals;
    unsigned long long waiter_end = 0;
    for (int i = 0; i < ctx.nwaiter; ++i) {
        const ProbeRecord& r = ctx.h_waiter_trace[(size_t)i];
        unsigned int expect = producerValue(epoch, (unsigned int)(i % ctx.nproducer));
        expect ^= reg_checksum;
        if (smem_bytes) expect ^= (unsigned int)(unsigned char)(epoch + (unsigned int)i);
        if (ctx.h_waiter[(size_t)i] != expect || !r.t_start || !r.t_wait_enter ||
            !r.t_wait_exit || !r.t_end || r.t_wait_exit < r.t_wait_enter) {
            fprintf(stderr, "waiter validation failed mode=%s block=%d\n",
                    resident ? "resident_wait" : "deferred_gate", i);
            valid = false;
            break;
        }
        waiter_end = std::max(waiter_end, r.t_end);
        if (resident) {
            if (r.t_wait_enter < producer_end) {
                ++sample.early_waiters;
                if (r.t_wait_exit < producer_end) {
                    fprintf(stderr,
                            "dependency violation block=%d exit=%llu producer_end=%llu\n",
                            i, r.t_wait_exit, producer_end);
                    valid = false;
                    break;
                }
                wait_us.push_back((double)(r.t_wait_exit - r.t_wait_enter) / 1000.0);
                wait_intervals.emplace_back(r.t_wait_enter, r.t_wait_exit);
            }
        } else {
            if (r.t_wait_enter < producer_end) {
                fprintf(stderr,
                        "deferred gate started early block=%d enter=%llu producer_end=%llu\n",
                        i, r.t_wait_enter, producer_end);
                valid = false;
                break;
            }
            ++sample.deferred_waiters;
            wait_us.push_back((double)(r.t_wait_exit - r.t_wait_enter) / 1000.0);
        }
    }
    sample.peak_waiters = peakConcurrent(wait_intervals);
    sample.waiter_median_us = medianDouble(wait_us);
    composite_end = std::max(composite_end, waiter_end);
    sample.end_to_end_ms = (double)(composite_end - anchor) / 1.0e6;
    if (resident && (sample.early_waiters == 0 || sample.peak_waiters == 0)) {
        fprintf(stderr, "no waiter entered before producer end; sample is inadmissible\n");
        valid = false;
    }
    if (!resident && sample.deferred_waiters != ctx.nwaiter) {
        fprintf(stderr, "deferred gate proof incomplete got=%d expected=%d\n",
                sample.deferred_waiters, ctx.nwaiter);
        valid = false;
    }

    sample.valid = valid;
    if (trace_path) {
        const char* mode = resident ? "resident_wait" : "deferred_gate";
        if (!dumpTrace(trace_path, cfg.tag, mode, ctx, trace_append)) {
            fprintf(stderr, "failed to write trace %s\n", trace_path);
            sample.valid = false;
        } else {
            printf("TRACE tier0=background path=%s mode=%s "
                   "semantics=paired_wait_enter_to_wait_exit\n", trace_path, mode);
        }
    }
    return sample;
}

template <int RegWords, int MinBlocksPerSM>
static int runConfig(const Config& cfg, const DeviceInfo& dev) {
    const size_t smem_bytes = (size_t)cfg.smem_kb * 1024;
    int optin_max = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                      dev.dev));
    if ((int)smem_bytes > optin_max) {
        fprintf(stderr, "requested smem=%zu exceeds device opt-in maximum=%d\n",
                smem_bytes, optin_max);
        return 2;
    }

    // Launch hygiene: opt in before both the occupancy query and every launch above the
    // default dynamic-smem ceiling.  The kernel never touches dynamic smem on smem=0.
    CUDA_CHECK(cudaFuncSetAttribute(dependencyWaiter<RegWords, MinBlocksPerSM>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    64 * 1024));
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, (const void*)dependencyWaiter<RegWords, MinBlocksPerSM>,
        kThreads, smem_bytes));
    if (occupancy <= 0) {
        fprintf(stderr, "waiter occupancy is zero\n");
        return 2;
    }

    cudaFuncAttributes attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, dependencyWaiter<RegWords, MinBlocksPerSM>));
    int waiter_blocks = occupancy * dev.sms;
    printf("RESOURCE tier0=background reg_tier=%s requested_reg_words=%d "
           "launch_bounds_min_blocks=%d actual_num_regs=%d local_bytes=%zu "
           "static_smem_bytes=%zu "
           "smem_kb=%d occ_per_sm=%d waiter_blocks=%d\n",
           cfg.reg_tier, RegWords, MinBlocksPerSM, attr.numRegs, attr.localSizeBytes,
           attr.sharedSizeBytes, cfg.smem_kb, occupancy, waiter_blocks);
    printf("SEMANTICS tier0=background trigger=producer_entry wait=griddepcontrol "
           "producer_role=single_cta_dependency_holder control=deferred_gate "
           "test=resident_wait same_waiter_kernel=1 poison=every_repeat "
           "validation=all_outputs timer=globaltimer "
           "anchor=earliest_producer_or_background_activity\n");

    RunContext ctx;
    initContext(&ctx, kProducerBlocks, dev.sms, waiter_blocks, cfg.bg_waves);
    bool all_valid = true;
    unsigned int epoch = 1u;

    for (int w = 0; w < cfg.warmup; ++w) {
        Sample a = runOnce<RegWords, MinBlocksPerSM>(cfg, ctx, false, epoch++, nullptr, false);
        Sample b = runOnce<RegWords, MinBlocksPerSM>(cfg, ctx, true, epoch++, nullptr, false);
        if (!a.valid || !b.valid) {
            all_valid = false;
            break;
        }
    }

    std::vector<Sample> controls, residents;
    controls.reserve((size_t)cfg.repeats);
    residents.reserve((size_t)cfg.repeats);
    for (int rep = 0; rep < cfg.repeats && all_valid; ++rep) {
        Sample control, resident;
        const bool resident_first = (rep & 1) != 0;
        const char* trace = (cfg.trace_path && rep == cfg.repeats - 1)
                          ? cfg.trace_path : nullptr;
        if (resident_first) {
            resident = runOnce<RegWords, MinBlocksPerSM>(
                cfg, ctx, true, epoch++, trace, false);
            control = runOnce<RegWords, MinBlocksPerSM>(
                cfg, ctx, false, epoch++, trace, trace != nullptr);
        } else {
            control = runOnce<RegWords, MinBlocksPerSM>(
                cfg, ctx, false, epoch++, trace, false);
            resident = runOnce<RegWords, MinBlocksPerSM>(
                cfg, ctx, true, epoch++, trace, trace != nullptr);
        }
        all_valid = control.valid && resident.valid;
        controls.push_back(control);
        residents.push_back(resident);
        printf("SAMPLE_TIER0_BG semantics=2 tag=%s rep=%d mode=deferred_gate "
               "bg_active_ms=%.6f bg_effective_ms=%.6f bg_gupdates_s=%.6f "
               "e2e_ms=%.6f wait_median_us=%.3f deferred_waiters=%d "
               "early_waiters=%d peak_waiters=%d bg_peak_ctas=%d valid=%d\n",
               cfg.tag, rep, control.background_active_ms,
               control.background_effective_ms, control.background_gupdates_s,
               control.end_to_end_ms, control.waiter_median_us,
               control.deferred_waiters, control.early_waiters, control.peak_waiters,
               control.peak_background_ctas,
               control.valid ? 1 : 0);
        printf("SAMPLE_TIER0_BG semantics=2 tag=%s rep=%d mode=resident_wait "
               "bg_active_ms=%.6f bg_effective_ms=%.6f bg_gupdates_s=%.6f "
               "e2e_ms=%.6f wait_median_us=%.3f early_waiters=%d "
               "peak_waiters=%d bg_peak_ctas=%d valid=%d\n",
               cfg.tag, rep, resident.background_active_ms,
               resident.background_effective_ms, resident.background_gupdates_s,
               resident.end_to_end_ms, resident.waiter_median_us,
               resident.early_waiters, resident.peak_waiters,
               resident.peak_background_ctas, resident.valid ? 1 : 0);
    }

    std::vector<double> control_throughput, resident_throughput;
    std::vector<double> control_active, resident_active;
    std::vector<double> control_bg_peak, resident_bg_peak;
    std::vector<double> control_e2e, resident_e2e, e2e_delta;
    std::vector<double> control_wait_us, resident_wait_us, loss_pct;
    std::vector<double> deferred_waiters, early_waiters, peak_waiters;
    for (size_t i = 0; i < controls.size(); ++i) {
        control_throughput.push_back(controls[i].background_gupdates_s);
        resident_throughput.push_back(residents[i].background_gupdates_s);
        control_active.push_back(controls[i].background_active_ms);
        resident_active.push_back(residents[i].background_active_ms);
        control_bg_peak.push_back((double)controls[i].peak_background_ctas);
        resident_bg_peak.push_back((double)residents[i].peak_background_ctas);
        control_e2e.push_back(controls[i].end_to_end_ms);
        resident_e2e.push_back(residents[i].end_to_end_ms);
        e2e_delta.push_back(residents[i].end_to_end_ms - controls[i].end_to_end_ms);
        control_wait_us.push_back(controls[i].waiter_median_us);
        resident_wait_us.push_back(residents[i].waiter_median_us);
        deferred_waiters.push_back((double)controls[i].deferred_waiters);
        early_waiters.push_back((double)residents[i].early_waiters);
        peak_waiters.push_back((double)residents[i].peak_waiters);
        double base = controls[i].background_gupdates_s;
        loss_pct.push_back(base > 0.0
            ? 100.0 * (base - residents[i].background_gupdates_s) / base : 0.0);
    }

    Stats ct = bootstrapMedian(control_throughput, 0x1001u + RegWords + cfg.smem_kb);
    Stats rt = bootstrapMedian(resident_throughput, 0x2001u + RegWords + cfg.smem_kb);
    Stats ca = bootstrapMedian(control_active, 0x3001u + RegWords + cfg.smem_kb);
    Stats ra = bootstrapMedian(resident_active, 0x4001u + RegWords + cfg.smem_kb);
    Stats cb = bootstrapMedian(control_bg_peak, 0x5001u + RegWords + cfg.smem_kb);
    Stats rb = bootstrapMedian(resident_bg_peak, 0x6001u + RegWords + cfg.smem_kb);
    Stats ce = bootstrapMedian(control_e2e, 0x7001u + RegWords + cfg.smem_kb);
    Stats re = bootstrapMedian(resident_e2e, 0x8001u + RegWords + cfg.smem_kb);
    Stats ed = bootstrapMedian(e2e_delta, 0x9001u + RegWords + cfg.smem_kb);
    Stats cw = bootstrapMedian(control_wait_us, 0xa001u + RegWords + cfg.smem_kb);
    Stats rw = bootstrapMedian(resident_wait_us, 0xb001u + RegWords + cfg.smem_kb);
    Stats lp = bootstrapMedian(loss_pct, 0xc001u + RegWords + cfg.smem_kb);
    Stats dw = bootstrapMedian(deferred_waiters, 0xd001u + RegWords + cfg.smem_kb);
    Stats ew = bootstrapMedian(early_waiters, 0xe001u + RegWords + cfg.smem_kb);
    Stats pw = bootstrapMedian(peak_waiters, 0xf001u + RegWords + cfg.smem_kb);

    printf("SUMMARY tier0=background semantics=2 tag=%s smem_kb=%d reg_tier=%s "
           "requested_reg_words=%d launch_bounds_min_blocks=%d actual_num_regs=%d "
           "local_bytes=%zu static_smem_bytes=%zu occ_per_sm=%d sms=%d waiter_blocks=%d "
           "producer_blocks=%d bg_blocks=%d threads_per_block=%d bg_total_threads=%zu "
           "bg_waves=%d bg_iters=%u producer_cycles=%llu repeats=%zu trigger=entry "
           "wait=griddepcontrol control_mode=deferred_gate resident_mode=resident_wait "
           "anchor=earliest_producer_or_background_activity",
           cfg.tag, cfg.smem_kb, cfg.reg_tier, RegWords, MinBlocksPerSM,
           attr.numRegs, attr.localSizeBytes, attr.sharedSizeBytes,
           occupancy, dev.sms, waiter_blocks, ctx.nproducer, ctx.nbackground,
           kThreads, ctx.background_threads, cfg.bg_waves, cfg.bg_iters,
           cfg.producer_cycles, controls.size());
    printf(" control_gupdates_s=%.6f control_gupdates_ci_low=%.6f "
           "control_gupdates_ci_high=%.6f resident_gupdates_s=%.6f "
           "resident_gupdates_ci_low=%.6f resident_gupdates_ci_high=%.6f "
           "throughput_loss_pct=%.4f throughput_loss_ci_low=%.4f "
           "throughput_loss_ci_high=%.4f",
           ct.median, ct.ci_low, ct.ci_high, rt.median, rt.ci_low, rt.ci_high,
           lp.median, lp.ci_low, lp.ci_high);
    printf(" control_bg_active_ms=%.6f control_bg_active_ci_low=%.6f "
           "control_bg_active_ci_high=%.6f resident_bg_active_ms=%.6f "
           "resident_bg_active_ci_low=%.6f resident_bg_active_ci_high=%.6f "
           "control_bg_peak_ctas_median=%.1f control_bg_peak_ctas_ci_low=%.1f "
           "control_bg_peak_ctas_ci_high=%.1f resident_bg_peak_ctas_median=%.1f "
           "resident_bg_peak_ctas_ci_low=%.1f resident_bg_peak_ctas_ci_high=%.1f",
           ca.median, ca.ci_low, ca.ci_high, ra.median, ra.ci_low, ra.ci_high,
           cb.median, cb.ci_low, cb.ci_high, rb.median, rb.ci_low, rb.ci_high);
    printf(" control_e2e_ms=%.6f "
           "control_e2e_ci_low=%.6f control_e2e_ci_high=%.6f "
           "resident_e2e_ms=%.6f resident_e2e_ci_low=%.6f "
           "resident_e2e_ci_high=%.6f e2e_delta_ms=%.6f "
           "e2e_delta_ci_low=%.6f e2e_delta_ci_high=%.6f",
           ce.median, ce.ci_low, ce.ci_high, re.median, re.ci_low, re.ci_high,
           ed.median, ed.ci_low, ed.ci_high);
    printf(" control_wait_median_us=%.3f control_wait_ci_low=%.3f "
           "control_wait_ci_high=%.3f wait_median_us=%.3f wait_ci_low=%.3f "
           "wait_ci_high=%.3f deferred_waiters_median=%.1f "
           "deferred_waiters_ci_low=%.1f deferred_waiters_ci_high=%.1f "
           "early_waiters_median=%.1f early_waiters_ci_low=%.1f "
           "early_waiters_ci_high=%.1f peak_waiters_median=%.1f "
           "peak_waiters_ci_low=%.1f peak_waiters_ci_high=%.1f valid=%d\n",
           cw.median, cw.ci_low, cw.ci_high, rw.median, rw.ci_low, rw.ci_high,
           dw.median, dw.ci_low, dw.ci_high, ew.median, ew.ci_low, ew.ci_high,
           pw.median, pw.ci_low, pw.ci_high,
           all_valid ? 1 : 0);

    freeContext(&ctx);
    return all_valid ? 0 : 1;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf(
            "usage: tier0_background [options]\n"
            "  --smem-kb K          0|8|16|32|64 (default 0)\n"
            "  --reg-tier T         low|mid|high (default low)\n"
            "  --repeats N          timed paired repeats (default 31)\n"
            "  --warmup N           paired warmups (default 3)\n"
            "  --bg-waves N         background grid as a nominal SM multiple; not observed scheduling waves (default 8)\n"
            "  --bg-iters N         LCG updates per background thread (default 1000000)\n"
            "  --producer-cycles N  readiness delay after entry trigger (default 4000000)\n"
            "  --tag STR            SUMMARY/SAMPLE tag\n"
            "  --trace PATH         dump final paired deferred/resident %%globaltimer trace\n"
            "  --allow-short        permit repeats <31 for FAST/smoke only\n");
        return 0;
    }

    Config cfg;
    cfg.smem_kb = (int)args.ll("--smem-kb", cfg.smem_kb);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.warmup = (int)args.ll("--warmup", cfg.warmup);
    cfg.bg_waves = (int)args.ll("--bg-waves", cfg.bg_waves);
    cfg.bg_iters = (unsigned int)args.ll("--bg-iters", cfg.bg_iters);
    cfg.producer_cycles = (unsigned long long)args.ll("--producer-cycles",
                                                       cfg.producer_cycles);
    cfg.allow_short = args.has("--allow-short");
    cfg.reg_tier = args.str("--reg-tier", cfg.reg_tier);
    cfg.tag = args.str("--tag", cfg.tag);
    cfg.trace_path = args.str("--trace", nullptr);

    bool smem_ok = cfg.smem_kb == 0 || cfg.smem_kb == 8 || cfg.smem_kb == 16 ||
                   cfg.smem_kb == 32 || cfg.smem_kb == 64;
    if (!smem_ok || cfg.repeats <= 0 || cfg.warmup < 0 || cfg.bg_waves <= 0 ||
        cfg.bg_iters == 0 || cfg.producer_cycles == 0) {
        fprintf(stderr, "invalid tier0_background configuration\n");
        return 2;
    }
    if (cfg.repeats < 31 && !cfg.allow_short) {
        fprintf(stderr,
                "formal Tier 0.3 requires >=31 repeats; use --allow-short only for FAST/smoke\n");
        return 2;
    }

    DeviceInfo dev = queryDevice();
    printDeviceBanner(dev);
    printf("CONFIG tier0=background device=\"%s\" sms=%d cc=%d.%d producer_blocks=%d smem_kb=%d "
           "reg_tier=%s repeats=%d warmup=%d bg_waves=%d bg_iters=%u "
           "producer_cycles=%llu\n",
           dev.name, dev.sms, dev.major, dev.minor, kProducerBlocks, cfg.smem_kb, cfg.reg_tier,
           cfg.repeats, cfg.warmup, cfg.bg_waves, cfg.bg_iters, cfg.producer_cycles);

    // Requested arrays are intentionally far enough apart to create auditable resource
    // tiers.  Runtime numRegs/localSizeBytes in RESOURCE/SUMMARY are the actual evidence.
    if (strcmp(cfg.reg_tier, "low") == 0)
        return runConfig<8, 8>(cfg, dev);
    if (strcmp(cfg.reg_tier, "mid") == 0)
        return runConfig<40, 4>(cfg, dev);
    if (strcmp(cfg.reg_tier, "high") == 0)
        return runConfig<80, 2>(cfg, dev);
    fprintf(stderr, "unknown --reg-tier %s (expected low|mid|high)\n", cfg.reg_tier);
    return 2;
}
