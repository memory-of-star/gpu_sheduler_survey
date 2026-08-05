// tier23_clc_scheduler.cu -- CLC-backed persistent scheduler for plan §7.6.
//
// Each physical CTA owns one launch token and uses clusterlaunchcontrol.try_cancel to take
// pending one-CTA clusters.  Running CTAs then spend exactly those tokens on a software task
// DAG containing N producer and N one-to-one consumer tiles.  The three safe policies differ
// only in task selection: producer priority, readiness-aware consumer priority, or locality
// first.  An unsafe consumer-first/no-readiness mode is retained solely as the required wrong
// Ceiling control.

#include "common/tier23_native.cuh"

#include <cuda/atomic>
#include <climits>
#include <fstream>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define T23_CLC_SUPPORTED 1
#else
#define T23_CLC_SUPPORTED 0
#endif

#if T23_CLC_SUPPORTED
__device__ __forceinline__ void t23_clc_try_cancel(void* response, void* barrier) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 "
        "[%0], [%1];"
        :: "l"(__cvta_generic_to_shared(response)),
           "l"(__cvta_generic_to_shared(barrier)) : "memory");
}

__device__ __forceinline__ bool t23_clc_is_canceled(const void* response) {
    unsigned value;
    asm volatile(
        "{\n\t.reg .b128 r;\n\t.reg .pred p;\n\t"
        "ld.shared.b128 r, [%1];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(value)
        : "l"(__cvta_generic_to_shared(const_cast<void*>(response))) : "memory");
    return value != 0;
}

__device__ __forceinline__ unsigned t23_clc_canceled_x(const void* response) {
    unsigned value;
    asm volatile(
        "{\n\t.reg .b128 r;\n\t"
        "ld.shared.b128 r, [%1];\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, r;\n\t}"
        : "=r"(value)
        : "l"(__cvta_generic_to_shared(const_cast<void*>(response))) : "memory");
    return value;
}

__device__ __forceinline__ void t23_mbar_init(void* barrier) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "l"(__cvta_generic_to_shared(barrier)) : "memory");
}

__device__ __forceinline__ void t23_mbar_expect(void* barrier) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;"
                 :: "l"(__cvta_generic_to_shared(barrier)) : "memory");
}

__device__ __forceinline__ void t23_mbar_wait(void* barrier, unsigned phase) {
    asm volatile(
        "{\n\t.reg .pred p;\n\tt23ClcWaitLoop:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra t23ClcWaitLoop;\n\t}"
        :: "l"(__cvta_generic_to_shared(barrier)), "r"(phase) : "memory");
}
#endif

enum CLCSchedule {
    CLC_PRODUCER_PRIORITY = 0,
    CLC_CONSUMER_PRIORITY = 1,
    CLC_LOCALITY = 2,
    CLC_NONE = 3,
};

static const char* clc_schedule_name(int mode) {
    switch (mode) {
        case CLC_PRODUCER_PRIORITY: return "producer-priority";
        case CLC_CONSUMER_PRIORITY: return "consumer-priority";
        case CLC_LOCALITY: return "locality";
        case CLC_NONE: return "none";
        default: return "?";
    }
}

__host__ __device__ __forceinline__ unsigned long long clc_consumer_value(
        unsigned long long epoch, unsigned int tile) {
    return t23_mix64(t23_value(epoch, tile, 0) ^ 0xc6bc279692b5cc83ull);
}

__global__ void clc_init(unsigned long long* producer,
                         unsigned long long* consumer,
                         unsigned long long* ready,
                         int* consumer_claim,
                         unsigned int* canceled,
                         unsigned int* executed,
                         T23TraceRecord* trace,
                         int tiles,
                         int launch_blocks,
                         unsigned long long epoch) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < tiles) {
        producer[i] = t23_mix64(epoch ^ (unsigned)i ^ 0xfeedfacecafebeefull);
        consumer[i] = t23_mix64(epoch ^ (unsigned)i ^ 0x123456789abcdef0ull);
        ready[i] = 0;
        consumer_claim[i] = 0;
    }
    if (i < launch_blocks) {
        canceled[i] = 0;
        executed[i] = 0;
    }
    if (i < 2 * tiles) trace[i] = T23TraceRecord{};
}

struct CLCDeviceState {
    unsigned long long* producer;
    unsigned long long* consumer;
    unsigned long long* ready;
    int* consumer_claim;
    int* next_producer;
    int* next_consumer;
    unsigned int* canceled;
    unsigned int* executed;
    unsigned long long* kernel_start;
    unsigned long long* kernel_end;
    unsigned long long* clc_cycles;
    unsigned long long* wait_polls;
    unsigned int* clc_attempts;
    unsigned int* clc_successes;
    unsigned int* tokens_processed;
    unsigned int* locality_hits;
    unsigned int* errors;
    T23TraceRecord* trace;
};

__device__ __forceinline__ void clc_run_producer(CLCDeviceState s, int tile,
                                                 int mode, unsigned long long epoch,
                                                 unsigned long long work_cycles) {
    T23TraceRecord* r = s.trace + tile;
    if (threadIdx.x == 0) {
        r->t_start = ctatrace_globaltimer();
        r->t_wait_begin = r->t_start;
        r->t_dep = r->t_start;
        r->block_id = (unsigned)tile;
        r->kernel_id = 0;
        r->sm_id = ctatrace_smid();
        r->aux = (unsigned)blockIdx.x;
    }
    __syncthreads();
    spin_cycles(work_cycles);
    if (threadIdx.x == 0) {
        s.producer[tile] = t23_value(epoch, (unsigned)tile, 0);
        r->t_ready = ctatrace_globaltimer();
        if (mode != CLC_NONE) {
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> f(s.ready[tile]);
            f.store(epoch, cuda::memory_order_release);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) r->t_end = ctatrace_globaltimer();
}

__device__ __forceinline__ void clc_consume_task(CLCDeviceState s, int tile, int tiles,
                                                 int mode, unsigned long long epoch,
                                                 unsigned long long work_cycles) {
    T23TraceRecord* r = s.trace + tiles + tile;
    unsigned long long observed = 0, polls = 0;
    if (threadIdx.x == 0) {
        r->t_start = ctatrace_globaltimer();
        r->t_wait_begin = r->t_start;
        r->block_id = (unsigned)tile;
        r->kernel_id = 1;
        r->sm_id = ctatrace_smid();
        r->aux = (unsigned)blockIdx.x;
        if (mode != CLC_NONE) {
            cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> f(s.ready[tile]);
            unsigned int ns = 64;
            for (;;) {
                ++polls;
                if (f.load(cuda::memory_order_acquire) >= epoch) break;
                __nanosleep(ns);
                ns = ns < 2048 ? ns * 2 : 2048;
            }
        }
        r->t_dep = ctatrace_globaltimer();
        r->t_ready = r->t_dep;
        r->poll_loads = polls;
        observed = s.producer[tile]; // RAW capture immediately after dependency point
        atomicAdd(s.wait_polls, polls);
    }
    __syncthreads();
    spin_cycles(work_cycles);
    if (threadIdx.x == 0) {
        s.consumer[tile] = t23_mix64(observed ^ 0xc6bc279692b5cc83ull);
        r->t_end = ctatrace_globaltimer();
    }
    __syncthreads();
}

__device__ __forceinline__ bool clc_ready(CLCDeviceState s, int tile,
                                          unsigned long long epoch) {
    cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> f(s.ready[tile]);
    return f.load(cuda::memory_order_acquire) >= epoch;
}

__global__ void clc_scheduler(CLCDeviceState state,
                              int tiles,
                              int mode,
                              unsigned long long epoch,
                              unsigned long long producer_cycles,
                              unsigned long long consumer_cycles) {
#if T23_CLC_SUPPORTED
    __shared__ alignas(16) char response[16];
    __shared__ alignas(8) char barrier[8];
    __shared__ unsigned int token_count;
    __shared__ int selected_kind;
    __shared__ int selected_tile;
    __shared__ int local_tile;

    if (threadIdx.x == 0) {
        token_count = 1; // this physically executing CTA owns its launch token
        local_tile = -1;
        state.executed[blockIdx.x] = 1;
        atomicMin(state.kernel_start, ctatrace_globaltimer());
        t23_mbar_init(barrier);
    }
    __syncthreads();

    unsigned phase = 0;
    for (;;) {
        bool got = false;
        unsigned canceled_id = 0;
        unsigned long long cycles = 0;
        if (threadIdx.x == 0) {
            unsigned long long begin = clock64();
            t23_mbar_expect(barrier);
            t23_clc_try_cancel(response, barrier);
            t23_mbar_wait(barrier, phase);
            phase ^= 1;
            cycles = (unsigned long long)(clock64() - begin);
            got = t23_clc_is_canceled(response);
            if (got) canceled_id = t23_clc_canceled_x(response);
            atomicAdd(state.clc_cycles, cycles);
            atomicAdd(state.clc_attempts, 1u);
            if (got) {
                atomicAdd(state.clc_successes, 1u);
                atomicAdd(&token_count, 1u);
                if (canceled_id < (unsigned)gridDim.x)
                    atomicAdd(&state.canceled[canceled_id], 1u);
                else
                    atomicAdd(state.errors, 1u);
            }
        }
        got = __syncthreads_or(got ? 1 : 0) != 0;
        if (!got) break;
    }

    for (unsigned int token = 0; token < token_count; ++token) {
        if (threadIdx.x == 0) {
            selected_kind = -1;
            selected_tile = -1;
            for (;;) {
                if (mode == CLC_PRODUCER_PRIORITY) {
                    int p = atomicAdd(state.next_producer, 1);
                    if (p < tiles) { selected_kind = 0; selected_tile = p; break; }
                    int c = atomicAdd(state.next_consumer, 1);
                    if (c < tiles) { selected_kind = 1; selected_tile = c; break; }
                } else if (mode == CLC_CONSUMER_PRIORITY) {
                    int c = atomicAdd(state.next_consumer, 0);
                    if (c < tiles && clc_ready(state, c, epoch) &&
                        atomicCAS(state.next_consumer, c, c + 1) == c) {
                        selected_kind = 1; selected_tile = c; break;
                    }
                    int p = atomicAdd(state.next_producer, 1);
                    if (p < tiles) { selected_kind = 0; selected_tile = p; break; }
                } else if (mode == CLC_LOCALITY) {
                    if (local_tile >= 0 && clc_ready(state, local_tile, epoch) &&
                        atomicCAS(&state.consumer_claim[local_tile], 0, 1) == 0) {
                        selected_kind = 1; selected_tile = local_tile; local_tile = -1;
                        atomicAdd(state.locality_hits, 1u); break;
                    }
                    int p = atomicAdd(state.next_producer, 1);
                    if (p < tiles) {
                        selected_kind = 0; selected_tile = p; local_tile = p; break;
                    }
                    unsigned begin = ((unsigned)blockIdx.x + token) % (unsigned)tiles;
                    for (int k = 0; k < tiles; ++k) {
                        int c = (int)((begin + (unsigned)k) % (unsigned)tiles);
                        if (clc_ready(state, c, epoch) &&
                            atomicCAS(&state.consumer_claim[c], 0, 1) == 0) {
                            selected_kind = 1; selected_tile = c; break;
                        }
                    }
                    if (selected_kind >= 0) break;
                } else { // deliberately unsafe consumer-first Ceiling
                    int c = atomicAdd(state.next_consumer, 1);
                    if (c < tiles) { selected_kind = 1; selected_tile = c; break; }
                    int p = atomicAdd(state.next_producer, 1);
                    if (p < tiles) { selected_kind = 0; selected_tile = p; break; }
                }
                __nanosleep(64);
            }
        }
        __syncthreads();
        if (selected_kind == 0)
            clc_run_producer(state, selected_tile, mode, epoch, producer_cycles);
        else
            clc_consume_task(state, selected_tile, tiles, mode, epoch, consumer_cycles);
        if (threadIdx.x == 0) atomicAdd(state.tokens_processed, 1u);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(state.kernel_end, ctatrace_globaltimer());
#else
    (void)state; (void)tiles; (void)mode; (void)epoch;
    (void)producer_cycles; (void)consumer_cycles;
#endif
}

struct CLCCfg {
    std::string tag = "clc";
    std::string trace_path = "tier23_clc_trace.csv";
    int tiles = 4096;
    int threads = 128;
    int repeats = 31;
    int warmup = 3;
    unsigned long long producer_cycles = 100000;
    unsigned long long consumer_cycles = 100000;
    bool allow_short = false;
};

struct CLCHostCtx {
    CLCDeviceState d{};
    cudaStream_t stream{};
};

struct CLCMetrics {
    double ms = 0.0;
    unsigned int attempts = 0, successes = 0, tokens = 0, locality_hits = 0;
    unsigned int token_coverage_errors = 0, stale = 0, errors = 0;
    unsigned long long clc_cycles = 0, wait_polls = 0;
    unsigned long long observed_digest = 0, expected_digest = 0;
    bool correct = false, trace_ok = false;
};

static std::vector<unsigned long long> clc_expected(const CLCCfg& cfg,
                                                    unsigned long long epoch) {
    std::vector<unsigned long long> v((size_t)cfg.tiles);
    for (int i = 0; i < cfg.tiles; ++i)
        v[(size_t)i] = clc_consumer_value(epoch, (unsigned)i);
    return v;
}

static void clc_reset(const CLCCfg& cfg, CLCHostCtx& ctx,
                      unsigned long long epoch) {
    int launch_blocks = 2 * cfg.tiles;
    int n = std::max(2 * cfg.tiles, launch_blocks);
    clc_init<<<(n + 255) / 256, 256, 0, ctx.stream>>>(
        ctx.d.producer, ctx.d.consumer, ctx.d.ready, ctx.d.consumer_claim,
        ctx.d.canceled, ctx.d.executed, ctx.d.trace, cfg.tiles, launch_blocks, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(ctx.d.next_producer, 0, sizeof(int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.next_consumer, 0, sizeof(int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.kernel_start, 0xff,
                              sizeof(unsigned long long), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.kernel_end, 0,
                              sizeof(unsigned long long), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.clc_cycles, 0,
                              sizeof(unsigned long long), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.wait_polls, 0,
                              sizeof(unsigned long long), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.clc_attempts, 0, sizeof(unsigned int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.clc_successes, 0, sizeof(unsigned int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.tokens_processed, 0, sizeof(unsigned int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.locality_hits, 0, sizeof(unsigned int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d.errors, 0, sizeof(unsigned int), ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

static CLCMetrics clc_collect(const CLCCfg& cfg, CLCHostCtx& ctx, int mode,
                              unsigned long long epoch) {
    std::vector<unsigned long long> observed((size_t)cfg.tiles);
    std::vector<unsigned long long> producer((size_t)cfg.tiles);
    std::vector<unsigned int> canceled((size_t)2 * cfg.tiles),
                              executed((size_t)2 * cfg.tiles);
    std::vector<T23TraceRecord> trace((size_t)2 * cfg.tiles);
    CUDA_CHECK(cudaMemcpy(observed.data(), ctx.d.consumer,
        observed.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(producer.data(), ctx.d.producer,
        producer.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(canceled.data(), ctx.d.canceled,
        canceled.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(executed.data(), ctx.d.executed,
        executed.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trace.data(), ctx.d.trace,
        trace.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    unsigned long long start = 0, end = 0;
    CLCMetrics m;
    CUDA_CHECK(cudaMemcpy(&start, ctx.d.kernel_start, sizeof(start), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&end, ctx.d.kernel_end, sizeof(end), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.clc_cycles, ctx.d.clc_cycles,
                          sizeof(m.clc_cycles), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.wait_polls, ctx.d.wait_polls,
                          sizeof(m.wait_polls), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.attempts, ctx.d.clc_attempts,
                          sizeof(m.attempts), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.successes, ctx.d.clc_successes,
                          sizeof(m.successes), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.tokens, ctx.d.tokens_processed,
                          sizeof(m.tokens), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.locality_hits, ctx.d.locality_hits,
                          sizeof(m.locality_hits), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&m.errors, ctx.d.errors,
                          sizeof(m.errors), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < canceled.size(); ++i)
        if (canceled[i] + executed[i] != 1u) ++m.token_coverage_errors;
    std::vector<unsigned long long> expected = clc_expected(cfg, epoch);
    m.observed_digest = t23_digest(observed);
    m.expected_digest = t23_digest(expected);
    m.correct = observed == expected;
    if (mode == CLC_NONE) {
        for (int i = 0; i < cfg.tiles; ++i)
            if (observed[(size_t)i] != expected[(size_t)i]) ++m.stale;
        m.correct = false;
    }
    m.trace_ok = start != ULLONG_MAX && end > start && m.tokens == (unsigned)(2 * cfg.tiles) &&
                 m.token_coverage_errors == 0 && m.errors == 0;
    for (int kind = 0; kind < 2; ++kind) {
        for (int tile = 0; tile < cfg.tiles; ++tile) {
            const auto& r = trace[(size_t)kind * cfg.tiles + tile];
            bool ok = r.block_id == (unsigned)tile && r.kernel_id == (unsigned)kind &&
                      r.t_start && r.t_ready && r.t_wait_begin && r.t_dep && r.t_end &&
                      r.t_start <= r.t_wait_begin && r.t_wait_begin <= r.t_dep &&
                      r.t_dep <= r.t_end && r.t_ready <= r.t_end;
            m.trace_ok = m.trace_ok && ok;
            if (kind == 1 && mode != CLC_NONE &&
                r.t_dep < trace[(size_t)tile].t_ready) m.trace_ok = false;
        }
    }
    m.ms = m.trace_ok ? (double)(end - start) / 1.0e6 : 0.0;
    return m;
}

static CLCMetrics clc_once(const CLCCfg& cfg, CLCHostCtx& ctx, int mode,
                           unsigned long long epoch) {
    clc_reset(cfg, ctx, epoch);
    clc_scheduler<<<2 * cfg.tiles, cfg.threads, 0, ctx.stream>>>(
        ctx.d, cfg.tiles, mode, epoch, cfg.producer_cycles, cfg.consumer_cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return clc_collect(cfg, ctx, mode, epoch);
}

static void clc_dump_trace(const CLCCfg& cfg, CLCHostCtx& ctx, int mode,
                           unsigned long long epoch, bool header) {
    std::vector<T23TraceRecord> rows((size_t)2 * cfg.tiles);
    std::vector<unsigned int> canceled((size_t)2 * cfg.tiles),
                              executed((size_t)2 * cfg.tiles);
    unsigned long long kernel_start = 0, kernel_end = 0;
    CUDA_CHECK(cudaMemcpy(rows.data(), ctx.d.trace,
        rows.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(canceled.data(), ctx.d.canceled,
        canceled.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(executed.data(), ctx.d.executed,
        executed.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&kernel_start, ctx.d.kernel_start,
                          sizeof(kernel_start), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&kernel_end, ctx.d.kernel_end,
                          sizeof(kernel_end), cudaMemcpyDeviceToHost));
    std::ofstream f(cfg.trace_path, header ? std::ios::out : std::ios::app);
    if (!f) { fprintf(stderr, "cannot open trace %s\n", cfg.trace_path.c_str()); exit(1); }
    if (header)
        f << "tag,experiment,mode,epoch,kernel_id,block_id,sm_id,t_start,t_ready,"
             "t_wait_begin,t_dep,t_end,poll_loads,metadata_loads,decode_ns,aux\n";
    for (const auto& r : rows)
        f << cfg.tag << ",clc," << clc_schedule_name(mode) << ',' << epoch << ','
          << r.kernel_id << ',' << r.block_id << ',' << r.sm_id << ',' << r.t_start << ','
          << r.t_ready << ',' << r.t_wait_begin << ',' << r.t_dep << ',' << r.t_end << ','
          << r.poll_loads << ',' << r.metadata_loads << ',' << r.decode_ns << ',' << r.aux
          << '\n';
    // Aggregate kernel interval binds makespan (including CLC claim overhead) to trace.
    f << cfg.tag << ",clc," << clc_schedule_name(mode) << ',' << epoch
      << ",2,0,0," << kernel_start << ',' << kernel_start << ',' << kernel_start << ','
      << kernel_start << ',' << kernel_end << ",0,0,0,0\n";
    // One launch-token row per original cluster.  poll_loads=canceled and
    // metadata_loads=executed let the validator independently enforce canceled+executed=1.
    for (int i = 0; i < 2 * cfg.tiles; ++i)
        f << cfg.tag << ",clc," << clc_schedule_name(mode) << ',' << epoch
          << ",3," << i << ",0," << kernel_start << ',' << kernel_start << ','
          << kernel_start << ',' << kernel_start << ',' << kernel_end << ','
          << canceled[(size_t)i] << ',' << executed[(size_t)i] << ",0,0\n";
}

static void clc_alloc_scalar(void** ptr, size_t bytes) { CUDA_CHECK(cudaMalloc(ptr, bytes)); }

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf("usage: tier23_clc_scheduler [--tag T --trace PATH --tiles N]\n"
               "  [--producer-cycles N --consumer-cycles N --repeats N --warmup N "
               "--allow-short]\n");
        return 0;
    }
    CLCCfg cfg;
    cfg.tag = args.str("--tag", cfg.tag.c_str());
    cfg.trace_path = args.str("--trace", cfg.trace_path.c_str());
    cfg.tiles = (int)args.ll("--tiles", cfg.tiles);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.warmup = (int)args.ll("--warmup", cfg.warmup);
    cfg.producer_cycles = (unsigned long long)args.ll("--producer-cycles",
                                                      cfg.producer_cycles);
    cfg.consumer_cycles = (unsigned long long)args.ll("--consumer-cycles",
                                                      cfg.consumer_cycles);
    cfg.allow_short = args.has("--allow-short");
    if (cfg.tiles <= 0 || cfg.threads <= 0 || cfg.threads > 1024 || cfg.repeats <= 0 ||
        cfg.warmup < 0 || cfg.producer_cycles == 0 || cfg.consumer_cycles == 0 ||
        cfg.trace_path.empty()) {
        fprintf(stderr, "invalid CLC scheduler configuration\n"); return 2;
    }
    if (!t23_short_allowed(cfg.repeats, cfg.warmup, cfg.allow_short)) {
        t23_print_short_error(cfg.repeats, cfg.warmup); return 2;
    }
    DeviceInfo dev = queryDevice();
    if (dev.major < 10) { fprintf(stderr, "CLC scheduler requires CC >= 10.0\n"); return 2; }
    printDeviceBanner(dev);
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, clc_scheduler, cfg.threads, 0));
    if (occupancy < 1 || 2 * cfg.tiles <= occupancy * dev.sms) {
        fprintf(stderr, "CLC formal point must have pending clusters beyond residency "
                        "(launch=%d resident_capacity=%d)\n",
                2 * cfg.tiles, occupancy * dev.sms);
        return 2;
    }
    printf("CONFIG_TIER23 semantics=%d experiment=clc tag=%s device=%s sm=%d cc=%d.%d "
           "tiles=%d launch_clusters=%d cluster_dim=1 threads=%d occupancy=%d "
           "producer_cycles=%llu consumer_cycles=%llu warmup=%d repeats=%d "
           "timer=globaltimer bootstrap=%d launch_eligibility=single-persistent-kernel "
           "trigger_floor=n/a trigger_impl=n/a trigger_ceiling=n/a "
           "publication_safe=after_data_release publication_ceiling=none "
           "floor=producer-priority impl=consumer-priority+locality ceiling=none "
           "policies=producer-priority,consumer-priority,locality validation=all_tasks "
           "token_conservation=executed_plus_canceled_equals_one "
           "poll_counter_semantics=logical_acquire_loads_not_l2_requests "
           "trace=%s\n", T23_SEMANTICS, cfg.tag.c_str(), dev.name, dev.sms,
           dev.major, dev.minor, cfg.tiles, 2 * cfg.tiles, cfg.threads, occupancy,
           cfg.producer_cycles, cfg.consumer_cycles, cfg.warmup, cfg.repeats,
           T23_BOOTSTRAPS, cfg.trace_path.c_str());

    CLCHostCtx ctx;
    clc_alloc_scalar((void**)&ctx.d.producer,
                     (size_t)cfg.tiles * sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.consumer,
                     (size_t)cfg.tiles * sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.ready,
                     (size_t)cfg.tiles * sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.consumer_claim, (size_t)cfg.tiles * sizeof(int));
    clc_alloc_scalar((void**)&ctx.d.next_producer, sizeof(int));
    clc_alloc_scalar((void**)&ctx.d.next_consumer, sizeof(int));
    clc_alloc_scalar((void**)&ctx.d.canceled,
                     (size_t)2 * cfg.tiles * sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.executed,
                     (size_t)2 * cfg.tiles * sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.kernel_start, sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.kernel_end, sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.clc_cycles, sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.wait_polls, sizeof(unsigned long long));
    clc_alloc_scalar((void**)&ctx.d.clc_attempts, sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.clc_successes, sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.tokens_processed, sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.locality_hits, sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.errors, sizeof(unsigned int));
    clc_alloc_scalar((void**)&ctx.d.trace,
                     (size_t)2 * cfg.tiles * sizeof(T23TraceRecord));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));

    const std::vector<int> modes{CLC_PRODUCER_PRIORITY, CLC_CONSUMER_PRIORITY,
                                 CLC_LOCALITY, CLC_NONE};
    unsigned long long epoch = 0;
    for (int mode : modes) {
        ++epoch;
        CLCMetrics m = clc_once(cfg, ctx, mode, epoch);
        bool pass = mode == CLC_NONE ? (m.stale > 0 && m.trace_ok)
                                     : (m.correct && m.trace_ok);
        printf("VALIDATION_TIER23 semantics=%d experiment=clc tag=%s mode=%s epoch=%llu "
               "validation=%s correct=%d ceiling_wrong=%d stale=%u observed_digest=%llu "
               "expected_digest=%llu attempts=%u successes=%u tokens=%u "
               "token_coverage_errors=%u trace_ok=%d status=%s\n", T23_SEMANTICS,
               cfg.tag.c_str(), clc_schedule_name(mode), epoch,
               mode == CLC_NONE ? "ceiling_stale" : "all_tasks",
               mode != CLC_NONE && m.correct, mode == CLC_NONE && m.stale > 0, m.stale,
               m.observed_digest, m.expected_digest, m.attempts, m.successes, m.tokens,
               m.token_coverage_errors, m.trace_ok, pass ? "PASS" : "FAIL");
        if (!pass) return 2;
    }
    for (int w = 0; w < cfg.warmup; ++w) {
        std::vector<int> order = modes;
        if (w & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            CLCMetrics m = clc_once(cfg, ctx, mode, epoch);
            bool pass = mode == CLC_NONE ? (m.stale > 0 && m.trace_ok)
                                         : (m.correct && m.trace_ok);
            printf("WARMUP_TIER23 semantics=%d experiment=clc tag=%s warmup=%d mode=%s "
                   "epoch=%llu status=%s\n", T23_SEMANTICS, cfg.tag.c_str(), w,
                   clc_schedule_name(mode), epoch, pass ? "PASS" : "FAIL");
            if (!pass) return 2;
        }
    }
    std::vector<std::vector<double>> times(4), success_rates(4), poll_values(4),
                                     locality_values(4), cycles_per_attempt(4),
                                     attempts_per_ms(4);
    bool header = true;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        std::vector<int> order = modes;
        if (rep & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            CLCMetrics m = clc_once(cfg, ctx, mode, epoch);
            bool pass = mode == CLC_NONE ? (m.stale > 0 && m.trace_ok)
                                         : (m.correct && m.trace_ok);
            if (!pass) return 2;
            times[(size_t)mode].push_back(m.ms);
            success_rates[(size_t)mode].push_back(
                m.attempts ? (double)m.successes / m.attempts : 0.0);
            poll_values[(size_t)mode].push_back((double)m.wait_polls);
            locality_values[(size_t)mode].push_back((double)m.locality_hits);
            cycles_per_attempt[(size_t)mode].push_back(
                m.attempts ? (double)m.clc_cycles / m.attempts : 0.0);
            attempts_per_ms[(size_t)mode].push_back(
                m.ms > 0.0 ? (double)m.attempts / m.ms : 0.0);
            printf("SAMPLE_TIER23 semantics=%d experiment=clc tag=%s rep=%d mode=%s "
                   "epoch=%llu ms=%.9f clc_attempts=%u clc_successes=%u "
                   "clc_success_rate=%.9f clc_cycles=%llu "
                   "clc_cycles_per_attempt=%.9f clc_attempts_per_ms=%.9f "
                   "wait_poll_loads=%llu "
                   "wait_poll_bytes=%llu tokens=%u locality_hits=%u "
                   "token_coverage_errors=%u observed_digest=%llu expected_digest=%llu "
                   "correct=%d ceiling_wrong=%d stale=%u trace_rows=%d trace_ok=%d\n",
                   T23_SEMANTICS, cfg.tag.c_str(), rep, clc_schedule_name(mode), epoch,
                   m.ms, m.attempts, m.successes,
                   m.attempts ? (double)m.successes / m.attempts : 0.0,
                   m.clc_cycles,
                   m.attempts ? (double)m.clc_cycles / m.attempts : 0.0,
                   m.ms > 0.0 ? (double)m.attempts / m.ms : 0.0,
                   m.wait_polls,
                   m.wait_polls * sizeof(unsigned long long), m.tokens, m.locality_hits,
                   m.token_coverage_errors, m.observed_digest, m.expected_digest,
                   mode != CLC_NONE && m.correct, mode == CLC_NONE && m.stale > 0,
                   m.stale, 4 * cfg.tiles + 1, m.trace_ok);
            if (rep == cfg.repeats - 1) {
                clc_dump_trace(cfg, ctx, mode, epoch, header);
                header = false;
            }
        }
    }
    for (int mode : modes) {
        T23CI tci = t23_bootstrap_median_ci(times[(size_t)mode], 0xc1c000 + mode);
        T23CI sci = t23_bootstrap_median_ci(success_rates[(size_t)mode], 0xc1c100 + mode);
        T23CI pci = t23_bootstrap_median_ci(poll_values[(size_t)mode], 0xc1c200 + mode);
        T23CI lci = t23_bootstrap_median_ci(locality_values[(size_t)mode], 0xc1c300 + mode);
        T23CI cci = t23_bootstrap_median_ci(cycles_per_attempt[(size_t)mode],
                                            0xc1c400 + mode);
        T23CI aci = t23_bootstrap_median_ci(attempts_per_ms[(size_t)mode],
                                            0xc1c500 + mode);
        printf("SUMMARY_TIER23 semantics=%d experiment=clc tag=%s mode=%s repeats=%d "
               "median_ms=%.9f ci_ms_lo=%.9f ci_ms_hi=%.9f "
               "median_clc_success_rate=%.9f ci_success_lo=%.9f ci_success_hi=%.9f "
               "median_wait_poll_loads=%.3f ci_poll_lo=%.3f ci_poll_hi=%.3f "
               "median_locality_hits=%.3f ci_locality_lo=%.3f ci_locality_hi=%.3f "
               "median_clc_cycles_per_attempt=%.9f ci_cycles_lo=%.9f ci_cycles_hi=%.9f "
               "median_clc_attempts_per_ms=%.9f ci_attempt_rate_lo=%.9f "
               "ci_attempt_rate_hi=%.9f "
               "valid=1\n", T23_SEMANTICS, cfg.tag.c_str(), clc_schedule_name(mode),
               cfg.repeats, t23_median(times[(size_t)mode]), tci.lo, tci.hi,
               t23_median(success_rates[(size_t)mode]), sci.lo, sci.hi,
               t23_median(poll_values[(size_t)mode]), pci.lo, pci.hi,
               t23_median(locality_values[(size_t)mode]), lci.lo, lci.hi,
               t23_median(cycles_per_attempt[(size_t)mode]), cci.lo, cci.hi,
               t23_median(attempts_per_ms[(size_t)mode]), aci.lo, aci.hi);
    }
    printf("TRACE_TIER23 semantics=%d experiment=clc tag=%s path=%s modes=4 "
           "rows_per_mode=%d final_epoch=%llu\n", T23_SEMANTICS, cfg.tag.c_str(),
           cfg.trace_path.c_str(), 4 * cfg.tiles + 1, epoch);

    cudaFree(ctx.d.producer); cudaFree(ctx.d.consumer); cudaFree(ctx.d.ready);
    cudaFree(ctx.d.consumer_claim); cudaFree(ctx.d.next_producer);
    cudaFree(ctx.d.next_consumer); cudaFree(ctx.d.canceled); cudaFree(ctx.d.executed);
    cudaFree(ctx.d.kernel_start); cudaFree(ctx.d.kernel_end); cudaFree(ctx.d.clc_cycles);
    cudaFree(ctx.d.wait_polls); cudaFree(ctx.d.clc_attempts); cudaFree(ctx.d.clc_successes);
    cudaFree(ctx.d.tokens_processed); cudaFree(ctx.d.locality_hits); cudaFree(ctx.d.errors);
    cudaFree(ctx.d.trace); cudaStreamDestroy(ctx.stream);
    return 0;
}
