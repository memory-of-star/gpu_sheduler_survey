// tier23_c1.cu -- C1 data-locality comparison for EXPERIMENT_PLAN.md §7.5.
//
// Correct variants:
//   fused-cluster   one two-CTA cluster, producer rank 0 -> consumer rank 1 via DSMEM
//   separate-persist / separate-default / separate-cv
//                   two kernels with identical grid-PDL dependency semantics
// Ceiling:
//   none            same separate kernels, entry trigger, no publication/wait; must be wrong
//
// `separate-cv` uses PTX ld.global.cv, the strongest software cache-discard load available
// on this target.  The harness reports requested bytes and retains profiler sidecars; it does
// not relabel software byte counts as measured DRAM transactions or L2 hit rate.

#include "common/tier23_native.cuh"

#include <cooperative_groups.h>
#include <climits>
#include <fstream>

namespace cg = cooperative_groups;

enum C1Mode {
    C1_FUSED_CLUSTER = 0,
    C1_SEPARATE_PERSIST = 1,
    C1_SEPARATE_DEFAULT = 2,
    C1_SEPARATE_CV = 3,
    C1_NONE = 4,
};

static const char* c1_mode_name(int mode) {
    switch (mode) {
        case C1_FUSED_CLUSTER: return "fused-cluster";
        case C1_SEPARATE_PERSIST: return "separate-persist";
        case C1_SEPARATE_DEFAULT: return "separate-default";
        case C1_SEPARATE_CV: return "separate-cv";
        case C1_NONE: return "none";
        default: return "?";
    }
}

__host__ __device__ __forceinline__ unsigned int c1_word(
        unsigned long long epoch, unsigned int tile, unsigned int word) {
    unsigned long long v = t23_mix64(epoch * 0x9e3779b97f4a7c15ull ^
        ((unsigned long long)tile + 1ull) * 0xd1b54a32d192ed03ull ^
        ((unsigned long long)word + 1ull) * 0x94d049bb133111ebull);
    return (unsigned int)(v ^ (v >> 32));
}

__device__ __forceinline__ unsigned int c1_load_cv(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__global__ void c1_init(unsigned int* intermediate,
                        unsigned int* output,
                        size_t words,
                        unsigned long long epoch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < words) {
        unsigned int poison = (unsigned int)t23_mix64(epoch ^ i ^ 0xa0761d6478bd642full);
        intermediate[i] = poison;
        output[i] = poison ^ 0xdeadbeefu;
    }
}

__global__ __cluster_dims__(2, 1, 1)
void c1_fused_cluster(unsigned int* output,
                      T23TraceRecord* trace,
                      int words_per_tile,
                      unsigned long long epoch,
                      unsigned long long ready_cycles,
                      unsigned long long tail_cycles) {
    extern __shared__ unsigned int local[];
    cg::cluster_group cluster = cg::this_cluster();
    int rank = (int)cluster.block_rank();
    int tile = (int)blockIdx.x / 2;
    int trace_index = 2 * tile + rank;
    t23_trace_begin(trace + trace_index, 0, (unsigned)rank, (unsigned)C1_FUSED_CLUSTER);
    T23TraceRecord* row = trace + trace_index;
    if (threadIdx.x == 0) row->block_id = (unsigned)tile;
    __syncthreads();

    if (threadIdx.x == 0) row->t_wait_begin = ctatrace_globaltimer();
    if (rank == 0) {
        spin_cycles(ready_cycles);
        for (int w = (int)threadIdx.x; w < words_per_tile; w += (int)blockDim.x)
            local[w] = c1_word(epoch, (unsigned)tile, (unsigned)w);
    }
    __syncthreads();
    if (rank == 0 && threadIdx.x == 0) row->t_ready = ctatrace_globaltimer();

    cluster.sync();
    if (threadIdx.x == 0) row->t_dep = ctatrace_globaltimer();
    if (rank == 1) {
        unsigned int* remote = cluster.map_shared_rank(local, 0);
        size_t base = (size_t)tile * words_per_tile;
        for (int w = (int)threadIdx.x; w < words_per_tile; w += (int)blockDim.x)
            output[base + w] = remote[w];
    }
    __syncthreads();
    cluster.sync();
    if (rank == 0) spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) {
        if (rank == 1) row->t_ready = row->t_dep;
        row->t_end = ctatrace_globaltimer();
    }
}

__global__ void c1_producer(unsigned int* intermediate,
                            T23TraceRecord* trace,
                            int words_per_tile,
                            int mode,
                            unsigned long long epoch,
                            unsigned long long ready_cycles,
                            unsigned long long tail_cycles) {
    int tile = (int)blockIdx.x;
    t23_trace_begin(trace, (unsigned)tile, 0, (unsigned)mode);
    T23TraceRecord* row = trace + tile;
    __syncthreads();
    if (mode == C1_NONE) cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(ready_cycles);
    size_t base = (size_t)tile * words_per_tile;
    for (int w = (int)threadIdx.x; w < words_per_tile; w += (int)blockDim.x)
        intermediate[base + w] = c1_word(epoch, (unsigned)tile, (unsigned)w);
    __syncthreads();
    if (threadIdx.x == 0) {
        row->t_wait_begin = row->t_start;
        row->t_dep = row->t_start;
        row->t_ready = ctatrace_globaltimer();
    }
    __syncthreads();
    if (mode != C1_NONE) cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) row->t_end = ctatrace_globaltimer();
}

__global__ void c1_consumer(unsigned int* output,
                            const unsigned int* intermediate,
                            T23TraceRecord* trace,
                            int words_per_tile,
                            int mode) {
    int tile = (int)blockIdx.x;
    t23_trace_begin(trace, (unsigned)tile, 1, (unsigned)mode);
    T23TraceRecord* row = trace + tile;
    __syncthreads();
    if (threadIdx.x == 0) {
        row->t_wait_begin = ctatrace_globaltimer();
        if (mode != C1_NONE) cudaGridDependencySynchronize();
        row->t_dep = ctatrace_globaltimer();
        row->t_ready = row->t_dep;
    }
    __syncthreads();
    size_t base = (size_t)tile * words_per_tile;
    for (int w = (int)threadIdx.x; w < words_per_tile; w += (int)blockDim.x) {
        const unsigned int* p = intermediate + base + w;
        output[base + w] = mode == C1_SEPARATE_CV ? c1_load_cv(p) : *p;
    }
    __syncthreads();
    if (threadIdx.x == 0) row->t_end = ctatrace_globaltimer();
}

struct C1Cfg {
    std::string tag = "c1";
    std::string trace_path = "tier23_c1_trace.csv";
    int tiles = 148;
    int threads = 128;
    int bytes_per_tile = 1024;
    int repeats = 31;
    int warmup = 3;
    unsigned long long ready_cycles = 500000;
    unsigned long long tail_cycles = 300000;
    bool allow_short = false;
};

struct C1Ctx {
    unsigned int *intermediate = nullptr, *output = nullptr;
    T23TraceRecord* trace = nullptr;
    cudaStream_t stream{};
};

struct C1Metrics {
    double ms = 0.0;
    double transfer_gbps = 0.0;
    unsigned int stale_words = 0;
    unsigned long long observed_digest = 0;
    unsigned long long expected_digest = 0;
    bool correct = false;
    bool trace_ok = false;
};

static unsigned long long c1_digest(const std::vector<unsigned int>& values) {
    unsigned long long h = 1469598103934665603ull;
    for (unsigned int v : values) {
        for (int b = 0; b < 4; ++b) {
            h ^= (v >> (8 * b)) & 0xffu;
            h *= 1099511628211ull;
        }
    }
    return h;
}

static std::vector<unsigned int> c1_expected(const C1Cfg& cfg,
                                             unsigned long long epoch) {
    int words = cfg.bytes_per_tile / (int)sizeof(unsigned int);
    std::vector<unsigned int> v((size_t)cfg.tiles * words);
    for (int tile = 0; tile < cfg.tiles; ++tile)
        for (int w = 0; w < words; ++w)
            v[(size_t)tile * words + w] = c1_word(epoch, (unsigned)tile, (unsigned)w);
    return v;
}

static void c1_set_persist_window(const C1Cfg& cfg, C1Ctx& ctx, bool enabled,
                                  size_t max_window) {
    cudaStreamAttrValue attr{};
    size_t bytes = (size_t)cfg.tiles * cfg.bytes_per_tile;
    attr.accessPolicyWindow.base_ptr = ctx.intermediate;
    attr.accessPolicyWindow.num_bytes = enabled ? std::min(bytes, max_window) : 0;
    attr.accessPolicyWindow.hitRatio = enabled ? 1.0f : 0.0f;
    attr.accessPolicyWindow.hitProp = enabled ? cudaAccessPropertyPersisting
                                             : cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CUDA_CHECK(cudaStreamSetAttribute(ctx.stream, cudaStreamAttributeAccessPolicyWindow,
                                      &attr));
}

static C1Metrics c1_collect(const C1Cfg& cfg, C1Ctx& ctx, int mode,
                            unsigned long long epoch) {
    int words_per_tile = cfg.bytes_per_tile / (int)sizeof(unsigned int);
    size_t words = (size_t)cfg.tiles * words_per_tile;
    std::vector<unsigned int> observed(words);
    std::vector<T23TraceRecord> trace((size_t)2 * cfg.tiles);
    CUDA_CHECK(cudaMemcpy(observed.data(), ctx.output, words * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(trace.data(), ctx.trace,
        trace.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    std::vector<unsigned int> expected = c1_expected(cfg, epoch);
    C1Metrics m;
    m.observed_digest = c1_digest(observed);
    m.expected_digest = c1_digest(expected);
    m.correct = observed == expected;
    if (mode == C1_NONE) {
        for (size_t i = 0; i < words; ++i) if (observed[i] != expected[i]) ++m.stale_words;
        m.correct = false;
    }
    unsigned long long first = ULLONG_MAX, last = 0, producer_end = 0;
    m.trace_ok = true;
    for (int rank = 0; rank < 2; ++rank) {
        for (int tile = 0; tile < cfg.tiles; ++tile) {
            const auto& r = trace[(size_t)(mode == C1_FUSED_CLUSTER ? 2 * tile + rank
                                                                    : rank * cfg.tiles + tile)];
            bool ok = r.block_id == (unsigned)tile && r.kernel_id == (unsigned)rank &&
                      r.aux == (unsigned)mode && r.t_start && r.t_ready && r.t_wait_begin &&
                      r.t_dep && r.t_end && r.t_start <= r.t_wait_begin &&
                      r.t_wait_begin <= r.t_dep && r.t_dep <= r.t_end &&
                      r.t_ready <= r.t_end;
            m.trace_ok = m.trace_ok && ok;
            first = std::min(first, r.t_start);
            last = std::max(last, r.t_end);
            if (rank == 0) producer_end = std::max(producer_end, r.t_end);
        }
    }
    if (mode != C1_FUSED_CLUSTER && mode != C1_NONE) {
        for (int tile = 0; tile < cfg.tiles; ++tile)
            if (trace[(size_t)cfg.tiles + tile].t_dep < producer_end) m.trace_ok = false;
    }
    if (first == ULLONG_MAX || last <= first) m.trace_ok = false;
    m.ms = m.trace_ok ? (double)(last - first) / 1.0e6 : 0.0;
    // One logical transfer is X bytes read plus X bytes output.  This is requested software
    // traffic, not a profiler-derived DRAM counter.
    m.transfer_gbps = (double)cfg.tiles * cfg.bytes_per_tile * 2.0 /
                      (double)(last - first);
    return m;
}

static C1Metrics c1_once(const C1Cfg& cfg, C1Ctx& ctx, int mode,
                         unsigned long long epoch, size_t persist_max) {
    int words_per_tile = cfg.bytes_per_tile / (int)sizeof(unsigned int);
    size_t words = (size_t)cfg.tiles * words_per_tile;
    c1_init<<<(words + 255) / 256, 256, 0, ctx.stream>>>(
        ctx.intermediate, ctx.output, words, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(ctx.trace, 0,
        (size_t)2 * cfg.tiles * sizeof(T23TraceRecord), ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    c1_set_persist_window(cfg, ctx, mode == C1_SEPARATE_PERSIST, persist_max);

    if (mode == C1_FUSED_CLUSTER) {
        c1_fused_cluster<<<2 * cfg.tiles, cfg.threads, cfg.bytes_per_tile, ctx.stream>>>(
            ctx.output, ctx.trace, words_per_tile, epoch, cfg.ready_cycles, cfg.tail_cycles);
        CUDA_CHECK(cudaGetLastError());
    } else {
        c1_producer<<<cfg.tiles, cfg.threads, 0, ctx.stream>>>(
            ctx.intermediate, ctx.trace, words_per_tile, mode, epoch,
            cfg.ready_cycles, cfg.tail_cycles);
        CUDA_CHECK(cudaGetLastError());
        T23TraceRecord* consumer_trace = ctx.trace + cfg.tiles;
        void* argv[] = {&ctx.output, &ctx.intermediate, &consumer_trace,
                        &words_per_tile, &mode};
        CUDA_CHECK(t23_launch_pss(ctx.stream, dim3(cfg.tiles), dim3(cfg.threads), 0,
                                 (const void*)c1_consumer, argv));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    C1Metrics m = c1_collect(cfg, ctx, mode, epoch);
    c1_set_persist_window(cfg, ctx, false, persist_max);
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());
    return m;
}

static void c1_dump_trace(const C1Cfg& cfg, C1Ctx& ctx, int mode,
                          unsigned long long epoch, bool header) {
    std::vector<T23TraceRecord> rows((size_t)2 * cfg.tiles);
    CUDA_CHECK(cudaMemcpy(rows.data(), ctx.trace,
        rows.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    std::ofstream f(cfg.trace_path, header ? std::ios::out : std::ios::app);
    if (!f) { fprintf(stderr, "cannot open trace %s\n", cfg.trace_path.c_str()); exit(1); }
    if (header)
        f << "tag,experiment,mode,epoch,kernel_id,block_id,sm_id,t_start,t_ready,"
             "t_wait_begin,t_dep,t_end,poll_loads,metadata_loads,decode_ns,aux\n";
    for (const auto& r : rows)
        f << cfg.tag << ",c1," << c1_mode_name(mode) << ',' << epoch << ',' << r.kernel_id
          << ',' << r.block_id << ',' << r.sm_id << ',' << r.t_start << ',' << r.t_ready
          << ',' << r.t_wait_begin << ',' << r.t_dep << ',' << r.t_end << ','
          << r.poll_loads << ',' << r.metadata_loads << ',' << r.decode_ns << ',' << r.aux
          << '\n';
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf("usage: tier23_c1 [--tag T --trace PATH --tiles N --bytes-per-tile N]\n"
               "  [--repeats N --warmup N --ready-cycles N --tail-cycles N --allow-short]\n");
        return 0;
    }
    C1Cfg cfg;
    cfg.tag = args.str("--tag", cfg.tag.c_str());
    cfg.trace_path = args.str("--trace", cfg.trace_path.c_str());
    cfg.tiles = (int)args.ll("--tiles", cfg.tiles);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.bytes_per_tile = (int)args.ll("--bytes-per-tile", cfg.bytes_per_tile);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.warmup = (int)args.ll("--warmup", cfg.warmup);
    cfg.ready_cycles = (unsigned long long)args.ll("--ready-cycles", cfg.ready_cycles);
    cfg.tail_cycles = (unsigned long long)args.ll("--tail-cycles", cfg.tail_cycles);
    cfg.allow_short = args.has("--allow-short");
    if (cfg.tiles <= 0 || cfg.threads <= 0 || cfg.threads > 1024 ||
        cfg.bytes_per_tile < 1024 || cfg.bytes_per_tile > 64 * 1024 ||
        cfg.bytes_per_tile % (int)sizeof(unsigned int) != 0 || cfg.repeats <= 0 ||
        cfg.warmup < 0 || cfg.ready_cycles == 0 || cfg.trace_path.empty()) {
        fprintf(stderr, "invalid C1 configuration\n"); return 2;
    }
    if (!t23_short_allowed(cfg.repeats, cfg.warmup, cfg.allow_short)) {
        t23_print_short_error(cfg.repeats, cfg.warmup); return 2;
    }
    DeviceInfo dev = queryDevice();
    if (dev.major < 9) { fprintf(stderr, "cluster DSMEM/grid PDL needs CC >= 9.0\n"); return 2; }
    if (cfg.tiles == 148) cfg.tiles = dev.sms;
    printDeviceBanner(dev);
    CUDA_CHECK(cudaFuncSetAttribute(c1_fused_cluster,
        cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.bytes_per_tile));
    int fused_blocks_per_sm = 0, separate_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fused_blocks_per_sm, c1_fused_cluster, cfg.threads, cfg.bytes_per_tile));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &separate_blocks_per_sm, c1_consumer, cfg.threads, 0));
    if (fused_blocks_per_sm < 2 || separate_blocks_per_sm < 1) {
        fprintf(stderr, "C1 occupancy does not admit a two-block cluster\n"); return 2;
    }
    int device = 0, persist_max_int = 0, window_max_int = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&persist_max_int,
        cudaDevAttrMaxPersistingL2CacheSize, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&window_max_int,
        cudaDevAttrMaxAccessPolicyWindowSize, device));
    size_t persist_max = (size_t)std::max(0, window_max_int);
    if (persist_max_int > 0)
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                     (size_t)persist_max_int));

    printf("CONFIG_TIER23 semantics=%d experiment=c1 tag=%s device=%s sm=%d cc=%d.%d "
           "tiles=%d threads=%d bytes_per_tile=%d words_per_tile=%d warmup=%d repeats=%d "
           "ready_cycles=%llu tail_cycles=%llu timer=globaltimer bootstrap=%d "
           "bracket=floor:separate-default,impl:separate-persist,ceiling:none,"
           "ideal:fused-cluster,lower-control:separate-cv "
           "cv_semantics=forced-refetch-pessimal-control trigger_floor=ready trigger_impl=ready "
           "trigger_ceiling=entry publication_ceiling=none dependency=grid-PDL "
           "fused_transport=cluster-DSMEM cluster_blocks=2 cluster_smem_bytes=%d "
           "fused_blocks_per_sm=%d separate_blocks_per_sm=%d persist_limit=%d "
           "access_window_limit=%d software_bytes_not_dram_counters=1 validation=all_words "
           "trace=%s\n", T23_SEMANTICS, cfg.tag.c_str(), dev.name, dev.sms,
           dev.major, dev.minor, cfg.tiles, cfg.threads, cfg.bytes_per_tile,
           cfg.bytes_per_tile / (int)sizeof(unsigned int), cfg.warmup, cfg.repeats,
           cfg.ready_cycles, cfg.tail_cycles, T23_BOOTSTRAPS, 2 * cfg.bytes_per_tile,
           fused_blocks_per_sm, separate_blocks_per_sm, persist_max_int, window_max_int,
           cfg.trace_path.c_str());

    C1Ctx ctx;
    size_t words = (size_t)cfg.tiles * cfg.bytes_per_tile / sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&ctx.intermediate, words * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx.output, words * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx.trace,
        (size_t)2 * cfg.tiles * sizeof(T23TraceRecord)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
    // Keep the operational bracket adjacent in both traversal directions.  Ideal and the
    // forced-refetch pessimal control flank it but are not mislabeled as CTA implementations.
    const std::vector<int> modes{C1_SEPARATE_DEFAULT, C1_SEPARATE_PERSIST, C1_NONE,
                                 C1_FUSED_CLUSTER, C1_SEPARATE_CV};
    unsigned long long epoch = 0;
    for (int mode : modes) {
        ++epoch;
        C1Metrics m = c1_once(cfg, ctx, mode, epoch, persist_max);
        bool pass = mode == C1_NONE ? (m.stale_words > 0 && m.trace_ok)
                                    : (m.correct && m.trace_ok);
        printf("VALIDATION_TIER23 semantics=%d experiment=c1 tag=%s mode=%s epoch=%llu "
               "validation=%s correct=%d ceiling_wrong=%d stale=%u observed_digest=%llu "
               "expected_digest=%llu trace_ok=%d status=%s\n", T23_SEMANTICS,
               cfg.tag.c_str(), c1_mode_name(mode), epoch,
               mode == C1_NONE ? "ceiling_stale" : "all_words",
               mode != C1_NONE && m.correct, mode == C1_NONE && m.stale_words > 0,
               m.stale_words, m.observed_digest, m.expected_digest, m.trace_ok,
               pass ? "PASS" : "FAIL");
        if (!pass) return 2;
    }
    for (int w = 0; w < cfg.warmup; ++w) {
        std::vector<int> order = modes;
        if (w & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            C1Metrics m = c1_once(cfg, ctx, mode, epoch, persist_max);
            bool pass = mode == C1_NONE ? (m.stale_words > 0 && m.trace_ok)
                                        : (m.correct && m.trace_ok);
            printf("WARMUP_TIER23 semantics=%d experiment=c1 tag=%s warmup=%d mode=%s "
                   "epoch=%llu status=%s\n", T23_SEMANTICS, cfg.tag.c_str(), w,
                   c1_mode_name(mode), epoch, pass ? "PASS" : "FAIL");
            if (!pass) return 2;
        }
    }
    std::vector<std::vector<double>> times(5), throughputs(5);
    bool header = true;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        std::vector<int> order = modes;
        if (rep & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            C1Metrics m = c1_once(cfg, ctx, mode, epoch, persist_max);
            bool pass = mode == C1_NONE ? (m.stale_words > 0 && m.trace_ok)
                                        : (m.correct && m.trace_ok);
            if (!pass) return 2;
            times[(size_t)mode].push_back(m.ms);
            throughputs[(size_t)mode].push_back(m.transfer_gbps);
            printf("SAMPLE_TIER23 semantics=%d experiment=c1 tag=%s rep=%d mode=%s "
                   "epoch=%llu ms=%.9f requested_read_bytes=%zu requested_write_bytes=%zu "
                   "software_transfer_gbps=%.9f observed_digest=%llu expected_digest=%llu "
                   "correct=%d ceiling_wrong=%d stale=%u trace_rows=%d trace_ok=%d\n",
                   T23_SEMANTICS, cfg.tag.c_str(), rep, c1_mode_name(mode), epoch, m.ms,
                   (size_t)cfg.tiles * cfg.bytes_per_tile,
                   (size_t)cfg.tiles * cfg.bytes_per_tile, m.transfer_gbps,
                   m.observed_digest, m.expected_digest, mode != C1_NONE && m.correct,
                   mode == C1_NONE && m.stale_words > 0, m.stale_words,
                   2 * cfg.tiles, m.trace_ok);
            if (rep == cfg.repeats - 1) {
                c1_dump_trace(cfg, ctx, mode, epoch, header);
                header = false;
            }
        }
    }
    for (int mode : modes) {
        T23CI tci = t23_bootstrap_median_ci(times[(size_t)mode], 0xc100 + mode);
        T23CI bci = t23_bootstrap_median_ci(throughputs[(size_t)mode], 0xc200 + mode);
        printf("SUMMARY_TIER23 semantics=%d experiment=c1 tag=%s mode=%s repeats=%d "
               "median_ms=%.9f ci_ms_lo=%.9f ci_ms_hi=%.9f "
               "median_software_transfer_gbps=%.9f ci_transfer_lo=%.9f "
               "ci_transfer_hi=%.9f valid=1\n", T23_SEMANTICS, cfg.tag.c_str(),
               c1_mode_name(mode), cfg.repeats, t23_median(times[(size_t)mode]),
               tci.lo, tci.hi, t23_median(throughputs[(size_t)mode]), bci.lo, bci.hi);
    }
    printf("TRACE_TIER23 semantics=%d experiment=c1 tag=%s path=%s modes=5 "
           "rows_per_mode=%d final_epoch=%llu\n", T23_SEMANTICS, cfg.tag.c_str(),
           cfg.trace_path.c_str(), 2 * cfg.tiles, epoch);
    cudaFree(ctx.intermediate); cudaFree(ctx.output); cudaFree(ctx.trace);
    cudaStreamDestroy(ctx.stream);
    return 0;
}
