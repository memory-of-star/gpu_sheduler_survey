// tier23_diamond.cu -- native CUDA diamond harness for EXPERIMENT_PLAN.md §7.4.
//
// Floor flattens K1->K2->K3->K4 into grid-PDL order.  The two software modes use the
// same four independent-stream launch path and differ by exactly one edge: ordered makes
// K3 wait for K2, while unordered lets K2/K3 both depend directly on K1.  K4 always waits
// for both branches.  Ceiling removes every readiness wait, is required to observe poison,
// and contributes timing only.

#include "common/tier23_native.cuh"

#include <cuda/atomic>
#include <climits>
#include <fstream>

enum DiamondMode {
    D_GRID_ORDERED = 0,
    D_CTA_ORDERED = 1,
    D_CTA_UNORDERED = 2,
    D_NONE = 3,
};

static const char* diamond_mode_name(int mode) {
    switch (mode) {
        case D_GRID_ORDERED: return "grid-ordered";
        case D_CTA_ORDERED: return "cta-ordered";
        case D_CTA_UNORDERED: return "cta-unordered";
        case D_NONE: return "none";
        default: return "?";
    }
}

__host__ __device__ __forceinline__ unsigned long long diamond_expected(
        unsigned long long epoch, unsigned int block, int stage) {
    unsigned long long x0 = t23_value(epoch, block, 0);
    if (stage == 0) return x0;
    unsigned long long x1 = t23_mix64(x0 ^ 0x1111111111111111ull);
    if (stage == 1) return x1;
    unsigned long long x2 = t23_mix64(x0 ^ 0x3333333333333333ull);
    if (stage == 2) return x2;
    return t23_mix64(x1 ^ ((x2 << 17) | (x2 >> 47)) ^ 0x4444444444444444ull);
}

__device__ __forceinline__ void diamond_wait_flag(const unsigned long long* flags,
                                                  int index,
                                                  unsigned long long epoch,
                                                  unsigned long long* loads) {
    cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> f(flags[index]);
    unsigned int ns = 64;
    for (;;) {
        ++*loads;
        if (f.load(cuda::memory_order_acquire) >= epoch) break;
        __nanosleep(ns);
        ns = ns < 2048 ? ns * 2 : 2048;
    }
}

__global__ void diamond_init(unsigned long long* data,
                             unsigned long long* flags,
                             int blocks,
                             unsigned long long epoch) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < 4 * blocks) {
        int stage = i / blocks;
        int block = i % blocks;
        data[i] = t23_mix64(diamond_expected(epoch, (unsigned)block, stage) ^
                            0xd6e8feb86659fd93ull ^
                            ((unsigned long long)stage + 1ull) * 0xa0761d6478bd642full);
        flags[i] = 0;
    }
}

__global__ void diamond_stage(unsigned long long* data,
                              unsigned long long* flags,
                              T23TraceRecord* trace,
                              int blocks,
                              int stage,
                              int mode,
                              unsigned long long epoch,
                              unsigned long long work_cycles,
                              unsigned long long tail_cycles,
                              int validate,
                              int* errors) {
    extern __shared__ unsigned char smem[];
    int block = (int)blockIdx.x;
    int index = stage * blocks + block;
    if (threadIdx.x == 0) smem[0] = (unsigned char)(stage + block);
    t23_trace_begin(trace + (size_t)stage * blocks, (unsigned)block,
                    (unsigned)stage, (unsigned)mode);
    T23TraceRecord* row = trace + index;
    __syncthreads();

    // Every software/Ceiling stage signals entry.  Floor stages delay the programmatic
    // trigger until this stage's datum is ready, preserving the grid-PDL baseline.
    if (mode != D_GRID_ORDERED) cudaTriggerProgrammaticLaunchCompletion();

    unsigned long long polls = 0;
    unsigned long long input0 = 0, input1 = 0, input2 = 0;
    if (threadIdx.x == 0) {
        row->t_wait_begin = ctatrace_globaltimer();
        if (stage > 0 && mode == D_GRID_ORDERED) {
            cudaGridDependencySynchronize();
        } else if (stage > 0 && mode != D_NONE) {
            if (stage == 1) {
                diamond_wait_flag(flags, block, epoch, &polls);            // K1 -> K2
            } else if (stage == 2) {
                int parent_stage = mode == D_CTA_ORDERED ? 1 : 0;
                diamond_wait_flag(flags, parent_stage * blocks + block,
                                  epoch, &polls);                           // order edge
            } else {
                diamond_wait_flag(flags, blocks + block, epoch, &polls);   // K2 -> K4
                diamond_wait_flag(flags, 2 * blocks + block, epoch, &polls); // K3 -> K4
            }
        }
        row->t_dep = ctatrace_globaltimer();
        row->poll_loads = polls;
        // Capture dependent inputs immediately at the dependency point.  Keeping the read
        // after work would let an unsafe Ceiling accidentally become correct while it burns
        // nominal compute cycles, which would no longer represent a true no-wait path.
        if (stage == 1 || stage == 2) input0 = data[block];
        if (stage == 3) {
            input1 = data[blocks + block];
            input2 = data[2 * blocks + block];
        }
    }
    __syncthreads();
    spin_cycles(work_cycles);

    if (threadIdx.x == 0) {
        unsigned long long observed;
        if (stage == 0) {
            observed = t23_value(epoch, (unsigned)block, 0);
        } else if (stage == 1) {
            observed = t23_mix64(input0 ^ 0x1111111111111111ull);
        } else if (stage == 2) {
            observed = t23_mix64(input0 ^ 0x3333333333333333ull);
        } else {
            observed = t23_mix64(input1 ^ ((input2 << 17) | (input2 >> 47)) ^
                                 0x4444444444444444ull);
        }
        data[index] = observed;
        if (validate && observed != diamond_expected(epoch, (unsigned)block, stage))
            atomicExch(errors, 1);
        if (mode != D_NONE) {
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> f(flags[index]);
            f.store(epoch, cuda::memory_order_release);
        }
        row->t_ready = ctatrace_globaltimer();
    }
    __syncthreads();
    if (mode == D_GRID_ORDERED) cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) row->t_end = ctatrace_globaltimer();
}

struct DiamondCfg {
    std::string tag = "diamond";
    std::string trace_path = "tier23_diamond_trace.csv";
    int blocks = 148;
    int threads = 128;
    int repeats = 31;
    int warmup = 3;
    int smem_kb = 16;
    int ratio = 1;
    unsigned long long base_cycles = 300000;
    unsigned long long tail_cycles = 300000;
    bool allow_short = false;
};

struct DiamondCtx {
    unsigned long long *data = nullptr, *flags = nullptr;
    int* errors = nullptr;
    T23TraceRecord* trace = nullptr;
    cudaStream_t streams[4]{};
};

struct DiamondMetrics {
    double ms = 0.0;
    double branch_overlap_ms = 0.0;
    unsigned long long polls = 0;
    unsigned int stale = 0;
    unsigned long long observed_digest = 0;
    unsigned long long expected_digest = 0;
    bool correct = false;
    bool trace_ok = false;
};

static std::vector<unsigned long long> diamond_expected_all(const DiamondCfg& cfg,
                                                            unsigned long long epoch) {
    std::vector<unsigned long long> v((size_t)4 * cfg.blocks);
    for (int stage = 0; stage < 4; ++stage)
        for (int block = 0; block < cfg.blocks; ++block)
            v[(size_t)stage * cfg.blocks + block] =
                diamond_expected(epoch, (unsigned)block, stage);
    return v;
}

static void diamond_reset(const DiamondCfg& cfg, DiamondCtx& ctx,
                          unsigned long long epoch) {
    diamond_init<<<(4 * cfg.blocks + 255) / 256, 256, 0, ctx.streams[0]>>>(
        ctx.data, ctx.flags, cfg.blocks, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(ctx.errors, 0, sizeof(int), ctx.streams[0]));
    CUDA_CHECK(cudaMemsetAsync(ctx.trace, 0,
        (size_t)4 * cfg.blocks * sizeof(T23TraceRecord), ctx.streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[0]));
}

static void diamond_launch_one(const DiamondCfg& cfg, DiamondCtx& ctx,
                               int stage, int mode, unsigned long long epoch,
                               int validate, cudaStream_t stream, bool pss) {
    unsigned long long cycles = cfg.base_cycles;
    if (stage == 2) cycles *= (unsigned long long)cfg.ratio;
    if (!pss) {
        diamond_stage<<<cfg.blocks, cfg.threads, (size_t)cfg.smem_kb * 1024u, stream>>>(
            ctx.data, ctx.flags, ctx.trace, cfg.blocks, stage, mode, epoch,
            cycles, cfg.tail_cycles, validate, ctx.errors);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    void* argv[] = {&ctx.data, &ctx.flags, &ctx.trace,
                    const_cast<int*>(&cfg.blocks), &stage, &mode, &epoch, &cycles,
                    const_cast<unsigned long long*>(&cfg.tail_cycles), &validate,
                    &ctx.errors};
    CUDA_CHECK(t23_launch_pss(stream, dim3(cfg.blocks), dim3(cfg.threads),
                             (size_t)cfg.smem_kb * 1024u,
                             (const void*)diamond_stage, argv));
}

static DiamondMetrics diamond_collect(const DiamondCfg& cfg, DiamondCtx& ctx,
                                      int mode, unsigned long long epoch) {
    std::vector<unsigned long long> observed((size_t)4 * cfg.blocks);
    std::vector<T23TraceRecord> rows((size_t)4 * cfg.blocks);
    CUDA_CHECK(cudaMemcpy(observed.data(), ctx.data,
        observed.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(rows.data(), ctx.trace,
        rows.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    int errors = 1;
    CUDA_CHECK(cudaMemcpy(&errors, ctx.errors, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> expected = diamond_expected_all(cfg, epoch);

    DiamondMetrics m;
    m.observed_digest = t23_digest(observed);
    m.expected_digest = t23_digest(expected);
    m.correct = errors == 0 && observed == expected;
    if (mode == D_NONE) {
        for (size_t i = (size_t)3 * cfg.blocks; i < observed.size(); ++i)
            if (observed[i] != expected[i]) ++m.stale;
        m.correct = false;
    }

    unsigned long long first = ULLONG_MAX, last = 0;
    unsigned long long k2_first = ULLONG_MAX, k2_last = 0;
    unsigned long long k3_first = ULLONG_MAX, k3_last = 0;
    m.trace_ok = true;
    for (int stage = 0; stage < 4; ++stage) {
        unsigned long long parent_grid_end = 0;
        if (stage > 0 && mode == D_GRID_ORDERED) {
            for (int b = 0; b < cfg.blocks; ++b)
                parent_grid_end = std::max(parent_grid_end,
                    rows[(size_t)(stage - 1) * cfg.blocks + b].t_end);
        }
        for (int block = 0; block < cfg.blocks; ++block) {
            const auto& r = rows[(size_t)stage * cfg.blocks + block];
            bool ok = r.block_id == (unsigned)block && r.kernel_id == (unsigned)stage &&
                      r.aux == (unsigned)mode && r.t_start && r.t_wait_begin && r.t_dep &&
                      r.t_ready && r.t_end && r.t_start <= r.t_wait_begin &&
                      r.t_wait_begin <= r.t_dep && r.t_dep <= r.t_ready &&
                      r.t_ready <= r.t_end;
            m.trace_ok = m.trace_ok && ok;
            first = std::min(first, r.t_start);
            last = std::max(last, r.t_end);
            m.polls += r.poll_loads;
            if (stage == 1) {
                k2_first = std::min(k2_first, r.t_dep);
                k2_last = std::max(k2_last, r.t_ready);
            }
            if (stage == 2) {
                k3_first = std::min(k3_first, r.t_dep);
                k3_last = std::max(k3_last, r.t_ready);
            }
            if (stage > 0 && mode == D_GRID_ORDERED && r.t_dep < parent_grid_end)
                m.trace_ok = false;
            if (stage > 0 && mode != D_GRID_ORDERED && mode != D_NONE) {
                auto ready = [&](int s) {
                    return rows[(size_t)s * cfg.blocks + block].t_ready;
                };
                if (stage == 1 && r.t_dep < ready(0)) m.trace_ok = false;
                if (stage == 2 && r.t_dep < ready(mode == D_CTA_ORDERED ? 1 : 0))
                    m.trace_ok = false;
                if (stage == 3 && (r.t_dep < ready(1) || r.t_dep < ready(2)))
                    m.trace_ok = false;
            }
        }
    }
    if (first == ULLONG_MAX || last <= first) m.trace_ok = false;
    m.ms = m.trace_ok ? (double)(last - first) / 1.0e6 : 0.0;
    unsigned long long overlap_begin = std::max(k2_first, k3_first);
    unsigned long long overlap_end = std::min(k2_last, k3_last);
    m.branch_overlap_ms = overlap_end > overlap_begin
        ? (double)(overlap_end - overlap_begin) / 1.0e6 : 0.0;
    return m;
}

static DiamondMetrics diamond_once(const DiamondCfg& cfg, DiamondCtx& ctx,
                                   int mode, unsigned long long epoch, bool validate) {
    diamond_reset(cfg, ctx, epoch);
    int validation = validate ? 1 : 0;
    if (mode == D_GRID_ORDERED) {
        diamond_launch_one(cfg, ctx, 0, mode, epoch, validation, ctx.streams[0], false);
        for (int stage = 1; stage < 4; ++stage)
            diamond_launch_one(cfg, ctx, stage, mode, epoch, validation,
                               ctx.streams[0], true);
        CUDA_CHECK(cudaStreamSynchronize(ctx.streams[0]));
    } else {
        // Enqueue dependents first so the software waits are real resident waits.  The
        // producer stream has higher priority and all four kernels have a verified mixed
        // resource envelope, so K1 retains a forward-progress slot.
        diamond_launch_one(cfg, ctx, 3, mode, epoch, validation, ctx.streams[3], false);
        diamond_launch_one(cfg, ctx, 2, mode, epoch, validation, ctx.streams[2], false);
        diamond_launch_one(cfg, ctx, 1, mode, epoch, validation, ctx.streams[1], false);
        diamond_launch_one(cfg, ctx, 0, mode, epoch, validation, ctx.streams[0], false);
        for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaStreamSynchronize(ctx.streams[i]));
    }
    return diamond_collect(cfg, ctx, mode, epoch);
}

static void diamond_dump_trace(const DiamondCfg& cfg, DiamondCtx& ctx, int mode,
                               unsigned long long epoch, bool header) {
    std::vector<T23TraceRecord> rows((size_t)4 * cfg.blocks);
    CUDA_CHECK(cudaMemcpy(rows.data(), ctx.trace,
        rows.size() * sizeof(T23TraceRecord), cudaMemcpyDeviceToHost));
    std::ofstream f(cfg.trace_path, header ? std::ios::out : std::ios::app);
    if (!f) { fprintf(stderr, "cannot open trace %s\n", cfg.trace_path.c_str()); exit(1); }
    if (header)
        f << "tag,experiment,mode,epoch,kernel_id,block_id,sm_id,t_start,t_ready,"
             "t_wait_begin,t_dep,t_end,poll_loads,metadata_loads,decode_ns,aux\n";
    for (const auto& r : rows)
        f << cfg.tag << ",diamond," << diamond_mode_name(mode) << ',' << epoch << ','
          << r.kernel_id << ',' << r.block_id << ',' << r.sm_id << ',' << r.t_start << ','
          << r.t_ready << ',' << r.t_wait_begin << ',' << r.t_dep << ',' << r.t_end << ','
          << r.poll_loads << ',' << r.metadata_loads << ',' << r.decode_ns << ',' << r.aux
          << '\n';
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf("usage: tier23_diamond [--tag T --trace PATH --blocks N --ratio 1..10]\n"
               "  [--base-cycles N --tail-cycles N --repeats N --warmup N --allow-short]\n");
        return 0;
    }
    DiamondCfg cfg;
    cfg.tag = args.str("--tag", cfg.tag.c_str());
    cfg.trace_path = args.str("--trace", cfg.trace_path.c_str());
    cfg.blocks = (int)args.ll("--blocks", cfg.blocks);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.warmup = (int)args.ll("--warmup", cfg.warmup);
    cfg.smem_kb = (int)args.ll("--smem-kb", cfg.smem_kb);
    cfg.ratio = (int)args.ll("--ratio", cfg.ratio);
    cfg.base_cycles = (unsigned long long)args.ll("--base-cycles", cfg.base_cycles);
    cfg.tail_cycles = (unsigned long long)args.ll("--tail-cycles", cfg.tail_cycles);
    cfg.allow_short = args.has("--allow-short");
    if (cfg.blocks <= 0 || cfg.threads <= 0 || cfg.threads > 1024 || cfg.repeats <= 0 ||
        cfg.warmup < 0 || cfg.smem_kb <= 0 || cfg.ratio < 1 || cfg.ratio > 10 ||
        cfg.base_cycles == 0 || cfg.trace_path.empty()) {
        fprintf(stderr, "invalid diamond configuration\n"); return 2;
    }
    if (!t23_short_allowed(cfg.repeats, cfg.warmup, cfg.allow_short)) {
        t23_print_short_error(cfg.repeats, cfg.warmup); return 2;
    }
    DeviceInfo dev = queryDevice();
    if (dev.major < 9) { fprintf(stderr, "grid PDL requires CC >= 9.0\n"); return 2; }
    if (cfg.blocks == 148) cfg.blocks = dev.sms;
    printDeviceBanner(dev);
    size_t smem = (size_t)cfg.smem_kb * 1024u;
    CUDA_CHECK(cudaFuncSetAttribute(diamond_stage,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, diamond_stage, cfg.threads, smem));
    if (occupancy < 4) {
        fprintf(stderr, "diamond needs >=4 resident stage CTAs/SM; observed %d\n", occupancy);
        return 2;
    }
    printf("CONFIG_TIER23 semantics=%d experiment=diamond tag=%s device=%s sm=%d cc=%d.%d "
           "blocks=%d threads=%d ratio=%d ratio_label=1:%d "
           "K2_cycles=%llu K3_cycles=%llu tail_cycles=%llu "
           "smem_kb=%d occupancy=%d warmup=%d repeats=%d timer=globaltimer bootstrap=%d "
           "trigger_floor=ready trigger_impl=entry trigger_ceiling=entry "
           "publication_impl=after_data_release publication_ceiling=none "
           "floor_edges=K1>K2>K3>K4 ordered_edge=K2>K3 "
           "unordered_edges=K1>K2,K1>K3,K2+K3>K4 validation=all_stages_all_blocks "
           "trace=%s\n", T23_SEMANTICS, cfg.tag.c_str(), dev.name, dev.sms,
           dev.major, dev.minor, cfg.blocks, cfg.threads, cfg.ratio, cfg.ratio, cfg.base_cycles,
           cfg.base_cycles * (unsigned long long)cfg.ratio, cfg.tail_cycles,
           cfg.smem_kb, occupancy, cfg.warmup, cfg.repeats, T23_BOOTSTRAPS,
           cfg.trace_path.c_str());

    DiamondCtx ctx;
    CUDA_CHECK(cudaMalloc(&ctx.data,
        (size_t)4 * cfg.blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.flags,
        (size_t)4 * cfg.blocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.errors, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.trace,
        (size_t)4 * cfg.blocks * sizeof(T23TraceRecord)));
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.streams[0], cudaStreamNonBlocking, greatest));
    for (int i = 1; i < 4; ++i)
        CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.streams[i], cudaStreamNonBlocking, least));

    const std::vector<int> modes{D_GRID_ORDERED, D_CTA_ORDERED, D_CTA_UNORDERED, D_NONE};
    unsigned long long epoch = 0;
    for (int mode : modes) {
        ++epoch;
        DiamondMetrics m = diamond_once(cfg, ctx, mode, epoch, mode != D_NONE);
        bool pass = mode == D_NONE ? (m.stale > 0 && m.trace_ok)
                                   : (m.correct && m.trace_ok);
        printf("VALIDATION_TIER23 semantics=%d experiment=diamond tag=%s mode=%s epoch=%llu "
               "validation=%s correct=%d ceiling_wrong=%d stale=%u observed_digest=%llu "
               "expected_digest=%llu trace_ok=%d status=%s\n", T23_SEMANTICS,
               cfg.tag.c_str(), diamond_mode_name(mode), epoch,
               mode == D_NONE ? "ceiling_stale" : "all_stages_all_blocks",
               mode != D_NONE && m.correct, mode == D_NONE && m.stale > 0, m.stale,
               m.observed_digest, m.expected_digest, m.trace_ok, pass ? "PASS" : "FAIL");
        if (!pass) return 2;
    }
    for (int w = 0; w < cfg.warmup; ++w) {
        std::vector<int> order = modes;
        if (w & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            DiamondMetrics m = diamond_once(cfg, ctx, mode, epoch, false);
            bool pass = mode == D_NONE ? (m.stale > 0 && m.trace_ok)
                                       : (m.correct && m.trace_ok);
            printf("WARMUP_TIER23 semantics=%d experiment=diamond tag=%s warmup=%d "
                   "mode=%s epoch=%llu status=%s\n", T23_SEMANTICS, cfg.tag.c_str(), w,
                   diamond_mode_name(mode), epoch, pass ? "PASS" : "FAIL");
            if (!pass) return 2;
        }
    }
    std::vector<std::vector<double>> times(4), overlaps(4), polls(4);
    bool trace_header = true;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        std::vector<int> order = modes;
        if (rep & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            DiamondMetrics m = diamond_once(cfg, ctx, mode, epoch, false);
            bool pass = mode == D_NONE ? (m.stale > 0 && m.trace_ok)
                                       : (m.correct && m.trace_ok);
            if (!pass) return 2;
            times[(size_t)mode].push_back(m.ms);
            overlaps[(size_t)mode].push_back(m.branch_overlap_ms);
            polls[(size_t)mode].push_back((double)m.polls);
            printf("SAMPLE_TIER23 semantics=%d experiment=diamond tag=%s rep=%d mode=%s "
                   "epoch=%llu ms=%.9f branch_overlap_ms=%.9f poll_loads=%llu "
                   "poll_bytes=%llu observed_digest=%llu expected_digest=%llu "
                   "correct=%d ceiling_wrong=%d stale=%u trace_rows=%d trace_ok=%d\n",
                   T23_SEMANTICS, cfg.tag.c_str(), rep, diamond_mode_name(mode), epoch,
                   m.ms, m.branch_overlap_ms, m.polls,
                   m.polls * sizeof(unsigned long long), m.observed_digest,
                   m.expected_digest, mode != D_NONE && m.correct,
                   mode == D_NONE && m.stale > 0, m.stale, 4 * cfg.blocks, m.trace_ok);
            if (rep == cfg.repeats - 1) {
                diamond_dump_trace(cfg, ctx, mode, epoch, trace_header);
                trace_header = false;
            }
        }
    }
    for (int mode : modes) {
        T23CI tci = t23_bootstrap_median_ci(times[(size_t)mode], 0x7400 + mode);
        T23CI oci = t23_bootstrap_median_ci(overlaps[(size_t)mode], 0x7500 + mode);
        T23CI pci = t23_bootstrap_median_ci(polls[(size_t)mode], 0x7600 + mode);
        printf("SUMMARY_TIER23 semantics=%d experiment=diamond tag=%s mode=%s repeats=%d "
               "median_ms=%.9f ci_ms_lo=%.9f ci_ms_hi=%.9f "
               "median_branch_overlap_ms=%.9f ci_overlap_lo=%.9f ci_overlap_hi=%.9f "
               "median_poll_loads=%.3f ci_poll_lo=%.3f ci_poll_hi=%.3f valid=1\n",
               T23_SEMANTICS, cfg.tag.c_str(), diamond_mode_name(mode), cfg.repeats,
               t23_median(times[(size_t)mode]), tci.lo, tci.hi,
               t23_median(overlaps[(size_t)mode]), oci.lo, oci.hi,
               t23_median(polls[(size_t)mode]), pci.lo, pci.hi);
    }
    printf("TRACE_TIER23 semantics=%d experiment=diamond tag=%s path=%s modes=4 "
           "rows_per_mode=%d final_epoch=%llu\n", T23_SEMANTICS, cfg.tag.c_str(),
           cfg.trace_path.c_str(), 4 * cfg.blocks, epoch);
    cudaFree(ctx.data); cudaFree(ctx.flags); cudaFree(ctx.errors); cudaFree(ctx.trace);
    for (auto s : ctx.streams) cudaStreamDestroy(s);
    return 0;
}
