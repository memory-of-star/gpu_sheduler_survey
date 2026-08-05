// tier23_protocol_encoding.cu -- admissible native CUDA harness for plan §7.1/§7.3.
//
// §7.1 fixes a one-to-one relation and compares grid PDL, fixed-interval polling,
// exponential backoff, and an identity-safe monotonic contiguous-prefix counter.  The
// historical global cardinality counter is deliberately not used.
//
// §7.3 compares interval, bitmask, and CSR dependency encodings over independently chosen
// interval/strided structures and degrees.  A dry metadata pass records decode latency; the
// actual wait is a second pass, and post-wait ordinary work remains O(1).  Full-edge checking
// is a separate untimed invocation.

#include "common/tier23_native.cuh"

#include <cuda/atomic>
#include <climits>
#include <fstream>
#include <numeric>

enum PEMode {
    PE_GRID = 0,
    PE_FIXED = 1,
    PE_BACKOFF = 2,
    PE_PREFIX = 3,
    PE_INTERVAL = 4,
    PE_BITMASK = 5,
    PE_CSR = 6,
    PE_NONE = 7,
};

static const char* pe_mode_name(int mode) {
    switch (mode) {
        case PE_GRID: return "grid";
        case PE_FIXED: return "fixed-spin";
        case PE_BACKOFF: return "backoff";
        case PE_PREFIX: return "monotonic-prefix";
        case PE_INTERVAL: return "interval";
        case PE_BITMASK: return "bitmask";
        case PE_CSR: return "csr";
        case PE_NONE: return "none";
        default: return "?";
    }
}

__device__ __forceinline__ unsigned long long pe_flag_load(
        const unsigned long long* flags, int parent, unsigned long long* loads) {
    cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> f(flags[parent]);
    ++*loads;
    return f.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void pe_wait_flag(const unsigned long long* flags,
                                             int parent,
                                             unsigned long long epoch,
                                             int policy,
                                             unsigned long long* loads) {
    unsigned int ns = 64;
    while (pe_flag_load(flags, parent, loads) < epoch) {
        if (policy == PE_FIXED) {
            __nanosleep(64);
        } else {
            __nanosleep(ns);
            ns = ns < 2048 ? ns * 2 : 2048;
        }
    }
}

__device__ __forceinline__ void pe_publish_flag(unsigned long long* flags,
                                                int block,
                                                unsigned long long epoch) {
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> f(flags[block]);
    f.store(epoch, cuda::memory_order_release);
}

// Advance only across a contiguous prefix whose individual identities have each been
// acquired.  This is a transitive release/acquire chain and cannot be fooled by a high-ID
// CTA finishing early, unlike the rejected completion-count >= hi+1 protocol.
__device__ __forceinline__ unsigned long long pe_advance_prefix(
        unsigned long long* prefix,
        const unsigned long long* flags,
        int nproducer,
        unsigned long long epoch) {
    unsigned long long scans = 0;
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> p(*prefix);
    for (;;) {
        unsigned long long cur = p.load(cuda::memory_order_acquire);
        if (cur >= (unsigned long long)nproducer) break;
        if (pe_flag_load(flags, (int)cur, &scans) < epoch) break;
        unsigned long long expected = cur;
        (void)p.compare_exchange_strong(expected, cur + 1,
                                        cuda::memory_order_acq_rel,
                                        cuda::memory_order_acquire);
    }
    return scans;
}

__global__ void pe_init(unsigned long long* data,
                        unsigned long long* out,
                        int* errors,
                        int nproducer,
                        int nconsumer,
                        unsigned long long epoch) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < nproducer) data[i] = ~t23_value(epoch, (unsigned)i, 0);
    if (i < nconsumer) out[i] = ~t23_value(epoch, (unsigned)i, 7);
    if (i == 0) *errors = 0;
}

__global__ void pe_producer(unsigned long long* data,
                            unsigned long long* flags,
                            unsigned long long* prefix,
                            const unsigned long long* ceiling_observed,
                            T23TraceRecord* trace,
                            int nproducer,
                            int ceiling_sentinel_parent,
                            int mode,
                            unsigned long long epoch,
                            unsigned long long ready_cycles,
                            unsigned long long tail_cycles,
                            int skew_bins) {
    int block = (int)blockIdx.x;
    t23_trace_begin(trace, (unsigned)block, 0);
    __syncthreads();

    // Impl/Ceiling launch eligibility is signaled at entry.  Floor deliberately delays the
    // trigger until this CTA's datum is ready.
    if (mode != PE_GRID) cudaTriggerProgrammaticLaunchCompletion();

    unsigned int bucket = skew_bins > 1
        ? dep_hash((unsigned int)block ^ (unsigned int)epoch) % (unsigned int)skew_bins
        : 0u;
    unsigned long long delay = ready_cycles +
        (skew_bins > 1 ? ready_cycles * bucket / (unsigned long long)skew_bins : 0ull);
    spin_cycles(delay);
    unsigned long long ceiling_proof_loads = 0;
    if (threadIdx.x == 0 && mode == PE_NONE && block == ceiling_sentinel_parent) {
        // Deterministic adversarial Ceiling schedule: this real parent retains the exact
        // producer store/work, but cannot publish its datum until child 0 has performed the
        // omitted-wait RAW snapshot.  Only the producer waits; the unsafe consumer does not.
        cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> observed(
            *ceiling_observed);
        while (true) {
            ++ceiling_proof_loads;
            if (observed.load(cuda::memory_order_acquire) >= epoch) break;
            __nanosleep(64);
        }
    }
    if (threadIdx.x == 0) data[block] = t23_value(epoch, (unsigned)block, 0);
    __syncthreads();

    unsigned long long prefix_scans = 0;
    if (threadIdx.x == 0) {
        if (mode != PE_NONE && mode != PE_GRID) pe_publish_flag(flags, block, epoch);
        // Record readiness immediately after the release publication (or after the datum
        // for grid/Ceiling), before prefix maintenance.  A consumer that acquires prefix
        // can therefore never appear to return before the trace's readiness timestamp.
        trace[block].t_ready = ctatrace_globaltimer();
        if (mode == PE_PREFIX)
            prefix_scans = pe_advance_prefix(prefix, flags, nproducer, epoch);
        trace[block].poll_loads = prefix_scans;
        trace[block].metadata_loads = ceiling_proof_loads;
    }
    __syncthreads();
    if (mode == PE_GRID) cudaTriggerProgrammaticLaunchCompletion();

    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) trace[block].t_end = ctatrace_globaltimer();
}

__device__ __forceinline__ unsigned long long pe_validation_value(
        const unsigned long long* data, const DepPattern& pat, int child) {
    unsigned long long h = 1469598103934665603ull;
    int degree = dep_degree(pat, child);
    for (int k = 0; k < degree; ++k) {
        int p = dep_parent(pat, child, k);
        h = t23_mix64(h ^ data[p] ^
                      ((unsigned long long)(unsigned)p + 1ull) * 0x9e3779b97f4a7c15ull);
    }
    return h;
}

__global__ void pe_consumer(unsigned long long* out,
                            unsigned long long* decode_sink,
                            unsigned long long* ceiling_observed,
                            const unsigned long long* data,
                            const unsigned long long* flags,
                            const unsigned long long* prefix,
                            const int* interval_lo,
                            const int* interval_hi,
                            const unsigned int* bitmask,
                            int mask_words,
                            const int* csr_offsets,
                            const int* csr_parents,
                            T23TraceRecord* trace,
                            DepPattern pat,
                            int ceiling_sentinel_parent,
                            int mode,
                            unsigned long long epoch,
                            unsigned long long prologue_cycles,
                            unsigned long long epilogue_cycles,
                            int validate,
                            int* errors) {
    extern __shared__ unsigned char shared_touch[];
    int child = (int)blockIdx.x;
    if (threadIdx.x == 0) shared_touch[0] = (unsigned char)child;
    t23_trace_begin(trace, (unsigned)child, 1);
    __syncthreads();

    unsigned long long poll_loads = 0;
    unsigned long long metadata_loads = 0;
    unsigned long long decode_checksum = 0;
    unsigned long long decode_begin = 0, decode_end = 0;
    if (threadIdx.x == 0) {
        trace[child].t_wait_begin = ctatrace_globaltimer();
        decode_begin = ctatrace_globaltimer();

        // First pass: representation decoding only.  The volatile sink prevents the
        // compiler from deleting it.  Waiting is a second pass so decode_ns excludes spin.
        if (mode == PE_INTERVAL) {
            int lo = interval_lo[child];
            int hi = interval_hi[child];
            metadata_loads += 2;
            decode_checksum = ((unsigned long long)(unsigned)lo << 32) ^ (unsigned)hi;
        } else if (mode == PE_BITMASK) {
            const unsigned int* row = bitmask + (size_t)child * mask_words;
            for (int w = 0; w < mask_words; ++w) {
                unsigned int bits = row[w];
                ++metadata_loads;
                decode_checksum = t23_mix64(decode_checksum ^
                    ((unsigned long long)bits << 32) ^ (unsigned)w);
            }
        } else if (mode == PE_CSR) {
            int begin = csr_offsets[child];
            int end = csr_offsets[child + 1];
            metadata_loads += 2;
            for (int i = begin; i < end; ++i) {
                int p = csr_parents[i];
                ++metadata_loads;
                decode_checksum = t23_mix64(decode_checksum ^ (unsigned)p);
            }
        } else if (mode == PE_FIXED || mode == PE_BACKOFF || mode == PE_PREFIX) {
            decode_checksum = (unsigned)child;
        }
        decode_end = ctatrace_globaltimer();
        decode_sink[child] = decode_checksum;

        if (mode == PE_GRID) {
            cudaGridDependencySynchronize();
        } else if (mode == PE_FIXED || mode == PE_BACKOFF) {
            pe_wait_flag(flags, child, epoch, mode, &poll_loads);
        } else if (mode == PE_PREFIX) {
            cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> p(*prefix);
            while (true) {
                ++poll_loads;
                if (p.load(cuda::memory_order_acquire) >= (unsigned long long)child + 1ull)
                    break;
                __nanosleep(64);
            }
        } else if (mode == PE_INTERVAL) {
            int lo = interval_lo[child];
            int hi = interval_hi[child];
            metadata_loads += 2;
            for (int p = lo; p <= hi; ++p)
                pe_wait_flag(flags, p, epoch, PE_BACKOFF, &poll_loads);
        } else if (mode == PE_BITMASK) {
            const unsigned int* row = bitmask + (size_t)child * mask_words;
            for (int w = 0; w < mask_words; ++w) {
                unsigned int bits = row[w];
                ++metadata_loads;
                while (bits) {
                    int bit = __ffs((int)bits) - 1;
                    int p = w * 32 + bit;
                    if (p < pat.n_producer)
                        pe_wait_flag(flags, p, epoch, PE_BACKOFF, &poll_loads);
                    bits &= bits - 1;
                }
            }
        } else if (mode == PE_CSR) {
            int begin = csr_offsets[child];
            int end = csr_offsets[child + 1];
            metadata_loads += 2;
            for (int i = begin; i < end; ++i) {
                int p = csr_parents[i];
                ++metadata_loads;
                pe_wait_flag(flags, p, epoch, PE_BACKOFF, &poll_loads);
            }
        }

        // The RAW snapshot belongs immediately after the dependency operation.  In
        // particular, PE_NONE must observe the still-poisoned datum before any common
        // consumer work can delay it until the producer happens to finish.  Correct modes
        // likewise snapshot immediately after their acquire/grid wait.  The common
        // prologue remains symmetric, but is intentionally downstream of this snapshot.
        if (validate) {
            unsigned long long observed = pe_validation_value(data, pat, child);
            out[child] = observed;
            // Device-side readiness proof: every true edge must carry this invocation's
            // epoch value.  Host and strict validator independently recompute the digest.
            int degree = dep_degree(pat, child);
            for (int k = 0; k < degree; ++k) {
                int p = dep_parent(pat, child, k);
                if (data[p] != t23_value(epoch, (unsigned)p, 0)) atomicExch(errors, 1);
            }
        } else {
            int degree = dep_degree(pat, child);
            int parent = degree > 0 ? dep_parent(pat, child, degree - 1) : -1;
            out[child] = parent >= 0 ? data[parent] : 0ull;  // O(1) post-wait payload
        }

        // t_dep is the actual post-wait/post-RAW-snapshot timestamp, rather than a marker
        // placed just before the load.  Record it before releasing the proof latch so the
        // final trace can establish child0.snapshot <= sentinel_parent.ready.
        trace[child].t_dep = ctatrace_globaltimer();
        trace[child].poll_loads = poll_loads;
        trace[child].metadata_loads = metadata_loads;
        trace[child].decode_ns = decode_end - decode_begin;
        if (mode == PE_NONE && child == 0) {
            int degree = dep_degree(pat, child);
            int parent = degree > 0 ? dep_parent(pat, child, degree - 1) : -1;
            if (parent != ceiling_sentinel_parent) atomicExch(errors, 1);
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> observed(
                *ceiling_observed);
            observed.store(epoch, cuda::memory_order_release);
        }
    }
    __syncthreads();
    spin_cycles(prologue_cycles);
    __syncthreads();
    spin_cycles(epilogue_cycles);
    __syncthreads();
    if (threadIdx.x == 0) trace[child].t_end = ctatrace_globaltimer();
}

__global__ void pe_background(unsigned int* buffer,
                              size_t words,
                              unsigned int iterations,
                              T23TraceRecord* trace) {
    int block = (int)blockIdx.x;
    t23_trace_begin(trace, (unsigned)block, 2);
    __syncthreads();
    size_t lane = (size_t)block * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned int x = (unsigned int)(lane + 1);
    for (unsigned int i = 0; i < iterations; ++i) {
        size_t index = (lane + (size_t)i * stride * 17u) % words;
        unsigned int v = buffer[index];
        x = dep_hash(x ^ v ^ i);
        buffer[index] = x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        trace[block].t_ready = trace[block].t_start;
        trace[block].t_wait_begin = trace[block].t_start;
        trace[block].t_dep = trace[block].t_start;
        trace[block].t_end = ctatrace_globaltimer();
    }
}

struct PECfg {
    std::string experiment = "protocol";
    std::string tag = "tier23";
    std::string trace_path;
    int nproducer = 148;
    int nconsumer = 148;
    int threads = 128;
    int repeats = 31;
    int warmup = 3;
    int structure = DEP_SELF;
    int degree = 1;
    int skew_bins = 8;
    int consumer_smem_kb = 32;
    int background_blocks = 296;
    unsigned int background_iterations = 50000;
    unsigned long long ready_cycles = 600000;
    unsigned long long tail_cycles = 800000;
    unsigned long long prologue_cycles = 100000;
    unsigned long long epilogue_cycles = 400000;
    bool allow_short = false;
};

struct PEContext {
    unsigned long long *data = nullptr, *out = nullptr, *flags = nullptr;
    unsigned long long *prefix = nullptr, *decode_sink = nullptr;
    unsigned long long *ceiling_observed = nullptr;
    int *error = nullptr, *ilo = nullptr, *ihi = nullptr;
    unsigned int* bitmask = nullptr;
    int *csr_offsets = nullptr, *csr_parents = nullptr;
    unsigned int* background = nullptr;
    T23TraceRecord *ptrace = nullptr, *ctrace = nullptr, *btrace = nullptr;
    int mask_words = 0;
    size_t background_words = 0;
    cudaStream_t dependency_stream{}, background_stream{};
};

struct PEMetrics {
    double ms = 0.0;
    double wait_ns = 0.0;
    double wake_ns = 0.0;
    double decode_ns = 0.0;
    double background_ms = 0.0;
    double background_gbps = 0.0;
    unsigned long long poll_loads = 0;
    unsigned long long metadata_loads = 0;
    unsigned long long ceiling_schedule_latch_loads = 0;
    unsigned long long observed_digest = 0;
    unsigned long long expected_digest = 0;
    unsigned int stale = 0;
    unsigned int trace_rows = 0;
    unsigned int background_overlap_rows = 0;
    bool correct = false;
    bool trace_ok = false;
};

static std::vector<unsigned long long> pe_expected_outputs(const PECfg& cfg,
                                                           const DepPattern& pat,
                                                           unsigned long long epoch,
                                                           bool validate) {
    std::vector<unsigned long long> values((size_t)cfg.nconsumer);
    for (int child = 0; child < cfg.nconsumer; ++child) {
        if (validate) {
            unsigned long long h = 1469598103934665603ull;
            int degree = dep_degree(pat, child);
            for (int k = 0; k < degree; ++k) {
                int p = dep_parent(pat, child, k);
                h = t23_mix64(h ^ t23_value(epoch, (unsigned)p, 0) ^
                              ((unsigned long long)(unsigned)p + 1ull) *
                                  0x9e3779b97f4a7c15ull);
            }
            values[(size_t)child] = h;
        } else {
            int degree = dep_degree(pat, child);
            int p = degree > 0 ? dep_parent(pat, child, degree - 1) : -1;
            values[(size_t)child] = p >= 0 ? t23_value(epoch, (unsigned)p, 0) : 0ull;
        }
    }
    return values;
}

static void pe_build_metadata(const PECfg& cfg, const DepPattern& pat,
                              std::vector<int>* ilo, std::vector<int>* ihi,
                              std::vector<unsigned int>* mask,
                              std::vector<int>* offsets,
                              std::vector<int>* parents) {
    ilo->resize((size_t)cfg.nconsumer);
    ihi->resize((size_t)cfg.nconsumer);
    int words = (cfg.nproducer + 31) / 32;
    mask->assign((size_t)cfg.nconsumer * words, 0u);
    offsets->resize((size_t)cfg.nconsumer + 1);
    parents->clear();
    for (int child = 0; child < cfg.nconsumer; ++child) {
        dep_interval(pat, child, &(*ilo)[(size_t)child], &(*ihi)[(size_t)child]);
        (*offsets)[(size_t)child] = (int)parents->size();
        int degree = dep_degree(pat, child);
        for (int k = 0; k < degree; ++k) {
            int p = dep_parent(pat, child, k);
            parents->push_back(p);
            (*mask)[(size_t)child * words + p / 32] |= 1u << (p % 32);
        }
    }
    (*offsets)[(size_t)cfg.nconsumer] = (int)parents->size();
}

static void pe_reset(const PECfg& cfg, PEContext& ctx, unsigned long long epoch) {
    int n = std::max(cfg.nproducer, cfg.nconsumer);
    pe_init<<<(n + 255) / 256, 256, 0, ctx.dependency_stream>>>(
        ctx.data, ctx.out, ctx.error, cfg.nproducer, cfg.nconsumer, epoch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(ctx.flags, 0,
                              (size_t)cfg.nproducer * sizeof(unsigned long long),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.prefix, 0, sizeof(unsigned long long),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.ceiling_observed, 0, sizeof(unsigned long long),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.decode_sink, 0,
                              (size_t)cfg.nconsumer * sizeof(unsigned long long),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.ptrace, 0,
                              (size_t)cfg.nproducer * sizeof(T23TraceRecord),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.ctrace, 0,
                              (size_t)cfg.nconsumer * sizeof(T23TraceRecord),
                              ctx.dependency_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.btrace, 0,
                              (size_t)cfg.background_blocks * sizeof(T23TraceRecord),
                              ctx.background_stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.background, (int)(epoch & 0xffu),
                              ctx.background_words * sizeof(unsigned int),
                              ctx.background_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.dependency_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.background_stream));
}

static int pe_required_parent_hi(int mode, const PECfg& cfg, const DepPattern& pat,
                                 int child, const std::vector<int>& ilo,
                                 const std::vector<int>& ihi) {
    if (mode == PE_GRID) return cfg.nproducer - 1;
    if (mode == PE_PREFIX) return child;
    if (mode == PE_FIXED || mode == PE_BACKOFF) return child;
    if (mode == PE_INTERVAL) return ihi[(size_t)child];
    int degree = dep_degree(pat, child);
    int last = -1;
    for (int k = 0; k < degree; ++k) last = std::max(last, dep_parent(pat, child, k));
    return last;
}

static PEMetrics pe_collect(const PECfg& cfg, const DepPattern& pat, PEContext& ctx,
                            int mode, unsigned long long epoch, bool validate,
                            const std::vector<int>& ilo, const std::vector<int>& ihi) {
    std::vector<T23TraceRecord> p((size_t)cfg.nproducer), c((size_t)cfg.nconsumer),
                                b((size_t)cfg.background_blocks);
    std::vector<unsigned long long> out((size_t)cfg.nconsumer);
    CUDA_CHECK(cudaMemcpy(p.data(), ctx.ptrace, p.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.data(), ctx.ctrace, c.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), ctx.btrace, b.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.data(), ctx.out, out.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    int device_error = 1;
    CUDA_CHECK(cudaMemcpy(&device_error, ctx.error, sizeof(int), cudaMemcpyDeviceToHost));

    PEMetrics m;
    unsigned long long first = ULLONG_MAX, last = 0, p_last = 0;
    std::vector<double> waits, wakes, decodes;
    for (int i = 0; i < cfg.nproducer; ++i) {
        const auto& r = p[(size_t)i];
        bool ok = r.block_id == (unsigned)i && r.kernel_id == 0 && r.t_start &&
                  r.t_ready && r.t_end && r.t_start <= r.t_ready && r.t_ready <= r.t_end;
        m.trace_ok = i == 0 ? ok : (m.trace_ok && ok);
        first = std::min(first, r.t_start);
        last = std::max(last, r.t_end);
        p_last = std::max(p_last, r.t_end);
        m.poll_loads += r.poll_loads;
        m.ceiling_schedule_latch_loads += r.metadata_loads;
        ++m.trace_rows;
    }
    for (int i = 0; i < cfg.nconsumer; ++i) {
        const auto& r = c[(size_t)i];
        bool ok = r.block_id == (unsigned)i && r.kernel_id == 1 && r.t_start &&
                  r.t_wait_begin && r.t_dep && r.t_end && r.t_start <= r.t_wait_begin &&
                  r.t_wait_begin <= r.t_dep && r.t_dep <= r.t_end;
        m.trace_ok = m.trace_ok && ok;
        first = std::min(first, r.t_start);
        last = std::max(last, r.t_end);
        waits.push_back((double)(r.t_dep - r.t_wait_begin));
        decodes.push_back((double)r.decode_ns);
        unsigned long long satisfied = 0;
        if (mode == PE_GRID) {
            satisfied = p_last;
        } else if (mode != PE_NONE) {
            if (mode == PE_INTERVAL) {
                for (int parent = ilo[(size_t)i]; parent <= ihi[(size_t)i]; ++parent)
                    satisfied = std::max(satisfied, p[(size_t)parent].t_ready);
            } else if (mode == PE_PREFIX || mode == PE_FIXED || mode == PE_BACKOFF) {
                int hi = pe_required_parent_hi(mode, cfg, pat, i, ilo, ihi);
                if (mode == PE_PREFIX) {
                    for (int parent = 0; parent <= hi; ++parent)
                        satisfied = std::max(satisfied, p[(size_t)parent].t_ready);
                } else {
                    satisfied = p[(size_t)hi].t_ready;
                }
            } else {
                int degree = dep_degree(pat, i);
                for (int k = 0; k < degree; ++k)
                    satisfied = std::max(satisfied,
                        p[(size_t)dep_parent(pat, i, k)].t_ready);
            }
            if (r.t_dep < satisfied) m.trace_ok = false;
            wakes.push_back((double)(r.t_dep - satisfied));
        }
    if (mode == PE_GRID && r.t_dep < p_last) m.trace_ok = false;
        m.poll_loads += r.poll_loads;
        m.metadata_loads += r.metadata_loads;
        ++m.trace_rows;
    }
    if (first == ULLONG_MAX || last <= first) m.trace_ok = false;
    m.ms = m.trace_ok ? (double)(last - first) / 1.0e6 : 0.0;
    m.wait_ns = t23_median(waits);
    m.wake_ns = wakes.empty() ? 0.0 : t23_median(wakes);
    m.decode_ns = t23_median(decodes);

    if (mode == PE_NONE) {
        int degree = dep_degree(pat, 0);
        int sentinel_parent = degree > 0 ? dep_parent(pat, 0, degree - 1) : -1;
        if (sentinel_parent < 0 || c[0].t_dep > p[(size_t)sentinel_parent].t_ready)
            m.trace_ok = false;
    }

    unsigned long long bg_first = ULLONG_MAX, bg_last = 0;
    for (int i = 0; i < cfg.background_blocks; ++i) {
        const auto& r = b[(size_t)i];
        bool ok = r.block_id == (unsigned)i && r.kernel_id == 2 && r.t_start && r.t_end &&
                  r.t_start <= r.t_end;
        m.trace_ok = m.trace_ok && ok;
        bg_first = std::min(bg_first, r.t_start);
        bg_last = std::max(bg_last, r.t_end);
        if (r.t_start < last && r.t_end > first) ++m.background_overlap_rows;
        ++m.trace_rows;
    }
    if (bg_first == ULLONG_MAX || bg_last <= bg_first || m.background_overlap_rows == 0)
        m.trace_ok = false;
    m.background_ms = (double)(bg_last - bg_first) / 1.0e6;
    double bytes = (double)cfg.background_blocks * cfg.threads *
                   cfg.background_iterations * 2.0 * sizeof(unsigned int);
    m.background_gbps = bytes / (double)(bg_last - bg_first);

    std::vector<unsigned long long> expected = pe_expected_outputs(cfg, pat, epoch, validate);
    m.observed_digest = t23_digest(out);
    m.expected_digest = t23_digest(expected);
    m.correct = device_error == 0 && out == expected;
    if (device_error != 0) m.trace_ok = false;
    if (mode == PE_NONE) {
        for (size_t i = 0; i < out.size(); ++i) if (out[i] != expected[i]) ++m.stale;
        m.correct = false;
    }
    return m;
}

static PEMetrics pe_once(const PECfg& cfg, const DepPattern& pat, PEContext& ctx,
                         int mode, unsigned long long epoch, bool validate,
                         const std::vector<int>& ilo, const std::vector<int>& ihi) {
    pe_reset(cfg, ctx, epoch);

    pe_background<<<cfg.background_blocks, cfg.threads, 0, ctx.background_stream>>>(
        ctx.background, ctx.background_words, cfg.background_iterations, ctx.btrace);
    CUDA_CHECK(cudaGetLastError());
    int degree0 = dep_degree(pat, 0);
    int ceiling_sentinel_parent = degree0 > 0 ? dep_parent(pat, 0, degree0 - 1) : -1;
    pe_producer<<<cfg.nproducer, cfg.threads, 0, ctx.dependency_stream>>>(
        ctx.data, ctx.flags, ctx.prefix, ctx.ceiling_observed, ctx.ptrace, cfg.nproducer,
        ceiling_sentinel_parent, mode, epoch,
        cfg.ready_cycles, cfg.tail_cycles, cfg.skew_bins);
    CUDA_CHECK(cudaGetLastError());

    void* args[] = {
        &ctx.out, &ctx.decode_sink, &ctx.ceiling_observed, &ctx.data, &ctx.flags, &ctx.prefix,
        &ctx.ilo, &ctx.ihi, &ctx.bitmask, &ctx.mask_words,
        &ctx.csr_offsets, &ctx.csr_parents, &ctx.ctrace,
        const_cast<DepPattern*>(&pat), &ceiling_sentinel_parent, &mode, &epoch,
        const_cast<unsigned long long*>(&cfg.prologue_cycles),
        const_cast<unsigned long long*>(&cfg.epilogue_cycles),
        &validate, &ctx.error,
    };
    int validate_arg = validate ? 1 : 0;
    args[19] = &validate_arg;
    CUDA_CHECK(t23_launch_pss(ctx.dependency_stream, dim3(cfg.nconsumer),
                             dim3(cfg.threads), (size_t)cfg.consumer_smem_kb * 1024u,
                             (const void*)pe_consumer, args));
    CUDA_CHECK(cudaStreamSynchronize(ctx.dependency_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.background_stream));
    return pe_collect(cfg, pat, ctx, mode, epoch, validate, ilo, ihi);
}

static void pe_dump_trace(const PECfg& cfg, int mode, unsigned long long epoch,
                          PEContext& ctx, bool header) {
    std::vector<T23TraceRecord> p((size_t)cfg.nproducer), c((size_t)cfg.nconsumer),
                                b((size_t)cfg.background_blocks);
    CUDA_CHECK(cudaMemcpy(p.data(), ctx.ptrace, p.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.data(), ctx.ctrace, c.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), ctx.btrace, b.size() * sizeof(T23TraceRecord),
                          cudaMemcpyDeviceToHost));
    std::ofstream f(cfg.trace_path, header ? std::ios::out : std::ios::app);
    if (!f) {
        fprintf(stderr, "cannot open trace path %s\n", cfg.trace_path.c_str());
        exit(1);
    }
    if (header)
        f << "tag,experiment,mode,epoch,kernel_id,block_id,sm_id,t_start,t_ready,"
             "t_wait_begin,t_dep,t_end,poll_loads,metadata_loads,decode_ns,aux\n";
    auto rows = [&](const std::vector<T23TraceRecord>& v) {
        for (const auto& r : v)
            f << cfg.tag << ',' << cfg.experiment << ',' << pe_mode_name(mode) << ','
              << epoch << ',' << r.kernel_id << ',' << r.block_id << ',' << r.sm_id << ','
              << r.t_start << ',' << r.t_ready << ',' << r.t_wait_begin << ',' << r.t_dep
              << ',' << r.t_end << ',' << r.poll_loads << ',' << r.metadata_loads << ','
              << r.decode_ns << ',' << r.aux << '\n';
    };
    rows(p); rows(c); rows(b);
}

static void pe_allocate(const PECfg& cfg, const std::vector<int>& ilo,
                        const std::vector<int>& ihi,
                        const std::vector<unsigned int>& mask,
                        const std::vector<int>& offsets,
                        const std::vector<int>& parents, PEContext* ctx) {
    ctx->mask_words = (cfg.nproducer + 31) / 32;
    ctx->background_words = (size_t)16 * 1024 * 1024; // 64 MiB L2/DRAM working set
    CUDA_CHECK(cudaMalloc(&ctx->data, (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->out, (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->flags, (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->prefix, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->ceiling_observed, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->decode_sink,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx->error, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->ilo, ilo.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->ihi, ihi.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->bitmask, mask.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->csr_offsets, offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->csr_parents, parents.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->background,
                          ctx->background_words * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx->ptrace,
                          (size_t)cfg.nproducer * sizeof(T23TraceRecord)));
    CUDA_CHECK(cudaMalloc(&ctx->ctrace,
                          (size_t)cfg.nconsumer * sizeof(T23TraceRecord)));
    CUDA_CHECK(cudaMalloc(&ctx->btrace,
                          (size_t)cfg.background_blocks * sizeof(T23TraceRecord)));
    CUDA_CHECK(cudaMemcpy(ctx->ilo, ilo.data(), ilo.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->ihi, ihi.data(), ihi.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->bitmask, mask.data(), mask.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->csr_offsets, offsets.data(), offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->csr_parents, parents.data(), parents.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx->dependency_stream,
                                            cudaStreamNonBlocking, greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx->background_stream,
                                            cudaStreamNonBlocking, least));
}

static void pe_free(PEContext& ctx) {
    cudaFree(ctx.data); cudaFree(ctx.out); cudaFree(ctx.flags); cudaFree(ctx.prefix);
    cudaFree(ctx.decode_sink); cudaFree(ctx.ceiling_observed); cudaFree(ctx.error);
    cudaFree(ctx.ilo); cudaFree(ctx.ihi);
    cudaFree(ctx.bitmask); cudaFree(ctx.csr_offsets); cudaFree(ctx.csr_parents);
    cudaFree(ctx.background); cudaFree(ctx.ptrace); cudaFree(ctx.ctrace); cudaFree(ctx.btrace);
    cudaStreamDestroy(ctx.dependency_stream); cudaStreamDestroy(ctx.background_stream);
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf("usage: tier23_protocol_encoding --experiment protocol|encoding [options]\n"
               "  --tag T --trace PATH --producers P --consumers C\n"
               "  --structure self|interval|strided --degree D\n"
               "  --repeats N --warmup N [--allow-short]\n"
               "  --ready CYC --tail CYC --prologue CYC --epilogue CYC\n"
               "  --background-blocks N --background-iterations N\n");
        return 0;
    }

    PECfg cfg;
    cfg.experiment = args.str("--experiment", cfg.experiment.c_str());
    cfg.tag = args.str("--tag", cfg.tag.c_str());
    cfg.trace_path = args.str("--trace", "tier23_protocol_encoding_trace.csv");
    cfg.nproducer = (int)args.ll("--producers", cfg.nproducer);
    cfg.nconsumer = (int)args.ll("--consumers", cfg.nproducer);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.warmup = (int)args.ll("--warmup", cfg.warmup);
    cfg.degree = (int)args.ll("--degree", cfg.degree);
    cfg.structure = depStructureFromName(args.str("--structure",
        cfg.experiment == "protocol" ? "self" : "interval"));
    cfg.skew_bins = (int)args.ll("--skew-bins", cfg.skew_bins);
    cfg.consumer_smem_kb = (int)args.ll("--consumer-smem-kb", cfg.consumer_smem_kb);
    cfg.background_blocks = (int)args.ll("--background-blocks", cfg.background_blocks);
    cfg.background_iterations = (unsigned int)args.ll(
        "--background-iterations", cfg.background_iterations);
    cfg.ready_cycles = (unsigned long long)args.ll("--ready", cfg.ready_cycles);
    cfg.tail_cycles = (unsigned long long)args.ll("--tail", cfg.tail_cycles);
    cfg.prologue_cycles = (unsigned long long)args.ll("--prologue", cfg.prologue_cycles);
    cfg.epilogue_cycles = (unsigned long long)args.ll("--epilogue", cfg.epilogue_cycles);
    cfg.allow_short = args.has("--allow-short");

    if ((cfg.experiment != "protocol" && cfg.experiment != "encoding") ||
        cfg.nproducer <= 0 || cfg.nconsumer <= 0 || cfg.nproducer != cfg.nconsumer ||
        cfg.threads <= 0 || cfg.threads > 1024 || cfg.degree <= 0 ||
        cfg.degree > cfg.nproducer || cfg.structure < 0 || cfg.repeats <= 0 ||
        cfg.warmup < 0 || cfg.background_blocks <= 0 || cfg.background_iterations == 0 ||
        cfg.consumer_smem_kb <= 0 || cfg.trace_path.empty()) {
        fprintf(stderr, "invalid Tier 2/3 protocol/encoding configuration\n");
        return 2;
    }
    if (!t23_short_allowed(cfg.repeats, cfg.warmup, cfg.allow_short)) {
        t23_print_short_error(cfg.repeats, cfg.warmup);
        return 2;
    }
    if (cfg.experiment == "protocol" &&
        (cfg.structure != DEP_SELF || cfg.degree != 1)) {
        fprintf(stderr, "protocol experiment is fixed to structure=self degree=1\n");
        return 2;
    }
    if (cfg.experiment == "encoding" &&
        (cfg.structure != DEP_INTERVAL && cfg.structure != DEP_STRIDED)) {
        fprintf(stderr, "encoding experiment requires interval or strided structure\n");
        return 2;
    }

    DeviceInfo dev = queryDevice();
    if (dev.major < 9) {
        fprintf(stderr, "grid PDL requires compute capability >= 9.0\n");
        return 2;
    }
    if (cfg.background_blocks == 296) cfg.background_blocks = 2 * dev.sms;
    printDeviceBanner(dev);
    DepPattern pat{cfg.structure, cfg.degree, cfg.nproducer, cfg.nconsumer, 0x12345u};
    if (!dep_parents_are_unique(pat)) {
        fprintf(stderr, "dependency metadata contains duplicate/out-of-range parents\n");
        return 2;
    }
    std::vector<int> ilo, ihi, offsets, parents;
    std::vector<unsigned int> mask;
    pe_build_metadata(cfg, pat, &ilo, &ihi, &mask, &offsets, &parents);
    int degree0 = dep_degree(pat, 0);
    int ceiling_sentinel_parent = degree0 > 0 ? dep_parent(pat, 0, degree0 - 1) : -1;
    if (ceiling_sentinel_parent < 0) {
        fprintf(stderr, "child0 has no sentinel parent\n");
        return 2;
    }

    size_t dynamic_smem = (size_t)cfg.consumer_smem_kb * 1024u;
    CUDA_CHECK(cudaFuncSetAttribute(pe_consumer,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynamic_smem));
    int producer_occ = 0, consumer_occ = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &producer_occ, pe_producer, cfg.threads, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &consumer_occ, pe_consumer, cfg.threads, dynamic_smem));
    if (producer_occ < 1 || consumer_occ < 1) {
        fprintf(stderr, "invalid producer/consumer occupancy\n");
        return 2;
    }

    const std::vector<int> modes = cfg.experiment == "protocol"
        ? std::vector<int>{PE_GRID, PE_FIXED, PE_BACKOFF, PE_PREFIX, PE_NONE}
        : std::vector<int>{PE_GRID, PE_INTERVAL, PE_BITMASK, PE_CSR, PE_NONE};
    printf("CONFIG_TIER23 semantics=%d experiment=%s tag=%s device=%s sm=%d cc=%d.%d "
           "P=%d C=%d structure=%s degree=%d eff_degree=%.6f tightness=%.9f "
           "threads=%d consumer_smem_kb=%d producer_occ=%d consumer_occ=%d "
           "warmup=%d repeats=%d timer=globaltimer bootstrap=%d "
           "trigger_floor=ready trigger_impl=entry trigger_ceiling=entry "
           "publication_impl=after_data_release publication_ceiling=none "
           "ceiling_schedule=deterministic_device_sentinel_RAW "
           "ceiling_proof=adversarial_device_sentinel_raw_before_store "
           "ceiling_sentinel_child=0 ceiling_sentinel_parent=%d "
           "ceiling_proof_timing=included "
           "postwait_payload=O(1) validation=full_edge_separate "
           "counter_semantics=identity_safe_contiguous_prefix "
           "poll_counter_semantics=logical_acquire_loads_not_l2_requests "
           "background_blocks=%d background_iterations=%u trace=%s\n",
           T23_SEMANTICS, cfg.experiment.c_str(), cfg.tag.c_str(), dev.name, dev.sms,
           dev.major, dev.minor, cfg.nproducer, cfg.nconsumer,
           depStructureName(cfg.structure), cfg.degree, dep_effective_degree(pat),
           dep_interval_tightness(pat), cfg.threads, cfg.consumer_smem_kb,
           producer_occ, consumer_occ, cfg.warmup, cfg.repeats, T23_BOOTSTRAPS,
           ceiling_sentinel_parent,
           cfg.background_blocks, cfg.background_iterations, cfg.trace_path.c_str());

    PEContext ctx;
    pe_allocate(cfg, ilo, ihi, mask, offsets, parents, &ctx);
    unsigned long long epoch = 0;
    bool all_ok = true;

    // Independent full-edge validation invocation for every correct mode.  Ceiling instead
    // gets an explicit stale/poison proof and is never labeled correct.
    for (int mode : modes) {
        ++epoch;
        bool validation = mode != PE_NONE;
        PEMetrics m = pe_once(cfg, pat, ctx, mode, epoch, validation, ilo, ihi);
        bool pass = validation ? (m.correct && m.trace_ok) : (m.stale > 0 && m.trace_ok);
        printf("VALIDATION_TIER23 semantics=%d experiment=%s tag=%s mode=%s epoch=%llu "
               "validation=%s correct=%d ceiling_wrong=%d stale=%u "
               "observed_digest=%llu expected_digest=%llu trace_ok=%d status=%s\n",
               T23_SEMANTICS, cfg.experiment.c_str(), cfg.tag.c_str(), pe_mode_name(mode),
               epoch, validation ? "full_edge" : "ceiling_stale", validation && m.correct,
               !validation && m.stale > 0, m.stale, m.observed_digest, m.expected_digest,
               m.trace_ok, pass ? "PASS" : "FAIL");
        all_ok = all_ok && pass;
    }
    if (!all_ok) {
        fprintf(stderr, "Tier 2/3 validation failed; refusing timed samples\n");
        pe_free(ctx);
        return 2;
    }

    for (int w = 0; w < cfg.warmup; ++w) {
        std::vector<int> order = modes;
        if (w & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            PEMetrics m = pe_once(cfg, pat, ctx, mode, epoch, false, ilo, ihi);
            bool pass = mode == PE_NONE ? (m.stale > 0 && m.trace_ok)
                                        : (m.correct && m.trace_ok);
            printf("WARMUP_TIER23 semantics=%d experiment=%s tag=%s warmup=%d mode=%s "
                   "epoch=%llu status=%s\n", T23_SEMANTICS, cfg.experiment.c_str(),
                   cfg.tag.c_str(), w, pe_mode_name(mode), epoch,
                   pass ? "PASS" : "FAIL");
            if (!pass) {
                pe_free(ctx);
                return 2;
            }
        }
    }

    std::vector<std::vector<double>> mode_ms(modes.size()), mode_wait(modes.size()),
                                     mode_wake(modes.size()), mode_bg(modes.size()),
                                     mode_poll(modes.size()), mode_meta(modes.size()),
                                     mode_decode(modes.size()), mode_latch(modes.size());
    bool wrote_trace = false;
    for (int rep = 0; rep < cfg.repeats; ++rep) {
        std::vector<int> order = modes;
        if (rep & 1) std::reverse(order.begin(), order.end());
        for (int mode : order) {
            ++epoch;
            PEMetrics m = pe_once(cfg, pat, ctx, mode, epoch, false, ilo, ihi);
            bool pass = mode == PE_NONE ? (m.stale > 0 && m.trace_ok)
                                        : (m.correct && m.trace_ok);
            if (!pass) {
                fprintf(stderr, "timed semantic failure tag=%s mode=%s rep=%d\n",
                        cfg.tag.c_str(), pe_mode_name(mode), rep);
                pe_free(ctx);
                return 2;
            }
            auto it = std::find(modes.begin(), modes.end(), mode);
            size_t idx = (size_t)std::distance(modes.begin(), it);
            mode_ms[idx].push_back(m.ms);
            mode_wait[idx].push_back(m.wait_ns);
            mode_wake[idx].push_back(m.wake_ns);
            mode_bg[idx].push_back(m.background_gbps);
            mode_poll[idx].push_back((double)m.poll_loads);
            mode_meta[idx].push_back((double)m.metadata_loads);
            mode_decode[idx].push_back(m.decode_ns);
            mode_latch[idx].push_back((double)m.ceiling_schedule_latch_loads);
            printf("SAMPLE_TIER23 semantics=%d experiment=%s tag=%s rep=%d mode=%s "
                   "epoch=%llu ms=%.9f wait_ns=%.3f wake_ns=%.3f decode_ns=%.3f "
                   "poll_loads=%llu poll_bytes=%llu metadata_loads=%llu metadata_bytes=%llu "
                   "ceiling_schedule_latch_loads=%llu "
                   "background_ms=%.9f background_gbps=%.9f background_overlap_rows=%u "
                   "trace_rows=%u observed_digest=%llu expected_digest=%llu "
                   "correct=%d ceiling_wrong=%d stale=%u trace_ok=%d\n",
                   T23_SEMANTICS, cfg.experiment.c_str(), cfg.tag.c_str(), rep,
                   pe_mode_name(mode), epoch, m.ms, m.wait_ns, m.wake_ns, m.decode_ns,
                   m.poll_loads, m.poll_loads * sizeof(unsigned long long),
                   m.metadata_loads, m.metadata_loads * sizeof(unsigned int),
                   m.ceiling_schedule_latch_loads,
                   m.background_ms, m.background_gbps, m.background_overlap_rows,
                   m.trace_rows, m.observed_digest, m.expected_digest,
                   mode != PE_NONE && m.correct, mode == PE_NONE && m.stale > 0,
                   m.stale, m.trace_ok);
            if (rep == cfg.repeats - 1) {
                pe_dump_trace(cfg, mode, epoch, ctx, !wrote_trace);
                wrote_trace = true;
            }
        }
    }

    for (size_t i = 0; i < modes.size(); ++i) {
        T23CI ms_ci = t23_bootstrap_median_ci(mode_ms[i], 0x230000ull + i);
        T23CI wait_ci = t23_bootstrap_median_ci(mode_wait[i], 0x231000ull + i);
        T23CI wake_ci = t23_bootstrap_median_ci(mode_wake[i], 0x232000ull + i);
        T23CI bg_ci = t23_bootstrap_median_ci(mode_bg[i], 0x233000ull + i);
        T23CI poll_ci = t23_bootstrap_median_ci(mode_poll[i], 0x234000ull + i);
        T23CI meta_ci = t23_bootstrap_median_ci(mode_meta[i], 0x235000ull + i);
        T23CI decode_ci = t23_bootstrap_median_ci(mode_decode[i], 0x236000ull + i);
        T23CI latch_ci = t23_bootstrap_median_ci(mode_latch[i], 0x237000ull + i);
        printf("SUMMARY_TIER23 semantics=%d experiment=%s tag=%s mode=%s repeats=%d "
               "median_ms=%.9f ci_ms_lo=%.9f ci_ms_hi=%.9f "
               "median_wait_ns=%.3f ci_wait_lo=%.3f ci_wait_hi=%.3f "
               "median_wake_ns=%.3f ci_wake_lo=%.3f ci_wake_hi=%.3f "
               "median_decode_ns=%.3f ci_decode_lo=%.3f ci_decode_hi=%.3f "
               "median_poll_loads=%.3f ci_poll_lo=%.3f ci_poll_hi=%.3f "
               "median_metadata_loads=%.3f ci_metadata_lo=%.3f ci_metadata_hi=%.3f "
               "median_ceiling_schedule_latch_loads=%.3f ci_latch_lo=%.3f "
               "ci_latch_hi=%.3f "
               "median_background_gbps=%.9f ci_background_lo=%.9f "
               "ci_background_hi=%.9f valid=1\n",
               T23_SEMANTICS, cfg.experiment.c_str(), cfg.tag.c_str(),
               pe_mode_name(modes[i]), cfg.repeats, t23_median(mode_ms[i]),
               ms_ci.lo, ms_ci.hi, t23_median(mode_wait[i]), wait_ci.lo, wait_ci.hi,
               t23_median(mode_wake[i]), wake_ci.lo, wake_ci.hi,
               t23_median(mode_decode[i]), decode_ci.lo, decode_ci.hi,
               t23_median(mode_poll[i]), poll_ci.lo, poll_ci.hi,
               t23_median(mode_meta[i]), meta_ci.lo, meta_ci.hi,
               t23_median(mode_latch[i]), latch_ci.lo, latch_ci.hi,
               t23_median(mode_bg[i]), bg_ci.lo, bg_ci.hi);
    }
    printf("TRACE_TIER23 semantics=%d experiment=%s tag=%s path=%s modes=%zu "
           "rows_per_mode=%d final_epoch=%llu\n", T23_SEMANTICS,
           cfg.experiment.c_str(), cfg.tag.c_str(), cfg.trace_path.c_str(), modes.size(),
           cfg.nproducer + cfg.nconsumer + cfg.background_blocks, epoch);
    pe_free(ctx);
    return 0;
}
