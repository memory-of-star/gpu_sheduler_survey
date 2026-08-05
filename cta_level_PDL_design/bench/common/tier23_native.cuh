// tier23_native.cuh -- common contracts for the admissible Tier 2/3 harnesses.
//
// This header is intentionally separate from dep_wait.cuh.  That historical header still
// contains the rejected cardinality-only completion counter; the native Tier 2/3 harnesses
// use identity-preserving flags or a contiguous-prefix counter whose advancement first
// acquires every corresponding per-CTA flag.

#pragma once

#include "bench_util.cuh"
#include "cta_trace.cuh"
#include "dep_pattern.cuh"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

static constexpr int T23_SEMANTICS = 1;
static constexpr int T23_WARMUP_FORMAL = 3;
static constexpr int T23_REPEATS_FORMAL = 31;
static constexpr int T23_BOOTSTRAPS = 2000;

__host__ __device__ __forceinline__ unsigned long long t23_mix64(
        unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

__host__ __device__ __forceinline__ unsigned long long t23_value(
        unsigned long long epoch, unsigned int index, unsigned int stage = 0) {
    return t23_mix64(epoch * 0x9e3779b97f4a7c15ull ^
                     ((unsigned long long)index + 1ull) * 0xd1b54a32d192ed03ull ^
                     ((unsigned long long)stage + 1ull) * 0x94d049bb133111ebull);
}

inline unsigned long long t23_digest(const std::vector<unsigned long long>& values) {
    unsigned long long h = 1469598103934665603ull;
    for (unsigned long long v : values) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (8 * b)) & 0xffull;
            h *= 1099511628211ull;
        }
    }
    return h;
}

inline double t23_median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct T23CI {
    double lo = 0.0;
    double hi = 0.0;
};

inline T23CI t23_bootstrap_median_ci(const std::vector<double>& values,
                                     unsigned long long seed) {
    T23CI ci;
    if (values.empty()) return ci;
    std::vector<double> medians;
    std::vector<double> sample(values.size());
    medians.reserve(T23_BOOTSTRAPS);
    for (int b = 0; b < T23_BOOTSTRAPS; ++b) {
        for (size_t i = 0; i < values.size(); ++i) {
            unsigned long long r = t23_mix64(seed ^
                ((unsigned long long)b + 1ull) * 0x9e3779b97f4a7c15ull ^
                ((unsigned long long)i + 1ull) * 0xd1b54a32d192ed03ull);
            sample[i] = values[(size_t)(r % values.size())];
        }
        medians.push_back(t23_median(sample));
    }
    std::sort(medians.begin(), medians.end());
    const size_t lo = (size_t)(0.025 * (double)(medians.size() - 1));
    const size_t hi = (size_t)(0.975 * (double)(medians.size() - 1));
    ci.lo = medians[lo];
    ci.hi = medians[hi];
    return ci;
}

// One trace row per logical CTA/task.  All fields are copied verbatim to CSV.  Integer
// equality, not a tolerance, is used by the strict validator for trace-derived metrics.
struct T23TraceRecord {
    unsigned long long t_start;
    unsigned long long t_ready;
    unsigned long long t_wait_begin;
    unsigned long long t_dep;
    unsigned long long t_end;
    unsigned long long poll_loads;
    unsigned long long metadata_loads;
    unsigned long long decode_ns;
    unsigned int block_id;
    unsigned int sm_id;
    unsigned int kernel_id;
    unsigned int aux;
};
static_assert(sizeof(T23TraceRecord) == 80, "unexpected Tier 2/3 trace ABI");

__device__ __forceinline__ void t23_trace_begin(T23TraceRecord* rows,
                                                unsigned int block,
                                                unsigned int kernel,
                                                unsigned int aux = 0) {
    if (threadIdx.x != 0 || rows == nullptr) return;
    T23TraceRecord& r = rows[block];
    r.t_start = ctatrace_globaltimer();
    r.t_ready = 0;
    r.t_wait_begin = 0;
    r.t_dep = 0;
    r.t_end = 0;
    r.poll_loads = 0;
    r.metadata_loads = 0;
    r.decode_ns = 0;
    r.block_id = block;
    r.sm_id = ctatrace_smid();
    r.kernel_id = kernel;
    r.aux = aux;
}

inline bool t23_short_allowed(int repeats, int warmup, bool allow_short) {
    if (repeats >= T23_REPEATS_FORMAL && warmup >= T23_WARMUP_FORMAL) return true;
    return allow_short;
}

inline void t23_print_short_error(int repeats, int warmup) {
    fprintf(stderr,
            "formal Tier 2/3 requires repeats>=%d and warmup>=%d "
            "(got %d/%d); pass --allow-short only for plumbing smoke\n",
            T23_REPEATS_FORMAL, T23_WARMUP_FORMAL, repeats, warmup);
}

inline cudaError_t t23_launch_pss(cudaStream_t stream, dim3 grid, dim3 block,
                                  size_t dynamic_smem, const void* kernel,
                                  void** args) {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = dynamic_smem;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelExC(&cfg, kernel, args);
}
