// cta_trace.cuh — Primitive 1: per-CTA timestamp instrumentation.
//
// Each CTA writes ONE record (thread 0 only) capturing when it started, when its
// dependency was satisfied, and when it finished, plus which SM it ran on. Offline
// analysis reconstructs the full CTA-level timeline from these records.
//
// CRITICAL: timestamps MUST come from %globaltimer, NOT clock64().
//   clock64()     -> per-SM clock counter. NOT comparable across SMs.
//   %globaltimer  -> global nanosecond timer, consistent across SMs.
// Timeline reconstruction is fundamentally a cross-SM event ordering problem, so
// using clock64() yields plausible-looking but silently wrong overlap relations.
//
// Overhead: 32 B per CTA, written once by thread 0, non-atomic. 10K CTAs = 320 KB.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- device intrinsics

__device__ __forceinline__ unsigned long long ctatrace_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    return t;
}

__device__ __forceinline__ unsigned int ctatrace_smid() {
    unsigned int s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

// ---------------------------------------------------------------- record layout

// 32 bytes, naturally aligned. One per CTA.
struct CtaRecord {
    unsigned long long t_launch;         // first instruction of the CTA body
    unsigned long long t_dep_satisfied;  // dependency wait returned (== t_launch if no wait)
    unsigned long long t_end;            // last instruction before exit
    unsigned int       block_id;         // linearized blockIdx
    unsigned short     kernel_id;        // which kernel in the chain (0 = producer, 1 = consumer, ...)
    unsigned short     sm_id;            // %smid
};
static_assert(sizeof(CtaRecord) == 32, "CtaRecord must stay 32 bytes");

// Per-kernel slice of a shared trace buffer. Pass by value into kernels.
struct CtaTrace {
    CtaRecord*   recs;      // device pointer to this kernel's slice (may be nullptr = disabled)
    unsigned int capacity;  // number of records in the slice
    unsigned short kernel_id;
};

// ---------------------------------------------------------------- device-side API
//
// Usage inside a kernel:
//     CtaTraceLocal tr = ctatrace_begin(trace);
//     ... independent prologue work ...
//     <dependency wait>
//     ctatrace_mark_dep(tr);
//     ... dependent work ...
//     ctatrace_end(tr);

struct CtaTraceLocal {
    CtaTrace     t;
    unsigned int block_id;
    unsigned long long t_launch;
    unsigned long long t_dep;
    bool         active;   // true only for thread 0 of a CTA with a valid slot
};

__device__ __forceinline__ CtaTraceLocal ctatrace_begin(const CtaTrace& t) {
    CtaTraceLocal l;
    l.t        = t;
    l.block_id = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    l.active   = (t.recs != nullptr) && (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                 && (l.block_id < t.capacity);
    l.t_launch = l.active ? ctatrace_globaltimer() : 0ull;
    l.t_dep    = l.t_launch;
    return l;
}

__device__ __forceinline__ void ctatrace_mark_dep(CtaTraceLocal& l) {
    if (l.active) l.t_dep = ctatrace_globaltimer();
}

__device__ __forceinline__ void ctatrace_end(const CtaTraceLocal& l) {
    if (!l.active) return;
    CtaRecord r;
    r.t_launch        = l.t_launch;
    r.t_dep_satisfied = l.t_dep;
    r.t_end           = ctatrace_globaltimer();
    r.block_id        = l.block_id;
    r.kernel_id       = l.t.kernel_id;
    r.sm_id           = (unsigned short)ctatrace_smid();
    l.t.recs[l.block_id] = r;   // one 32 B store, no atomics needed (unique slot per CTA)
}

// ---------------------------------------------------------------- host-side helper

struct CtaTraceBuffer {
    CtaRecord*   d_recs = nullptr;
    unsigned int total  = 0;      // total records across all kernel slices
    unsigned int nslices = 0;
    unsigned int per_slice = 0;
};

// Allocate one buffer holding `nslices` contiguous slices of `per_slice` records each.
inline cudaError_t ctatrace_alloc(CtaTraceBuffer* b, unsigned int nslices, unsigned int per_slice) {
    b->nslices   = nslices;
    b->per_slice = per_slice;
    b->total     = nslices * per_slice;
    return cudaMalloc(&b->d_recs, (size_t)b->total * sizeof(CtaRecord));
}

inline CtaTrace ctatrace_slice(const CtaTraceBuffer& b, unsigned int idx) {
    CtaTrace t;
    t.recs      = b.d_recs ? (b.d_recs + (size_t)idx * b.per_slice) : nullptr;
    t.capacity  = b.per_slice;
    t.kernel_id = (unsigned short)idx;
    return t;
}

inline CtaTrace ctatrace_disabled() {
    CtaTrace t; t.recs = nullptr; t.capacity = 0; t.kernel_id = 0; return t;
}

inline cudaError_t ctatrace_reset(CtaTraceBuffer* b) {
    if (!b->d_recs) return cudaSuccess;
    return cudaMemset(b->d_recs, 0, (size_t)b->total * sizeof(CtaRecord));
}

inline void ctatrace_free(CtaTraceBuffer* b) {
    if (b->d_recs) cudaFree(b->d_recs);
    b->d_recs = nullptr;
}

// Dump to a CSV consumed by tools/cta_timeline.py.
// Records with t_end == 0 were never written (CTA slot unused) and are skipped.
inline bool ctatrace_dump_csv(const CtaTraceBuffer& b, const char* path, const char* tag) {
    if (!b.d_recs) return false;
    CtaRecord* h = (CtaRecord*)malloc((size_t)b.total * sizeof(CtaRecord));
    if (!h) return false;
    if (cudaMemcpy(h, b.d_recs, (size_t)b.total * sizeof(CtaRecord),
                   cudaMemcpyDeviceToHost) != cudaSuccess) { free(h); return false; }

    FILE* f = fopen(path, "w");
    if (!f) { free(h); return false; }
    fprintf(f, "tag,kernel_id,block_id,sm_id,t_launch,t_dep_satisfied,t_end\n");
    for (unsigned int i = 0; i < b.total; ++i) {
        const CtaRecord& r = h[i];
        if (r.t_end == 0ull) continue;
        fprintf(f, "%s,%u,%u,%u,%llu,%llu,%llu\n",
                tag, (unsigned)r.kernel_id, r.block_id, (unsigned)r.sm_id,
                (unsigned long long)r.t_launch,
                (unsigned long long)r.t_dep_satisfied,
                (unsigned long long)r.t_end);
    }
    fclose(f);
    free(h);
    return true;
}
