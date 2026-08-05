// dep_pattern.cuh — Primitive 2 (part 1): parameterized inter-CTA dependency patterns.
//
// WHY THIS EXISTS
// ---------------
// BlockMaestro Fig.12 injected dependencies as "n-group fully connected", which grows
// dependency DEGREE and structural COMPLEXITY together. Its conclusion ("degree > 32 =>
// no benefit") therefore cannot separate the two causes.
//
// Real workloads routinely have HIGH degree with SIMPLE structure:
//   - LLM FFN GEMM chain : output tile (m,n) depends on token row range [m*BM, (m+1)*BM)
//                          => degree ~= BM (128/256) but a CONTIGUOUS INTERVAL, O(1) to encode
//   - DSA indexer->topk  : degree = L/key_block (thousands at 1M ctx), also a contiguous interval
//
// So this header makes STRUCTURE and DEGREE independent axes:
//   structure = INTERVAL | GROUPED | STRIDED | RANDOM ...   (how scattered the parents are)
//   degree    = how many parent CTAs a child depends on
//
// Patterns are evaluated in CLOSED FORM on the device (no materialized graph), which also
// mirrors the "parameterized pattern template" option of design-space dimension A3.

#pragma once

#include <cstdint>
#include <vector>

enum DepStructure {
    DEP_INTERVAL = 0,  // parents form ONE contiguous range  -> encodable as [lo,hi], O(1)
    DEP_GROUPED  = 1,  // n-group fully connected (BlockMaestro's injection shape)
    DEP_STRIDED  = 2,  // parents evenly strided across the whole producer grid
    DEP_RANDOM   = 3,  // pseudo-random parents (worst case, needs full adjacency)
    DEP_SELF     = 4,  // self-dependency: child j depends only on parent j (IKRA's common case)
    DEP_ALL      = 5,  // fully connected (degenerate; equivalent to a grid-level barrier)
    DEP_NONE     = 6,  // independent
    DEP_NSTRUCT  = 7
};

__host__ __device__ __forceinline__ const char* depStructureName(int s) {
    switch (s) {
        case DEP_INTERVAL: return "interval";
        case DEP_GROUPED:  return "grouped";
        case DEP_STRIDED:  return "strided";
        case DEP_RANDOM:   return "random";
        case DEP_SELF:     return "self";
        case DEP_ALL:      return "all";
        case DEP_NONE:     return "none";
        default:           return "?";
    }
}

__host__ inline int depStructureFromName(const char* s) {
    for (int i = 0; i < DEP_NSTRUCT; ++i) {
        const char* n = depStructureName(i);
        const char* a = s; const char* b = n;
        while (*a && *b && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return i;
    }
    return -1;
}

// Fully describes an inter-kernel CTA dependency relation without materializing it.
struct DepPattern {
    int structure;      // DepStructure
    int degree;         // parents per child (clamped to n_producer)
    int n_producer;     // producer grid size in CTAs
    int n_consumer;     // consumer grid size in CTAs
    unsigned int seed;  // for DEP_RANDOM
};

__host__ __device__ __forceinline__ unsigned int dep_hash(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ __forceinline__ unsigned int dep_gcd(unsigned int a,
                                                          unsigned int b) {
    while (b != 0u) {
        unsigned int r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// Number of parents of `child`.
__host__ __device__ __forceinline__ int dep_degree(const DepPattern& p, int child) {
    switch (p.structure) {
        case DEP_NONE: return 0;
        case DEP_SELF: return (child < p.n_producer) ? 1 : 0;
        case DEP_ALL:  return p.n_producer;
        default: {
            int d = p.degree < 1 ? 1 : p.degree;
            return d > p.n_producer ? p.n_producer : d;
        }
    }
}

// The k-th parent (0 <= k < dep_degree) of `child`. Returns -1 when there is none.
__host__ __device__ __forceinline__ int dep_parent(const DepPattern& p, int child, int k) {
    const int P = p.n_producer;
    if (P <= 0) return -1;
    switch (p.structure) {
        case DEP_NONE: return -1;
        case DEP_SELF: return child < P ? child : -1;
        case DEP_ALL:  return k < P ? k : -1;

        case DEP_INTERVAL: {
            // Child j owns a contiguous window of parents; windows slide with j.
            // This is the LLM GEMM / DSA indexer shape: high degree, O(1) encoding.
            int d = dep_degree(p, child);
            long long span = (long long)P - d;                  // last valid window start
            long long lo   = (p.n_consumer > 1)
                           ? (span * (long long)child) / (long long)(p.n_consumer - 1)
                           : 0;
            if (lo < 0) lo = 0;
            long long r = lo + k;
            return (int)(r < P ? r : P - 1);
        }

        case DEP_GROUPED: {
            // n-group fully connected: consumers are bucketed, each bucket depends on a
            // matching bucket of producers. This is BlockMaestro's injection shape.
            int d     = dep_degree(p, child);
            int group = (d > 0) ? (child / d) : 0;
            long long base = (long long)group * d;
            long long r    = base + k;
            return (int)(r < P ? r : (r % P));
        }

        case DEP_STRIDED: {
            // Same degree as INTERVAL but parents are NOT contiguous, so interval encoding
            // becomes lossy. Isolates "structure" from "degree".
            int d      = dep_degree(p, child);
            int stride = (d > 0) ? (P / d) : 1;
            if (stride < 1) stride = 1;
            return (int)(((long long)child + (long long)k * stride) % P);
        }

        case DEP_RANDOM:
        default: {
            // Generate a per-child affine permutation modulo P and take its first d
            // elements. A raw hash(k) % P can collide, silently making the actual
            // dependency degree smaller than requested. A step coprime with P makes
            // k -> offset + k*step a permutation (AGENTS.md validity rule 10).
            if (P == 1) return 0;
            unsigned int up = (unsigned int)P;
            unsigned int offset = dep_hash((unsigned int)child ^ p.seed) % up;
            unsigned int step = 1u + dep_hash((unsigned int)child * 0x9e3779b9u ^
                                               p.seed ^ 0xa511e9b3u) % (up - 1u);
            while (dep_gcd(step, up) != 1u) {
                ++step;
                if (step >= up) step = 1u;
            }
            return (int)(((unsigned long long)offset +
                          (unsigned long long)(unsigned int)k * step) % up);
        }
    }
}

// Host-side audit helper. The pilot calls this before spending GPU time so a future
// pattern change cannot re-introduce duplicate parents without failing loudly.
__host__ inline bool dep_parents_are_unique(const DepPattern& p) {
    if (p.n_producer <= 0 || p.n_consumer <= 0) return false;
    std::vector<int> seen((size_t)p.n_producer, -1);
    for (int child = 0; child < p.n_consumer; ++child) {
        int d = dep_degree(p, child);
        for (int i = 0; i < d; ++i) {
            int a = dep_parent(p, child, i);
            if (a < 0 || a >= p.n_producer) return false;
            if (seen[(size_t)a] == child) return false;
            seen[(size_t)a] = child;
        }
    }
    return true;
}

// Conservative [lo,hi] cover of the parent set — what an interval-encoded implementation
// actually waits on. Exact for DEP_INTERVAL / DEP_SELF; over-approximates otherwise, and
// the excess is the "false edge" cost that dimensions A3 / E1 want to quantify.
__host__ __device__ __forceinline__ void dep_interval(const DepPattern& p, int child,
                                                      int* lo, int* hi) {
    int d = dep_degree(p, child);
    if (d <= 0) { *lo = 0; *hi = -1; return; }

    // Closed form for the structures where it is known, loop otherwise.
    if (p.structure == DEP_INTERVAL) {
        int a = dep_parent(p, child, 0);
        *lo = a; *hi = a + d - 1;
        if (*hi >= p.n_producer) *hi = p.n_producer - 1;
        return;
    }
    if (p.structure == DEP_SELF) { *lo = child; *hi = child; return; }
    if (p.structure == DEP_ALL)  { *lo = 0; *hi = p.n_producer - 1; return; }

    int mn = dep_parent(p, child, 0);
    int mx = mn;
    for (int k = 1; k < d; ++k) {
        int r = dep_parent(p, child, k);
        if (r < mn) mn = r;
        if (r > mx) mx = r;
    }
    *lo = mn; *hi = mx;
}

// Interval tightness = degree / interval_width, averaged over consumers.
// 1.0 => interval encoding is exact. Small values => an interval-based wait pulls in many
// false dependencies, i.e. the encoding choice (A3) actually costs performance.
__host__ inline double dep_interval_tightness(const DepPattern& p) {
    if (p.structure == DEP_NONE) return 1.0;
    double acc = 0.0; long long n = 0;
    for (int j = 0; j < p.n_consumer; ++j) {
        int d = dep_degree(p, j);
        if (d <= 0) continue;
        int lo, hi; dep_interval(p, j, &lo, &hi);
        int width = hi - lo + 1;
        acc += (double)d / (double)(width > 0 ? width : 1);
        ++n;
    }
    return n ? acc / (double)n : 1.0;
}

// Average number of parents actually waited on when using interval encoding.
__host__ inline double dep_effective_degree(const DepPattern& p) {
    double acc = 0.0; long long n = 0;
    for (int j = 0; j < p.n_consumer; ++j) {
        int d = dep_degree(p, j);
        if (d <= 0) continue;
        int lo, hi; dep_interval(p, j, &lo, &hi);
        acc += (double)(hi - lo + 1);
        ++n;
    }
    return n ? acc / (double)n : 0.0;
}
