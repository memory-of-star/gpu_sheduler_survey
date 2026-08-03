// dep_wait.cuh — Primitive 2 (part 2): interchangeable CTA-level dependency wait protocols.
//
// These are the concrete options of design-space dimension B1 (synchronization protocol),
// implemented so a benchmark can swap ONE enum and keep everything else fixed.
//
// All of them are [S] (pure software on sm_90+): the producer publishes per-CTA completion
// via a release store to a global bitmap; the consumer waits with an acquire load. The
// release/acquire pair is what makes this correct under the CUDA memory model — no new
// hardware and no reliance on griddepcontrol's grid-wide flush.
//
//   producer CTA k:  ... writes ...  --release-->  done[k] = 1
//                                                       | synchronizes-with
//   consumer CTA j:  acquire load done[k] == 1  -->  ... reads ...
//
// thread_scope_device is sufficient (single GPU). Cross-GPU / host would need system scope.

#pragma once

#include <cuda/atomic>
#include "dep_pattern.cuh"

enum WaitMode {
    WAIT_NONE     = 0,  // Ceiling: no dependency enforced at all (RESULTS ARE WRONG, timing only)
    WAIT_GRID     = 1,  // Floor: griddepcontrol.wait, whole-grid all-or-nothing
    WAIT_SPIN     = 2,  // per-CTA interval wait, tight spin
    WAIT_BACKOFF  = 3,  // per-CTA interval wait, exponential __nanosleep backoff
    WAIT_COUNTER  = 4,  // monotonic completion counter, one compare covers the whole interval
    WAIT_EXACT    = 5,  // per-CTA exact parent set (no interval over-approximation)
    WAIT_NMODES   = 6
};

__host__ __device__ __forceinline__ const char* waitModeName(int m) {
    switch (m) {
        case WAIT_NONE:    return "none(ceiling)";
        case WAIT_GRID:    return "grid(griddepcontrol)";
        case WAIT_SPIN:    return "cta-spin";
        case WAIT_BACKOFF: return "cta-backoff";
        case WAIT_COUNTER: return "cta-counter";
        case WAIT_EXACT:   return "cta-exact";
        default:           return "?";
    }
}

__host__ inline int waitModeFromName(const char* s) {
    for (int i = 0; i < WAIT_NMODES; ++i) {
        const char* n = waitModeName(i);
        const char* a = s; const char* b = n;
        while (*a && *b && *a == *b) { ++a; ++b; }
        if (*a == 0 && *b == 0) return i;
    }
    return -1;
}

// ---------------------------------------------------------------- producer side

// Publish "my CTA is done". Must be the LAST thing the producer CTA does.
// __syncthreads() first so every thread's writes precede the release store.
__device__ __forceinline__ void dep_publish(int* done, unsigned long long* counter, int my_cta) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(done[my_cta]);
        flag.store(1, cuda::memory_order_release);
        if (counter) {
            // Monotonic completion count for WAIT_COUNTER. Release ordering again so the
            // counter never runs ahead of the data.
            cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> c(*counter);
            c.fetch_add(1ull, cuda::memory_order_release);
        }
    }
}

// ---------------------------------------------------------------- consumer side

__device__ __forceinline__ void dep_spin_one(const int* done, int parent, bool backoff) {
    cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[parent]);
    unsigned int ns = 32;
    while (flag.load(cuda::memory_order_acquire) == 0) {
        if (backoff) {
            __nanosleep(ns);
            ns = ns < 1024 ? ns * 2 : 1024;   // cap so wakeup latency stays bounded
        }
    }
}

// Wait for every parent in [lo, hi]. Leader thread polls; __syncthreads() propagates the
// acquire to the whole CTA.
__device__ __forceinline__ void dep_wait_interval(const int* done, int lo, int hi, bool backoff) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        for (int p = lo; p <= hi; ++p) dep_spin_one(done, p, backoff);
    }
    __syncthreads();
}

// Wait for the exact parent set (no interval over-approximation).
__device__ __forceinline__ void dep_wait_exact(const int* done, const DepPattern& pat,
                                               int child, bool backoff) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        int d = dep_degree(pat, child);
        for (int k = 0; k < d; ++k) {
            int p = dep_parent(pat, child, k);
            if (p >= 0) dep_spin_one(done, p, backoff);
        }
    }
    __syncthreads();
}

// Monotonic completion counter: if producer CTAs retire in roughly increasing order, then
// "counter >= hi+1" implies every parent in [0,hi] is done, so ONE compare replaces the
// per-parent poll. Correctness does NOT depend on the ordering assumption -- the counter is
// a lower bound on progress, so this is conservative (it may wait longer than necessary).
__device__ __forceinline__ void dep_wait_counter(const unsigned long long* counter,
                                                 int hi, bool backoff) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        cuda::atomic_ref<const unsigned long long, cuda::thread_scope_device> c(*counter);
        unsigned long long need = (unsigned long long)(hi + 1);
        unsigned int ns = 32;
        while (c.load(cuda::memory_order_acquire) < need) {
            if (backoff) { __nanosleep(ns); ns = ns < 1024 ? ns * 2 : 1024; }
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------- unified entry
//
// One call site in the consumer kernel; the mode picks the protocol. Returns immediately for
// WAIT_NONE (ceiling) and defers to griddepcontrol for WAIT_GRID (floor).

__device__ __forceinline__ void dep_wait(int mode,
                                         const int* done,
                                         const unsigned long long* counter,
                                         const DepPattern& pat,
                                         int child) {
    switch (mode) {
        case WAIT_NONE:
            return;                                  // ceiling: unsafe on purpose
        case WAIT_GRID:
            cudaGridDependencySynchronize();         // -> griddepcontrol.wait
            return;
        case WAIT_EXACT:
            dep_wait_exact(done, pat, child, true);
            return;
        case WAIT_COUNTER: {
            int lo, hi; dep_interval(pat, child, &lo, &hi);
            if (hi >= lo) dep_wait_counter(counter, hi, true);
            return;
        }
        case WAIT_SPIN:
        case WAIT_BACKOFF:
        default: {
            int lo, hi; dep_interval(pat, child, &lo, &hi);
            if (hi >= lo) dep_wait_interval(done, lo, hi, mode == WAIT_BACKOFF);
            return;
        }
    }
}

// Does this mode need the per-CTA done[] bitmap / the counter?
__host__ __device__ __forceinline__ bool waitNeedsBitmap(int m) {
    return m == WAIT_SPIN || m == WAIT_BACKOFF || m == WAIT_EXACT;
}
__host__ __device__ __forceinline__ bool waitNeedsCounter(int m) {
    return m == WAIT_COUNTER;
}
// Modes that enforce a real dependency (i.e. results must verify).
__host__ __device__ __forceinline__ bool waitIsCorrect(int m) {
    return m != WAIT_NONE;
}
