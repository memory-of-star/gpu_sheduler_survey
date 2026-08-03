// clc_probe.cu — Tier 0.4: measure CLC (clusterlaunchcontrol) characteristics on sm_100+.
//
// WHY THIS MATTERS
// ----------------
// CLC is the ONLY hardware primitive on Blackwell that lets software take over CTA/cluster
// scheduling. Two design-space dimensions depend on its measured cost:
//   B4  scheduling policy  -- the TB scheduler is not programmable, so the only way to compare
//                             producer-priority / consumer-priority / locality-first on real
//                             hardware is to rebuild them inside a persistent kernel driven
//                             by try_cancel.
//   B3  window depth       -- a persistent consumer sidesteps PDL's "all producer CTAs must
//                             have triggered" launch gate, but loses the deadlock-freedom
//                             argument that gate provides (see design space §B4.3).
//
// This probe answers: what does a try_cancel attempt cost, what fraction succeed under
// contention, and what throughput does the single-winner arbiter sustain.
//
// Requires sm_100 or higher. On older devices it prints a skip line and exits 0 so the
// unattended driver script does not abort.
//
// Build: ./build.sh    Run: ./clc_probe [--clusters N] [--repeats N]

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define CLC_SUPPORTED 1
#else
#define CLC_SUPPORTED 0
#endif

// ---------------------------------------------------------------- CLC wrappers
//
// clusterlaunchcontrol.try_cancel writes a 16-byte opaque response into shared memory and
// signals an mbarrier; query_cancel then decodes it. Both are PTX-only (no first-class CUDA
// runtime wrapper), so they go through inline asm. See PTX ISA 9.3 §9.7.14.18 / .19 and the
// interface notes in docs/cuda_13.4_pdl_clc_interfaces.md §2.

#if CLC_SUPPORTED
__device__ __forceinline__ void clc_try_cancel(void* smem_resp, void* smem_mbar) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 "
        "[%0], [%1];"
        :: "l"(__cvta_generic_to_shared(smem_resp)),
           "l"(__cvta_generic_to_shared(smem_mbar))
        : "memory");
}

__device__ __forceinline__ bool clc_is_canceled(const void* smem_resp) {
    unsigned pred;
    // Load the 128-bit response, then decode the success predicate out of it.
    asm volatile(
        "{\n\t"
        ".reg .b128 resp;\n\t"
        ".reg .pred p;\n\t"
        "ld.shared.b128 resp, [%1];\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, resp;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(pred)
        : "l"(__cvta_generic_to_shared(const_cast<void*>(smem_resp)))
        : "memory");
    return pred != 0;
}

__device__ __forceinline__ unsigned clc_first_ctaid_x(const void* smem_resp) {
    unsigned x;
    asm volatile(
        "{\n\t"
        ".reg .b128 resp;\n\t"
        "ld.shared.b128 resp, [%1];\n\t"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, resp;\n\t"
        "}"
        : "=r"(x)
        : "l"(__cvta_generic_to_shared(const_cast<void*>(smem_resp)))
        : "memory");
    return x;
}

__device__ __forceinline__ void mbar_init(void* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "l"(__cvta_generic_to_shared(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(void* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "l"(__cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(void* bar, unsigned phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "waitLoop:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra waitLoop;\n\t"
        "}"
        :: "l"(__cvta_generic_to_shared(bar)), "r"(phase) : "memory");
}
#endif // CLC_SUPPORTED

// Persistent kernel that repeatedly claims not-yet-launched clusters via try_cancel.
// Each CTA records how many attempts it made, how many succeeded, and the total cycles spent
// inside try_cancel + wait.
__global__ void clcProbeK(unsigned int* __restrict__ claimed,
                          unsigned long long* __restrict__ cycles,
                          unsigned int* __restrict__ attempts,
                          unsigned int* __restrict__ successes,
                          unsigned int n_clusters) {
#if CLC_SUPPORTED
    __shared__ alignas(16) char resp[16];
    __shared__ alignas(8)  char bar[8];

    const int leader = (threadIdx.x == 0);
    unsigned int myAttempts = 0, mySucc = 0;
    unsigned long long myCycles = 0;
    unsigned phase = 0;

    if (leader) mbar_init(bar, 1);
    __syncthreads();

    // Claim loop: try to cancel a pending cluster; on success record which one we took.
    for (;;) {
        bool got = false;
        unsigned cid = 0;
        if (leader) {
            long long t0 = clock64();
            mbar_expect_tx(bar, 16);
            clc_try_cancel(resp, bar);
            mbar_wait(bar, phase);
            phase ^= 1;
            myCycles += (unsigned long long)(clock64() - t0);
            ++myAttempts;
            got = clc_is_canceled(resp);
            if (got) { cid = clc_first_ctaid_x(resp); ++mySucc; }
        }
        // Broadcast the leader's outcome to the whole CTA.
        got = __syncthreads_or(got ? 1 : 0) != 0;
        if (!got) break;                       // a failed try_cancel means the queue is drained
        if (leader && cid < n_clusters) atomicAdd(&claimed[cid], 1u);
        __syncthreads();
    }

    if (leader) {
        atomicAdd(cycles,    myCycles);
        atomicAdd(attempts,  myAttempts);
        atomicAdd(successes, mySucc);
    }
#else
    (void)claimed; (void)cycles; (void)attempts; (void)successes; (void)n_clusters;
#endif
}

int main(int argc, char** argv) {
    Args A(argc, argv);
    if (A.has("--help")) {
        printf("usage: clc_probe [--clusters N] [--repeats N]\n"
               "  Measures clusterlaunchcontrol.try_cancel latency / success rate / arbiter\n"
               "  throughput. Requires sm_100+; skips cleanly otherwise.\n");
        return 0;
    }

    DeviceInfo dev = queryDevice();
    printDeviceBanner(dev);

    if (dev.major < 10) {
        printf("SKIP: CLC needs compute capability >= 10.0, device is %d.%d\n",
               dev.major, dev.minor);
        printf("SUMMARY tier0=clc status=skipped cc=%d.%d\n", dev.major, dev.minor);
        return 0;   // exit 0 so the unattended driver keeps going
    }

    int nClusters = (int)A.ll("--clusters", 4096);
    int repeats   = (int)A.ll("--repeats", 10);
    const int threads = 128;

    unsigned int *d_claimed = nullptr, *d_attempts = nullptr, *d_succ = nullptr;
    unsigned long long* d_cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&d_claimed,  (size_t)nClusters * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_attempts, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_succ,     sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_cycles,   sizeof(unsigned long long)));

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    printf("\n=== 0.4 CLC try_cancel characteristics ===\n");
    printf("%-12s %14s %14s %14s %16s\n",
           "clusters", "median(ms)", "attempts", "successes", "cyc_per_attempt");

    std::vector<float> v;
    unsigned int hAtt = 0, hSucc = 0; unsigned long long hCyc = 0;
    for (int r = 0; r < repeats + 3; ++r) {
        CUDA_CHECK(cudaMemsetAsync(d_claimed, 0, (size_t)nClusters * sizeof(unsigned int), s));
        CUDA_CHECK(cudaMemsetAsync(d_attempts, 0, sizeof(unsigned int), s));
        CUDA_CHECK(cudaMemsetAsync(d_succ,     0, sizeof(unsigned int), s));
        CUDA_CHECK(cudaMemsetAsync(d_cycles,   0, sizeof(unsigned long long), s));

        CUDA_CHECK(cudaEventRecord(e0, s));
        clcProbeK<<<nClusters, threads, 0, s>>>(d_claimed, d_cycles, d_attempts, d_succ,
                                                (unsigned)nClusters);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1, s));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r >= 3) v.push_back(ms);

        CUDA_CHECK(cudaMemcpy(&hAtt,  d_attempts, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hSucc, d_succ,     sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hCyc,  d_cycles,   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }

    // Every pending cluster must be claimed by exactly one winner -- verify the arbiter.
    std::vector<unsigned int> hClaimed(nClusters);
    CUDA_CHECK(cudaMemcpy(hClaimed.data(), d_claimed,
                          (size_t)nClusters * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int dup = 0, zero = 0;
    for (int i = 0; i < nClusters; ++i) {
        if (hClaimed[i] > 1) ++dup;
        if (hClaimed[i] == 0) ++zero;
    }

    double med = medianOf(v);
    double cycPer = hAtt ? (double)hCyc / (double)hAtt : 0.0;
    double succRate = hAtt ? (double)hSucc / (double)hAtt : 0.0;

    printf("%-12d %14.4f %14u %14u %16.1f\n", nClusters, med, hAtt, hSucc, cycPer);
    printf("  single-winner check: duplicate_claims=%u unclaimed=%u %s\n",
           dup, zero, (dup == 0) ? "(arbiter OK)" : "(ARBITER VIOLATION)");
    printf("SUMMARY tier0=clc status=ok clusters=%d median_ms=%.5f attempts=%u successes=%u "
           "success_rate=%.4f cyc_per_attempt=%.2f duplicate_claims=%u unclaimed=%u\n",
           nClusters, med, hAtt, hSucc, succRate, cycPer, dup, zero);

    cudaFree(d_claimed); cudaFree(d_attempts); cudaFree(d_succ); cudaFree(d_cycles);
    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
