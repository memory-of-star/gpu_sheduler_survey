// tier0_facts.cu — Tier 0 base facts. Run this FIRST on B300.
//
// Nothing here evaluates a design option; it establishes the ground truth that every later
// experiment's interpretation depends on. See docs/cta_pdl_eval_plan.md §3.
//
//   0.1  How many kernels of a same-stream chain actually overlap on B300?
//        -> decides which B3 window-depth options are even REACHABLE on this hardware.
//   0.3  Occupancy cost of a CTA that waits while resident.
//        -> the ONLY pricing basis for dimension B2 ("what would pre-dispatch gating buy?").
//   0.5  Fence scope cost calibration (.cta / .gpu / .sys).
//        -> feeds the "Ideal" point of the four-point bracket.
//
// 0.2 (PDL eager-cross-stream behaviour) is covered by re-running the existing
// 跨stream_PDL调研/bench/pdl_bench on B300; 0.4 (CLC try_cancel) needs sm_100+ PTX and is
// split into clc_probe.cu.
//
// Build: ./build.sh   Run: ./tier0_facts [--repeats N]

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"

// ---------------------------------------------------------------- 0.1 chain overlap depth

// A link in a K1->K2->...->Kn chain. Each kernel triggers its dependents early, waits for
// its predecessor, then burns `work` cycles. With PDL the waits should let stage i+1's
// prologue overlap stage i's tail; how many stages overlap SIMULTANEOUSLY is the question.
__global__ void chainK(float* __restrict__ buf, int stage, unsigned long long work,
                       CtaTrace trace, int use_pdl) {
    CtaTraceLocal tr = ctatrace_begin(trace);

    spin_cycles(work / 2);                        // independent prologue
    if (use_pdl) cudaGridDependencySynchronize(); // wait for predecessor
    ctatrace_mark_dep(tr);

    if (threadIdx.x == 0) buf[blockIdx.x] += (float)stage;

    if (use_pdl) cudaTriggerProgrammaticLaunchCompletion();
    spin_cycles(work / 2);                        // overlappable tail
    ctatrace_end(tr);
}

static void chainOverlap(const DeviceInfo& dev, int repeats) {
    const int STAGES = 6;
    const int blocks = dev.sms;                   // one wave, so overlap is not resource-bound
    const int threads = 128;
    const unsigned long long work = 2000000ull;

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, (size_t)blocks * sizeof(float)));

    CtaTraceBuffer tb{};
    CUDA_CHECK(ctatrace_alloc(&tb, STAGES, (unsigned)blocks));

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    printf("\n=== 0.1 same-stream chain overlap depth ===\n");
    printf("%-8s %14s %14s %12s %14s\n",
           "stages", "pdl_off(ms)", "pdl_on(ms)", "speedup", "implied_depth");

    for (int n = 1; n <= STAGES; ++n) {
        double t[2];
        for (int pdl = 0; pdl <= 1; ++pdl) {
            std::vector<float> v;
            for (int r = 0; r < repeats + 3; ++r) {
                CUDA_CHECK(cudaMemsetAsync(d_buf, 0, (size_t)blocks * sizeof(float), s));
                CUDA_CHECK(cudaMemsetAsync(tb.d_recs, 0,
                            (size_t)tb.total * sizeof(CtaRecord), s));
                CUDA_CHECK(cudaEventRecord(e0, s));
                for (int st = 0; st < n; ++st) {
                    CtaTrace tr = ctatrace_slice(tb, st);
                    if (pdl) {
                        cudaLaunchAttribute a{};
                        a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
                        a.val.programmaticStreamSerializationAllowed = 1;
                        cudaLaunchConfig_t lc{};
                        lc.gridDim = dim3(blocks); lc.blockDim = dim3(threads);
                        lc.stream = s; lc.attrs = &a; lc.numAttrs = 1;
                        CUDA_CHECK(cudaLaunchKernelEx(&lc, chainK, d_buf, st, work, tr, 1));
                    } else {
                        chainK<<<blocks, threads, 0, s>>>(d_buf, st, work, tr, 0);
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                CUDA_CHECK(cudaEventRecord(e1, s));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (r >= 3) v.push_back(ms);
            }
            t[pdl] = medianOf(v);
        }
        double sp = t[1] > 0 ? t[0] / t[1] : 0.0;
        // With perfect pairwise overlap an n-stage chain costs (n+1)/2 stage-times, so the
        // speedup saturates at 2n/(n+1). Inverting gives the effective overlap depth.
        double implied = (sp < 2.0) ? sp / (2.0 - sp) : (double)n;
        printf("%-8d %14.4f %14.4f %12.3f %14.2f\n", n, t[0], t[1], sp, implied);
        printf("SUMMARY tier0=chain stages=%d pdl_off_ms=%.5f pdl_on_ms=%.5f speedup=%.4f implied_depth=%.3f\n",
               n, t[0], t[1], sp, implied);
    }

    ctatrace_dump_csv(tb, "tier0_chain_trace.csv", "chain6");
    printf("  [trace] tier0_chain_trace.csv (use tools/cta_timeline.py to see true overlap)\n");

    ctatrace_free(&tb);
    cudaFree(d_buf);
    CUDA_CHECK(cudaStreamDestroy(s));
}

// ---------------------------------------------------------------- 0.3 occupancy cost of waiting

// A CTA that simply occupies its slot for `wait` cycles, holding `smem` shared memory.
// Models a consumer CTA that is resident but blocked on a dependency.
extern __shared__ char g_smem[];
__global__ void waiterK(float* __restrict__ sink, unsigned long long wait) {
    if (threadIdx.x == 0) g_smem[0] = (char)blockIdx.x;   // touch smem so it is not elided
    spin_cycles(wait);
    if (threadIdx.x == 0) sink[blockIdx.x] = (float)g_smem[0];
}

static void occupancyCost(const DeviceInfo& dev, int repeats) {
    printf("\n=== 0.3 occupancy cost of a resident-but-waiting CTA ===\n");
    printf("This is the pricing basis for B2: how much does 'wait while holding a slot' cost,\n");
    printf("as a function of the resources the waiting CTA holds.\n\n");
    printf("%-10s %-10s %14s %16s %14s\n",
           "smem_kb", "threads", "ctas_per_sm", "concurrent_ctas", "median(ms)");

    const int smemKB[]   = {0, 8, 16, 32, 64};
    const int threadSet[] = {128, 256};
    const unsigned long long wait = 1000000ull;

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, (size_t)(dev.sms * 64) * sizeof(float)));

    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    for (int ti = 0; ti < 2; ++ti) {
        for (int si = 0; si < 5; ++si) {
            int threads = threadSet[ti];
            size_t smem = (size_t)smemKB[si] * 1024;

            int occ = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)waiterK,
                                                          threads, smem);
            if (occ <= 0) {
                printf("%-10d %-10d %14s %16s %14s\n", smemKB[si], threads, "n/a", "n/a", "skip");
                continue;
            }
            int blocks = occ * dev.sms;   // exactly fill the machine

            std::vector<float> v;
            for (int r = 0; r < repeats + 3; ++r) {
                CUDA_CHECK(cudaEventRecord(e0, s));
                waiterK<<<blocks, threads, smem, s>>>(d_sink, wait);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(e1, s));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                if (r >= 3) v.push_back(ms);
            }
            double med = medianOf(v);
            printf("%-10d %-10d %14d %16d %14.4f\n", smemKB[si], threads, occ, blocks, med);
            printf("SUMMARY tier0=occupancy smem_kb=%d threads=%d ctas_per_sm=%d concurrent_ctas=%d median_ms=%.5f\n",
                   smemKB[si], threads, occ, blocks, med);
        }
    }
    cudaFree(d_sink);
    CUDA_CHECK(cudaStreamDestroy(s));
}

// ---------------------------------------------------------------- 0.5 fence scope cost

__global__ void fenceK(float* __restrict__ buf, int scope, int iters) {
    float acc = 0.f;
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x == 0) buf[blockIdx.x] = (float)i;
        switch (scope) {
            case 0: __threadfence_block(); break;   // .cta
            case 1: __threadfence();       break;   // .gpu
            case 2: __threadfence_system();break;   // .sys
            default: break;                         // none
        }
        acc += buf[blockIdx.x];
    }
    if (threadIdx.x == 0 && acc == 1e30f) buf[blockIdx.x] = acc;   // keep the loop alive
}

static void fenceCost(const DeviceInfo& dev, int repeats) {
    printf("\n=== 0.5 fence scope cost ===\n");
    const char* names[] = {"cta", "gpu", "sys", "none"};
    const int iters = 2000;
    const int blocks = dev.sms, threads = 128;

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, (size_t)blocks * sizeof(float)));
    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    double base = 0.0;
    printf("%-8s %14s %16s\n", "scope", "median(ms)", "ns_per_fence");
    for (int sc = 3; sc >= 0; --sc) {   // 'none' first to get the baseline
        std::vector<float> v;
        for (int r = 0; r < repeats + 3; ++r) {
            CUDA_CHECK(cudaEventRecord(e0, s));
            fenceK<<<blocks, threads, 0, s>>>(d_buf, sc, iters);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(e1, s));
            CUDA_CHECK(cudaEventSynchronize(e1));
            float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            if (r >= 3) v.push_back(ms);
        }
        double med = medianOf(v);
        if (sc == 3) base = med;
        double nsPer = (med - base) * 1e6 / iters;
        printf("%-8s %14.4f %16.2f\n", names[sc], med, sc == 3 ? 0.0 : nsPer);
        printf("SUMMARY tier0=fence scope=%s median_ms=%.5f ns_per_fence=%.3f\n",
               names[sc], med, sc == 3 ? 0.0 : nsPer);
    }
    cudaFree(d_buf);
    CUDA_CHECK(cudaStreamDestroy(s));
}

int main(int argc, char** argv) {
    Args A(argc, argv);
    if (A.has("--help")) {
        printf("usage: tier0_facts [--repeats N]\n"
               "  Runs Tier 0 base-fact probes 0.1 / 0.3 / 0.5.\n"
               "  0.2 = re-run 跨stream_PDL调研/bench/pdl_bench on this device.\n"
               "  0.4 = ./clc_probe (needs sm_100+).\n");
        return 0;
    }
    int repeats = (int)A.ll("--repeats", 15);

    DeviceInfo dev = queryDevice();
    printDeviceBanner(dev);
    printf("SUMMARY tier0=device name=\"%s\" sms=%d cc=%d.%d ghz=%.3f\n",
           dev.name, dev.sms, dev.major, dev.minor, dev.ghz);

    chainOverlap(dev, repeats);
    occupancyCost(dev, repeats);
    fenceCost(dev, repeats);

    printf("\nDone. Feed the SUMMARY lines to tools/parse_summary.py.\n");
    return 0;
}
