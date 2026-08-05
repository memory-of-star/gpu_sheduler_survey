// cta_dep_pilot.cu — corrected, bounded CTA-level PDL screening pilot.
//
// This pilot exists because cta_dep_bench publishes every done[] flag before
// the PDL trigger.  Since a dependent grid becomes eligible only after every
// producer CTA triggers, those waits are all already satisfied and the claimed
// CTA-granular benefit map degenerates into protocol overhead.
//
// Corrected multi-wave execution:
//   Floor: producer --programmatic CUDA-Graph edge--> consumer/grid wait.  Each producer
//          CTA triggers only after its datum is ready.
//   Impl/Ceiling: producer and consumer use independent priority streams.  Every consumer
//          reserves 64 KiB dynamic shared memory, which leaves a checked producer resource
//          slot on each SM. Software modes release/acquire done[cta]; Ceiling never waits.
//
// Timed epilogue work is O(1) in dependency degree.  A separate, untimed
// validation launch checks every exact parent flag and datum immediately after
// each supposedly-correct wait.  The unsound global completion-counter protocol
// is deliberately excluded.
//
// Every CTA records %globaltimer entry/end stamps. A nominal P,C>SM sample is accepted only
// when at least one consumer started before a producer CTA started, every producer completed,
// the Floor graph launched during producer tails, and full-edge validation passed. CUDA does
// not guarantee fair concurrent-kernel scheduling, so resource arithmetic alone never opens
// the gate; an observed timeline is mandatory and failures are non-zero/fail-closed.

#include "common/bench_util.cuh"
#include "common/cta_trace.cuh"
#include "common/dep_pattern.cuh"

#include <cuda/atomic>
#include <chrono>
#include <limits>
#include <thread>

enum PilotWait {
    PILOT_NONE = 0,
    PILOT_GRID = 1,
    PILOT_INTERVAL_SPIN = 2,
    PILOT_INTERVAL_BACKOFF = 3,
    PILOT_EXACT_BACKOFF = 4,
    PILOT_NMODES = 5,
};

static const char* pilotWaitName(int mode) {
    switch (mode) {
        case PILOT_NONE:             return "none";
        case PILOT_GRID:             return "grid";
        case PILOT_INTERVAL_SPIN:    return "interval-spin";
        case PILOT_INTERVAL_BACKOFF: return "interval-backoff";
        case PILOT_EXACT_BACKOFF:    return "exact-backoff";
        default:                     return "?";
    }
}

__device__ __forceinline__ void pilot_wait_one(const int* done, int parent,
                                                bool backoff) {
    cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[parent]);
    unsigned int ns = 32;
    while (flag.load(cuda::memory_order_acquire) == 0) {
        if (backoff) {
            __nanosleep(ns);
            ns = ns < 1024 ? ns * 2 : 1024;
        }
    }
}

__device__ __forceinline__ void pilot_wait_interval(const int* done,
                                                     const DepPattern& pat,
                                                     int child,
                                                     bool backoff) {
    if (threadIdx.x == 0) {
        int lo, hi;
        dep_interval(pat, child, &lo, &hi);
        for (int p = lo; p <= hi; ++p) pilot_wait_one(done, p, backoff);
    }
}

__device__ __forceinline__ void pilot_wait_exact(const int* done,
                                                  const DepPattern& pat,
                                                  int child) {
    if (threadIdx.x == 0) {
        int degree = dep_degree(pat, child);
        for (int k = 0; k < degree; ++k) {
            int p = dep_parent(pat, child, k);
            if (p >= 0) pilot_wait_one(done, p, true);
        }
    }
}

__global__ void pilotProducer(float* data,
                              int* done,
                              unsigned long long* t_start,
                              unsigned long long* t_ready,
                              unsigned long long* t_end,
                              int wait_mode,
                              unsigned long long ready_cycles,
                              unsigned long long tail_cycles,
                              int skew_bins,
                              unsigned int seed) {
    int cta = (int)blockIdx.x;
    if (threadIdx.x == 0) t_start[cta] = ctatrace_globaltimer();
    __syncthreads();

    // The independent-stream modes do not rely on the PDL signal, but execute it at entry
    // to preserve the declared software coordinate. The production Floor delays it until
    // this CTA's datum is genuinely ready.
    if (wait_mode != PILOT_GRID) cudaTriggerProgrammaticLaunchCompletion();

    unsigned int h = dep_hash((unsigned int)cta ^ seed);
    unsigned int bucket = skew_bins > 1 ? h % (unsigned int)skew_bins : 0u;
    unsigned long long delay = ready_cycles;
    if (skew_bins > 1)
        delay += (ready_cycles * (unsigned long long)bucket) /
                 (unsigned long long)skew_bins;
    spin_cycles(delay);

    if (threadIdx.x == 0) data[cta] = (float)cta * 2.0f + 1.0f;
    __syncthreads();

    bool software_wait = wait_mode == PILOT_INTERVAL_SPIN ||
                         wait_mode == PILOT_INTERVAL_BACKOFF ||
                         wait_mode == PILOT_EXACT_BACKOFF;
    if (software_wait && threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> flag(done[cta]);
        flag.store(1, cuda::memory_order_release);
    }
    if (threadIdx.x == 0) t_ready[cta] = ctatrace_globaltimer();
    __syncthreads();
    if (wait_mode == PILOT_GRID) cudaTriggerProgrammaticLaunchCompletion();

    // Work independent of the published datum.  CTA-level consumers may run
    // dependent work during this tail; griddepcontrol.wait cannot.
    spin_cycles(tail_cycles);
    __syncthreads();
    if (threadIdx.x == 0) t_end[cta] = ctatrace_globaltimer();
}

__global__ void pilotConsumer(float* out,
                              const float* data,
                              const int* done,
                              unsigned long long* t_start,
                              unsigned long long* t_dep,
                              unsigned long long* t_end,
                              DepPattern pat,
                              int wait_mode,
                              unsigned long long prologue_cycles,
                              unsigned long long epilogue_cycles,
                              int validate,
                              int* error,
                              unsigned int* stale) {
    extern __shared__ unsigned char pilot_consumer_smem[];
    int child = (int)blockIdx.x;
    if (threadIdx.x == 0) {
        pilot_consumer_smem[0] = (unsigned char)child;
        t_start[child] = ctatrace_globaltimer();
    }
    __syncthreads();
    spin_cycles(prologue_cycles);

    switch (wait_mode) {
        case PILOT_NONE:
            break;
        case PILOT_GRID:
            if (threadIdx.x == 0) cudaGridDependencySynchronize();
            break;
        case PILOT_INTERVAL_SPIN:
            pilot_wait_interval(done, pat, child, false);
            break;
        case PILOT_INTERVAL_BACKOFF:
            pilot_wait_interval(done, pat, child, true);
            break;
        case PILOT_EXACT_BACKOFF:
            pilot_wait_exact(done, pat, child);
            break;
    }
    __syncthreads();
    if (threadIdx.x == 0) t_dep[child] = ctatrace_globaltimer();

    // Validation is intentionally outside timed samples.  It checks every true
    // dependency without adding O(degree) ordinary work to the benefit sweep.
    if (validate && threadIdx.x == 0) {
        int degree = dep_degree(pat, child);
        for (int k = 0; k < degree; ++k) {
            int p = dep_parent(pat, child, k);
            if (p < 0) continue;
            bool software_wait = wait_mode == PILOT_INTERVAL_SPIN ||
                                 wait_mode == PILOT_INTERVAL_BACKOFF ||
                                 wait_mode == PILOT_EXACT_BACKOFF;
            int ready = 1;
            if (software_wait) {
                cuda::atomic_ref<const int, cuda::thread_scope_device> flag(done[p]);
                ready = flag.load(cuda::memory_order_acquire);
            }
            float expected = (float)p * 2.0f + 1.0f;
            if (!ready || data[p] != expected) atomicExch(error, 1);
        }
    }
    __syncthreads();

    // Constant timed post-wait memory work: one representative parent read,
    // independent of degree. Capture it immediately, so Ceiling records poison/stale data
    // instead of accidentally becoming correct while its epilogue runs.
    if (threadIdx.x == 0 && child < pat.n_consumer) {
        int degree = dep_degree(pat, child);
        int p = degree > 0 ? dep_parent(pat, child, degree - 1) : -1;
        float observed = p >= 0 ? data[p] : 0.0f;
        out[child] = observed;
        if (wait_mode == PILOT_NONE && p >= 0) {
            float expected = (float)p * 2.0f + 1.0f;
            if (observed != expected) atomicAdd(stale, 1u);
        }
    }

    spin_cycles(epilogue_cycles);
    __syncthreads();
    if (threadIdx.x == 0) t_end[child] = ctatrace_globaltimer();
}

struct PilotCfg {
    int nproducer = 1024;
    int nconsumer = 1024;
    int structure = DEP_INTERVAL;
    int degree = 1;
    int threads = 128;
    int repeats = 31;
    int skew_bins = 8;
    int consumer_smem_kb = 64;
    unsigned int seed = 12345u;
    unsigned long long ready = 400000ull;
    unsigned long long tail = 1000000ull;
    unsigned long long prologue = 200000ull;
    unsigned long long epilogue = 1000000ull;
    const char* tag = "pilot";
};

struct PilotCtx {
    float* data = nullptr;
    float* out = nullptr;
    int* done = nullptr;
    int* error = nullptr;
    unsigned int* stale = nullptr;
    unsigned long long *p_start = nullptr, *p_ready = nullptr, *p_end = nullptr;
    unsigned long long *c_start = nullptr, *c_dep = nullptr, *c_end = nullptr;
    cudaStream_t graph_stream{}, producer_stream{}, consumer_stream{};
    cudaEvent_t begin{}, graph_end{}, producer_end{}, consumer_end{};
};

struct PilotGraph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
};

static PilotGraph buildFloorGraph(const PilotCfg& cfg,
                                  const DepPattern& pat,
                                  PilotCtx& ctx,
                                  bool validate) {
    PilotGraph built;
    CUDA_CHECK(cudaGraphCreate(&built.graph, 0));
    int mode = PILOT_GRID;
    void* producer_args[] = {
        &ctx.data, &ctx.done, &ctx.p_start, &ctx.p_ready, &ctx.p_end,
        &mode, const_cast<unsigned long long*>(&cfg.ready),
        const_cast<unsigned long long*>(&cfg.tail),
        const_cast<int*>(&cfg.skew_bins), const_cast<unsigned int*>(&cfg.seed),
    };
    cudaKernelNodeParams producer_params{};
    producer_params.func = (void*)pilotProducer;
    producer_params.gridDim = dim3(cfg.nproducer);
    producer_params.blockDim = dim3(cfg.threads);
    producer_params.kernelParams = producer_args;
    cudaGraphNode_t producer_node{};
    CUDA_CHECK(cudaGraphAddKernelNode(&producer_node, built.graph, nullptr, 0,
                                      &producer_params));

    int validate_arg = validate ? 1 : 0;
    DepPattern pat_arg = pat;
    void* consumer_args[] = {
        &ctx.out, &ctx.data, &ctx.done, &ctx.c_start, &ctx.c_dep, &ctx.c_end,
        &pat_arg, &mode, const_cast<unsigned long long*>(&cfg.prologue),
        const_cast<unsigned long long*>(&cfg.epilogue), &validate_arg,
        &ctx.error, &ctx.stale,
    };
    cudaKernelNodeParams consumer_params{};
    consumer_params.func = (void*)pilotConsumer;
    consumer_params.gridDim = dim3(cfg.nconsumer);
    consumer_params.blockDim = dim3(cfg.threads);
    consumer_params.sharedMemBytes = (unsigned)cfg.consumer_smem_kb * 1024u;
    consumer_params.kernelParams = consumer_args;
    cudaGraphNode_t consumer_node{};
    CUDA_CHECK(cudaGraphAddKernelNode(&consumer_node, built.graph, nullptr, 0,
                                      &consumer_params));

    cudaGraphEdgeData edge{};
    edge.type = cudaGraphDependencyTypeProgrammatic;
    edge.from_port = cudaGraphKernelNodePortProgrammatic;
    CUDA_CHECK(cudaGraphAddDependencies(built.graph, &producer_node,
                                        &consumer_node, &edge, 1));
    CUDA_CHECK(cudaGraphInstantiate(&built.exec, built.graph, 0));
    return built;
}

static void destroyPilotGraph(PilotGraph& graph) {
    if (graph.exec) cudaGraphExecDestroy(graph.exec);
    if (graph.graph) cudaGraphDestroy(graph.graph);
    graph.exec = nullptr;
    graph.graph = nullptr;
}

using PilotDeadline = std::chrono::steady_clock::time_point;

static bool waitEventUntil(cudaEvent_t event, const PilotDeadline& deadline) {
    for (;;) {
        cudaError_t status = cudaEventQuery(event);
        if (status == cudaSuccess) return true;
        if (status != cudaErrorNotReady) {
            fprintf(stderr, "event query failed: %s\n", cudaGetErrorString(status));
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static bool waitEventBounded(cudaEvent_t event, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    return waitEventUntil(event, deadline);
}

static bool waitEventsBounded(cudaEvent_t first, cudaEvent_t second, int timeout_ms) {
    // Both streams consume the same wall-clock budget.  Giving each event a fresh
    // deadline would silently turn a documented 10 s/sample watchdog into 20 s.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    return waitEventUntil(first, deadline) && waitEventUntil(second, deadline);
}

static void resetPilot(const PilotCfg& cfg, PilotCtx& ctx) {
    // cudaMemset may return before the device-side clear completes.  All launch streams
    // below are cudaStreamNonBlocking, so they do not inherit legacy-stream ordering.
    // Without a post-reset barrier, a late timestamp memset can race a newly launched
    // kernel and overwrite a valid slot with zero (the first formal campaign's rare
    // trace-incomplete failure).  Both barriers are outside the timed interval.
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx.data, 0xff, (size_t)cfg.nproducer * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx.out, 0xff, (size_t)cfg.nconsumer * sizeof(float)));
    CUDA_CHECK(cudaMemset(ctx.done, 0, (size_t)cfg.nproducer * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx.error, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx.stale, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(ctx.p_start, 0,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.p_ready, 0,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.p_end, 0,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.c_start, 0,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.c_dep, 0,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(ctx.c_end, 0,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct PilotEvidence {
    float ms = 0.0f;              // %globaltimer makespan, authoritative sample
    float event_ms = 0.0f;        // launch-path diagnostic only
    unsigned producers_unstarted = 0;
    unsigned producers_unfinished = 0;
    unsigned stale_outputs = 0;
    unsigned missing_p_start = 0;
    unsigned missing_p_ready = 0;
    unsigned missing_p_end = 0;
    unsigned missing_c_start = 0;
    unsigned missing_c_dep = 0;
    unsigned missing_c_end = 0;
    unsigned invalid_p_order = 0;
    unsigned invalid_c_order = 0;
    unsigned trace_attempts = 1;
    bool trace_complete = false;
};

static PilotEvidence collectPilotEvidence(const PilotCfg& cfg, PilotCtx& ctx) {
    std::vector<unsigned long long> ps((size_t)cfg.nproducer),
                                    pr((size_t)cfg.nproducer),
                                    pe((size_t)cfg.nproducer);
    std::vector<unsigned long long> cs((size_t)cfg.nconsumer),
                                    cd((size_t)cfg.nconsumer),
                                    ce((size_t)cfg.nconsumer);
    CUDA_CHECK(cudaMemcpy(ps.data(), ctx.p_start,
                          ps.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pr.data(), ctx.p_ready,
                          pr.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pe.data(), ctx.p_end,
                          pe.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cs.data(), ctx.c_start,
                          cs.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cd.data(), ctx.c_dep,
                          cd.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ce.data(), ctx.c_end,
                          ce.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    unsigned long long first = ~0ull, last = 0, first_consumer = ~0ull;
    PilotEvidence evidence;
    for (int i = 0; i < cfg.nproducer; ++i) {
        if (ps[(size_t)i] == 0) {
            ++evidence.missing_p_start;
        } else if (ps[(size_t)i] < first) {
            first = ps[(size_t)i];
        }
        if (pr[(size_t)i] == 0) ++evidence.missing_p_ready;
        if (pe[(size_t)i] == 0) {
            ++evidence.missing_p_end;
        } else if (pe[(size_t)i] > last) {
            last = pe[(size_t)i];
        }
        if (ps[(size_t)i] != 0 && pr[(size_t)i] != 0 && pe[(size_t)i] != 0 &&
            !(ps[(size_t)i] <= pr[(size_t)i] && pr[(size_t)i] <= pe[(size_t)i])) {
            ++evidence.invalid_p_order;
        }
    }
    for (int i = 0; i < cfg.nconsumer; ++i) {
        if (cs[(size_t)i] == 0) {
            ++evidence.missing_c_start;
        } else {
            if (cs[(size_t)i] < first) first = cs[(size_t)i];
            if (cs[(size_t)i] < first_consumer) first_consumer = cs[(size_t)i];
        }
        if (cd[(size_t)i] == 0) ++evidence.missing_c_dep;
        if (ce[(size_t)i] == 0) {
            ++evidence.missing_c_end;
        } else if (ce[(size_t)i] > last) {
            last = ce[(size_t)i];
        }
        if (cs[(size_t)i] != 0 && cd[(size_t)i] != 0 && ce[(size_t)i] != 0 &&
            !(cs[(size_t)i] <= cd[(size_t)i] && cd[(size_t)i] <= ce[(size_t)i])) {
            ++evidence.invalid_c_order;
        }
    }

    if (first_consumer != ~0ull) {
        for (int i = 0; i < cfg.nproducer; ++i) {
            if (ps[(size_t)i] > first_consumer) ++evidence.producers_unstarted;
            if (pe[(size_t)i] > first_consumer) ++evidence.producers_unfinished;
        }
    }
    evidence.trace_complete =
        evidence.missing_p_start == 0 && evidence.missing_p_ready == 0 &&
        evidence.missing_p_end == 0 && evidence.missing_c_start == 0 &&
        evidence.missing_c_dep == 0 && evidence.missing_c_end == 0 &&
        evidence.invalid_p_order == 0 && evidence.invalid_c_order == 0 &&
        first != ~0ull && first_consumer != ~0ull;
    evidence.ms = evidence.trace_complete ? (float)((double)(last - first) / 1.0e6) : 0.0f;
    CUDA_CHECK(cudaMemcpy(&evidence.stale_outputs, ctx.stale,
                          sizeof(evidence.stale_outputs), cudaMemcpyDeviceToHost));
    return evidence;
}

static PilotEvidence pilotOnce(const PilotCfg& cfg,
                               const DepPattern& pat,
                               PilotCtx& ctx,
                               const PilotGraph& floor_graph,
                               int mode) {
    resetPilot(cfg, ctx);
    if (mode == PILOT_GRID) {
        CUDA_CHECK(cudaEventRecord(ctx.begin, ctx.graph_stream));
        CUDA_CHECK(cudaGraphLaunch(floor_graph.exec, ctx.graph_stream));
        CUDA_CHECK(cudaEventRecord(ctx.graph_end, ctx.graph_stream));
        if (!waitEventBounded(ctx.graph_end, 10000)) {
            fprintf(stderr, "Floor graph timeout; refusing a hung/incomplete sample\n");
            exit(124);
        }
        // A successful event query proves execution reached the marker.  Make the
        // stream-completion / host-visibility boundary explicit before copying the six
        // timestamp arrays; rare zero slots observed in the first formal run occurred
        // after event query success, not after a timeout.
        CUDA_CHECK(cudaStreamSynchronize(ctx.graph_stream));
        PilotEvidence evidence = collectPilotEvidence(cfg, ctx);
        CUDA_CHECK(cudaEventElapsedTime(&evidence.event_ms, ctx.begin, ctx.graph_end));
        return evidence;
    }

    CUDA_CHECK(cudaEventRecord(ctx.begin, ctx.graph_stream));
    CUDA_CHECK(cudaEventSynchronize(ctx.begin));
    const size_t smem = (size_t)cfg.consumer_smem_kb * 1024u;
    pilotConsumer<<<cfg.nconsumer, cfg.threads, smem, ctx.consumer_stream>>>(
        ctx.out, (const float*)ctx.data, (const int*)ctx.done,
        ctx.c_start, ctx.c_dep, ctx.c_end, pat, mode,
        cfg.prologue, cfg.epilogue, 0, ctx.error, ctx.stale);
    CUDA_CHECK(cudaGetLastError());
    pilotProducer<<<cfg.nproducer, cfg.threads, 0, ctx.producer_stream>>>(
        ctx.data, ctx.done, ctx.p_start, ctx.p_ready, ctx.p_end,
        mode, cfg.ready, cfg.tail, cfg.skew_bins, cfg.seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ctx.consumer_end, ctx.consumer_stream));
    CUDA_CHECK(cudaEventRecord(ctx.producer_end, ctx.producer_stream));
    if (!waitEventsBounded(ctx.consumer_end, ctx.producer_end, 10000)) {
        fprintf(stderr,
                "independent-stream timeout; refusing possible resident-wait deadlock\n");
        exit(124);
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.consumer_stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.producer_stream));
    PilotEvidence evidence = collectPilotEvidence(cfg, ctx);
    float consumer_ms = 0.0f, producer_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&consumer_ms, ctx.begin, ctx.consumer_end));
    CUDA_CHECK(cudaEventElapsedTime(&producer_ms, ctx.begin, ctx.producer_end));
    evidence.event_ms = consumer_ms > producer_ms ? consumer_ms : producer_ms;
    return evidence;
}

// A trace slot is diagnostic metadata, not the measured workload result.  The first
// formal campaign exposed rare zero slots after otherwise-normal event completion.  A
// rejected trace attempt is therefore rerun, but it is never published as SAMPLE and
// never enters a median.  Timeouts, CUDA errors, failed validation, missing overlap,
// missing Ceiling corruption, and performance outliers are deliberately not retryable.
static constexpr int PILOT_TRACE_RETRY_LIMIT = 3;
static constexpr int PILOT_TRACE_MAX_ATTEMPTS = 1 + PILOT_TRACE_RETRY_LIMIT;

static PilotEvidence pilotTraceCompleteSample(
        const PilotCfg& cfg,
        const DepPattern& pat,
        PilotCtx& ctx,
        const PilotGraph& floor_graph,
        int mode,
        int rep,
        unsigned long long* trace_retries,
        int* max_attempts_observed) {
    for (int attempt = 1; attempt <= PILOT_TRACE_MAX_ATTEMPTS; ++attempt) {
        if (attempt > 1) ++*trace_retries;
        if (attempt > *max_attempts_observed) *max_attempts_observed = attempt;
        PilotEvidence evidence = pilotOnce(cfg, pat, ctx, floor_graph, mode);
        evidence.trace_attempts = (unsigned)attempt;
        if (evidence.trace_complete) return evidence;

        printf("REJECTED_ATTEMPT tag=%s mode=%s rep=%d attempt=%d "
               "max_attempts=%d reason=trace_incomplete event_ms=%.6f "
               "missing_p_start=%u missing_p_ready=%u missing_p_end=%u "
               "missing_c_start=%u missing_c_dep=%u missing_c_end=%u "
               "invalid_p_order=%u invalid_c_order=%u "
               "producers_unstarted_at_consumer=%u "
               "producers_unfinished_at_consumer=%u stale_outputs=%u\n",
               cfg.tag, pilotWaitName(mode), rep, attempt,
               PILOT_TRACE_MAX_ATTEMPTS, evidence.event_ms,
               evidence.missing_p_start, evidence.missing_p_ready,
               evidence.missing_p_end, evidence.missing_c_start,
               evidence.missing_c_dep, evidence.missing_c_end,
               evidence.invalid_p_order, evidence.invalid_c_order,
               evidence.producers_unstarted, evidence.producers_unfinished,
               evidence.stale_outputs);
        fflush(stdout);
    }

    fprintf(stderr,
            "TRACE_RETRY_EXHAUSTED tag=%s mode=%s rep=%d max_attempts=%d; "
            "no SAMPLE emitted\n",
            cfg.tag, pilotWaitName(mode), rep, PILOT_TRACE_MAX_ATTEMPTS);
    exit(1);
}

static bool validateMode(const PilotCfg& cfg,
                         const DepPattern& pat,
                         PilotCtx& ctx,
                         const PilotGraph& floor_validation,
                         int mode) {
    if (mode == PILOT_NONE) return true;
    resetPilot(cfg, ctx);
    if (mode == PILOT_GRID) {
        CUDA_CHECK(cudaGraphLaunch(floor_validation.exec, ctx.graph_stream));
        CUDA_CHECK(cudaEventRecord(ctx.graph_end, ctx.graph_stream));
        if (!waitEventBounded(ctx.graph_end, 10000)) return false;
        CUDA_CHECK(cudaStreamSynchronize(ctx.graph_stream));
    } else {
        const size_t smem = (size_t)cfg.consumer_smem_kb * 1024u;
        pilotConsumer<<<cfg.nconsumer, cfg.threads, smem, ctx.consumer_stream>>>(
            ctx.out, (const float*)ctx.data, (const int*)ctx.done,
            ctx.c_start, ctx.c_dep, ctx.c_end, pat, mode,
            cfg.prologue, cfg.epilogue, 1, ctx.error, ctx.stale);
        CUDA_CHECK(cudaGetLastError());
        pilotProducer<<<cfg.nproducer, cfg.threads, 0, ctx.producer_stream>>>(
            ctx.data, ctx.done, ctx.p_start, ctx.p_ready, ctx.p_end,
            mode, cfg.ready, cfg.tail, cfg.skew_bins, cfg.seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ctx.consumer_end, ctx.consumer_stream));
        CUDA_CHECK(cudaEventRecord(ctx.producer_end, ctx.producer_stream));
        if (!waitEventsBounded(ctx.consumer_end, ctx.producer_end, 10000)) return false;
        CUDA_CHECK(cudaStreamSynchronize(ctx.consumer_stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.producer_stream));
    }
    int error = 1;
    CUDA_CHECK(cudaMemcpy(&error, ctx.error, sizeof(int), cudaMemcpyDeviceToHost));
    return error == 0;
}

int main(int argc, char** argv) {
    Args args(argc, argv);
    if (args.has("--help")) {
        printf(
            "usage: cta_dep_pilot [options]\n"
            "  --producers N --consumers N   (may exceed SM count; multi-wave is §5.3)\n"
            "  --structure interval|grouped|strided|random|self\n"
            "  --degree D --repeats N --threads N\n"
            "  --consumer-smem-kb N        (default/fixed campaign coordinate: 64)\n"
            "  --ready CYC --tail CYC --prologue CYC --epilogue CYC\n"
            "  --skew-bins N --seed N --tag STR\n");
        return 0;
    }

    PilotCfg cfg;
    cfg.nproducer = (int)args.ll("--producers", cfg.nproducer);
    cfg.nconsumer = (int)args.ll("--consumers", cfg.nproducer);
    cfg.degree = (int)args.ll("--degree", cfg.degree);
    cfg.threads = (int)args.ll("--threads", cfg.threads);
    cfg.repeats = (int)args.ll("--repeats", cfg.repeats);
    cfg.skew_bins = (int)args.ll("--skew-bins", cfg.skew_bins);
    cfg.consumer_smem_kb =
        (int)args.ll("--consumer-smem-kb", cfg.consumer_smem_kb);
    cfg.seed = (unsigned int)args.ll("--seed", cfg.seed);
    cfg.ready = (unsigned long long)args.ll("--ready", cfg.ready);
    cfg.tail = (unsigned long long)args.ll("--tail", cfg.tail);
    cfg.prologue = (unsigned long long)args.ll("--prologue", cfg.prologue);
    cfg.epilogue = (unsigned long long)args.ll("--epilogue", cfg.epilogue);
    cfg.tag = args.str("--tag", cfg.tag);
    cfg.structure = depStructureFromName(args.str("--structure", "interval"));

    if (cfg.nproducer <= 0 || cfg.nconsumer <= 0 || cfg.degree <= 0 ||
        cfg.threads <= 0 || cfg.threads > 1024 || cfg.repeats <= 0 ||
        cfg.skew_bins <= 0 || cfg.consumer_smem_kb <= 0 || cfg.structure < 0) {
        fprintf(stderr, "invalid pilot configuration\n");
        return 2;
    }
    if (cfg.structure == DEP_ALL || cfg.structure == DEP_NONE) {
        fprintf(stderr,
                "pilot excludes all/none structures; use interval/grouped/strided/random/self\n");
        return 2;
    }

    DeviceInfo dev = queryDevice();
    // Multi-wave (P,C > SM) is required by EXPERIMENT_PLAN.md §5.3. Trigger semantics
    // already satisfy §3.1: software modes trigger at entry then publish done[cta] only
    // after this CTA's readiness work; Floor triggers after readiness. Later producer
    // waves cannot publish early — they have not run yet. Do not re-introduce a P,C<=SM
    // cap here; that was the project's top measurement gap, not a correctness requirement.
    const int grid_max = cfg.nproducer > cfg.nconsumer ? cfg.nproducer : cfg.nconsumer;
    if (grid_max > 64 * dev.sms) {
        fprintf(stderr,
                "refusing producers/consumers=%d > 64*SM (%d); plan §5.3 tops out at 32*SM\n",
                grid_max, 64 * dev.sms);
        return 2;
    }
    const char* wave_regime = "underfilled";
    if (grid_max > dev.sms) wave_regime = "multi";
    else if (grid_max == dev.sms) wave_regime = "single_full";

    const size_t consumer_smem = (size_t)cfg.consumer_smem_kb * 1024u;
    CUDA_CHECK(cudaFuncSetAttribute(pilotConsumer,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)consumer_smem));
    int producer_occ = ctasPerSM(pilotProducer, {0, cfg.threads});
    int consumer_occ = ctasPerSM(
        pilotConsumer, {cfg.consumer_smem_kb, cfg.threads});
    cudaFuncAttributes producer_attr{}, consumer_attr{};
    CUDA_CHECK(cudaFuncGetAttributes(&producer_attr, pilotProducer));
    CUDA_CHECK(cudaFuncGetAttributes(&consumer_attr, pilotConsumer));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev.dev));
    int regs_per_sm = 0, blocks_per_sm = 0, smem_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&regs_per_sm,
                                      cudaDevAttrMaxRegistersPerMultiprocessor, dev.dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&blocks_per_sm,
                                      cudaDevAttrMaxBlocksPerMultiprocessor, dev.dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_sm,
                                      cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev.dev));
    const long long mixed_threads =
        (long long)(consumer_occ + 1) * cfg.threads;
    const long long mixed_regs =
        (long long)consumer_occ * consumer_attr.numRegs * cfg.threads +
        (long long)producer_attr.numRegs * cfg.threads;
    const long long mixed_smem =
        (long long)consumer_occ *
            ((long long)consumer_smem + consumer_attr.sharedSizeBytes) +
        producer_attr.sharedSizeBytes;
    const bool resource_envelope =
        consumer_occ >= 1 && producer_occ >= 1 &&
        consumer_occ + 1 <= blocks_per_sm &&
        mixed_threads <= prop.maxThreadsPerMultiProcessor &&
        mixed_regs <= regs_per_sm && mixed_smem <= smem_per_sm;
    if (!resource_envelope) {
        fprintf(stderr,
                "consumer resource envelope leaves no verified producer slot: "
                "consumer_occ=%d producer_occ=%d\n",
                consumer_occ, producer_occ);
        return 2;
    }
    printDeviceBanner(dev);
    DepPattern pat{cfg.structure, cfg.degree, cfg.nproducer, cfg.nconsumer, cfg.seed};
    if (!dep_parents_are_unique(pat)) {
        fprintf(stderr,
                "dependency generator produced duplicate/out-of-range parents; refusing invalid degree\n");
        return 2;
    }
    int actual_degree = dep_degree(pat, 0);
    double tightness = dep_interval_tightness(pat);
    double effdegree = dep_effective_degree(pat);
    printf("Pilot semantics=2 tag=%s structure=%s degree=%d requested_degree=%d "
           "effective_degree=%.2f "
           "tightness=%.4f grid=%d wave=%s ready=%llu tail=%llu prologue=%llu "
           "epilogue=%llu skew_bins=%d repeats=%d producer_occ=%d consumer_occ=%d "
           "consumer_smem_kb=%d producer_slot_reserved=1 timer=globaltimer "
           "floor_path=programmatic_graph noedge_path=priority_streams\n",
           cfg.tag, depStructureName(cfg.structure), actual_degree, cfg.degree, effdegree,
           tightness, cfg.nproducer, wave_regime, cfg.ready, cfg.tail, cfg.prologue,
           cfg.epilogue, cfg.skew_bins, cfg.repeats, producer_occ, consumer_occ,
           cfg.consumer_smem_kb);
    printf("RESOURCE mixed_threads=%lld/%d mixed_regs=%lld/%d mixed_smem=%lld/%d "
           "mixed_blocks=%d/%d producer_slot_reserved=1\n",
           mixed_threads, prop.maxThreadsPerMultiProcessor, mixed_regs, regs_per_sm,
           mixed_smem, smem_per_sm, consumer_occ + 1, blocks_per_sm);

    PilotCtx ctx;
    CUDA_CHECK(cudaMalloc(&ctx.data, (size_t)cfg.nproducer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.out, (size_t)cfg.nconsumer * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.done, (size_t)cfg.nproducer * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.error, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.stale, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&ctx.p_start,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.p_ready,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.p_end,
                          (size_t)cfg.nproducer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.c_start,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.c_dep,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.c_end,
                          (size_t)cfg.nconsumer * sizeof(unsigned long long)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.graph_stream, cudaStreamNonBlocking));
    int least_priority = 0, greatest_priority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.producer_stream,
                                            cudaStreamNonBlocking, greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.consumer_stream,
                                            cudaStreamNonBlocking, least_priority));
    CUDA_CHECK(cudaEventCreate(&ctx.begin));
    CUDA_CHECK(cudaEventCreate(&ctx.graph_end));
    CUDA_CHECK(cudaEventCreate(&ctx.producer_end));
    CUDA_CHECK(cudaEventCreate(&ctx.consumer_end));

    PilotGraph floor_timed = buildFloorGraph(cfg, pat, ctx, false);
    PilotGraph floor_validation = buildFloorGraph(cfg, pat, ctx, true);

    double med[PILOT_NMODES] = {};
    double low[PILOT_NMODES] = {};
    bool valid[PILOT_NMODES] = {};
    bool any_failure = false;
    std::vector<float> samples[PILOT_NMODES];
    for (int mode = 0; mode < PILOT_NMODES; ++mode)
        samples[mode].reserve((size_t)cfg.repeats);

    // Warm every path equally. Timed repeats below interleave Floor/Impl/Ceiling first so the
    // bracket points remain adjacent in one process; the two protocol controls follow.
    const int execution_order[PILOT_NMODES] = {
        PILOT_GRID, PILOT_INTERVAL_BACKOFF, PILOT_NONE,
        PILOT_INTERVAL_SPIN, PILOT_EXACT_BACKOFF,
    };
    for (int warm = 0; warm < 3; ++warm)
        for (int index = 0; index < PILOT_NMODES; ++index)
            (void)pilotOnce(cfg, pat, ctx, floor_timed, execution_order[index]);

    bool timeline_complete = true;
    bool floor_early_launch = true;
    bool ceiling_wrong = true;
    unsigned min_unstarted = std::numeric_limits<unsigned>::max();
    unsigned long long trace_retries = 0;
    int trace_max_attempts_observed = 1;
    const bool is_multiwave = cfg.nproducer > dev.sms && cfg.nconsumer > dev.sms;

    for (int rep = 0; rep < cfg.repeats; ++rep) {
        for (int index = 0; index < PILOT_NMODES; ++index) {
            // Alternate forward/reverse order to balance thermal/order bias while keeping the
            // three bracket points contiguous in either direction.
            const int ordered_index = (rep & 1) ? (PILOT_NMODES - 1 - index) : index;
            const int mode = execution_order[ordered_index];
            PilotEvidence evidence = pilotTraceCompleteSample(
                cfg, pat, ctx, floor_timed, mode, rep, &trace_retries,
                &trace_max_attempts_observed);
            samples[mode].push_back(evidence.ms);
            if (mode == PILOT_GRID) {
                if (evidence.producers_unfinished == 0) floor_early_launch = false;
            } else {
                if (evidence.producers_unstarted < min_unstarted)
                    min_unstarted = evidence.producers_unstarted;
                if (is_multiwave && evidence.producers_unstarted == 0)
                    timeline_complete = false;
            }
            if (mode == PILOT_NONE && evidence.stale_outputs == 0)
                ceiling_wrong = false;
            printf("SAMPLE tag=%s mode=%s rep=%d ms=%.6f "
                   "event_ms=%.6f producers_unstarted_at_consumer=%u "
                   "producers_unfinished_at_consumer=%u stale_outputs=%u "
                   "trace_attempts=%u trace_complete=1\n",
                   cfg.tag, pilotWaitName(mode), rep, evidence.ms, evidence.event_ms,
                   evidence.producers_unstarted, evidence.producers_unfinished,
                   evidence.stale_outputs, evidence.trace_attempts);
        }
    }

    printf("%-20s %12s %12s %10s\n", "mode", "median_ms", "min_ms", "valid");
    for (int mode = 0; mode < PILOT_NMODES; ++mode) {
        med[mode] = medianOf(samples[mode]);
        low[mode] = minOf(samples[mode]);
        valid[mode] = validateMode(cfg, pat, ctx, floor_validation, mode);
        if (mode != PILOT_NONE && !valid[mode]) any_failure = true;
        printf("%-20s %12.5f %12.5f %10s\n", pilotWaitName(mode),
               med[mode], low[mode],
               mode == PILOT_NONE ? "n/a" : (valid[mode] ? "PASS" : "FAIL"));
    }
    if (!timeline_complete || !floor_early_launch || !ceiling_wrong)
        any_failure = true;
    if (min_unstarted == std::numeric_limits<unsigned>::max()) min_unstarted = 0;
    const int multiwave_overlap_proven =
        (!is_multiwave || (timeline_complete && min_unstarted > 0)) ? 1 : 0;

    double floor_ms = med[PILOT_GRID];
    double ceiling_ms = med[PILOT_NONE];
    const int impl_mode = PILOT_INTERVAL_BACKOFF;
    double impl_ms = med[impl_mode];
    double space_pct = floor_ms > 0.0
        ? 100.0 * (floor_ms - ceiling_ms) / floor_ms : 0.0;
    double captured_pct = floor_ms > 0.0 && valid[impl_mode]
        ? 100.0 * (floor_ms - impl_ms) / floor_ms : 0.0;
    double of_space_pct = space_pct > 0.0
        ? 100.0 * captured_pct / space_pct : 0.0;
    const bool trace_gate_ok = timeline_complete && floor_early_launch &&
                               (!is_multiwave || min_unstarted > 0);
    const char* launch_gate = trace_gate_ok ? "trace_verified" : "trace_failed";

    printf("BRACKET floor=%.5f ceiling=%.5f space=%.3f%% "
           "impl=%s impl_ms=%.5f captured=%.3f%% of_space=%.1f%%\n",
           floor_ms, ceiling_ms, space_pct, pilotWaitName(impl_mode), impl_ms,
           captured_pct, of_space_pct);
    printf("SUMMARY_PILOT semantics=2 tag=%s structure=%s degree=%d requested_degree=%d "
           "eff_degree=%.2f "
           "tightness=%.4f producers=%d consumers=%d threads=%d sms=%d "
           "wave=%s launch_gate=%s multiwave_overlap_proven=%d "
           "producers_unstarted_at_consumer=%u producer_progress_complete=%d "
           "floor_early_launch_proven=%d ceiling_wrong=%d unique_parents=1 "
           "producer_occ=%d consumer_occ=%d consumer_smem_kb=%d "
           "producer_slot_reserved=1 floor_path=programmatic_graph "
           "impl_path=priority_streams ceiling_path=priority_streams timer=globaltimer "
           "trigger_floor=ready "
           "trigger_impl=entry trigger_ceiling=entry "
           "trace_retries=%llu trace_retry_limit=%d trace_max_attempts=%d "
           "trace_max_attempts_observed=%d "
           "ready=%llu tail=%llu prologue=%llu epilogue=%llu skew_bins=%d "
           "repeats=%d floor_ms=%.6f ceiling_ms=%.6f interval_spin_ms=%.6f "
           "interval_backoff_ms=%.6f exact_backoff_ms=%.6f impl=%s "
           "impl_ms=%.6f space_pct=%.4f captured_pct=%.4f "
           "of_space_pct=%.3f valid=%d\n",
           cfg.tag, depStructureName(cfg.structure), actual_degree, cfg.degree, effdegree,
           tightness, cfg.nproducer, cfg.nconsumer, cfg.threads, dev.sms,
           wave_regime, launch_gate, multiwave_overlap_proven, min_unstarted,
           timeline_complete ? 1 : 0, floor_early_launch ? 1 : 0,
           ceiling_wrong ? 1 : 0, producer_occ, consumer_occ, cfg.consumer_smem_kb,
           trace_retries, PILOT_TRACE_RETRY_LIMIT, PILOT_TRACE_MAX_ATTEMPTS,
           trace_max_attempts_observed,
           cfg.ready, cfg.tail, cfg.prologue, cfg.epilogue, cfg.skew_bins,
           cfg.repeats, floor_ms, ceiling_ms, med[PILOT_INTERVAL_SPIN],
           med[PILOT_INTERVAL_BACKOFF], med[PILOT_EXACT_BACKOFF],
           pilotWaitName(impl_mode), impl_ms, space_pct, captured_pct, of_space_pct,
           any_failure ? 0 : 1);

    destroyPilotGraph(floor_timed);
    destroyPilotGraph(floor_validation);
    cudaEventDestroy(ctx.begin);
    cudaEventDestroy(ctx.graph_end);
    cudaEventDestroy(ctx.producer_end);
    cudaEventDestroy(ctx.consumer_end);
    cudaStreamDestroy(ctx.graph_stream);
    cudaStreamDestroy(ctx.producer_stream);
    cudaStreamDestroy(ctx.consumer_stream);
    cudaFree(ctx.data);
    cudaFree(ctx.out);
    cudaFree(ctx.done);
    cudaFree(ctx.error);
    cudaFree(ctx.stale);
    cudaFree(ctx.p_start);
    cudaFree(ctx.p_ready);
    cudaFree(ctx.p_end);
    cudaFree(ctx.c_start);
    cudaFree(ctx.c_dep);
    cudaFree(ctx.c_end);
    return any_failure ? 1 : 0;
}
