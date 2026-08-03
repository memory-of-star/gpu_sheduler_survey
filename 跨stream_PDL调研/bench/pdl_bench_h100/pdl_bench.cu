// pdl_bench.cu — Confirm whether CROSS-STREAM Programmatic Dependent Launch (PDL)
// actually overlaps a producer's tail with a consumer's independent prologue on B300.
//
// Design (built so the THEORETICAL benefit is ~2x):
//   producer:  write output (tiny "body") -> trigger -> long "tail" spin
//   consumer:  long independent "prologue" spin -> wait -> read producer output (tiny "epilogue")
// With tail == prologue == T and body/epilogue ~ 0:
//   BASE (serialized dependency): ~ 2T
//   PDL  (prologue overlaps tail): ~ 1T   => ~2x speedup if PDL works.
//
// Modes measured (single dependent pair, fully synchronized between repeats so there is
// NO cross-iteration pipelining confounder — the ONLY overlap is prologue||tail):
//   BASE        cross-stream, ordinary event dependency (no early launch)    <- baseline
//   PDL_XS      cross-stream, programmatic event + cudaStreamWaitEvent       <- eager cross-stream
//   PDL_CAPTURE SAME sA/sB two-stream pattern captured (cudaStreamBeginCapture) into a graph
//   PDL_GRAPH   CUDA graph with a programmatic edge, built directly (cross-NODE PDL)
//   PDL_SS      same-stream,  programmatic stream serialization (canonical)  <- cross-check
//   CONC        no dependency (unsafe, results wrong) — pure concurrency ceiling ~1T
//
// Note: cudaLaunchAttributeProgrammaticEvent is documented as a way to express a programmatic
// dependency during STREAM CAPTURE (i.e. it becomes a graph edge). In eager stream execution
// the early-launch overlap may not engage; PDL_GRAPH tests the graph path directly.
//
// Build:  ./build.sh           (defaults to -arch=sm_103 for B300)
// Run:    ./pdl_bench --help

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); \
    exit(1);} } while(0)

// Burn a fixed number of GPU clock cycles (wall-time proportional, independent of ALU throughput).
__device__ __forceinline__ void spin_cycles(unsigned long long cyc) {
    long long start = clock64();
    while ((unsigned long long)(clock64() - start) < cyc) {
        asm volatile("" ::: "memory");   // keep the loop; no dead-code elimination
    }
}

__global__ void producer(float* pout, const float* in, int n, unsigned long long tail_cyc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pout[i] = in[i] * 2.0f;              // body: data the consumer will read
    cudaTriggerProgrammaticLaunchCompletion();      // -> griddepcontrol.launch_dependents
    spin_cycles(tail_cyc);                          // TAIL: overlappable independent work
}

__global__ void consumer(float* out, const float* pout, int n, unsigned long long prologue_cyc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    spin_cycles(prologue_cyc);                      // PROLOGUE: overlappable independent work
    cudaGridDependencySynchronize();                // -> griddepcontrol.wait (producer done+visible)
    if (i < n) out[i] = pout[i] + 1.0f;             // EPILOGUE: depends on producer output
}

enum Mode { BASE = 0, PDL_XS = 1, PDL_SS = 2, CONC = 3, PDL_GRAPH = 4, PDL_CAPTURE = 5, NMODES = 6 };
static const char* modeName(Mode m) {
    switch (m) { case BASE:return "BASE(xstream,ordinary-event)";
                 case PDL_XS:return "PDL_XS(xstream,prog-event,eager)";
                 case PDL_SS:return "PDL_SS(same-stream,serialize)";
                 case PDL_GRAPH:return "PDL_GRAPH(prog-edge,built)";
                 case PDL_CAPTURE:return "PDL_CAPTURE(sA/sB captured->graph)";
                 case CONC:return "CONC(no-dep,ceiling)";
                 default:return "?"; }
}

struct Ctx {
    float *in, *pout, *out;
    int n;
    dim3 grid, block;
    unsigned long long tail, prologue;
    cudaStream_t sA, sB;
    cudaEvent_t evStart, evStopA, evStopB;  // timing (timing enabled)
    cudaEvent_t evNorm, evProg, evJoin;     // dependency / capture-join events (timing disabled)
};

// Run ONE producer->consumer pair for the given mode, return elapsed ms (producer-start .. all-done).
static float time_pair(Mode m, const Ctx& c) {
    CUDA_CHECK(cudaEventRecord(c.evStart, c.sA));

    // ---- producer launch (always in sA) ----
    if (m == PDL_XS) {
        cudaLaunchAttribute a{};
        a.id = cudaLaunchAttributeProgrammaticEvent;
        a.val.programmaticEvent.event = c.evProg;   // fires when all blocks hit the trigger (early)
        a.val.programmaticEvent.flags = 0;
        a.val.programmaticEvent.triggerAtBlockStart = 0;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = c.grid; cfg.blockDim = c.block; cfg.stream = c.sA;
        cfg.attrs = &a; cfg.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, producer, c.pout, (const float*)c.in, c.n, c.tail));
    } else {
        producer<<<c.grid, c.block, 0, c.sA>>>(c.pout, c.in, c.n, c.tail);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- dependency wiring + consumer launch ----
    if (m == PDL_SS) {
        // Canonical same-stream PDL: consumer in sA with stream-serialization attribute.
        cudaLaunchAttribute a{};
        a.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        a.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = c.grid; cfg.blockDim = c.block; cfg.stream = c.sA;
        cfg.attrs = &a; cfg.numAttrs = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, consumer, c.out, (const float*)c.pout, c.n, c.prologue));
        CUDA_CHECK(cudaEventRecord(c.evStopA, c.sA));
        CUDA_CHECK(cudaEventSynchronize(c.evStopA));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.evStart, c.evStopA));
        return ms;
    }

    if (m == BASE) {
        // Ordinary event: completes only when producer fully finishes -> consumer starts late.
        CUDA_CHECK(cudaEventRecord(c.evNorm, c.sA));
        CUDA_CHECK(cudaStreamWaitEvent(c.sB, c.evNorm, 0));
    } else if (m == PDL_XS) {
        // Programmatic event was armed by the launch attribute; it fires EARLY at the trigger.
        CUDA_CHECK(cudaStreamWaitEvent(c.sB, c.evProg, 0));
    }
    // CONC: no wait at all (unsafe; timing ceiling only).

    consumer<<<c.grid, c.block, 0, c.sB>>>(c.out, c.pout, c.n, c.prologue);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(c.evStopB, c.sB));
    CUDA_CHECK(cudaEventRecord(c.evStopA, c.sA));   // capture producer completion too
    CUDA_CHECK(cudaEventSynchronize(c.evStopB));
    CUDA_CHECK(cudaEventSynchronize(c.evStopA));
    float msB = 0, msA = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msB, c.evStart, c.evStopB));
    CUDA_CHECK(cudaEventElapsedTime(&msA, c.evStart, c.evStopA));
    return msB > msA ? msB : msA;   // wall time until BOTH streams are done
}

// Build a CUDA graph: producer --(programmatic edge)--> consumer. This is the documented
// cross-NODE PDL path (equivalent to capturing cudaLaunchAttributeProgrammaticEvent). The
// programmatic edge lets the consumer launch early and makes the producer visible to the
// consumer's cudaGridDependencySynchronize().
static cudaGraphExec_t build_pdl_graph(Ctx& c) {
    cudaGraph_t g; CUDA_CHECK(cudaGraphCreate(&g, 0));

    void* pArgs[] = { (void*)&c.pout, (void*)&c.in, (void*)&c.n, (void*)&c.tail };
    cudaKernelNodeParams pp{};
    pp.func = (void*)producer; pp.gridDim = c.grid; pp.blockDim = c.block;
    pp.sharedMemBytes = 0; pp.kernelParams = pArgs; pp.extra = nullptr;
    cudaGraphNode_t pNode; CUDA_CHECK(cudaGraphAddKernelNode(&pNode, g, nullptr, 0, &pp));

    void* cArgs[] = { (void*)&c.out, (void*)&c.pout, (void*)&c.n, (void*)&c.prologue };
    cudaKernelNodeParams cp{};
    cp.func = (void*)consumer; cp.gridDim = c.grid; cp.blockDim = c.block;
    cp.sharedMemBytes = 0; cp.kernelParams = cArgs; cp.extra = nullptr;
    cudaGraphNode_t cNode; CUDA_CHECK(cudaGraphAddKernelNode(&cNode, g, nullptr, 0, &cp));

    cudaGraphEdgeData edge{};
    edge.type = cudaGraphDependencyTypeProgrammatic;
    edge.from_port = cudaGraphKernelNodePortProgrammatic;   // fires at griddepcontrol.launch_dependents
    CUDA_CHECK(cudaGraphAddDependencies(g, &pNode, &cNode, &edge, 1));

    cudaGraphExec_t exec; CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return exec;
}

static float time_graph(cudaGraphExec_t exec, const Ctx& c) {
    CUDA_CHECK(cudaEventRecord(c.evStart, c.sA));
    CUDA_CHECK(cudaGraphLaunch(exec, c.sA));
    CUDA_CHECK(cudaEventRecord(c.evStopA, c.sA));
    CUDA_CHECK(cudaEventSynchronize(c.evStopA));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.evStart, c.evStopA));
    return ms;
}

// Capture the LITERAL cross-stream pattern into a graph:
//   producer in sA (programmatic event) --> sB waits that event --> consumer in sB --> join back to sA
// Per the CUDA docs, the (programmatic event + cudaStreamWaitEvent) pattern is captured as a
// programmatic graph edge. This proves the SAME two-stream code that shows 1x in eager mode
// yields the overlap once it is captured into a graph.
static cudaGraphExec_t build_pdl_capture(Ctx& c) {
    CUDA_CHECK(cudaStreamBeginCapture(c.sA, cudaStreamCaptureModeThreadLocal));

    cudaLaunchAttribute a{};
    a.id = cudaLaunchAttributeProgrammaticEvent;
    a.val.programmaticEvent.event = c.evProg;
    a.val.programmaticEvent.flags = 0;
    a.val.programmaticEvent.triggerAtBlockStart = 0;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = c.grid; cfg.blockDim = c.block; cfg.stream = c.sA;
    cfg.attrs = &a; cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, producer, c.pout, (const float*)c.in, c.n, c.tail));

    // sB forks into the capture by waiting on the producer's programmatic event.
    CUDA_CHECK(cudaStreamWaitEvent(c.sB, c.evProg, 0));
    consumer<<<c.grid, c.block, 0, c.sB>>>(c.out, c.pout, c.n, c.prologue);
    CUDA_CHECK(cudaGetLastError());

    // Join sB back to the origin stream sA so capture ends cleanly.
    CUDA_CHECK(cudaEventRecord(c.evJoin, c.sB));
    CUDA_CHECK(cudaStreamWaitEvent(c.sA, c.evJoin, 0));

    cudaGraph_t g; CUDA_CHECK(cudaStreamEndCapture(c.sA, &g));
    cudaGraphExec_t exec; CUDA_CHECK(cudaGraphInstantiate(&exec, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
    return exec;
}

static bool verify(const Ctx& c, std::vector<float>& hostIn) {
    std::vector<float> hostOut(c.n);
    CUDA_CHECK(cudaMemcpy(hostOut.data(), c.out, c.n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < c.n; ++i) {
        float expect = hostIn[i] * 2.0f + 1.0f;
        if (fabsf(hostOut[i] - expect) > 1e-3f) {
            fprintf(stderr, "  verify FAIL at %d: got %f expect %f\n", i, hostOut[i], expect);
            return false;
        }
    }
    return true;
}

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size()/2];
}

int main(int argc, char** argv) {
    long long repeats = 50, tail = 20000000, prologue = -1, blocks = -1, threads = 128;
    for (int i = 1; i < argc; ++i) {
        auto next = [&](long long& dst){ if (i+1<argc) dst = atoll(argv[++i]); };
        if      (!strcmp(argv[i], "--repeats"))  next(repeats);
        else if (!strcmp(argv[i], "--tail"))     next(tail);
        else if (!strcmp(argv[i], "--prologue")) next(prologue);
        else if (!strcmp(argv[i], "--blocks"))   next(blocks);
        else if (!strcmp(argv[i], "--threads"))  next(threads);
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--repeats N] [--tail CYC] [--prologue CYC] [--blocks B] [--threads T]\n"
                   "  tail/prologue are GPU clock cycles to spin (default tail=20e6, prologue=tail)\n"
                   "  blocks default = #SMs (one wave), threads default = 128\n", argv[0]);
            return 0;
        }
    }
    if (prologue < 0) prologue = tail;

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int clockKHz = 0;                                  // cudaDeviceProp::clockRate removed in CUDA 13
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev);
    if (blocks  < 0) blocks  = prop.multiProcessorCount;
    if (threads < 0) threads = 128;

    Ctx c{};
    c.grid = dim3((unsigned)blocks); c.block = dim3((unsigned)threads);
    c.n = (int)(blocks * threads);
    c.tail = (unsigned long long)tail; c.prologue = (unsigned long long)prologue;

    double ghz = clockKHz / 1e6;                       // clockKHz is kHz
    double tail_ms = ghz > 0 ? (double)tail / (ghz * 1e6) : 0.0;
    double prol_ms = ghz > 0 ? (double)prologue / (ghz * 1e6) : 0.0;

    printf("Device: %s  | SMs=%d  | SM clock~%.2f GHz  | CC %d.%d\n",
           prop.name, prop.multiProcessorCount, ghz, prop.major, prop.minor);
    printf("Config: blocks=%lld threads=%lld (n=%d) | tail=%lld cyc (~%.2f ms) | prologue=%lld cyc (~%.2f ms) | repeats=%lld\n\n",
           blocks, threads, c.n, tail, tail_ms, prologue, prol_ms, repeats);
    if (prop.major < 9)
        printf("WARNING: PDL needs compute capability >= 9.0; this device is %d.%d\n\n", prop.major, prop.minor);

    std::vector<float> hostIn(c.n);
    for (int i = 0; i < c.n; ++i) hostIn[i] = (float)(i % 97) * 0.5f + 1.0f;
    CUDA_CHECK(cudaMalloc(&c.in,   c.n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.pout, c.n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.out,  c.n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.in, hostIn.data(), c.n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaStreamCreateWithFlags(&c.sA, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&c.sB, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&c.evStart));
    CUDA_CHECK(cudaEventCreate(&c.evStopA));
    CUDA_CHECK(cudaEventCreate(&c.evStopB));
    CUDA_CHECK(cudaEventCreateWithFlags(&c.evNorm, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&c.evProg, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&c.evJoin, cudaEventDisableTiming));

    double med[NMODES] = {0}, best[NMODES] = {0};
    bool   ok[NMODES];
    for (int i = 0; i < NMODES; ++i) ok[i] = true;

    // ---- stream-based modes ----
    Mode streamModes[] = { BASE, PDL_XS, PDL_SS, CONC };
    for (Mode m : streamModes) {
        for (int w = 0; w < 5; ++w) (void)time_pair(m, c);   // warmup
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (long long r = 0; r < repeats; ++r) { t.push_back(time_pair(m, c)); CUDA_CHECK(cudaDeviceSynchronize()); }
        med[m]  = median(t);
        best[m] = *std::min_element(t.begin(), t.end());
        if (m != CONC) ok[m] = verify(c, hostIn);
    }

    // ---- CUDA graph programmatic-edge mode (cross-node PDL) ----
    {
        cudaGraphExec_t exec = build_pdl_graph(c);
        for (int w = 0; w < 5; ++w) (void)time_graph(exec, c);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (long long r = 0; r < repeats; ++r) { t.push_back(time_graph(exec, c)); CUDA_CHECK(cudaDeviceSynchronize()); }
        med[PDL_GRAPH]  = median(t);
        best[PDL_GRAPH] = *std::min_element(t.begin(), t.end());
        ok[PDL_GRAPH]   = verify(c, hostIn);
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    }

    // ---- LITERAL cross-stream (sA/sB) captured into a graph ----
    {
        cudaGraphExec_t exec = build_pdl_capture(c);
        for (int w = 0; w < 5; ++w) (void)time_graph(exec, c);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> t;
        for (long long r = 0; r < repeats; ++r) { t.push_back(time_graph(exec, c)); CUDA_CHECK(cudaDeviceSynchronize()); }
        med[PDL_CAPTURE]  = median(t);
        best[PDL_CAPTURE] = *std::min_element(t.begin(), t.end());
        ok[PDL_CAPTURE]   = verify(c, hostIn);
        CUDA_CHECK(cudaGraphExecDestroy(exec));
    }

    Mode order[] = { BASE, PDL_XS, PDL_CAPTURE, PDL_GRAPH, PDL_SS, CONC };
    printf("%-34s  %10s  %10s  %8s  %s\n", "mode", "median(ms)", "min(ms)", "vs BASE", "correct");
    for (Mode m : order) {
        double sp = med[m] > 0 ? med[BASE] / med[m] : 0.0;
        printf("%-34s  %10.3f  %10.3f  %7.2fx  %s\n",
               modeName(m), med[m], best[m], sp, (m==CONC?"n/a":(ok[m]?"PASS":"FAIL")));
    }

    double spXS  = med[PDL_XS]      > 0 ? med[BASE]/med[PDL_XS]      : 0.0;
    double spCap = med[PDL_CAPTURE] > 0 ? med[BASE]/med[PDL_CAPTURE] : 0.0;
    double spGR  = med[PDL_GRAPH]   > 0 ? med[BASE]/med[PDL_GRAPH]   : 0.0;
    double spSS  = med[PDL_SS]      > 0 ? med[BASE]/med[PDL_SS]      : 0.0;
    printf("\nSUMMARY tail=%lld prologue=%lld blocks=%lld threads=%lld | "
           "BASE=%.3f PDL_XS=%.3f PDL_CAPTURE=%.3f PDL_GRAPH=%.3f PDL_SS=%.3f CONC=%.3f ms | "
           "speedup_xs=%.2f speedup_capture=%.2f speedup_graph=%.2f speedup_ss=%.2f\n",
           tail, prologue, blocks, threads,
           med[BASE], med[PDL_XS], med[PDL_CAPTURE], med[PDL_GRAPH], med[PDL_SS], med[CONC],
           spXS, spCap, spGR, spSS);

    printf("VERDICT: eager cross-stream (PDL_XS)=%.2fx | SAME two streams captured into a graph "
           "(PDL_CAPTURE)=%.2fx | built graph (PDL_GRAPH)=%.2fx | same-stream (PDL_SS)=%.2fx | "
           "ceiling ~%.2fx.%s\n",
           spXS, spCap, spGR, spSS, (med[CONC]>0?med[BASE]/med[CONC]:0.0),
           ((ok[PDL_XS] && ok[PDL_CAPTURE] && ok[PDL_GRAPH]) ? "" : "  [WARNING: a PDL result was INCORRECT]"));

    cudaFree(c.in); cudaFree(c.pout); cudaFree(c.out);
    return 0;
}
